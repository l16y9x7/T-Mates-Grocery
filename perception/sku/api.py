from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import uvicorn
from fastapi import Body, FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException


PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
if str(PERCEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_ROOT))
from config import SERVICE_BIND_HOST  # noqa: E402


ROOT = Path(__file__).resolve().parent
DEFAULT_CATALOG_PATH = ROOT / "products.json"
DEFAULT_INSPECTION_CANDIDATES_PATH = ROOT / "inspection_candidates.json"
IMAGES_ROOT = (ROOT / "images_new").resolve()
LOCATION_PATTERN = re.compile(
    r"^H(?P<shelf>[1-3])_L(?P<level>0[1-5])_C(?P<column>\d{2})$"
)
INSPECTION_TARGET_PATTERN = re.compile(r"^H(?:1|12|2|23|3)_INSPECT$")
CandidatePoseType = Literal["", "SHELF_VIEW_UPPER", "SHELF_VIEW_LOWER"]
InspectionPoseType = Literal["SHELF_VIEW_UPPER", "SHELF_VIEW_LOWER"]


class HealthResponse(BaseModel):
    status: Literal["READY"]


class ProductResponse(BaseModel):
    sku_id: str
    name: str
    images: list[str]
    locations: list[str]


class SlotProductResponse(ProductResponse):
    location_id: str


class CandidateSkuRequest(BaseModel):
    location_id: str
    pose_type: CandidatePoseType


class InspectionCandidateSkuRequest(BaseModel):
    location_id: str
    pose_type: InspectionPoseType


class ErrorResponse(BaseModel):
    error_code: str


ERROR_RESPONSES = {
    400: {"model": ErrorResponse},
    404: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
}


class ApiError(Exception):
    def __init__(self, status_code: int, error_code: str) -> None:
        self.status_code = status_code
        self.error_code = error_code


class SkuCatalog:
    def __init__(self, payload: dict[str, Any], catalog_root: Path = ROOT) -> None:
        products = payload.get("products")
        if not isinstance(products, list):
            raise ValueError("products.json 中的 products 必须是数组")

        self._by_name: dict[str, dict[str, Any]] = {}
        self._by_name_alias: dict[str, dict[str, Any]] = {}
        self._by_sku: dict[str, dict[str, Any]] = {}
        self._by_location: dict[str, dict[str, Any]] = {}
        self._inspection_candidate_rows: dict[str, dict[int, tuple[str, ...]]] = {}

        for product in products:
            if not isinstance(product, dict):
                raise ValueError("products 中存在非对象元素")

            name = product.get("name")
            sku_id = product.get("sku_id")
            locations = product.get("locations")
            images = product.get("images")
            if not isinstance(sku_id, str) or not sku_id.strip():
                raise ValueError("存在无效 SKU ID")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("存在无效商品名称")
            if not isinstance(locations, list) or not locations:
                raise ValueError(f"商品 {name!r} 没有有效位置")
            if not isinstance(images, list):
                raise ValueError(f"商品 {name!r} 的 images 必须是数组")

            for image in images:
                if not isinstance(image, str) or not image:
                    raise ValueError(f"商品 {name!r} 存在无效图片路径")
                normalized_image = image.replace("\\", "/")
                image_path = PurePosixPath(normalized_image)
                if (
                    not normalized_image.startswith("images/")
                    or image_path.is_absolute()
                    or ".." in image_path.parts
                ):
                    raise ValueError(
                        f"商品 {name!r} 的图片必须是 images/ 下的相对路径"
                    )
                physical_image_path = (
                    image_path.with_suffix(".jpg")
                    if image_path.suffix.lower() == ".png"
                    else image_path
                )
                resolved_image = (catalog_root / "images_new").joinpath(
                    *physical_image_path.parts[1:]
                )
                if not resolved_image.is_file() or resolved_image.stat().st_size == 0:
                    raise ValueError(f"商品 {name!r} 的图片不存在: {image}")

            # Keep products.json unchanged while returning the actual JPEG URL.
            product["images"] = [
                str(PurePosixPath(image).with_suffix(".jpg"))
                if PurePosixPath(image).suffix.lower() == ".png"
                else image
                for image in images
            ]

            normalized_name = name.strip()
            normalized_sku = sku_id.strip().upper()
            if normalized_sku in self._by_sku:
                raise ValueError(f"SKU ID 重复: {normalized_sku}")
            if normalized_name in self._by_name:
                raise ValueError(f"商品名称重复: {normalized_name}")
            normalized_name_alias = self._name_lookup_key(normalized_name)
            if normalized_name_alias in self._by_name_alias:
                raise ValueError(f"商品名称规范化后重复: {normalized_name}")
            self._by_sku[normalized_sku] = product
            self._by_name[normalized_name] = product
            self._by_name_alias[normalized_name_alias] = product

            for location in locations:
                if not isinstance(location, str) or not location.strip():
                    raise ValueError(f"商品 {normalized_name!r} 存在无效位置")
                normalized_location = location.strip().upper()
                if normalized_location in self._by_location:
                    raise ValueError(f"位置重复: {normalized_location}")
                self._by_location[normalized_location] = product

    @classmethod
    def load(cls, path: Path) -> "SkuCatalog":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(payload, path.resolve().parent)

    def load_inspection_candidates(self, path: Path) -> None:
        """Load explicit per-view, per-shelf-level candidates for inspection."""

        payload = json.loads(path.read_text(encoding="utf-8"))
        targets = payload.get("inspection_targets")
        if not isinstance(targets, dict):
            raise ValueError(
                "inspection_candidates.json inspection_targets must be an object"
            )

        loaded: dict[str, dict[int, tuple[str, ...]]] = {}
        for raw_target, target_payload in targets.items():
            if not isinstance(raw_target, str):
                raise ValueError("inspection target ID must be a string")
            target = raw_target.strip().upper()
            if INSPECTION_TARGET_PATTERN.fullmatch(target) is None:
                raise ValueError(f"invalid inspection target ID: {raw_target!r}")
            if not isinstance(target_payload, dict):
                raise ValueError(f"inspection target {target} must be an object")
            raw_rows = target_payload.get("rows")
            if not isinstance(raw_rows, dict):
                raise ValueError(f"inspection target {target} is missing rows")

            rows: dict[int, tuple[str, ...]] = {}
            for level in range(1, 6):
                raw_candidates = raw_rows.get(str(level))
                if not isinstance(raw_candidates, list):
                    raise ValueError(
                        f"inspection target {target} row {level} must be a list"
                    )
                sku_ids: list[str] = []
                seen_skus: set[str] = set()
                for candidate in raw_candidates:
                    if not isinstance(candidate, dict):
                        raise ValueError(
                            f"inspection target {target} row {level} contains an invalid candidate"
                        )
                    sku_id = candidate.get("sku_id")
                    name = candidate.get("name")
                    if not isinstance(sku_id, str) or not sku_id.strip():
                        raise ValueError(
                            f"inspection target {target} row {level} candidate is missing sku_id"
                        )
                    normalized_sku = sku_id.strip().upper()
                    product = self._by_sku.get(normalized_sku)
                    if product is None:
                        raise ValueError(
                            f"inspection target {target} row {level} references unknown SKU {normalized_sku}"
                        )
                    if name != product["name"]:
                        raise ValueError(
                            f"inspection target {target} row {level} name does not match {normalized_sku}"
                        )
                    if normalized_sku in seen_skus:
                        raise ValueError(
                            f"inspection target {target} row {level} duplicates {normalized_sku}"
                        )
                    seen_skus.add(normalized_sku)
                    sku_ids.append(normalized_sku)
                rows[level] = tuple(sku_ids)
            loaded[target] = rows
        self._inspection_candidate_rows = loaded

    def product_for_sku(self, sku: str) -> dict[str, Any] | None:
        return self._copy_product(self._by_sku.get(sku.strip().upper()))

    def product_for_name(self, name: str) -> dict[str, Any] | None:
        normalized_name = name.strip()
        product = self._by_name.get(normalized_name)
        if product is None:
            product = self._by_name_alias.get(self._name_lookup_key(normalized_name))
        return self._copy_product(product)

    def product_for_location(self, location: str) -> dict[str, Any] | None:
        return self._copy_product(self._by_location.get(location.strip().upper()))

    @property
    def has_inspection_candidates(self) -> bool:
        return bool(self._inspection_candidate_rows)

    def candidate_products(
        self,
        location_id: str,
        pose_type: CandidatePoseType,
    ) -> list[list[dict[str, Any]]] | None:
        """Return unique products for each visible row, ordered left to right."""

        normalized_location = location_id.strip().upper()
        requested = LOCATION_PATTERN.fullmatch(normalized_location)
        if requested is None:
            raise ValueError("invalid location_id")
        if normalized_location not in self._by_location:
            return None

        if pose_type == "SHELF_VIEW_UPPER":
            levels = (1, 2)
        elif pose_type == "SHELF_VIEW_LOWER":
            levels = (3, 4, 5)
        else:
            levels = (int(requested.group("level")),)

        shelf = requested.group("shelf")
        rows: list[list[dict[str, Any]]] = []
        for level in levels:
            slots: list[tuple[int, dict[str, Any]]] = []
            for location, product in self._by_location.items():
                parsed = LOCATION_PATTERN.fullmatch(location)
                if parsed is None:
                    continue
                if (
                    parsed.group("shelf") == shelf
                    and int(parsed.group("level")) == level
                ):
                    slots.append((int(parsed.group("column")), product))
            slots.sort(key=lambda item: item[0])

            seen_skus: set[str] = set()
            row: list[dict[str, Any]] = []
            for _, product in slots:
                sku_id = product["sku_id"]
                if sku_id in seen_skus:
                    continue
                seen_skus.add(sku_id)
                copied = self._copy_product(product)
                if copied is not None:
                    row.append(copied)
            rows.append(row)
        return rows

    def row_layout(self, location_id: str) -> list[dict[str, Any]] | None:
        """Return every physical column in one row, preserving repeated SKUs."""

        normalized_location = location_id.strip().upper()
        requested = LOCATION_PATTERN.fullmatch(normalized_location)
        if requested is None:
            raise ValueError("invalid location_id")
        if normalized_location not in self._by_location:
            return None

        shelf = requested.group("shelf")
        level = int(requested.group("level"))
        slots: list[tuple[int, str, dict[str, Any]]] = []
        for location, product in self._by_location.items():
            parsed = LOCATION_PATTERN.fullmatch(location)
            if parsed is None:
                continue
            if (
                parsed.group("shelf") == shelf
                and int(parsed.group("level")) == level
            ):
                slots.append((int(parsed.group("column")), location, product))
        slots.sort(key=lambda item: item[0])
        row: list[dict[str, Any]] = []
        for _, location, product in slots:
            copied = self._copy_product(product)
            if copied is not None:
                row.append({"location_id": location, **copied})
        return row

    def inspection_candidate_products(
        self,
        location_id: str,
        pose_type: InspectionPoseType,
    ) -> list[list[dict[str, Any]]] | None:
        """Return the manually configured candidates for one inspection view."""

        target = location_id.strip().upper()
        if INSPECTION_TARGET_PATTERN.fullmatch(target) is None:
            raise ValueError("invalid inspection location_id")
        levels = (1, 2) if pose_type == "SHELF_VIEW_UPPER" else (3, 4, 5)
        configured_rows = self._inspection_candidate_rows.get(target)
        if configured_rows is None:
            return [self._derived_inspection_row(target, level) for level in levels]
        return [
            [
                self._copy_product(self._by_sku[sku_id])
                for sku_id in configured_rows[level]
            ]
            for level in levels
        ]

    def _derived_inspection_row(
        self, target: str, level: int
    ) -> list[dict[str, Any]]:
        """Build candidates from the five-point shelf geometry when no manual view exists."""

        def visible(shelf: int, column: int) -> bool:
            return (
                (target == "H1_INSPECT" and shelf == 1)
                or (target == "H12_INSPECT" and ((shelf == 1 and column >= 4) or (shelf == 2 and column <= 2)))
                or (target == "H2_INSPECT" and shelf == 2)
                or (target == "H23_INSPECT" and ((shelf == 2 and column >= 5) or (shelf == 3 and column <= 2)))
                or (target == "H3_INSPECT" and shelf == 3)
            )

        slots: list[tuple[int, int, dict[str, Any]]] = []
        for location, product in self._by_location.items():
            parsed = LOCATION_PATTERN.fullmatch(location)
            if parsed is None or int(parsed.group("level")) != level:
                continue
            shelf = int(parsed.group("shelf"))
            column = int(parsed.group("column"))
            if visible(shelf, column):
                slots.append((shelf, column, product))
        slots.sort(key=lambda item: (item[0], item[1]))
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for _, _, product in slots:
            if product["sku_id"] in seen:
                continue
            seen.add(product["sku_id"])
            copied = self._copy_product(product)
            if copied is not None:
                result.append(copied)
        return result

    def images_for_name(self, name: str) -> list[str] | None:
        normalized_name = name.strip()
        product = self._by_name.get(normalized_name)
        if product is None:
            product = self._by_name_alias.get(self._name_lookup_key(normalized_name))
        if product is None:
            return None
        return list(product["images"])

    def all_names(self) -> list[str]:
        return list(self._by_name)

    @staticmethod
    def _name_lookup_key(name: str) -> str:
        return (
            name.strip()
            .replace("’", "'")
            .replace("‘", "'")
            .replace("'", "")
            .casefold()
        )

    @staticmethod
    def _copy_product(product: dict[str, Any] | None) -> dict[str, Any] | None:
        if product is None:
            return None
        return {
            "sku_id": product["sku_id"],
            "name": product["name"],
            "images": list(product["images"]),
            "locations": list(product["locations"]),
        }


def create_app(
    catalog_path: Path = DEFAULT_CATALOG_PATH,
    inspection_candidates_path: Path = DEFAULT_INSPECTION_CANDIDATES_PATH,
) -> FastAPI:
    catalog = SkuCatalog.load(catalog_path)
    catalog.load_inspection_candidates(inspection_candidates_path)
    app = FastAPI(
        title="感知模块 SKU 查询服务",
        version="2.0.0",
        description="按 SKU ID、商品名称或标准货位查询完整 SKU 信息。",
    )

    @app.exception_handler(ApiError)
    async def handle_api_error(_: Request, error: ApiError) -> JSONResponse:
        return JSONResponse(
            status_code=error.status_code,
            content={"error_code": error.error_code},
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _: Request, __: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={"error_code": "INVALID_REQUEST"},
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(
        _: Request, error: StarletteHTTPException
    ) -> JSONResponse:
        error_code = "ENDPOINT_NOT_FOUND" if error.status_code == 404 else "HTTP_ERROR"
        return JSONResponse(
            status_code=error.status_code,
            content={"error_code": error_code},
        )

    @app.get("/sku/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="READY")

    @app.get(
        "/sku/search_by_SKU",
        response_model=ProductResponse,
        responses=ERROR_RESPONSES,
    )
    def search_by_sku(sku: str = Query(min_length=1)) -> ProductResponse:
        product = catalog.product_for_sku(sku)
        if product is None:
            raise ApiError(404, "SKU_NOT_FOUND")
        return ProductResponse(**product)

    @app.get(
        "/sku/search_by_name",
        response_model=ProductResponse,
        responses=ERROR_RESPONSES,
    )
    def search_by_name(name: str = Query(min_length=1)) -> ProductResponse:
        product = catalog.product_for_name(name)
        if product is None:
            raise ApiError(404, "SKU_NOT_FOUND")
        return ProductResponse(**product)

    @app.get(
        "/sku/search_by_location",
        response_model=ProductResponse,
        responses=ERROR_RESPONSES,
    )
    def search_by_location(location: str = Query(min_length=1)) -> ProductResponse:
        product = catalog.product_for_location(location)
        if product is None:
            raise ApiError(404, "LOCATION_NOT_FOUND")
        return ProductResponse(**product)

    @app.get(
        "/sku/get_image",
        response_model=list[str],
        responses=ERROR_RESPONSES,
    )
    def get_image_paths(name: str = Query(min_length=1)) -> list[str]:
        images = catalog.images_for_name(name)
        if images is None:
            raise ApiError(404, "SKU_NOT_FOUND")
        return images

    @app.get(
        "/sku/get_candidate_SKU",
        response_model=list[list[ProductResponse]],
        responses=ERROR_RESPONSES,
    )
    def get_candidate_sku(
        request: CandidateSkuRequest = Body(...),
    ) -> list[list[ProductResponse]]:
        try:
            rows = catalog.candidate_products(request.location_id, request.pose_type)
        except ValueError as error:
            raise ApiError(400, "INVALID_LOCATION_ID") from error
        if rows is None:
            raise ApiError(404, "LOCATION_NOT_FOUND")
        return [
            [ProductResponse(**product) for product in row]
            for row in rows
        ]

    @app.get(
        "/sku/get_row_layout",
        response_model=list[SlotProductResponse],
        responses=ERROR_RESPONSES,
    )
    def get_row_layout(
        request: CandidateSkuRequest = Body(...),
    ) -> list[SlotProductResponse]:
        try:
            slots = catalog.row_layout(request.location_id)
        except ValueError as error:
            raise ApiError(400, "INVALID_LOCATION_ID") from error
        if slots is None:
            raise ApiError(404, "LOCATION_NOT_FOUND")
        return [SlotProductResponse(**slot) for slot in slots]

    @app.get(
        "/sku/get_inspection_candidate_SKU",
        response_model=list[list[ProductResponse]],
        responses=ERROR_RESPONSES,
    )
    def get_inspection_candidate_sku(
        request: InspectionCandidateSkuRequest = Body(...),
    ) -> list[list[ProductResponse]]:
        try:
            rows = catalog.inspection_candidate_products(
                request.location_id,
                request.pose_type,
            )
        except ValueError as error:
            raise ApiError(400, "INVALID_LOCATION_ID") from error
        if rows is None:
            raise ApiError(404, "LOCATION_NOT_FOUND")
        return [
            [ProductResponse(**product) for product in row if product is not None]
            for row in rows
        ]

    @app.get("/sku/get_all_names", response_model=list[str])
    def get_all_names() -> list[str]:
        return catalog.all_names()

    @app.get(
        "/images/{image_path:path}",
        response_class=FileResponse,
        responses=ERROR_RESPONSES,
    )
    def get_image(image_path: str) -> FileResponse:
        relative = PurePosixPath(image_path)
        if not image_path or relative.is_absolute() or ".." in relative.parts:
            raise ApiError(400, "INVALID_IMAGE_PATH")

        resolved_path = (IMAGES_ROOT / Path(*relative.parts)).resolve()
        if os.path.commonpath((str(IMAGES_ROOT), str(resolved_path))) != str(IMAGES_ROOT):
            raise ApiError(400, "INVALID_IMAGE_PATH")
        if not resolved_path.is_file() and resolved_path.suffix.lower() == ".png":
            # Preserve compatibility with old image URLs after PNG conversion.
            resolved_path = resolved_path.with_suffix(".jpg")
        if not resolved_path.is_file():
            raise ApiError(404, "IMAGE_NOT_FOUND")

        media_type = mimetypes.guess_type(resolved_path.name)[0]
        return FileResponse(resolved_path, media_type=media_type)

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="感知模块 SKU 查询服务")
    parser.add_argument("--host", default=SERVICE_BIND_HOST)
    parser.add_argument("--port", type=int, default=25540)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument(
        "--inspection-candidates",
        type=Path,
        default=DEFAULT_INSPECTION_CANDIDATES_PATH,
    )
    args = parser.parse_args()

    uvicorn.run(
        create_app(args.catalog, args.inspection_candidates),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
