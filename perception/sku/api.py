from __future__ import annotations

import argparse
import json
import mimetypes
import os
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException


ROOT = Path(__file__).resolve().parent
DEFAULT_CATALOG_PATH = ROOT / "products.json"
IMAGES_ROOT = (ROOT / "images").resolve()


class HealthResponse(BaseModel):
    status: Literal["READY"]


class ProductResponse(BaseModel):
    sku_id: str
    name: str
    images: list[str]
    locations: list[str]


class ErrorResponse(BaseModel):
    error_code: str


ERROR_RESPONSES = {
    400: {"model": ErrorResponse},
    404: {"model": ErrorResponse},
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
                resolved_image = catalog_root.joinpath(*image_path.parts)
                if not resolved_image.is_file() or resolved_image.stat().st_size == 0:
                    raise ValueError(f"商品 {name!r} 的图片不存在: {image}")

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


def create_app(catalog_path: Path = DEFAULT_CATALOG_PATH) -> FastAPI:
    catalog = SkuCatalog.load(catalog_path)
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
        if not resolved_path.is_file():
            raise ApiError(404, "IMAGE_NOT_FOUND")

        media_type = mimetypes.guess_type(resolved_path.name)[0]
        return FileResponse(resolved_path, media_type=media_type)

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="感知模块 SKU 查询服务")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=25540)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    args = parser.parse_args()

    uvicorn.run(create_app(args.catalog), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
