"""Strict validation and deterministic business-output projection."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from .errors import SchemaValidationError


ALLOWED_STATUSES = {"ok", "needs_review", "unreadable"}
ALLOWED_REVIEW_REASONS = {
    "name_unclear",
    "specification_unclear",
    "other",
}


@dataclass(frozen=True)
class LineItem:
    name: str
    specification: str | None
    needs_review: bool = False
    reason: None = None


@dataclass(frozen=True)
class ReviewItem:
    name: str | None
    specification: str | None
    needs_review: bool
    reason: str


@dataclass(frozen=True)
class ReceiptResult:
    receipt_status: str
    line_items: tuple[LineItem, ...]
    review_items: tuple[ReviewItem, ...]
    reported_receipt_status: str

    def diagnostics_dict(self) -> dict[str, Any]:
        return {
            "receipt_status": self.receipt_status,
            "reported_receipt_status": self.reported_receipt_status,
            "status_normalized": (
                self.receipt_status != self.reported_receipt_status
            ),
            "line_items": [asdict(item) for item in self.line_items],
            "review_items": [asdict(item) for item in self.review_items],
        }

    def business_items(self) -> list[dict[str, Any]]:
        return [
            {
                "name": item.name,
                "specification": item.specification,
            }
            for item in self.line_items
        ]


def parse_receipt_result(raw_text: str) -> ReceiptResult:
    try:
        value = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise SchemaValidationError(
            f"模型输出不是严格 JSON：第 {exc.lineno} 行第 {exc.colno} 列"
        ) from exc

    if not isinstance(value, dict):
        raise SchemaValidationError("模型输出顶层必须是 JSON 对象。")

    status = value.get("receipt_status")
    if status not in ALLOWED_STATUSES:
        raise SchemaValidationError(
            "receipt_status 必须是 ok、needs_review 或 unreadable。"
        )

    line_items_raw = value.get("line_items")
    review_items_raw = value.get("review_items")
    if not isinstance(line_items_raw, list):
        raise SchemaValidationError("line_items 必须是数组。")
    if not isinstance(review_items_raw, list):
        raise SchemaValidationError("review_items 必须是数组。")

    line_items = tuple(
        _parse_line_item(item, index)
        for index, item in enumerate(line_items_raw)
    )
    review_items = tuple(
        _parse_review_item(item, index)
        for index, item in enumerate(review_items_raw)
    )

    if review_items:
        normalized_status = "needs_review"
    elif line_items:
        normalized_status = "ok"
    elif status == "unreadable":
        normalized_status = "unreadable"
    else:
        raise SchemaValidationError(
            "line_items 和 review_items 不能同时为空，"
            "除非 receipt_status 为 unreadable。"
        )

    return ReceiptResult(
        receipt_status=normalized_status,
        line_items=line_items,
        review_items=review_items,
        reported_receipt_status=status,
    )


def _parse_line_item(value: Any, index: int) -> LineItem:
    label = f"line_items[{index}]"
    if not isinstance(value, dict):
        raise SchemaValidationError(f"{label} 必须是对象。")

    name = value.get("name")
    if "specification" not in value:
        raise SchemaValidationError(
            f"{label}.specification 是必填字段。"
        )
    specification = value["specification"]
    needs_review = value.get("needs_review")
    reason = value.get("reason")

    if not isinstance(name, str) or not name.strip():
        raise SchemaValidationError(f"{label}.name 必须是非空字符串。")
    if specification is not None and (
        not isinstance(specification, str) or not specification.strip()
    ):
        raise SchemaValidationError(
            f"{label}.specification 必须是非空字符串或 null。"
        )
    if needs_review is not False:
        raise SchemaValidationError(f"{label}.needs_review 必须为 false。")
    if reason is not None:
        raise SchemaValidationError(f"{label}.reason 必须为 null。")

    return LineItem(
        name=name.strip(),
        specification=(
            specification.strip()
            if isinstance(specification, str)
            else None
        ),
    )


def _parse_review_item(value: Any, index: int) -> ReviewItem:
    label = f"review_items[{index}]"
    if not isinstance(value, dict):
        raise SchemaValidationError(f"{label} 必须是对象。")

    name = value.get("name")
    if "specification" not in value:
        raise SchemaValidationError(
            f"{label}.specification 是必填字段。"
        )
    specification = value["specification"]
    needs_review = value.get("needs_review")
    reason = value.get("reason")

    if name is not None and (
        not isinstance(name, str) or not name.strip()
    ):
        raise SchemaValidationError(
            f"{label}.name 必须是非空字符串或 null。"
        )
    if specification is not None and (
        not isinstance(specification, str) or not specification.strip()
    ):
        raise SchemaValidationError(
            f"{label}.specification 必须是非空字符串或 null。"
        )
    if needs_review is not True:
        raise SchemaValidationError(f"{label}.needs_review 必须为 true。")
    if reason not in ALLOWED_REVIEW_REASONS:
        raise SchemaValidationError(
            f"{label}.reason 不在允许值中。"
        )

    return ReviewItem(
        name=name.strip() if isinstance(name, str) else None,
        specification=(
            specification.strip()
            if isinstance(specification, str)
            else None
        ),
        needs_review=True,
        reason=reason,
    )
