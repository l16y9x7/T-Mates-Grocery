"""Inventory matching helpers for post-recognition validation.

This module intentionally does not change model output.  It only checks
whether a recognized receipt item can be matched to a known SKU name.
"""

from __future__ import annotations

import csv
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class InventorySku:
    sku_name: str
    normalized_name: str


@dataclass(frozen=True)
class ItemInventoryMatch:
    item: dict[str, Any]
    match_status: str
    matched_sku_name: str | None
    candidate_sku_names: tuple[str, ...]
    suggested_sku_names: tuple[str, ...]
    evidence: tuple[str, ...]
    suggestion_evidence: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "item": self.item,
            "match_status": self.match_status,
            "matched_sku_name": self.matched_sku_name,
            "candidate_sku_names": list(self.candidate_sku_names),
            "suggested_sku_names": list(self.suggested_sku_names),
            "evidence": list(self.evidence),
            "suggestion_evidence": list(self.suggestion_evidence),
        }


def normalize_product_text(value: str | None) -> str:
    """Normalize OCR/model text for SKU matching.

    The goal is not semantic standardization.  We only remove visual noise that
    commonly differs between receipts and inventory tables: spaces, punctuation
    and full-width/half-width variants.  Chinese characters, letters and digits
    are preserved.
    """

    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower()
    text = (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("`", "'")
        .replace("´", "'")
    )

    kept: list[str] = []
    for char in text:
        category = unicodedata.category(char)
        if category.startswith(("L", "N")):
            kept.append(char)
    return "".join(kept)


def load_inventory_csv(path: Path) -> list[InventorySku]:
    """Load an inventory CSV with a required ``sku_name`` column."""

    rows: list[InventorySku] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "sku_name" not in reader.fieldnames:
            raise ValueError("库存 CSV 必须包含 sku_name 列。")

        for row in reader:
            sku_name = (row.get("sku_name") or "").strip()
            normalized = normalize_product_text(sku_name)
            if not sku_name or not normalized or normalized in seen:
                continue
            seen.add(normalized)
            rows.append(
                InventorySku(
                    sku_name=sku_name,
                    normalized_name=normalized,
                )
            )
    return rows


def item_text_candidates(item: dict[str, Any]) -> list[str]:
    """Return receipt texts worth matching against inventory names."""

    name = _clean_string(item.get("name"))
    candidates: list[str] = []

    # New schema keeps flavor/variant inside the full product name, so the
    # model-provided name should be the primary exact-match text.
    if name:
        _append_candidate(candidates, name)
    return candidates


def source_text_candidates(source_text: str) -> list[str]:
    """Deprecated compatibility hook.

    The current business JSON no longer keeps ``source_text``.  Inventory
    validation intentionally uses only the structured ``name`` field.
    """

    return []


def match_inventory_item(
    item: dict[str, Any],
    inventory: Iterable[InventorySku],
) -> ItemInventoryMatch:
    """Match one recognized item to inventory.

    Status values:
    - ``matched``: exactly one SKU was found by exact text matching.
    - ``not_found``: no exact SKU match was found.
    - ``ambiguous``: more than one exact SKU match was found.
    Fuzzy containment results are suggestions only and never count as matched.
    """

    candidates = item_text_candidates(item)
    normalized_candidates = [
        normalize_product_text(candidate) for candidate in candidates
    ]
    normalized_candidates = [
        candidate for candidate in _dedupe(normalized_candidates) if candidate
    ]

    exact_matches: dict[str, set[str]] = {}
    for sku in inventory:
        for candidate in normalized_candidates:
            if candidate == sku.normalized_name:
                exact_matches.setdefault(sku.sku_name, set()).add("exact")

    suggestions: dict[str, set[str]] = {}
    if not exact_matches:
        for sku in inventory:
            for candidate in normalized_candidates:
                # Suggestions are useful for diagnostics, but they are not
                # accepted as validation results.
                shorter = min(len(candidate), len(sku.normalized_name))
                if shorter >= 4 and (
                    candidate in sku.normalized_name
                    or sku.normalized_name in candidate
                ):
                    suggestions.setdefault(sku.sku_name, set()).add("contains")

    matched_names = tuple(exact_matches.keys())
    if len(matched_names) == 1:
        status = "matched"
        matched_sku_name = matched_names[0]
    elif len(matched_names) == 0:
        status = "not_found"
        matched_sku_name = None
    else:
        status = "ambiguous"
        matched_sku_name = None

    evidence = tuple(
        f"{sku_name}:{','.join(sorted(reasons))}"
        for sku_name, reasons in exact_matches.items()
    )
    suggestion_evidence = tuple(
        f"{sku_name}:{','.join(sorted(reasons))}"
        for sku_name, reasons in suggestions.items()
    )
    return ItemInventoryMatch(
        item=item,
        match_status=status,
        matched_sku_name=matched_sku_name,
        candidate_sku_names=matched_names,
        suggested_sku_names=tuple(suggestions.keys()),
        evidence=evidence,
        suggestion_evidence=suggestion_evidence,
    )


def validate_receipt_items(
    items: list[dict[str, Any]],
    inventory: Iterable[InventorySku],
) -> dict[str, Any]:
    """Validate recognized receipt items against known inventory names."""

    inventory_list = list(inventory)
    matches = [
        match_inventory_item(item, inventory_list)
        for item in items
    ]
    unique_sku_names = _dedupe(
        match.matched_sku_name
        for match in matches
        if match.matched_sku_name is not None
    )
    return {
        "summary": {
            "total_items": len(items),
            "matched_items": sum(
                1 for match in matches if match.match_status == "matched"
            ),
            "not_found_items": sum(
                1 for match in matches if match.match_status == "not_found"
            ),
            "ambiguous_items": sum(
                1 for match in matches if match.match_status == "ambiguous"
            ),
            "suggested_items": sum(
                1 for match in matches if match.suggested_sku_names
            ),
            "unique_sku_count": len(unique_sku_names),
            "is_unique_pair": len(unique_sku_names) == 2,
        },
        "unique_sku_names": unique_sku_names,
        "matches": [match.to_dict() for match in matches],
    }


def _clean_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _append_candidate(candidates: list[str], value: str) -> None:
    normalized = normalize_product_text(value)
    if value and normalized:
        for existing in candidates:
            if normalize_product_text(existing) == normalized:
                return
        candidates.append(value)


def _dedupe(values: Iterable[str | None]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
