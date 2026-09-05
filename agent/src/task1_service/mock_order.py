"""Mock ordering-system adapter used by Task 1.

The adapter deliberately loads the SKU catalog for every order so the running
SKU service remains the source of truth. A caller may also pass one or two
names shown in the web console; those names are revalidated against the
current catalog before the order is accepted.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from random import Random
from uuid import uuid4

from task1_service.models import Task1ServiceError


CatalogLoader = Callable[[], Awaitable[Sequence[str]]]


@dataclass(frozen=True)
class MockOrder:
    """One one- or two-product order returned by :class:`MockOrderSystem`."""

    order_id: str
    source: str
    catalog_size: int
    product_names: list[str]
    available_product_names: list[str] = field(default_factory=list)


class MockOrderSystem:
    """Create a one- or two-product mock order from the active SKU catalog."""

    def __init__(
        self,
        catalog_loader: CatalogLoader,
        *,
        rng: Random | None = None,
    ) -> None:
        self._catalog_loader = catalog_loader
        self._rng = rng if rng is not None else Random()

    async def create_order(
        self,
        requested_product_names: Sequence[str] | None = None,
        order_id: str | None = None,
    ) -> MockOrder:
        """Return a two-product random order or validate a manual order."""

        catalog = self._normalize_catalog(await self._catalog_loader())
        if not catalog:
            raise Task1ServiceError(
                "MOCK_ORDER_CATALOG_UNAVAILABLE",
                "mock order catalog must contain at least one unique product name",
                status_code=422,
            )

        if requested_product_names is None:
            if len(catalog) < 2:
                raise Task1ServiceError(
                    "MOCK_ORDER_CATALOG_UNAVAILABLE",
                    "random mock orders require at least two unique product names",
                    status_code=422,
                )
            sampled = set(self._rng.sample(catalog, 2))
            # Sampling decides which products are selected; catalog order keeps
            # downstream behavior deterministic for a fixed selected pair.
            product_names = [name for name in catalog if name in sampled]
        else:
            product_names = self._validate_requested_products(
                requested_product_names,
                catalog,
            )

        resolved_order_id = order_id.strip() if isinstance(order_id, str) else ""
        if order_id is not None and not resolved_order_id:
            raise Task1ServiceError(
                "INVALID_MOCK_ORDER_ID",
                "mock order id must not be empty",
                status_code=422,
            )

        return MockOrder(
            order_id=resolved_order_id or uuid4().hex,
            source="mock_random",
            catalog_size=len(catalog),
            product_names=product_names,
            available_product_names=list(catalog),
        )

    @staticmethod
    def _normalize_catalog(raw_catalog: Sequence[str]) -> list[str]:
        if isinstance(raw_catalog, (str, bytes)) or not isinstance(raw_catalog, Sequence):
            raise Task1ServiceError(
                "INVALID_MOCK_ORDER_CATALOG",
                "mock order catalog must be a sequence of product names",
                status_code=422,
            )

        product_names: list[str] = []
        seen: set[str] = set()
        for raw_name in raw_catalog:
            if not isinstance(raw_name, str):
                raise Task1ServiceError(
                    "INVALID_MOCK_ORDER_CATALOG",
                    "mock order catalog must contain only product names",
                    status_code=422,
                )
            name = raw_name.strip()
            if not name or name in seen:
                continue
            seen.add(name)
            product_names.append(name)
        return product_names

    @staticmethod
    def _validate_requested_products(
        requested_product_names: Sequence[str],
        catalog: Sequence[str],
    ) -> list[str]:
        if isinstance(requested_product_names, (str, bytes)):
            raise Task1ServiceError(
                "INVALID_MOCK_ORDER_PRODUCTS",
                "mock order must contain one or two product names",
                status_code=422,
            )

        requested = list(requested_product_names)
        if len(requested) not in {1, 2} or any(
            not isinstance(name, str) for name in requested
        ):
            raise Task1ServiceError(
                "INVALID_MOCK_ORDER_PRODUCTS",
                "mock order must contain one or two product names",
                status_code=422,
            )

        normalized = [name.strip() for name in requested]
        if any(not name for name in normalized) or len(set(normalized)) != len(
            normalized
        ):
            raise Task1ServiceError(
                "INVALID_MOCK_ORDER_PRODUCTS",
                "mock order product names must be non-empty and distinct",
                status_code=422,
            )

        catalog_names = set(catalog)
        unknown = [name for name in normalized if name not in catalog_names]
        if unknown:
            raise Task1ServiceError(
                "MOCK_ORDER_PRODUCT_NOT_FOUND",
                f"mock order products are not in the active SKU catalog: {unknown}",
                status_code=422,
            )
        return normalized
