"""Domain-specific errors with safe, actionable messages."""

from __future__ import annotations


class ReceiptRecognizerError(Exception):
    """Base class for expected application errors."""


class ConfigurationError(ReceiptRecognizerError):
    """Configuration is missing or invalid."""


class InputFileError(ReceiptRecognizerError):
    """The local image or PDF cannot be processed."""


class APIConnectionError(ReceiptRecognizerError):
    """The API could not be reached."""


class APIResponseError(ReceiptRecognizerError):
    """The API returned an HTTP or protocol-level error."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class ModelOutputError(ReceiptRecognizerError):
    """The model response did not match the required receipt schema."""


class SchemaValidationError(ModelOutputError):
    """Structured receipt JSON failed strict validation."""


class SKUConnectionError(ReceiptRecognizerError):
    """The SKU lookup service could not be reached."""


class SKUNotFoundError(ReceiptRecognizerError):
    """A recognized product name does not exist in the SKU catalog."""

    def __init__(self, name: str, error_code: str = "SKU_NOT_FOUND") -> None:
        super().__init__(f"商品名称未在 SKU 库中找到：{name}")
        self.name = name
        self.error_code = error_code


class SKUResponseError(ReceiptRecognizerError):
    """The SKU lookup service returned an unexpected response."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
