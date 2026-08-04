"""Receipt recognition through an OpenAI-compatible vision API."""

from .config import Settings
from .service import Recognition, ReceiptRecognizer

__all__ = ["Recognition", "ReceiptRecognizer", "Settings"]

