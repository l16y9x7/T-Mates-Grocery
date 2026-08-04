"""Receipt recognition orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .api import ChatResponse, OpenAICompatibleClient
from .config import Settings
from .errors import ModelOutputError, SchemaValidationError
from .media import multimodal_content, prepare_input
from .prompts import (
    CORRECTION_PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
    USER_PROMPT,
)
from .schema import ReceiptResult, parse_receipt_result


@dataclass(frozen=True)
class Recognition:
    business_items: list[dict[str, Any]]
    diagnostics: dict[str, Any]
    finish_reason: str | None
    usage: dict[str, Any] | None
    corrected_once: bool
    page_count: int


class ReceiptRecognizer:
    def __init__(
        self,
        settings: Settings,
        *,
        client: OpenAICompatibleClient | None = None,
    ) -> None:
        self.settings = settings
        self.client = client or OpenAICompatibleClient(settings)

    def recognize_file(
        self,
        input_path: Path,
        *,
        max_edge: int = 2200,
        pdf_dpi: int = 180,
        max_pdf_pages: int = 1,
        temperature: float = 0.0,
    ) -> Recognition:
        data_urls = prepare_input(
            input_path,
            max_edge=max_edge,
            pdf_dpi=pdf_dpi,
            max_pdf_pages=max_pdf_pages,
        )
        return self.recognize_data_urls(
            data_urls,
            temperature=temperature,
        )

    def recognize_data_urls(
        self,
        data_urls: list[str],
        *,
        temperature: float = 0.0,
    ) -> Recognition:
        if not data_urls:
            raise ModelOutputError("至少需要一张图片。")
        _validate_temperature(temperature)

        first = self.client.create_chat_completion(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": multimodal_content(data_urls, USER_PROMPT),
                },
            ],
            temperature=temperature,
            max_tokens=1400,
        )

        try:
            parsed = parse_receipt_result(first.content)
        except SchemaValidationError as first_error:
            second = self._correct_output(
                first.content,
                first_error,
                temperature=temperature,
            )
            try:
                parsed = parse_receipt_result(second.content)
            except SchemaValidationError as second_error:
                raise ModelOutputError(
                    "模型连续两次未返回合法结构化 JSON。"
                    f"首次错误：{first_error}；纠正后错误：{second_error}"
                ) from second_error
            return self._build_recognition(
                parsed,
                second,
                corrected_once=True,
                page_count=len(data_urls),
                temperature=temperature,
            )

        return self._build_recognition(
            parsed,
            first,
            corrected_once=False,
            page_count=len(data_urls),
            temperature=temperature,
        )

    def _correct_output(
        self,
        raw_output: str,
        validation_error: SchemaValidationError,
        *,
        temperature: float,
    ) -> ChatResponse:
        correction_prompt = CORRECTION_PROMPT_TEMPLATE.format(
            raw_output=raw_output[:12000],
            validation_error=str(validation_error),
        )
        return self.client.create_chat_completion(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": correction_prompt},
            ],
            temperature=temperature,
            max_tokens=1400,
        )

    @staticmethod
    def _build_recognition(
        parsed: ReceiptResult,
        response: ChatResponse,
        *,
        corrected_once: bool,
        page_count: int,
        temperature: float,
    ) -> Recognition:
        diagnostics = parsed.diagnostics_dict()
        diagnostics["corrected_once"] = corrected_once
        diagnostics["page_count"] = page_count
        diagnostics["temperature"] = temperature
        diagnostics["finish_reason"] = response.finish_reason
        diagnostics["usage"] = response.usage
        return Recognition(
            business_items=parsed.business_items(),
            diagnostics=diagnostics,
            finish_reason=response.finish_reason,
            usage=response.usage,
            corrected_once=corrected_once,
            page_count=page_count,
        )


def _validate_temperature(temperature: float) -> None:
    if temperature < 0:
        raise ModelOutputError("temperature 不能小于 0。")
