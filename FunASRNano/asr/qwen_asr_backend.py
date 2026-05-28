"""Qwen3-ASR backend through an OpenAI-compatible llama.cpp server."""

from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from FunASRNano.llamacpp_runtime import LlamaCppServerManager
from FunASRNano.schemas import QwenAsrSettings


class QwenAsrBackend:
    name = "qwen_asr"
    device = "llama.cpp"

    def __init__(self, settings: QwenAsrSettings, logger) -> None:
        self.settings = settings
        self.logger = logger
        self.model_name = settings.model
        self.server_manager = LlamaCppServerManager(
            settings.to_llamacpp_settings(),
            logger,
            project_root=Path.cwd(),
            service_name="Qwen3-ASR llama.cpp",
            log_prefix="qwen_asr_server",
        )
        self._server_checked = False

    def transcribe(self, audio_path: str | Path) -> str:
        self._ensure_server()
        endpoint = self.settings.endpoint.strip().lower()
        if endpoint not in ("chat_completions", "chat/completions"):
            raise ValueError(
                "QwenAsrBackend 当前支持 chat_completions endpoint；"
                f"收到: {self.settings.endpoint}"
            )
        return self._clean_output(self._dictate_chat_completions(Path(audio_path)))

    def shutdown(self) -> None:
        self.server_manager.shutdown_started_server()
        self._server_checked = False

    def _ensure_server(self) -> None:
        if self._server_checked:
            return
        self.server_manager.ensure_server()
        self._server_checked = True

    def _dictate_chat_completions(self, audio_path: Path) -> str:
        audio_bytes = audio_path.read_bytes()
        audio_base64 = base64.b64encode(audio_bytes).decode("ascii")
        payload = {
            "model": self.settings.model,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.settings.dictation_prompt},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_base64,
                                "format": self.settings.audio_format,
                            },
                        },
                    ],
                }
            ],
        }
        request = urllib.request.Request(
            self._url("chat/completions"),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                **self._headers(),
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.settings.request_timeout_seconds,
            ) as response:
                response_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {error_body}") from exc
        return self._extract_message_content(json.loads(response_body))

    def _headers(self) -> dict[str, str]:
        if not self.settings.api_key:
            return {}
        return {"Authorization": f"Bearer {self.settings.api_key}"}

    def _url(self, endpoint: str) -> str:
        return f"{self.settings.base_url.rstrip('/')}/{endpoint.lstrip('/')}"

    @staticmethod
    def _extract_message_content(payload: Any) -> str:
        if isinstance(payload, dict):
            choices = payload.get("choices")
            if isinstance(choices, list) and choices:
                first_choice = choices[0]
                if isinstance(first_choice, dict):
                    message = first_choice.get("message")
                    if isinstance(message, dict):
                        content = message.get("content")
                        if isinstance(content, str):
                            return content
                        if isinstance(content, list):
                            return "".join(
                                item.get("text", "")
                                for item in content
                                if isinstance(item, dict)
                            )
                    text = first_choice.get("text")
                    if isinstance(text, str):
                        return text
            text = payload.get("text")
            if isinstance(text, str):
                return text
        if isinstance(payload, str):
            return payload
        return ""

    @staticmethod
    def _clean_output(text: str) -> str:
        text = text.strip()
        text = re.sub(r"^language\s+\w+\s*", "", text, flags=re.IGNORECASE)
        text = text.replace("<asr_text>", "")
        text = text.replace("</asr_text>", "")
        text = re.sub(r"<\|[^>]+?\|>", "", text)
        return text.strip()
