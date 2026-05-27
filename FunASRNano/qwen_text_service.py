"""Qwen text post-processing service."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from FunASRNano.schemas import QwenTextSettings, TranscriptSegment


class QwenTextProcessor:
    def __init__(self, settings: QwenTextSettings, logger) -> None:
        self.settings = settings
        self.logger = logger

    def cleanup_segments(self, segments: list[TranscriptSegment]) -> None:
        if not self.settings.enabled or not self.settings.enable_segment_cleanup:
            return

        for segment in segments:
            raw_text = segment.text.strip()
            if not raw_text:
                continue
            try:
                cleaned = self._chat(
                    prompt=f"{self.settings.cleanup_prompt.strip()}\n\n{raw_text}",
                    max_tokens=self.settings.segment_cleanup_max_tokens,
                ).strip()
            except Exception as exc:
                self.logger.warning("Qwen3.6 片段整理失败，保留原始文本: %s", exc)
                continue
            if cleaned:
                segment.text = cleaned

    def summarize(self, segments: list[TranscriptSegment]) -> str:
        if not self.settings.enabled or not self.settings.enable_summary:
            return ""

        transcript = self._build_transcript_text(segments)
        if not transcript:
            return ""

        try:
            return self._chat(
                prompt=f"{self.settings.summary_prompt.strip()}\n\n{transcript}",
                max_tokens=self.settings.summary_max_tokens,
            ).strip()
        except Exception as exc:
            self.logger.warning("Qwen3.6 总结失败，跳过 summary metadata: %s", exc)
            return ""

    def structured_output(self, segments: list[TranscriptSegment]) -> Any:
        if not self.settings.enabled or not self.settings.enable_structured_output:
            return None

        transcript = self._build_transcript_text(segments)
        if not transcript:
            return None

        try:
            raw_text = self._chat(
                prompt=f"{self.settings.structured_prompt.strip()}\n\n{transcript}",
                max_tokens=self.settings.structured_max_tokens,
            ).strip()
        except Exception as exc:
            self.logger.warning("Qwen3.6 结构化输出失败，跳过 structured metadata: %s", exc)
            return None

        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            return {"raw": raw_text}

    def _build_transcript_text(self, segments: list[TranscriptSegment]) -> str:
        lines: list[str] = []
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            speaker = segment.speaker_name or segment.speaker_id or segment.speaker_label
            prefix = f"{speaker}: " if speaker else ""
            lines.append(f"{prefix}{text}")

        transcript = "\n".join(lines).strip()
        if len(transcript) > self.settings.summary_input_max_chars:
            return transcript[: self.settings.summary_input_max_chars]
        return transcript

    def _chat(self, prompt: str, max_tokens: int) -> str:
        payload = {
            "model": self.settings.model,
            "temperature": self.settings.temperature,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self._url("chat/completions"),
            data=data,
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
                    text = first_choice.get("text")
                    if isinstance(text, str):
                        return text
            text = payload.get("text")
            if isinstance(text, str):
                return text
        if isinstance(payload, str):
            return payload
        return ""
