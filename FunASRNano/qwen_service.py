"""Qwen local-model transcription and text understanding service."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import requests

from FunASRNano.schemas import QwenSettings, TranscriptSegment


class QwenTranscriber:
    def __init__(self, settings: QwenSettings, logger) -> None:
        self.settings = settings
        self.logger = logger

    def transcribe(self, audio_path: str | Path) -> str:
        audio_file = Path(audio_path)
        self.logger.debug("开始 Qwen 听写片段: %s", audio_file.resolve())
        raw_text = self._dictate_audio(audio_file).strip()
        if not raw_text:
            return ""

        if not self.settings.enable_text_refinement:
            return raw_text

        try:
            refined_text = self._refine_text(raw_text).strip()
        except Exception as exc:
            self.logger.warning("Qwen 文本整理失败，保留原始听写文本: %s", exc)
            return raw_text
        return refined_text or raw_text

    def summarize(self, segments: list[TranscriptSegment]) -> str:
        if not self.settings.enable_summary:
            return ""

        transcript_lines = []
        for segment in segments:
            speaker = segment.speaker_name or segment.speaker_id or segment.speaker_label
            prefix = f"{speaker}: " if speaker else ""
            text = segment.text.strip()
            if text:
                transcript_lines.append(f"{prefix}{text}")

        transcript = "\n".join(transcript_lines).strip()
        if not transcript:
            return ""
        if len(transcript) > self.settings.summary_input_max_chars:
            transcript = transcript[: self.settings.summary_input_max_chars]

        prompt = f"{self.settings.summary_prompt.strip()}\n\n{transcript}"
        try:
            return self._chat(
                base_url=self.settings.llm_base_url,
                model=self.settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.settings.summary_max_tokens,
            ).strip()
        except Exception as exc:
            self.logger.warning("Qwen 总结失败，跳过 summary metadata: %s", exc)
            return ""

    def _dictate_audio(self, audio_path: Path) -> str:
        endpoint = self.settings.asr_endpoint.strip().lower()
        if endpoint == "audio_transcriptions":
            return self._dictate_audio_transcriptions(audio_path)
        return self._dictate_chat_completions(audio_path)

    def _dictate_chat_completions(self, audio_path: Path) -> str:
        audio_bytes = audio_path.read_bytes()
        audio_base64 = base64.b64encode(audio_bytes).decode("ascii")
        content = [
            {"type": "text", "text": self.settings.dictation_prompt},
            {
                "type": "input_audio",
                "input_audio": {
                    "data": audio_base64,
                    "format": self.settings.audio_format,
                },
            },
        ]
        return self._chat(
            base_url=self.settings.asr_base_url,
            model=self.settings.asr_model,
            messages=[{"role": "user", "content": content}],
            max_tokens=self.settings.asr_max_tokens,
        )

    def _dictate_audio_transcriptions(self, audio_path: Path) -> str:
        url = self._url(self.settings.asr_base_url, "audio/transcriptions")
        headers = self._headers()
        with audio_path.open("rb") as file_obj:
            files = {
                "file": (
                    audio_path.name,
                    file_obj,
                    f"audio/{self.settings.audio_format}",
                )
            }
            data = {
                "model": self.settings.asr_model,
                "prompt": self.settings.dictation_prompt,
                "response_format": "json",
                "temperature": str(self.settings.temperature),
            }
            response = requests.post(
                url,
                headers=headers,
                data=data,
                files=files,
                timeout=self.settings.request_timeout_seconds,
            )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            text = payload.get("text")
            if isinstance(text, str):
                return text
        if isinstance(payload, str):
            return payload
        return ""

    def _refine_text(self, raw_text: str) -> str:
        prompt = f"{self.settings.refinement_prompt.strip()}\n\n{raw_text}"
        return self._chat(
            base_url=self.settings.llm_base_url,
            model=self.settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.settings.refinement_max_tokens,
        )

    def _chat(
        self,
        *,
        base_url: str,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
    ) -> str:
        payload = {
            "model": model,
            "temperature": self.settings.temperature,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        response = requests.post(
            self._url(base_url, "chat/completions"),
            headers={
                **self._headers(),
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.settings.request_timeout_seconds,
        )
        response.raise_for_status()
        return self._extract_message_content(response.json())

    def _headers(self) -> dict[str, str]:
        if not self.settings.api_key:
            return {}
        return {"Authorization": f"Bearer {self.settings.api_key}"}

    @staticmethod
    def _url(base_url: str, endpoint: str) -> str:
        return f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"

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
                            parts = [
                                item.get("text", "")
                                for item in content
                                if isinstance(item, dict)
                            ]
                            return "".join(parts)
                    text = first_choice.get("text")
                    if isinstance(text, str):
                        return text
            text = payload.get("text")
            if isinstance(text, str):
                return text
        if isinstance(payload, str):
            return payload
        return ""
