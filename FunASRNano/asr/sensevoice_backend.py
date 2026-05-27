"""SenseVoice backend adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from FunASRNano.schemas import SenseVoiceSettings


class SenseVoiceBackend:
    name = "sensevoice"

    def __init__(self, settings: SenseVoiceSettings, device: str, logger) -> None:
        self.settings = settings
        self.device = device
        self.logger = logger
        self.model_name = settings.model
        self._model: Any | None = None
        self._postprocess: Any | None = None

    @property
    def model(self) -> Any:
        if self._model is None:
            self._load()
        return self._model

    def _load(self) -> None:
        from funasr import AutoModel

        try:
            from funasr.utils.postprocess_utils import rich_transcription_postprocess
        except Exception:
            rich_transcription_postprocess = None

        kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "hub": self.settings.hub,
            "device": self.device,
        }
        if self.settings.trust_remote_code is not None:
            kwargs["trust_remote_code"] = self.settings.trust_remote_code

        self.logger.info(
            "加载 SenseVoice 模型: %s, device=%s",
            self.settings.model,
            self.device,
        )
        self._model = AutoModel(**kwargs)
        self._postprocess = rich_transcription_postprocess

    def transcribe(self, audio_path: str | Path) -> str:
        generate_kwargs = {
            "input": str(audio_path),
            "batch_size_s": self.settings.batch_size_s,
            "hotword": self.settings.hotword or None,
        }
        if self.settings.language:
            generate_kwargs["language"] = self.settings.language
        if self.settings.itn is not None:
            generate_kwargs["itn"] = self.settings.itn

        result = self.model.generate(**generate_kwargs)
        text = self._extract_text(result).strip()
        if text and self._postprocess is not None:
            try:
                text = self._postprocess(text).strip()
            except Exception:
                pass
        return text

    @staticmethod
    def _extract_text(result: Any) -> str:
        payload = result[0] if isinstance(result, list) and result else result
        if isinstance(payload, dict):
            text = payload.get("text")
            if isinstance(text, str):
                return text
            value = payload.get("value")
            if isinstance(value, str):
                return value
            if isinstance(value, list):
                return "".join(str(item) for item in value)
        if isinstance(payload, str):
            return payload
        return ""
