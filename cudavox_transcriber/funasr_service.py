"""FunASR transcription service."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cudavox_transcriber.schemas import FunASRSettings


class FunASRTranscriber:
    def __init__(self, settings: FunASRSettings, device: str, logger) -> None:
        self.settings = settings
        self.device = device
        self.logger = logger
        self._model = None
        self._postprocess = None

    @property
    def model(self):
        if self._model is None:
            self._load()
        return self._model

    def _load(self) -> None:
        from funasr import AutoModel

        try:
            from funasr.utils.postprocess_utils import rich_transcription_postprocess
        except Exception:
            rich_transcription_postprocess = None

        kwargs = {
            "model": self.settings.model,
            "vad_model": self.settings.vad_model,
            "punc_model": self.settings.punc_model,
            "hub": self.settings.hub,
            "device": self.device,
        }
        if self.settings.max_single_segment_time > 0:
            kwargs["vad_kwargs"] = {
                "max_single_segment_time": self.settings.max_single_segment_time
            }

        self.logger.info("加载 FunASR 模型: %s", self.settings.model)
        self._model = AutoModel(**kwargs)
        self._postprocess = rich_transcription_postprocess

    def transcribe(self, audio_path: str | Path) -> str:
        self.logger.debug("开始转写片段: %s", Path(audio_path).resolve())
        result = self.model.generate(
            input=str(audio_path),
            batch_size_s=self.settings.batch_size_s,
            hotword=self.settings.hotword or None,
        )
        text = self._extract_text(result).strip()
        if text and self._postprocess is not None:
            try:
                text = self._postprocess(text).strip()
            except Exception:
                pass
        self.logger.debug("转写完成: %s, 文本长度=%s", Path(audio_path).name, len(text))
        return text

    def _extract_text(self, result: Any) -> str:
        payload = result[0] if isinstance(result, list) and result else result
        if isinstance(payload, dict):
            text = payload.get("text")
            if isinstance(text, str):
                return text

            sentence_info = payload.get("sentence_info") or payload.get("sentences")
            if isinstance(sentence_info, list):
                parts = [
                    item.get("text", "").strip()
                    for item in sentence_info
                    if isinstance(item, dict) and item.get("text")
                ]
                if parts:
                    return "".join(parts)

            value = payload.get("value")
            if isinstance(value, str):
                return value
            if isinstance(value, list):
                return "".join(str(item) for item in value)

        if isinstance(payload, str):
            return payload
        return ""
