"""FunASR transcription service."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, cast

from FunASRNano.schemas import FunASRSettings


class FunASRTranscriber:
    def __init__(self, settings: FunASRSettings, device: str, logger) -> None:
        self.settings = settings
        self.device = device
        self.logger = logger
        self._model: Any | None = None
        self._postprocess: Any | None = None

    @property
    def model(self) -> Any:
        if self._model is None:
            self._load()
        return self._model

    def _load(self) -> None:
        from funasr import AutoModel
        from funasr.register import tables

        try:
            from funasr.utils.postprocess_utils import rich_transcription_postprocess
        except Exception:
            rich_transcription_postprocess = None

        if self._uses_fun_asr_nano():
            self._ensure_fun_asr_nano_registered(tables)

        kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "hub": self.settings.hub,
            "device": self.device,
        }
        trust_remote_code = self.settings.trust_remote_code
        if trust_remote_code is not None:
            kwargs["trust_remote_code"] = trust_remote_code
        if self.settings.vad_model and not self._uses_fun_asr_nano():
            kwargs["vad_model"] = self.settings.vad_model
        if self.settings.punc_model and not self._uses_fun_asr_nano():
            kwargs["punc_model"] = self.settings.punc_model
        if self.settings.max_single_segment_time > 0:
            kwargs["vad_kwargs"] = {
                "max_single_segment_time": self.settings.max_single_segment_time
            }

        self.logger.info("加载 FunASR 模型: %s", self.settings.model)
        self._model = AutoModel(**kwargs)
        self._postprocess = rich_transcription_postprocess

    def transcribe(self, audio_path: str | Path) -> str:
        self.logger.debug("开始转写片段: %s", Path(audio_path).resolve())
        if self._uses_fun_asr_nano():
            generate_kwargs = {
                "input": [str(audio_path)],
                "cache": {},
                "batch_size": 1,
            }
            if self.settings.hotword:
                generate_kwargs["hotwords"] = [self.settings.hotword]
        else:
            generate_kwargs = {
                "input": str(audio_path),
                "batch_size_s": self.settings.batch_size_s,
                "hotword": self.settings.hotword or None,
            }
        language = self.settings.language or self._default_language()
        if language:
            generate_kwargs["language"] = language
        itn = self.settings.itn
        if itn is None and self._uses_fun_asr_nano():
            itn = True
        if itn is not None:
            generate_kwargs["itn"] = itn

        model = self.model
        result = model.generate(**generate_kwargs)
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

    def _uses_fun_asr_nano(self) -> bool:
        return "fun-asr-nano" in self.settings.model.lower()

    def _default_language(self) -> str:
        if self._uses_fun_asr_nano():
            return "中文"
        return ""

    def _ensure_fun_asr_nano_registered(self, tables) -> None:
        if tables.model_classes.get("FunASRNano") is not None:
            return

        import funasr

        funasr_file = getattr(funasr, "__file__", None)
        if not funasr_file:
            raise FileNotFoundError("无法定位 funasr 包目录。")
        nano_dir = Path(cast(str, funasr_file)).resolve().parent / "models" / "fun_asr_nano"
        if not nano_dir.exists():
            raise FileNotFoundError(f"找不到 Fun-ASR-Nano 代码目录: {nano_dir}")

        nano_dir_str = str(nano_dir)
        if nano_dir_str not in sys.path:
            sys.path.insert(0, nano_dir_str)

        importlib.import_module("model")
