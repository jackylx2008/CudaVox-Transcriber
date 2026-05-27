"""ASR backend factory."""

from __future__ import annotations

from FunASRNano.asr.base import AsrBackend
from FunASRNano.asr.funasr_backend import FunAsrBackend
from FunASRNano.asr.sensevoice_backend import SenseVoiceBackend
from FunASRNano.schemas import Settings


def create_asr_backend(settings: Settings, device: str, logger) -> AsrBackend:
    backend = settings.asr.backend.strip().lower()
    if backend in ("funasr", "fun-asr"):
        selected = FunAsrBackend(settings.funasr, device, logger)
    elif backend in ("sensevoice", "sense-voice"):
        selected = SenseVoiceBackend(settings.sensevoice, device, logger)
    else:
        raise ValueError(
            f"不支持的 ASR_BACKEND: {settings.asr.backend}. "
            "可选值: funasr, sensevoice"
        )

    logger.info(
        "ASR backend 已选择: backend=%s, model=%s, device=%s",
        selected.name,
        selected.model_name,
        selected.device,
    )
    return selected
