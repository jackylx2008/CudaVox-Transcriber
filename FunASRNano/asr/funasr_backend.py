"""FunASR backend adapter."""

from __future__ import annotations

from pathlib import Path

from FunASRNano.funasr_service import FunASRTranscriber
from FunASRNano.schemas import FunASRSettings


class FunAsrBackend:
    name = "funasr"

    def __init__(self, settings: FunASRSettings, device: str, logger) -> None:
        self.settings = settings
        self.device = device
        self.logger = logger
        self.model_name = settings.model
        self._transcriber = FunASRTranscriber(settings, device, logger)

    def transcribe(self, audio_path: str | Path) -> str:
        return self._transcriber.transcribe(audio_path)
