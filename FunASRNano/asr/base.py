"""ASR backend protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class AsrBackend(Protocol):
    name: str
    model_name: str
    device: str

    def transcribe(self, audio_path: str | Path) -> str:
        """Transcribe a single audio clip."""
        ...
