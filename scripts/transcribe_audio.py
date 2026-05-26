"""Standalone audio-to-transcript workflow."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from FunASRNano.audio import ensure_dir, resolve_audio_files  # noqa: E402
from FunASRNano.config import load_settings  # noqa: E402
from FunASRNano.logging_config import get_logger  # noqa: E402
from FunASRNano.logging_utils import setup_project_logger  # noqa: E402
from FunASRNano.pipeline import CudaVoxPipeline  # noqa: E402
from FunASRNano.transcript_io import write_transcript_outputs  # noqa: E402


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe audio files into TranscriptDocument outputs.",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--env-file", default="common.env", help="Path to common.env")
    parser.add_argument(
        "--input",
        help="Audio file or directory. Overrides configured input settings.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to write transcript outputs. Defaults to configured output_dir.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep normalized WAV and ASR segment audio files.",
    )
    return parser.parse_args()


def main() -> int:
    setup_project_logger(log_level=logging.INFO, reset_log=True)
    args = parse_args()
    settings = load_settings(args.config, args.env_file)
    level = getattr(logging, settings.app.log_level.upper(), logging.INFO)
    setup_project_logger(log_level=level, reset_log=True)

    if args.input:
        settings.app.input_path = args.input
        settings.app.input_files = [args.input]
    if args.output_dir:
        settings.app.output_dir = args.output_dir
        settings.app.temp_dir = str(Path(args.output_dir) / "tmp")
        settings.campp.db_dir = str(Path(args.output_dir) / "voiceprints")
    if args.keep_temp:
        settings.app.cleanup_temp = False

    files = resolve_audio_files(
        settings.app.input_files,
        settings.app.input_path,
        settings.app.supported_extensions,
    )
    pipeline = CudaVoxPipeline(settings)

    failed = 0
    for file_path in files:
        try:
            document = pipeline.transcribe_file(file_path)
            output_dir = ensure_dir(Path(settings.app.output_dir) / file_path.stem)
            written = write_transcript_outputs(
                document=document,
                output_dir=output_dir,
                output_stem=file_path.stem,
                settings=settings.output,
                logger=LOGGER,
            )
            if settings.app.cleanup_temp:
                temp_dir = Path(settings.app.temp_dir) / file_path.stem
                shutil.rmtree(temp_dir, ignore_errors=True)
            LOGGER.info("转写完成: %s, 输出=%s", file_path, written)
        except Exception:
            failed += 1
            LOGGER.exception("转写失败: %s", file_path)

    LOGGER.info("独立转写工作流结束，成功=%s，失败=%s", len(files) - failed, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
