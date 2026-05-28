"""Internal audio transcription workflow entrypoint.

Use the project root `main.py` for normal CLI runs.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from logging_config import get_logger, setup_logger  # noqa: E402

from FunASRNano.audio import resolve_audio_files  # noqa: E402
from FunASRNano.config import load_settings  # noqa: E402
from FunASRNano.pipeline import CudaVoxPipeline  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe audio files with the configured ASR backend."
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--env-file", default="common.env", help="Path to common.env")
    parser.add_argument("--input", help="Audio file or directory")
    parser.add_argument("--output-dir", help="Directory for transcript outputs")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temp audio")
    return parser


def main(argv: list[str] | None = None) -> int:
    setup_logger(log_level=logging.INFO, reset_log=True)
    logger = get_logger(__name__)
    args = build_parser().parse_args(argv)
    settings = load_settings(args.config, args.env_file)

    if args.input:
        settings.app.input_path = args.input
        settings.app.input_files = [args.input]
    if args.output_dir:
        settings.app.output_dir = args.output_dir
        settings.app.temp_dir = str(Path(args.output_dir) / "tmp")
        settings.campp.db_dir = str(Path(args.output_dir) / "voiceprints")
    if args.keep_temp:
        settings.app.cleanup_temp = False

    pipeline = CudaVoxPipeline(settings)
    files = resolve_audio_files(
        settings.app.input_files,
        settings.app.input_path,
        settings.app.supported_extensions,
    )
    logger.info("待处理音频数量: %s", len(files))

    failed = 0
    for file_path in files:
        try:
            result = pipeline.process_file(file_path)
            logger.info("输出文件: %s", result)
        except Exception:
            failed += 1
            logger.exception("处理失败: %s", file_path)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
