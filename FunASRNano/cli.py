"""CLI entrypoint."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from logging_config import get_logger, setup_logger

from FunASRNano.audio import resolve_audio_files
from FunASRNano.config import load_settings
from FunASRNano.pipeline import CudaVoxPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用 FunASR + pyannote + CAM++ 进行中文语音转写和说话人区分。"
    )
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--env-file", default="common.env", help="环境变量文件路径")
    parser.add_argument("--input", help="单个音频文件或目录，优先覆盖配置项")
    parser.add_argument("--output-dir", help="输出目录，优先覆盖配置项")
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="保留中间切分音频和标准化 wav",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    setup_logger(log_level=logging.INFO, reset_log=True)
    logger = get_logger(__name__)
    args = build_parser().parse_args(argv)
    logger.info("CLI 启动，开始加载配置。")
    settings = load_settings(args.config, args.env_file)
    level = getattr(logging, settings.app.log_level.upper(), logging.INFO)
    setup_logger(log_level=level, reset_log=True)
    logger = get_logger(__name__)

    if args.input:
        settings.app.input_path = args.input
        settings.app.input_files = [args.input]
    if args.output_dir:
        settings.app.output_dir = args.output_dir
        settings.app.temp_dir = str(Path(args.output_dir) / "tmp")
        settings.campp.db_dir = str(Path(args.output_dir) / "voiceprints")
    if args.keep_temp:
        settings.app.cleanup_temp = False

    logger.info(
        "运行参数已生效: input=%s, input_files=%s, output_dir=%s, keep_temp=%s",
        settings.app.input_path,
        len(settings.app.input_files),
        settings.app.output_dir,
        not settings.app.cleanup_temp,
    )
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
            pipeline.logger.info("输出文件: %s", result)
        except Exception:
            failed += 1
            pipeline.logger.exception("处理失败: %s", file_path)

    logger.info("CLI 结束，成功=%s，失败=%s", len(files) - failed, failed)
    return 1 if failed else 0
