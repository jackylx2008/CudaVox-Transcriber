"""Cut audio clips from an SRT timeline."""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from FunASRNano.audio import cut_audio_clip, ensure_dir, ensure_ffmpeg  # noqa: E402
from FunASRNano.config import load_settings  # noqa: E402
from FunASRNano.logging_config import get_logger  # noqa: E402
from FunASRNano.logging_utils import setup_project_logger  # noqa: E402
from FunASRNano.schemas import TranscriptSegment  # noqa: E402
from FunASRNano.transcript_io import load_srt_segments  # noqa: E402


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cut audio clips by SRT subtitle time ranges.",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--env-file", default="common.env", help="Path to common.env")
    parser.add_argument("--audio", required=True, help="Source audio file")
    parser.add_argument("--srt", required=True, help="SRT subtitle file")
    parser.add_argument(
        "--output-dir",
        help="Directory to write clips. Defaults to output/srt_clips/<audio>__<srt>.",
    )
    parser.add_argument(
        "--manifest-name",
        default="clips.csv",
        help="CSV manifest file name inside the output directory.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.05,
        help="Skip SRT ranges shorter than this many seconds.",
    )
    parser.add_argument(
        "--prefix",
        default="clip",
        help="Filename prefix for generated clips.",
    )
    return parser.parse_args()


def sanitize_filename(value: str) -> str:
    normalized = re.sub(r"[^\w\-.]+", "_", value, flags=re.UNICODE).strip("_")
    return normalized or "segment"


def default_output_dir(audio_path: Path, srt_path: Path) -> Path:
    name = f"{sanitize_filename(audio_path.stem)}__{sanitize_filename(srt_path.stem)}"
    return Path("output") / "srt_clips" / name


def clip_filename(prefix: str, index: int, segment: TranscriptSegment) -> str:
    speaker = sanitize_filename(segment.speaker_name or segment.speaker_label)
    start_ms = int(round(segment.start * 1000))
    parts = [sanitize_filename(prefix), f"{index:04d}"]
    if speaker:
        parts.append(speaker)
    parts.append(f"{start_ms:010d}")
    return "__".join(parts) + ".wav"


def write_manifest(
    manifest_path: Path,
    rows: list[dict[str, str | int | float]],
) -> None:
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=[
                "index",
                "audio_path",
                "source_audio",
                "srt_path",
                "start",
                "end",
                "duration",
                "speaker_name",
                "text",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    setup_project_logger(log_level=logging.INFO, reset_log=True)
    args = parse_args()
    settings = load_settings(args.config, args.env_file)
    level = getattr(logging, settings.app.log_level.upper(), logging.INFO)
    setup_project_logger(log_level=level, reset_log=True)

    audio_path = Path(args.audio)
    srt_path = Path(args.srt)
    if not audio_path.exists():
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")
    if not srt_path.exists():
        raise FileNotFoundError(f"SRT 文件不存在: {srt_path}")

    ensure_ffmpeg(settings.app.ffmpeg_bin)
    output_dir = ensure_dir(args.output_dir or default_output_dir(audio_path, srt_path))
    manifest_path = output_dir / args.manifest_name
    segments = load_srt_segments(srt_path)

    rows: list[dict[str, str | int | float]] = []
    skipped = 0
    for index, segment in enumerate(segments, start=1):
        duration = segment.duration
        if duration < args.min_duration:
            skipped += 1
            continue

        output_path = output_dir / clip_filename(args.prefix, index, segment)
        cut_audio_clip(
            input_path=audio_path,
            start=segment.start,
            duration=duration,
            output_path=output_path,
            ffmpeg_bin=settings.app.ffmpeg_bin,
        )
        rows.append(
            {
                "index": index,
                "audio_path": str(output_path.resolve()),
                "source_audio": str(audio_path.resolve()),
                "srt_path": str(srt_path.resolve()),
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "duration": round(duration, 3),
                "speaker_name": segment.speaker_name,
                "text": segment.text,
            }
        )

    write_manifest(manifest_path, rows)
    LOGGER.info(
        "SRT 切片完成: 输入片段=%s, 导出=%s, 跳过=%s, 清单=%s",
        len(segments),
        len(rows),
        skipped,
        manifest_path.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
