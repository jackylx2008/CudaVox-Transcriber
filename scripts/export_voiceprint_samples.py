"""Export manual-review audio samples for voiceprint name mapping."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from logging_config import get_logger, setup_logger  # noqa: E402

from FunASRNano.audio import cut_audio_clip, ensure_dir, ensure_ffmpeg  # noqa: E402
from FunASRNano.config import load_settings  # noqa: E402


LOGGER = get_logger(__name__)


@dataclass
class SampleCandidate:
    speaker_id: str
    speaker_name: str
    input_file: Path
    result_json: Path
    start: float
    end: float
    duration: float
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export per-speaker review samples from output JSON files.",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--env-file", default="common.env", help="Path to common.env")
    parser.add_argument(
        "--results-root",
        default="output",
        help="Directory containing transcription result JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/voiceprint_samples",
        help="Directory to write sample WAV files and CSV manifest.",
    )
    parser.add_argument(
        "--manifest-name",
        default="voiceprint_samples.csv",
        help="CSV manifest file name inside the output directory.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=10.0,
        help="Minimum raw segment duration in seconds.",
    )
    parser.add_argument(
        "--samples-per-speaker",
        type=int,
        default=3,
        help="Maximum number of samples to export per speaker_id.",
    )
    return parser.parse_args()


def iter_result_jsons(results_root: Path) -> list[Path]:
    json_files = []
    for path in sorted(results_root.rglob("*.json")):
        if path.name == "speakers.json" and path.parent.name == "voiceprints":
            continue
        json_files.append(path)
    return json_files


def load_candidates(
    results_root: Path, min_duration: float
) -> dict[str, list[SampleCandidate]]:
    grouped: dict[str, list[SampleCandidate]] = {}
    for path in iter_result_jsons(results_root):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.warning("跳过无法读取的 JSON: %s, error=%s", path.resolve(), exc)
            continue

        input_file = payload.get("input_file")
        raw_segments = payload.get("raw_segments")
        if not input_file or not isinstance(raw_segments, list):
            LOGGER.debug("跳过非转写结果 JSON: %s", path.resolve())
            continue

        input_path = Path(input_file)
        for item in raw_segments:
            speaker_id = str(item.get("speaker_id") or "").strip()
            text = str(item.get("text") or "").strip()
            if not speaker_id or not text:
                continue

            start = float(item.get("start", 0.0))
            end = float(item.get("end", 0.0))
            duration = float(item.get("duration", max(0.0, end - start)))
            if duration < min_duration:
                continue

            candidate = SampleCandidate(
                speaker_id=speaker_id,
                speaker_name=str(item.get("speaker_name") or speaker_id).strip()
                or speaker_id,
                input_file=input_path,
                result_json=path,
                start=start,
                end=end,
                duration=duration,
                text=text,
            )
            grouped.setdefault(speaker_id, []).append(candidate)
    return grouped


def select_samples(
    grouped: dict[str, list[SampleCandidate]],
    samples_per_speaker: int,
) -> dict[str, list[SampleCandidate]]:
    selected: dict[str, list[SampleCandidate]] = {}
    for speaker_id, candidates in grouped.items():
        ordered = sorted(
            candidates,
            key=lambda item: (-item.duration, str(item.input_file).lower(), item.start),
        )
        selected[speaker_id] = ordered[:samples_per_speaker]
    return selected


def sanitize_filename(name: str) -> str:
    normalized = re.sub(r"[^\w\-.]+", "_", name, flags=re.UNICODE).strip("_")
    return normalized or "sample"


def export_samples(
    selected: dict[str, list[SampleCandidate]],
    output_dir: Path,
    manifest_path: Path,
    ffmpeg_bin: str,
) -> tuple[int, int]:
    written_rows: list[dict[str, str | float | int]] = []
    exported_count = 0
    skipped_missing_input = 0

    ensure_dir(output_dir)
    for speaker_id, samples in selected.items():
        speaker_dir = ensure_dir(output_dir / speaker_id)
        for index, sample in enumerate(samples, start=1):
            if not sample.input_file.exists():
                skipped_missing_input += 1
                LOGGER.warning("原始音频不存在，跳过: %s", sample.input_file.resolve())
                continue

            source_stem = sanitize_filename(sample.input_file.stem)
            output_file = speaker_dir / (
                f"{speaker_id}__{index:02d}__{source_stem}__"
                f"{int(round(sample.start * 1000)):010d}.wav"
            )
            cut_audio_clip(
                input_path=sample.input_file,
                start=sample.start,
                duration=sample.duration,
                output_path=output_file,
                ffmpeg_bin=ffmpeg_bin,
            )
            written_rows.append(
                {
                    "speaker_id": speaker_id,
                    "speaker_name": sample.speaker_name,
                    "sample_index": index,
                    "audio_path": str(output_file.resolve()),
                    "source_file": str(sample.input_file.resolve()),
                    "result_json": str(sample.result_json.resolve()),
                    "start": round(sample.start, 3),
                    "end": round(sample.end, 3),
                    "duration": round(sample.duration, 3),
                    "text": sample.text,
                }
            )
            exported_count += 1

    with manifest_path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=[
                "speaker_id",
                "speaker_name",
                "sample_index",
                "audio_path",
                "source_file",
                "result_json",
                "start",
                "end",
                "duration",
                "text",
            ],
        )
        writer.writeheader()
        writer.writerows(written_rows)

    return exported_count, skipped_missing_input


def main() -> int:
    args = parse_args()
    settings = load_settings(args.config, args.env_file)
    level = getattr(logging, settings.app.log_level.upper(), logging.INFO)
    setup_logger(log_level=level)

    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    manifest_path = output_dir / args.manifest_name

    ensure_ffmpeg(settings.app.ffmpeg_bin)
    candidates = load_candidates(results_root, args.min_duration)
    selected = select_samples(candidates, args.samples_per_speaker)
    exported_count, skipped_missing_input = export_samples(
        selected=selected,
        output_dir=output_dir,
        manifest_path=manifest_path,
        ffmpeg_bin=settings.app.ffmpeg_bin,
    )

    LOGGER.info(
        "声纹样本导出完成: 说话人数=%s, 导出样本=%s, 缺失原始音频=%s, 清单=%s",
        len(selected),
        exported_count,
        skipped_missing_input,
        manifest_path.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
