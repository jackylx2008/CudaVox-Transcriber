"""CAM++ voiceprint matching service."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict, cast

from FunASRNano.schemas import CamppSettings, VoiceprintIdentity


class SpeakerInfo(TypedDict):
    speaker_id: str
    speaker_name: str
    local_speaker_first_seen: str
    embedding_path: str
    num_samples: int
    created_at: str
    updated_at: str
    source_files: list[str]


class VoiceprintMetadata(TypedDict):
    speakers: dict[str, SpeakerInfo]


class RankedCandidate(TypedDict):
    speaker_id: str
    info: SpeakerInfo
    score: float


class VoiceprintStore:
    def __init__(self, settings: CamppSettings, device: str, logger) -> None:
        self.settings = settings
        self.device = device
        self.logger = logger
        self.db_dir = Path(settings.db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.db_dir / settings.metadata_file
        self._pipeline: Any | None = None
        self._embedding_cache: dict[str, Any] = {}
        self.metadata = self._load_metadata()
        self._sync_speaker_names()
        self.logger.info(
            "声纹库初始化完成: db_dir=%s, 已登记说话人=%s",
            self.db_dir.resolve(),
            len(self.metadata["speakers"]),
        )

    def _load_metadata(self) -> VoiceprintMetadata:
        if not self.metadata_path.exists():
            self.logger.info("声纹元数据不存在，创建新库: %s", self.metadata_path.resolve())
            return {"speakers": {}}
        self.logger.debug("加载声纹元数据: %s", self.metadata_path.resolve())
        data = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        return cast(VoiceprintMetadata, data)

    def save(self) -> None:
        self.metadata_path.write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.logger.debug("已保存声纹元数据: %s", self.metadata_path.resolve())

    def _sync_speaker_names(self) -> None:
        name_map = self.settings.speaker_name_map
        if not name_map:
            return

        changed = 0
        for speaker_id, speaker_name in name_map.items():
            info = self.metadata.get("speakers", {}).get(speaker_id)
            if info is None:
                self.logger.warning(
                    "VOICEPRINT_NAME_MAP 中的声纹 ID 不存在，已跳过: %s",
                    speaker_id,
                )
                continue
            if info.get("speaker_name") == speaker_name:
                continue
            info["speaker_name"] = speaker_name
            changed += 1

        if changed:
            self.save()
            self.logger.info("已从配置同步声纹姓名映射: %s 个", changed)

    @property
    def pipeline(self) -> Any:
        if self._pipeline is None:
            self._load_pipeline()
        return self._pipeline

    def _load_pipeline(self) -> None:
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        self.logger.info("加载 CAM++ 模型: %s", self.settings.model)
        try:
            self._pipeline = pipeline(
                task=Tasks.speaker_verification,
                model=self.settings.model,
                device=self.device,
            )
        except TypeError:
            self.logger.warning(
                "当前 ModelScope pipeline 不接受 device 参数，改为使用默认设备初始化。"
            )
            self._pipeline = pipeline(
                task=Tasks.speaker_verification,
                model=self.settings.model,
            )

    def extract_embedding(self, audio_path: str | Path):
        import numpy as np

        self.logger.debug("开始提取声纹 embedding: %s", Path(audio_path).resolve())
        pipeline = self.pipeline
        result = cast(dict[str, Any], pipeline([str(audio_path)], output_emb=True))
        embedding = result.get("embs")
        if embedding is None:
            raise RuntimeError("CAM++ 未返回声纹 embedding。")
        array = np.asarray(embedding[0], dtype=np.float32)
        self.logger.debug("声纹 embedding 提取完成: 维度=%s", array.shape[0])
        return self._normalize(array)

    def match_or_register(
        self,
        embedding,
        source_file: str,
        local_speaker: str,
    ) -> VoiceprintIdentity:
        ranked = self._rank_candidates(embedding)
        best = ranked[0] if ranked else None

        if best is not None and best["score"] >= self.settings.similarity_threshold:
            return self._match_existing_candidate(
                best,
                embedding=embedding,
                source_file=source_file,
                match_mode="strict",
            )

        relaxed_candidate = self._select_relaxed_candidate(ranked)
        if relaxed_candidate is not None:
            return self._match_existing_candidate(
                relaxed_candidate,
                embedding=embedding,
                source_file=source_file,
                match_mode="relaxed",
            )

        named_candidate = self._select_named_candidate(ranked)
        if named_candidate is not None:
            return self._match_existing_candidate(
                named_candidate,
                embedding=embedding,
                source_file=source_file,
                match_mode="named",
            )

        return self._create_new(
            embedding,
            source_file,
            local_speaker,
            best["score"] if best is not None else -1.0,
            best["speaker_id"] if best is not None else None,
        )

    def create_transient_identity(self, local_speaker: str) -> VoiceprintIdentity:
        return VoiceprintIdentity(
            speaker_id=local_speaker,
            speaker_name=local_speaker,
            similarity=None,
            is_new=False,
            transient=True,
        )

    def _create_new(
        self,
        embedding,
        source_file: str,
        local_speaker: str,
        best_score: float,
        best_candidate_id: str | None,
    ) -> VoiceprintIdentity:
        speaker_id = f"{self.settings.speaker_prefix}{len(self.metadata['speakers']) + 1:04d}"
        embedding_path = self.db_dir / f"{speaker_id}.npy"
        self._save_embedding(embedding_path, embedding)
        now = self._now()
        self.metadata["speakers"][speaker_id] = {
            "speaker_id": speaker_id,
            "speaker_name": self.settings.speaker_name_map.get(speaker_id, speaker_id),
            "local_speaker_first_seen": local_speaker,
            "embedding_path": str(embedding_path.resolve()),
            "num_samples": 1,
            "created_at": now,
            "updated_at": now,
            "source_files": [source_file],
        }
        self._embedding_cache[speaker_id] = embedding
        self.save()
        self.logger.info(
            "注册新声纹: %s, source=%s, best_similarity=%s, best_candidate=%s",
            speaker_id,
            source_file,
            round(best_score, 4) if best_score >= 0 else None,
            best_candidate_id,
        )
        return VoiceprintIdentity(
            speaker_id=speaker_id,
            speaker_name=self.settings.speaker_name_map.get(speaker_id, speaker_id),
            similarity=round(best_score, 4) if best_score >= 0 else None,
            is_new=True,
            embedding_path=str(embedding_path.resolve()),
        )

    def _update_existing(self, speaker_id: str, embedding, source_file: str):
        info = self.metadata["speakers"][speaker_id]
        current = self._load_embedding(Path(info["embedding_path"]))
        count = int(info.get("num_samples", 1))
        updated = self._normalize((current * count + embedding) / (count + 1))
        self._save_embedding(Path(info["embedding_path"]), updated)
        info["num_samples"] = count + 1
        info["updated_at"] = self._now()
        source_files = set(info.get("source_files", []))
        source_files.add(source_file)
        info["source_files"] = sorted(source_files)
        self.save()
        self.logger.debug(
            "更新已有声纹: %s, 样本数=%s",
            speaker_id,
            info["num_samples"],
        )
        return updated

    def _load_embedding(self, path: Path):
        import numpy as np

        cache_key = str(path.resolve())
        cached = self._embedding_cache.get(cache_key)
        if cached is not None:
            return cached
        embedding = np.load(path)
        self._embedding_cache[cache_key] = embedding
        return embedding

    def _save_embedding(self, path: Path, embedding) -> None:
        import numpy as np

        np.save(path, embedding)
        self._embedding_cache[str(path.resolve())] = embedding

    def _rank_candidates(self, embedding) -> list[RankedCandidate]:
        ranked: list[RankedCandidate] = []
        for speaker_id, info in self.metadata["speakers"].items():
            ref = self._load_embedding(Path(info["embedding_path"]))
            score = self._cosine_similarity(embedding, ref)
            ranked.append(
                {
                    "speaker_id": speaker_id,
                    "info": info,
                    "score": score,
                }
            )
        ranked.sort(key=lambda item: float(item["score"]), reverse=True)
        return ranked

    def _select_relaxed_candidate(
        self,
        ranked: list[RankedCandidate],
    ) -> RankedCandidate | None:
        if not ranked:
            return None
        candidate = ranked[0]
        if float(candidate["score"]) < self.settings.relaxed_similarity_threshold:
            return None
        info = candidate["info"]
        if int(info.get("num_samples", 1)) < self.settings.relaxed_min_samples:
            return None

        second_score = float(ranked[1]["score"]) if len(ranked) > 1 else -1.0
        margin = float(candidate["score"]) - second_score
        if margin < self.settings.relaxed_similarity_margin:
            return None
        return candidate

    def _select_named_candidate(
        self,
        ranked: list[RankedCandidate],
    ) -> RankedCandidate | None:
        if not ranked:
            return None
        candidate = ranked[0]
        info = candidate["info"]
        speaker_id = str(candidate["speaker_id"])
        speaker_name = info.get("speaker_name", speaker_id)
        if speaker_name == speaker_id:
            return None
        if int(info.get("num_samples", 1)) < self.settings.named_min_samples:
            return None
        if float(candidate["score"]) < self.settings.named_similarity_threshold:
            return None

        second_score = float(ranked[1]["score"]) if len(ranked) > 1 else -1.0
        margin = float(candidate["score"]) - second_score
        if margin < self.settings.named_similarity_margin:
            return None
        return candidate

    def _match_existing_candidate(
        self,
        candidate: RankedCandidate,
        embedding,
        source_file: str,
        match_mode: str,
    ) -> VoiceprintIdentity:
        speaker_id = str(candidate["speaker_id"])
        info = candidate["info"]
        score = float(candidate["score"])
        new_embedding = self._update_existing(speaker_id, embedding, source_file)
        self._embedding_cache[speaker_id] = new_embedding
        self.logger.info(
            "命中已有声纹(%s): %s, similarity=%.4f, source=%s",
            match_mode,
            speaker_id,
            score,
            source_file,
        )
        return VoiceprintIdentity(
            speaker_id=speaker_id,
            speaker_name=info.get("speaker_name", speaker_id),
            similarity=round(score, 4),
            is_new=False,
            embedding_path=info["embedding_path"],
        )

    @staticmethod
    def _normalize(vector):
        import numpy as np

        array = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(array)
        if norm == 0:
            return array
        return array / norm

    @staticmethod
    def _cosine_similarity(left, right) -> float:
        import numpy as np

        left = np.asarray(left)
        right = np.asarray(right)
        left_norm = np.linalg.norm(left)
        right_norm = np.linalg.norm(right)
        if left_norm == 0 or right_norm == 0:
            return -1.0
        return float((left @ right) / (left_norm * right_norm))

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()
