"""CAM++ voiceprint matching service."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from cudavox_transcriber.schemas import CamppSettings, VoiceprintIdentity


class VoiceprintStore:
    def __init__(self, settings: CamppSettings, device: str, logger) -> None:
        self.settings = settings
        self.device = device
        self.logger = logger
        self.db_dir = Path(settings.db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.db_dir / settings.metadata_file
        self._pipeline = None
        self._embedding_cache: dict[str, object] = {}
        self.metadata = self._load_metadata()
        self.logger.info(
            "声纹库初始化完成: db_dir=%s, 已登记说话人=%s",
            self.db_dir.resolve(),
            len(self.metadata.get('speakers', {})),
        )

    def _load_metadata(self) -> dict:
        if not self.metadata_path.exists():
            self.logger.info("声纹元数据不存在，创建新库: %s", self.metadata_path.resolve())
            return {"speakers": {}}
        self.logger.debug("加载声纹元数据: %s", self.metadata_path.resolve())
        return json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def save(self) -> None:
        self.metadata_path.write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.logger.debug("已保存声纹元数据: %s", self.metadata_path.resolve())

    @property
    def pipeline(self):
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
        result = self.pipeline([str(audio_path)], output_emb=True)
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
        best_id = None
        best_score = -1.0

        for speaker_id, info in self.metadata["speakers"].items():
            ref = self._load_embedding(Path(info["embedding_path"]))
            score = self._cosine_similarity(embedding, ref)
            if score > best_score:
                best_score = score
                best_id = speaker_id

        if best_id and best_score >= self.settings.similarity_threshold:
            info = self.metadata["speakers"][best_id]
            new_embedding = self._update_existing(best_id, embedding, source_file)
            self._embedding_cache[best_id] = new_embedding
            self.logger.info(
                "命中已有声纹: %s, similarity=%.4f, source=%s",
                best_id,
                best_score,
                source_file,
            )
            return VoiceprintIdentity(
                speaker_id=best_id,
                speaker_name=info.get("speaker_name", best_id),
                similarity=round(best_score, 4),
                is_new=False,
                embedding_path=info["embedding_path"],
            )

        return self._create_new(embedding, source_file, local_speaker, best_score)

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
    ) -> VoiceprintIdentity:
        speaker_id = f"{self.settings.speaker_prefix}{len(self.metadata['speakers']) + 1:04d}"
        embedding_path = self.db_dir / f"{speaker_id}.npy"
        self._save_embedding(embedding_path, embedding)
        now = self._now()
        self.metadata["speakers"][speaker_id] = {
            "speaker_id": speaker_id,
            "speaker_name": speaker_id,
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
            "注册新声纹: %s, source=%s, best_similarity=%s",
            speaker_id,
            source_file,
            round(best_score, 4) if best_score >= 0 else None,
        )
        return VoiceprintIdentity(
            speaker_id=speaker_id,
            speaker_name=speaker_id,
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
