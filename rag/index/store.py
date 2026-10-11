"""向量存储：numpy 暴力检索（默认）与 faiss（可选）。

语料规模在十万块以内时，float32 归一化矩阵的矩阵乘法（``matrix @ q``）在 CPU 上通常
几毫秒内完成，比引入 faiss 更简单也更可复现；因此默认后端是 numpy，faiss 通过
``index.backend=faiss`` 打开，用于十几万块以上的场景。

持久化目录结构与 chunk 元数据分离：
    <index_dir>/vectors.npy   # (n, dim) float32
    <index_dir>/chunks.jsonl  # 每行一个 Chunk
    <index_dir>/meta.json     # dim / backend / count / embedder / chunking / corpus_signature
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..types import Chunk, Scored

__all__ = ["BaseVectorStore", "NumpyVectorStore", "FaissVectorStore", "build_store", "load_store"]


class BaseVectorStore:
    backend = "base"

    def __init__(self, dim: int):
        self.dim = int(dim)
        self.chunks: List[Chunk] = []
        self.matrix: Optional[np.ndarray] = None
        # 索引元数据：无论是新建还是从磁盘加载，这个属性都必须存在
        self.meta: Dict[str, Any] = {}

    def __len__(self) -> int:
        return len(self.chunks)

    # ---- 写入 ---------------------------------------------------------------
    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> int:
        # 先规整形状再比数量：单个向量（如 encode_query 的返回值）是 1-D 的，
        # 若先做长度比较，会把「1 个块 + 1 个向量」误判成数量不一致
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.size == 0:
            return 0
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if len(chunks) != len(vectors):
            raise ValueError("chunks 与 vectors 数量不一致")
        if self.dim and vectors.shape[1] != self.dim:
            raise ValueError(f"向量维度不一致：索引 {self.dim}，新数据 {vectors.shape[1]}")
        if self.matrix is None:
            self.matrix = vectors
        else:
            self.matrix = np.vstack([self.matrix, vectors])
        self.chunks.extend(chunks)
        return len(chunks)

    # ---- 查询 ---------------------------------------------------------------
    def search(self, query_vector: np.ndarray, top_n: int = 10) -> List[Scored]:
        if self.matrix is None or not self.chunks:
            return []
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.matrix.shape[1]:
            raise ValueError(f"查询向量维度 {query.shape[0]} 与索引 {self.matrix.shape[1]} 不一致")
        scores = self.matrix @ query
        k = max(1, min(int(top_n), scores.shape[0]))
        if k < scores.shape[0]:
            candidate = np.argpartition(-scores, k - 1)[:k]
        else:
            candidate = np.arange(scores.shape[0])
        order = candidate[np.argsort(-scores[candidate])]
        return [
            Scored(self.chunks[int(i)], float(scores[int(i)]), source="vector", detail={"rank": rank})
            for rank, i in enumerate(order, start=1)
        ]

    def get(self, chunk_id: str) -> Optional[Chunk]:
        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None

    # ---- 持久化 -------------------------------------------------------------
    def save(self, directory: str, extra: Optional[Dict[str, Any]] = None) -> str:
        os.makedirs(directory, exist_ok=True)
        if self.matrix is None:
            self.matrix = np.zeros((0, self.dim), dtype=np.float32)
        np.save(os.path.join(directory, "vectors.npy"), self.matrix.astype(np.float32))
        with open(os.path.join(directory, "chunks.jsonl"), "w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        meta = {
            "dim": self.dim,
            "backend": self.backend,
            "count": len(self.chunks),
        }
        meta.update(extra or {})
        with open(os.path.join(directory, "meta.json"), "w", encoding="utf-8") as handle:
            json.dump(meta, handle, ensure_ascii=False, indent=2)
        self.meta = meta
        return directory

    @classmethod
    def load(cls, directory: str) -> "BaseVectorStore":
        with open(os.path.join(directory, "meta.json"), "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        store = cls(dim=int(meta.get("dim", 0) or 0))
        store.meta = meta
        vectors_path = os.path.join(directory, "vectors.npy")
        if os.path.isfile(vectors_path):
            matrix = np.load(vectors_path)
            store.matrix = np.asarray(matrix, dtype=np.float32)
            if not store.dim and store.matrix.size:
                store.dim = int(store.matrix.shape[1])
        chunks_path = os.path.join(directory, "chunks.jsonl")
        if os.path.isfile(chunks_path):
            with open(chunks_path, "r", encoding="utf-8") as handle:
                store.chunks = [Chunk.from_dict(json.loads(line)) for line in handle if line.strip()]
        return store

    def stats(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "dim": self.dim,
            "count": len(self.chunks),
        }


class NumpyVectorStore(BaseVectorStore):
    backend = "numpy"


class FaissVectorStore(BaseVectorStore):
    """faiss 加速后端；faiss 不可用时自动退回 numpy 暴力检索。"""

    backend = "faiss"

    def __init__(self, dim: int):
        super().__init__(dim)
        self._index = None
        self._dirty = True

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> int:
        added = super().add(chunks, vectors)
        self._dirty = True
        return added

    def _ensure_index(self):
        if not self._dirty and self._index is not None:
            return self._index
        try:
            import faiss  # type: ignore
        except ImportError:
            self._index = None
            return None
        index = faiss.IndexFlatIP(self.matrix.shape[1] if self.matrix is not None else self.dim)
        if self.matrix is not None and self.matrix.size:
            index.add(self.matrix)
        self._index = index
        self._dirty = False
        return index

    def search(self, query_vector: np.ndarray, top_n: int = 10) -> List[Scored]:
        index = self._ensure_index()
        if index is None or self.matrix is None or not self.chunks:
            return super().search(query_vector, top_n)
        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        k = max(1, min(int(top_n), len(self.chunks)))
        scores, indices = index.search(query, k)
        results: List[Scored] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0:
                continue
            results.append(
                Scored(self.chunks[int(idx)], float(score), source="vector", detail={"rank": rank})
            )
        return results


def build_store(cfg, dim: int, log=None) -> BaseVectorStore:
    backend = str(cfg.get("index.backend", "auto")).lower()
    if backend in ("faiss", "auto"):
        try:
            import faiss  # type: ignore  # noqa: F401

            return FaissVectorStore(dim=dim)
        except Exception as exc:
            if backend == "faiss" and log:
                log.warning("faiss 不可用（%s），退回 numpy 后端", exc)
    return NumpyVectorStore(dim=dim)


def load_store(directory: str, log=None) -> BaseVectorStore:
    meta_path = os.path.join(directory, "meta.json")
    backend = "numpy"
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as handle:
                backend = str(json.load(handle).get("backend", "numpy")).lower()
        except Exception:
            backend = "numpy"
    cls = FaissVectorStore if backend == "faiss" else NumpyVectorStore
    return cls.load(directory)
