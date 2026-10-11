"""自写 BM25（Okapi），不依赖 rank_bm25。

公式（面试必背）：

    score(q, d) = Σ_{t ∈ q} IDF(t) · (tf(t,d) · (k1 + 1)) / (tf(t,d) + k1 · (1 - b + b · |d| / avgdl))
    IDF(t)      = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

用 ``ln(1 + x)`` 形式的 IDF 是为了保证 IDF 恒为正，避免高频词出现负分（经典 BM25 的
``ln((N-df+0.5)/(df+0.5))`` 在 df > N/2 时会取负值，把命中高频词的文档往下压）。

倒排索引 + 只在命中词上累积分数，因此查询复杂度是 O(命中文档数) 而不是 O(全库)。
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

from ..text import tokenize

__all__ = ["BM25Index"]


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = float(k1)
        self.b = float(b)
        self.ids: List[str] = []
        self.doc_len: List[int] = []
        self.postings: Dict[str, List[Tuple[int, int]]] = {}
        self._df: Dict[str, int] = {}
        self._idf_cache: Dict[str, float] = {}
        self.avgdl: float = 0.0

    # ---- 构建 ---------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg) -> "BM25Index":
        return cls(
            k1=float(cfg.get("retrieval.bm25.k1", 1.5)),
            b=float(cfg.get("retrieval.bm25.b", 0.75)),
        )

    def fit(self, chunk_ids: Sequence[str], texts: Sequence[str]) -> "BM25Index":
        if len(chunk_ids) != len(texts):
            raise ValueError("chunk_ids 与 texts 长度必须一致")
        self.ids = [str(i) for i in chunk_ids]
        self.doc_len = []
        self.postings = {}
        self._df = {}
        self._idf_cache = {}
        for doc_index, text in enumerate(texts):
            counts: Dict[str, int] = {}
            for token in tokenize(text):
                counts[token] = counts.get(token, 0) + 1
            self.doc_len.append(sum(counts.values()))
            for token, tf in counts.items():
                self.postings.setdefault(token, []).append((doc_index, tf))
                self._df[token] = self._df.get(token, 0) + 1
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        return self

    # ---- 查询 ---------------------------------------------------------------
    @property
    def size(self) -> int:
        return len(self.ids)

    @property
    def vocabulary(self) -> int:
        return len(self.postings)

    @property
    def df(self) -> Dict[str, int]:
        return self._df

    def idf(self, term: str) -> float:
        if term in self._idf_cache:
            return self._idf_cache[term]
        n = len(self.ids)
        d = self._df.get(term, 0)
        if n == 0 or d == 0:
            value = 0.0
        else:
            value = math.log(1.0 + (n - d + 0.5) / (d + 0.5))
        self._idf_cache[term] = value
        return value

    def search(self, query: str, top_n: int = 20) -> List[Tuple[str, float]]:
        if not self.ids:
            return []
        scores: Dict[int, float] = {}
        avgdl = self.avgdl or 1.0
        for term in set(tokenize(query)):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf(term)
            if idf <= 0:
                continue
            for doc_index, tf in postings:
                dl = self.doc_len[doc_index] or 1
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / avgdl)
                scores[doc_index] = scores.get(doc_index, 0.0) + idf * (tf * (self.k1 + 1.0)) / denom
        if not scores:
            return []
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return [(self.ids[idx], float(score)) for idx, score in ranked[: max(1, int(top_n))]]

    # ---- 持久化 -------------------------------------------------------------
    def save(self, directory: str) -> str:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "bm25.json")
        payload = {
            "k1": self.k1,
            "b": self.b,
            "ids": self.ids,
            "doc_len": self.doc_len,
            "avgdl": self.avgdl,
            "postings": {term: [[i, tf] for i, tf in plist] for term, plist in self.postings.items()},
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        return path

    @classmethod
    def load(cls, directory: str) -> "BM25Index":
        path = os.path.join(directory, "bm25.json")
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        index = cls(k1=float(payload.get("k1", 1.5)), b=float(payload.get("b", 0.75)))
        index.ids = list(payload.get("ids", []))
        index.doc_len = [int(x) for x in payload.get("doc_len", [])]
        index.avgdl = float(payload.get("avgdl", 0.0))
        index.postings = {
            term: [(int(i), int(tf)) for i, tf in plist]
            for term, plist in (payload.get("postings") or {}).items()
        }
        index._df = {term: len(plist) for term, plist in index.postings.items()}
        index._idf_cache = {}
        return index

    def stats(self) -> Dict[str, float]:
        return {
            "docs": float(len(self.ids)),
            "vocabulary": float(len(self.postings)),
            "avgdl": round(self.avgdl, 2),
            "k1": self.k1,
            "b": self.b,
        }
