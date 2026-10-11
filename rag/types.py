"""贯穿全链路的数据结构。

约定：
- chunk_id 形如 ``<doc_id>#<序号>``，序号从 0 开始，保证同一份语料多次切分结果稳定。
- 所有对外输出的对象都提供 ``to_dict()``，方便 CLI 打印 JSON、API 序列化、评测落盘。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = [
    "Chunk",
    "Scored",
    "Citation",
    "Answer",
    "RetrieveResult",
    "IngestReport",
]


@dataclass
class Chunk:
    """一个可检索的最小单元。"""

    chunk_id: str
    doc_id: str
    text: str
    start: int = 0
    end: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def heading(self) -> str:
        return str(self.meta.get("heading", ""))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        return cls(
            chunk_id=str(data["chunk_id"]),
            doc_id=str(data["doc_id"]),
            text=str(data.get("text", "")),
            start=int(data.get("start", 0) or 0),
            end=int(data.get("end", 0) or 0),
            meta=dict(data.get("meta") or {}),
        )


@dataclass
class Scored:
    """带分数的候选片段；``source`` 标明它来自哪条链路。"""

    chunk: Chunk
    score: float
    source: str = "fused"
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def doc_id(self) -> str:
        return self.chunk.doc_id

    def to_dict(self, with_text: bool = True) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "chunk_id": self.chunk.chunk_id,
            "doc_id": self.chunk.doc_id,
            "score": round(float(self.score), 6),
            "source": self.source,
            "detail": dict(self.detail),
        }
        if with_text:
            out["text"] = self.chunk.text
            out["meta"] = dict(self.chunk.meta)
        return out


@dataclass
class Citation:
    """答案里的 [n] 标记与原始片段之间的可追溯映射。"""

    index: int
    doc_id: str
    chunk_id: str
    quote: str
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "quote": self.quote,
            "score": round(float(self.score), 6),
        }


@dataclass
class Answer:
    question: str
    text: str
    citations: List[Citation] = field(default_factory=list)
    refused: bool = False
    contexts: List[Scored] = field(default_factory=list)
    latency_ms: Dict[str, float] = field(default_factory=dict)
    debug: Dict[str, Any] = field(default_factory=dict)

    @property
    def cited_doc_ids(self) -> List[str]:
        seen: List[str] = []
        for cite in self.citations:
            if cite.doc_id not in seen:
                seen.append(cite.doc_id)
        return seen

    def to_dict(self, with_contexts: bool = True) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "question": self.question,
            "answer": self.text,
            "text": self.text,
            "citations": [c.to_dict() for c in self.citations],
            "refused": bool(self.refused),
            "latency_ms": {k: round(float(v), 3) for k, v in self.latency_ms.items()},
            "debug": self.debug,
        }
        if with_contexts:
            out["contexts"] = [c.to_dict(with_text=True) for c in self.contexts]
        return out


@dataclass
class RetrieveResult:
    question: str
    queries: List[str] = field(default_factory=list)
    vector_hits: List[Scored] = field(default_factory=list)
    bm25_hits: List[Scored] = field(default_factory=list)
    fused: List[Scored] = field(default_factory=list)
    ranked: List[Scored] = field(default_factory=list)
    latency_ms: Dict[str, float] = field(default_factory=dict)
    debug: Dict[str, Any] = field(default_factory=dict)

    @property
    def ranked_doc_ids(self) -> List[str]:
        seen: List[str] = []
        for item in self.ranked:
            if item.doc_id not in seen:
                seen.append(item.doc_id)
        return seen

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "queries": list(self.queries),
            "vector_hits": [s.to_dict(with_text=False) for s in self.vector_hits],
            "bm25_hits": [s.to_dict(with_text=False) for s in self.bm25_hits],
            "fused": [s.to_dict(with_text=False) for s in self.fused],
            "ranked": [s.to_dict(with_text=True) for s in self.ranked],
            "latency_ms": {k: round(float(v), 3) for k, v in self.latency_ms.items()},
            "debug": self.debug,
        }


@dataclass
class IngestReport:
    docs: int = 0
    chunks: int = 0
    index_dir: str = ""
    elapsed_ms: float = 0.0
    files: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "docs": self.docs,
            "chunks": self.chunks,
            "index_dir": self.index_dir,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "files": list(self.files),
            "skipped": list(self.skipped),
        }

    def summary(self) -> str:
        return (
            f"索引完成：{self.docs} 篇文档 -> {self.chunks} 个分块，"
            f"耗时 {self.elapsed_ms:.0f} ms，索引目录 {self.index_dir}"
        )
