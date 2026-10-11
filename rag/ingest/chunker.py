"""分块策略。

三种策略（对应面试里必被问的「chunk 大小怎么定」）：

- ``structural``（默认）：先按 Markdown 标题切「节」，再在节内按段落打包到 ``max_chars``，
  保留 ``overlap_chars`` 重叠。优点：不跨越语义边界，标题路径会写进元数据并拼进正文，
  显著改善「一片片段里不知道在讲什么」的问题。
- ``fixed``：不管结构，纯固定窗口 + 重叠。作为消融实验的 baseline 保留。
- ``recursive``：段落 -> 句子 -> 硬切 的递归回退，适合结构混乱的纯文本。

所有策略都保证 ``chunk_id = <doc_id>#<序号>``，同一份语料多次切分结果稳定可复现。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..text import normalize_ws, split_sentences
from ..types import Chunk
from .loader import Document

__all__ = ["Chunker", "ChunkingStats", "chunk_documents", "STRATEGIES"]

STRATEGIES = ("structural", "fixed", "recursive")

_HEADING_RE = re.compile(r"^(\s{0,3})(#{1,6})\s+(.*?)\s*#*\s*$")


@dataclass
class ChunkingStats:
    strategy: str = "structural"
    docs: int = 0
    chunks: int = 0
    total_chars: int = 0

    @property
    def avg_chars(self) -> float:
        return (self.total_chars / self.chunks) if self.chunks else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "docs": self.docs,
            "chunks": self.chunks,
            "avg_chars": round(self.avg_chars, 1),
        }


@dataclass
class _Section:
    heading: str
    body: str
    start: int


def _tail(text: str, n: int) -> str:
    """取尾部 n 个字符，尽量从句子/段落边界开始，避免把句子切两半。"""
    if n <= 0 or len(text) <= n:
        return "" if n <= 0 else text
    tail = text[-n:]
    for i, ch in enumerate(tail):
        if ch in "\n。！？；!?;":
            return tail[i + 1 :].lstrip()
    return tail


def _split_long(text: str, max_chars: int) -> List[str]:
    """把文本拆成**不超过 max_chars 的语义单元（句子级）**，但不做窗口打包。

    这里刻意「只拆不打包」：如果在这一步就把句子填满到 max_chars，后面的滑动窗口就再也
    没有空间留给 overlap，重叠会退化成 0。打包交给 ``_pack`` 统一处理。
    """
    text = text.strip()
    if not text:
        return []
    units: List[str] = []
    for sentence in split_sentences(text):
        if len(sentence) <= max_chars:
            units.append(sentence)
        else:
            for i in range(0, len(sentence), max_chars):
                units.append(sentence[i : i + max_chars])
    return units


def _pack(
    blocks: Sequence[str],
    max_chars: int,
    overlap_chars: int,
    min_chars: int,
) -> List[str]:
    """把段落序列打包成带重叠的窗口。"""
    pieces: List[str] = []
    for block in blocks:
        block = block.strip()
        if block:
            pieces.extend(_split_long(block, max_chars))

    windows: List[str] = []
    current = ""
    for piece in pieces:
        if not current:
            current = piece
            continue
        if len(current) + 2 + len(piece) <= max_chars:
            current = f"{current}\n\n{piece}"
        else:
            windows.append(current)
            tail = _tail(current, overlap_chars)
            # 新窗口是 「上一窗口的尾部重叠 + 本段」，必须按剩余预算裁剪重叠，
            # 而不是直接丢掉（直接丢会导致块大到接近 max_chars 时完全没有重叠）
            budget = max_chars - 2 - len(piece)
            if budget <= 0:
                tail = ""
            elif len(tail) > budget:
                tail = tail[-budget:]
            current = f"{tail}\n\n{piece}" if tail else piece
    if current:
        windows.append(current)

    merged: List[str] = []
    for window in windows:
        if merged and len(window) < min_chars and len(merged[-1]) + len(window) <= max_chars * 1.25:
            merged[-1] = f"{merged[-1]}\n{window}"
        else:
            merged.append(window)
    return [w for w in merged if w.strip()]


def _sections(text: str) -> List[_Section]:
    """按 Markdown 标题切节，heading 为标题路径（如 ``A > B``）。"""
    sections: List[_Section] = []
    stack: List[Tuple[int, str]] = []
    buffer: List[str] = []
    buffer_start = 0
    offset = 0
    heading = ""

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append(_Section(heading=heading, body=body, start=buffer_start))

    for line in text.split("\n"):
        match = _HEADING_RE.match(line)
        if match:
            flush()
            level = len(match.group(2))
            title = match.group(3).strip()
            stack = [item for item in stack if item[0] < level]
            stack.append((level, title))
            heading = " > ".join(title_ for _, title_ in stack)
            buffer = [line.strip()]
            buffer_start = offset
        else:
            if not buffer:
                buffer_start = offset
            buffer.append(line)
        offset += len(line) + 1
    flush()
    return sections


class Chunker:
    def __init__(
        self,
        strategy: str = "structural",
        max_chars: int = 512,
        overlap_chars: int = 96,
        min_chars: int = 80,
        prepend_heading: bool = True,
    ):
        if strategy not in STRATEGIES:
            raise ValueError(f"未知分块策略：{strategy}（可选 {STRATEGIES}）")
        self.strategy = strategy
        self.max_chars = max(64, int(max_chars))
        self.overlap_chars = max(0, min(int(overlap_chars), self.max_chars // 2))
        self.min_chars = max(1, int(min_chars))
        self.prepend_heading = prepend_heading

    # ---- 对外 ---------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg) -> "Chunker":
        return cls(
            strategy=str(cfg.get("chunking.strategy", "structural")),
            max_chars=int(cfg.get("chunking.max_chars", 512)),
            overlap_chars=int(cfg.get("chunking.overlap_chars", 96)),
            min_chars=int(cfg.get("chunking.min_chars", 80)),
        )

    def chunk_document(self, doc: Document) -> List[Chunk]:
        if self.strategy == "fixed":
            windows = _pack(
                [doc.text], self.max_chars, self.overlap_chars, self.min_chars
            )
            blocks: List[Tuple[str, str, int]] = [
                (doc.title or doc.doc_id, window, 0) for window in windows
            ]
        elif self.strategy == "recursive":
            windows = _pack(
                [p for p in doc.text.split("\n\n") if p.strip()],
                self.max_chars,
                self.overlap_chars,
                self.min_chars,
            )
            blocks = [(doc.title or doc.doc_id, window, 0) for window in windows]
        else:
            blocks = []
            for section in _sections(doc.text):
                body = section.body
                # 去掉节内首个标题行（标题已单独记录在 heading 里）
                lines = body.split("\n")
                if lines and _HEADING_RE.match(lines[0]):
                    body = "\n".join(lines[1:]).strip()
                if not body:
                    continue
                for window in _pack(
                    [p for p in body.split("\n\n") if p.strip()],
                    self.max_chars,
                    self.overlap_chars,
                    self.min_chars,
                ):
                    blocks.append((section.heading, window, section.start))

        chunks: List[Chunk] = []
        seen: Dict[str, int] = {}
        for heading, window, start in blocks:
            text = window.strip()
            if not text:
                continue
            fingerprint = text[:200]
            if fingerprint in seen:  # 同一文档内的重复片段只保留一次
                continue
            seen[fingerprint] = len(chunks)
            index = len(chunks)
            chunk_text = (
                f"{heading}\n{text}" if (self.prepend_heading and heading and heading not in text[: len(heading) + 2]) else text
            )
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#{index}",
                    doc_id=doc.doc_id,
                    text=chunk_text,
                    start=int(start),
                    end=int(start) + len(text),
                    meta={
                        "title": doc.title,
                        "heading": heading,
                        "source": doc.meta.get("source", ""),
                        "doc_path": doc.path,
                        "strategy": self.strategy,
                        "n_chars": len(chunk_text),
                        "chunk_index": index,
                    },
                )
            )
        return chunks

    def chunk_documents(self, docs: Iterable[Document]) -> Tuple[List[Chunk], ChunkingStats]:
        chunks: List[Chunk] = []
        stats = ChunkingStats(strategy=self.strategy)
        for doc in docs:
            doc_chunks = self.chunk_document(doc)
            chunks.extend(doc_chunks)
            stats.docs += 1
            stats.chunks += len(doc_chunks)
            stats.total_chars += sum(c.meta.get("n_chars", len(c.text)) for c in doc_chunks)
        return chunks, stats


def chunk_documents(docs: Iterable[Document], cfg) -> Tuple[List[Chunk], ChunkingStats]:
    return Chunker.from_config(cfg).chunk_documents(docs)
