"""查询改写：identity / keyword / multi_query / hyde。

为什么需要它：用户提问的用词分布与文档的用词分布常常对不上（词汇鸿沟）。BM25 对
「同义不同词」无能为力，向量对「专有名词、缩写、数字」不敏感，而查询改写是在
**检索前**用很低的成本同时改善两条链路的召回。

四种策略的适用条件：
- ``identity``   ：baseline，用于消融实验对照；
- ``keyword``    ：离线可用，把问题压成关键词，缓解长问句里噪声词稀释 BM25 分数；
- ``multi_query``：LLM 生成 N 个改写问句，覆盖不同表述，召回并集更大；
- ``hyde``       ：LLM 先写一段「假想答案」，用它做向量检索，把 query-to-doc 匹配
                   变成 doc-to-doc 匹配，对短问题收益明显。

``auto``：有可用 LLM 时用 multi_query，否则退化为 keyword —— 保证离线也能跑。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..text import keywords

__all__ = ["RewriteResult", "QueryRewriter", "STRATEGIES"]

STRATEGIES = ("auto", "identity", "keyword", "multi_query", "hyde")

_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)、])\s*")


@dataclass
class RewriteResult:
    queries: List[str] = field(default_factory=list)
    strategy: str = "identity"
    debug: Dict[str, Any] = field(default_factory=dict)


class QueryRewriter:
    def __init__(self, cfg, llm=None):
        self.cfg = cfg
        self.llm = llm
        self.enabled = bool(cfg.get("retrieval.rewrite.enabled", True))
        self.strategy = str(cfg.get("retrieval.rewrite.strategy", "auto")).lower()
        self.num_queries = int(cfg.get("retrieval.rewrite.num_queries", 3))
        self.max_keywords = int(cfg.get("retrieval.rewrite.max_keywords", 8))
        self.bm25 = None  # 由 pipeline 注入，用于带 IDF 的关键词抽取

    @classmethod
    def from_config(cls, cfg, llm=None) -> "QueryRewriter":
        return cls(cfg, llm=llm)

    # ---- 各策略 -------------------------------------------------------------
    def _identity(self, question: str) -> List[str]:
        return [question]

    def _keyword(self, question: str) -> List[str]:
        df = self.bm25.df if self.bm25 is not None else None
        total = self.bm25.size if self.bm25 is not None else 0
        terms = keywords(question, limit=self.max_keywords, df=df, total_docs=total)
        if not terms:
            return [question]
        squeezed = " ".join(terms)
        return [question, squeezed] if squeezed.strip() and squeezed != question else [question]

    def _call_llm(self, system: str, user: str) -> Optional[str]:
        if self.llm is None or not getattr(self.llm, "available", False):
            return None
        try:
            return self.llm.complete(system=system, user=user, temperature=0.3, max_tokens=400)
        except Exception:
            return None

    def _multi_query(self, question: str) -> List[str]:
        system = (
            "你是检索系统的查询改写器。请把用户问题改写成多个不同表述的检索查询，"
            "覆盖同义词、术语全称、可能出现在文档里的关键词。"
            f"输出 {self.num_queries} 行，每行一个查询，不要编号，不要解释。"
        )
        text = self._call_llm(system, question)
        if not text:
            return self._keyword(question)
        lines: List[str] = []
        for raw in text.splitlines():
            line = _BULLET.sub("", raw).strip().strip('"').strip()
            if len(line) < 2 or line in lines:
                continue
            lines.append(line)
        if not lines:
            return self._keyword(question)
        return [question] + lines[: self.num_queries - 1]

    def _hyde(self, question: str) -> List[str]:
        system = (
            "请基于常识写一段 100 字以内的、假设性的答案段落，用于向量检索；"
            "不要提问、不要解释、不要说我无法确定。"
        )
        text = self._call_llm(system, question)
        if not text:
            return self._keyword(question)
        return [question, text.strip().replace("\n", " ")[:300]]

    # ---- 对外 ---------------------------------------------------------------
    def rewrite(self, question: str) -> RewriteResult:
        question = (question or "").strip()
        if not question:
            return RewriteResult(queries=[], strategy="identity", debug={"note": "空问题"})
        if not self.enabled:
            return RewriteResult(queries=[question], strategy="identity", debug={"enabled": False})

        strategy = self.strategy
        if strategy == "auto":
            # auto 的语义：**只在能真正改写时才改写**。
            # 无 LLM 时退化为 identity 而不是 keyword —— 实测（12 题、哈希与 BGE 两套基线）
            # 关键词压缩产生的第二路查询会稀释 RRF 融合排名：BGE 基线上 Recall@1 0.90→0.75、
            # nDCG@10 1.000→0.920，是明确的负收益。keyword 仍作为显式选项保留供消融对比。
            strategy = (
                "multi_query"
                if (self.llm is not None and getattr(self.llm, "available", False))
                else "identity"
            )

        handler = {
            "identity": self._identity,
            "keyword": self._keyword,
            "multi_query": self._multi_query,
            "hyde": self._hyde,
        }.get(strategy, self._identity)
        queries = [q for q in handler(question) if q and q.strip()]
        # 去重且保持顺序
        seen = set()
        unique: List[str] = []
        for query in queries:
            key = query.strip()
            if key and key not in seen:
                seen.add(key)
                unique.append(key)
        if question not in unique:
            unique.insert(0, question)
        return RewriteResult(
            queries=unique,
            strategy=strategy,
            debug={"strategy": strategy, "num_queries": len(unique)},
        )
