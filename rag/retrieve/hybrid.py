"""混合检索：向量召回 ∥ BM25 召回 -> RRF 融合 -> 重排 -> 多跳迭代。

为什么用 RRF（Reciprocal Rank Fusion）而不是分数加权：
- 向量余弦相似度（0~1）与 BM25 分数（0~几十，随语料变化）不在同一量纲，归一化会引入
  新的超参（min-max / z-score 各自的截断点），且在不同查询上不稳定；
- RRF 只用「排名」，天然免归一化：

      score(d) = Σ_i  w_i / (k + rank_i(d))，默认 k = 60

  k 起平滑作用：k 越大，排名差异被压平得越多，头部结果越「保守」。

多跳（multi-hop）：第一轮检索出的证据里往往含有第二轮才需要的关键词（桥接实体）。
本项目用「离线关键词桥接」或「LLM 生成后续问题」构造第二轮查询，再把多轮的排名做
一次 RRF 聚合，最后统一重排 —— 这是把单跳检索器变成迭代式检索器的最小改动。
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..text import tokenize
from ..types import Chunk, RetrieveResult, Scored
from .rerank import NoneReranker, Reranker
from .rewrite import QueryRewriter

__all__ = ["rrf_fuse", "HybridRetriever"]


def rrf_fuse(
    ranked_lists: Sequence[Sequence[Tuple[str, float]]],
    k: int = 60,
    weights: Optional[Sequence[float]] = None,
    names: Optional[Sequence[str]] = None,
) -> List[Tuple[str, float, Dict[str, Any]]]:
    """倒数排名融合。返回 ``[(chunk_id, score, detail)]``，按分数降序。"""
    k = max(1, int(k))
    weights = list(weights) if weights else [1.0] * len(ranked_lists)
    names = list(names) if names else [f"list{i}" for i in range(len(ranked_lists))]
    scores: Dict[str, float] = {}
    detail: Dict[str, Dict[str, Any]] = {}
    for list_index, ranked in enumerate(ranked_lists):
        weight = weights[list_index] if list_index < len(weights) else 1.0
        name = names[list_index] if list_index < len(names) else f"list{list_index}"
        for rank, item in enumerate(ranked, start=1):
            chunk_id = item[0]
            contribution = weight / (k + rank)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + contribution
            entry = detail.setdefault(chunk_id, {"ranks": {}, "raw": {}})
            entry["ranks"][name] = rank
            entry["raw"][name] = round(float(item[1]), 6)
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(chunk_id, float(score), detail[chunk_id]) for chunk_id, score in ordered]


class HybridRetriever:
    def __init__(
        self,
        cfg,
        store,
        bm25,
        embedder=None,
        rewriter: Optional[QueryRewriter] = None,
        reranker: Optional[Reranker] = None,
        llm=None,
        log=None,
    ):
        self.cfg = cfg
        self.store = store
        self.bm25 = bm25
        # 向量模型不随索引持久化，必须在装配时显式注入
        self.embedder = embedder if embedder is not None else getattr(store, "embedder", None)
        self.rewriter = rewriter or QueryRewriter(cfg, llm=llm)
        if getattr(self.rewriter, "bm25", None) is None:
            self.rewriter.bm25 = bm25
        self.reranker = reranker or NoneReranker()
        self.llm = llm
        self.log = log
        self.rrf_k = int(cfg.get("retrieval.rrf_k", 60))
        self.top_k = int(cfg.get("retrieval.top_k", 5))
        self.top_n_vector = int(cfg.get("retrieval.top_n_vector", 20))
        self.top_n_bm25 = int(cfg.get("retrieval.top_n_bm25", 20))
        self.top_n_rerank = int(cfg.get("retrieval.rerank.top_n", 20))
        self.weights = cfg.section("retrieval").get("weights", {}) or {}
        self.w_vector = float(self.weights.get("vector", 1.0))
        self.w_bm25 = float(self.weights.get("bm25", 1.0))
        self.multihop_enabled = bool(cfg.get("multihop.enabled", False))
        self.max_rounds = max(1, int(cfg.get("multihop.max_rounds", 2)))
        self.bridge_terms = max(1, int(cfg.get("multihop.bridge_terms", 3)))
        self._chunk_map: Dict[str, Chunk] = {c.chunk_id: c for c in getattr(store, "chunks", [])}

    # ---- 单条链路 -----------------------------------------------------------
    def _vector_search(self, query: str, top_n: int) -> List[Scored]:
        if self.embedder is None:
            return []
        vector = self.embedder.encode_query(query)
        return self.store.search(vector, top_n=top_n)

    def _bm25_search(self, query: str, top_n: int) -> List[Scored]:
        out: List[Scored] = []
        for rank, (chunk_id, score) in enumerate(self.bm25.search(query, top_n=top_n), start=1):
            chunk = self._chunk_map.get(chunk_id)
            if chunk is None:
                continue
            out.append(Scored(chunk, float(score), source="bm25", detail={"rank": rank}))
        return out

    def _fuse(self, queries: Sequence[str]):
        vector_lists: List[List[Tuple[str, float]]] = []
        bm25_lists: List[List[Tuple[str, float]]] = []
        vector_hits: List[Scored] = []
        bm25_hits: List[Scored] = []
        names: List[str] = []
        weights: List[float] = []

        for index, query in enumerate(queries, start=1):
            # 每条查询只编码一次，向量与 BM25 各跑一次，结果同时用于统计与融合
            vec_ranked = self._vector_search(query, self.top_n_vector)
            bm25_ranked = self._bm25_search(query, self.top_n_bm25)
            vector_hits.extend(vec_ranked)
            bm25_hits.extend(bm25_ranked)
            vector_lists.append([(s.chunk_id, s.score) for s in vec_ranked])
            bm25_lists.append([(s.chunk_id, s.score) for s in bm25_ranked])
            names.extend([f"vector:q{index}", f"bm25:q{index}"])
            weights.extend([self.w_vector, self.w_bm25])

        lists: List[List[Tuple[str, float]]] = []
        for vec, bm in zip(vector_lists, bm25_lists):
            lists.append(vec)
            lists.append(bm)

        fused_raw = rrf_fuse(lists, k=self.rrf_k, weights=weights, names=names)
        fused: List[Scored] = []
        for chunk_id, score, detail in fused_raw:
            chunk = self._chunk_map.get(chunk_id)
            if chunk is None:
                continue
            fused.append(Scored(chunk, score, source="fused", detail=detail))

        def dedupe(items: Sequence[Scored]) -> List[Scored]:
            seen = set()
            out: List[Scored] = []
            for item in items:
                if item.chunk_id in seen:
                    continue
                seen.add(item.chunk_id)
                out.append(item)
            return out

        return dedupe(vector_hits), dedupe(bm25_hits), fused

    # ---- 桥接查询（多跳的第二轮） -------------------------------------------
    def _bridge_query(self, question: str, candidates: Sequence[Scored]) -> Optional[str]:
        if not candidates:
            return None
        if self.llm is not None and getattr(self.llm, "available", False):
            context = "\n".join(f"- {c.chunk.text[:200]}" for c in candidates[:2])
            system = (
                "你在做多跳检索。请基于已有证据，提出一个能补全推理链的后续检索问题，"
                "只输出一行问题本身，不要解释。"
            )
            user = f"原问题：{question}\n已检索到的证据：\n{context}"
            try:
                follow = (self.llm.complete(system=system, user=user, temperature=0.3, max_tokens=120) or "").strip()
                follow = follow.splitlines()[0].strip() if follow else ""
                if len(follow) >= 4 and follow != question:
                    return follow
            except Exception:
                pass

        # 离线桥接：挑出证据里出现、但原问题没有的关键词
        question_terms = set(tokenize(question))
        counts: Dict[str, int] = {}
        for item in candidates[:3]:
            for token in tokenize(item.chunk.text):
                if token in question_terms or (len(token) == 1 and not token.isdigit()):
                    continue
                counts[token] = counts.get(token, 0) + 1
        if not counts:
            return None
        ranked = sorted(counts, key=lambda t: (-counts[t] * (1 + len(t)), t))
        picked: List[str] = []
        for token in ranked:
            if any(token in p or p in token for p in picked):
                continue
            picked.append(token)
            if len(picked) >= self.bridge_terms:
                break
        if not picked:
            return None
        return f"{question} {' '.join(picked)}"

    # ---- 对外 ---------------------------------------------------------------
    def retrieve(
        self,
        question: str,
        top_k: Optional[int] = None,
        rounds: Optional[int] = None,
    ) -> RetrieveResult:
        question = (question or "").strip()
        top_k = int(top_k or self.top_k)
        total_rounds = int(rounds or (self.max_rounds if self.multihop_enabled else 1))
        total_rounds = max(1, total_rounds)

        started = time.perf_counter()
        rewrite_started = time.perf_counter()
        rewrite = self.rewriter.rewrite(question)
        rewrite_ms = (time.perf_counter() - rewrite_started) * 1000.0
        queries = list(rewrite.queries) or [question]

        round_fused: List[List[Scored]] = []
        vector_hits: List[Scored] = []
        bm25_hits: List[Scored] = []
        executed: List[str] = []
        multihop_trace: List[Dict[str, Any]] = []

        for round_index in range(total_rounds):
            batch = queries if round_index == 0 else [queries[-1]]
            executed.extend([q for q in batch if q not in executed])
            vec, bm, fused = self._fuse(batch)
            vector_hits.extend(vec)
            bm25_hits.extend(bm)
            round_fused.append(fused)
            if round_index + 1 < total_rounds:
                bridge = self._bridge_query(question, fused)
                multihop_trace.append(
                    {
                        "round": round_index + 1,
                        "bridge_query": bridge,
                        "top_chunks": [s.chunk_id for s in fused[:3]],
                    }
                )
                if not bridge or bridge in queries:
                    break
                queries.append(bridge)

        retrieve_ms = (time.perf_counter() - started) * 1000.0

        # 多轮融合：把所有轮次（以及多查询）的排名统一做一次 RRF
        rerank_started = time.perf_counter()
        if len(round_fused) <= 1:
            final_fused = round_fused[0] if round_fused else []
        else:
            lists = [[(s.chunk_id, s.score) for s in rnd] for rnd in round_fused]
            weights = [1.0] * len(lists)
            names = [f"round{i + 1}" for i in range(len(lists))]
            final_fused = []
            for chunk_id, score, detail in rrf_fuse(lists, k=self.rrf_k, weights=weights, names=names):
                chunk = self._chunk_map.get(chunk_id)
                if chunk is not None:
                    final_fused.append(Scored(chunk, score, source="fused", detail=detail))

        candidates = final_fused[: max(top_k, self.top_n_rerank)]
        ranked = self.reranker.rerank(question, candidates, top_k=top_k)
        rerank_ms = (time.perf_counter() - rerank_started) * 1000.0

        return RetrieveResult(
            question=question,
            queries=executed,
            vector_hits=vector_hits,
            bm25_hits=bm25_hits,
            fused=final_fused,
            ranked=ranked,
            latency_ms={
                "rewrite": rewrite_ms,
                "retrieve": retrieve_ms,
                "rerank": rerank_ms,
            },
            debug={
                "rewrite": rewrite.debug,
                "rounds": len(round_fused),
                "multihop": multihop_trace,
                "reranker": self.reranker.name,
                "rrf_k": self.rrf_k,
                "weights": {"vector": self.w_vector, "bm25": self.w_bm25},
            },
        )
