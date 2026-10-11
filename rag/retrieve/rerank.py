"""重排：none / lexical / cross_encoder / llm。

召回（recall）和排序（precision）是两件事：向量 + BM25 的 RRF 融合擅长「把相关的东西
捞进候选集」，但 top-k 的先后顺序仍然粗糙。重排器对 (query, chunk) 成对打分，把真正
相关的片段顶上来，代价是必须限制候选数量（``retrieval.rerank.top_n``）。

- ``none``       ：baseline，消融对照；
- ``lexical``    ：离线实现，词覆盖率 + 短语命中 + 长度惩罚，零依赖但确实有效；
- ``cross_encoder``：sentence-transformers 的 CrossEncoder（如 bge-reranker-base），效果最好；
- ``llm``        ：listwise，让 LLM 直接给候选打分，成本最高，用于对比实验。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

from ..device import resolve_device
from ..text import tokenize
from ..types import Scored

__all__ = [
    "Reranker",
    "NoneReranker",
    "LexicalReranker",
    "CrossEncoderReranker",
    "LLMReranker",
    "build_reranker",
]

_JSON_ARRAY = re.compile(r"\[.*\]", re.DOTALL)


class Reranker:
    kind = "none"
    name = "none"
    available = True
    fallback_reason: Optional[str] = None

    def rerank(self, query: str, candidates: Sequence[Scored], top_k: int) -> List[Scored]:
        raise NotImplementedError

    def status(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "available": bool(self.available),
            "fallback_reason": self.fallback_reason,
        }


class NoneReranker(Reranker):
    kind = "none"
    name = "none"

    def rerank(self, query: str, candidates: Sequence[Scored], top_k: int) -> List[Scored]:
        return list(candidates)[: max(1, int(top_k))]


class LexicalReranker(Reranker):
    """离线重排：IDF 加权的查询词覆盖 + 长短语命中 + 长度惩罚，再与原始融合分插值。

    为什么必须做 IDF 加权：中文按「单字 + 二元组」分词后，常见字词（的、是、量、度…）
    几乎出现在每个片段里。等权覆盖率会让「什么词都有但都不是重点」的长片段压过真正
    相关的片段——实测这会让重排把 RRF 的好顺序搞坏（recall@1 从 0.90 掉到 0.70）。
    """

    kind = "lexical"
    name = "lexical-coverage"

    def __init__(self, phrase_len: int = 4, alpha: float = 0.75, idf=None):
        self.phrase_len = int(phrase_len)
        self.alpha = float(alpha)
        # idf: Optional[Callable[[str], float]]，由 pipeline 注入 BM25 的 IDF
        self.idf = idf

    def _weight(self, token: str) -> float:
        if self.idf is None:
            return 1.0
        try:
            value = float(self.idf(token))
        except Exception:
            return 1.0
        return value if value > 0 else 1e-3  # 未登录词给一个很小的正数而不是 0

    @staticmethod
    def _phrase_hit(question: str, text: str) -> float:
        lowered = text.lower()
        for n in (8, 6, 4):
            for i in range(0, max(0, len(question) - n + 1)):
                if question[i : i + n] in lowered:
                    return 1.0
        return 0.0

    def rerank(self, query: str, candidates: Sequence[Scored], top_k: int) -> List[Scored]:
        candidates = list(candidates)
        if not candidates:
            return []
        # 与抽取式生成器同一套过滤规则：丢单个 CJK 字，保留单字母术语（b / k / n）
        query_terms = {t for t in tokenize(query) if len(t) > 1 or t.isascii()}
        total_weight = sum(self._weight(t) for t in query_terms) or 1.0
        max_score = max((c.score for c in candidates), default=0.0) or 1.0
        reranked: List[Scored] = []
        lowered_query = query.lower()
        for item in candidates:
            doc_terms = set(tokenize(item.chunk.text))
            matched_weight = sum(self._weight(t) for t in (query_terms & doc_terms))
            coverage = matched_weight / total_weight
            phrase = self._phrase_hit(lowered_query, item.chunk.text)
            length_penalty = 1.0 / (1.0 + max(0, len(item.chunk.text) - 800) / 2000.0)
            lexical = (0.6 * coverage + 0.4 * phrase) * length_penalty
            prior = max(0.0, item.score) / max_score
            combined = self.alpha * lexical + (1.0 - self.alpha) * prior
            detail = dict(item.detail)
            detail.update(
                {
                    "lexical": round(lexical, 4),
                    "coverage": round(coverage, 4),
                    "phrase": phrase,
                    "prior": round(prior, 4),
                    "idf_weighted": self.idf is not None,
                    "pre_rerank_score": round(item.score, 6),
                }
            )
            reranked.append(Scored(item.chunk, float(combined), source="rerank", detail=detail))
        reranked.sort(key=lambda s: (-s.score, s.chunk_id))
        return reranked[: max(1, int(top_k))]


class CrossEncoderReranker(Reranker):
    kind = "cross_encoder"

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str = "auto"):
        self.model_name = model_name
        self.device = resolve_device(device)  # auto -> 有 CUDA/MPS 就用
        self.name = f"cross-encoder:{model_name}"
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        from sentence_transformers import CrossEncoder  # type: ignore

        self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank(self, query: str, candidates: Sequence[Scored], top_k: int) -> List[Scored]:
        candidates = list(candidates)
        # 先加载模型：否则候选为空时提前返回，模型从未被加载，
        # build_reranker 里的「不可用 -> 降级」分支永远不可达，异常会推迟到 ask 期才爆。
        model = self._ensure_model()
        if not candidates:
            return []
        pairs = [(query, c.chunk.text) for c in candidates]
        scores = model.predict(pairs)
        results: List[Scored] = []
        for item, score in zip(candidates, scores):
            detail = dict(item.detail)
            detail["cross_encoder"] = float(score)
            detail["pre_rerank_score"] = round(item.score, 6)
            results.append(Scored(item.chunk, float(score), source="rerank", detail=detail))
        results.sort(key=lambda s: (-s.score, s.chunk_id))
        return results[: max(1, int(top_k))]


class LLMReranker(Reranker):
    """listwise 重排：一次请求给所有候选打分，省 token 但依赖模型稳定性。"""

    kind = "llm"

    def __init__(self, llm, max_chars: int = 200):
        self.llm = llm
        self.max_chars = int(max_chars)
        self.name = f"llm:{getattr(llm, 'name', 'unknown')}"
        self.available = bool(getattr(llm, "available", False))

    def rerank(self, query: str, candidates: Sequence[Scored], top_k: int) -> List[Scored]:
        candidates = list(candidates)
        if not candidates or not self.available:
            return LexicalReranker().rerank(query, candidates, top_k)
        listing = "\n".join(
            f"[{i}] {c.chunk.text[: self.max_chars]}" for i, c in enumerate(candidates, start=1)
        )
        system = (
            "你是检索结果重排器。根据问题与候选片段的相关性给每个片段打 0-10 分，"
            '只输出 JSON 数组，形如 [{"id":1,"score":8}, ...]，不要任何解释。'
        )
        user = f"问题：{query}\n候选片段：\n{listing}"
        try:
            text = self.llm.complete(system=system, user=user, temperature=0.0, max_tokens=600)
            match = _JSON_ARRAY.search(text or "")
            payload = json.loads(match.group(0)) if match else []
        except Exception:
            return LexicalReranker().rerank(query, candidates, top_k)

        scores: Dict[int, float] = {}
        for item in payload if isinstance(payload, list) else []:
            try:
                scores[int(item["id"])] = float(item["score"])
            except Exception:
                continue
        if not scores:
            return LexicalReranker().rerank(query, candidates, top_k)
        results: List[Scored] = []
        for idx, item in enumerate(candidates, start=1):
            score = scores.get(idx, 0.0)
            detail = dict(item.detail)
            detail["llm_score"] = score
            result = Scored(item.chunk, score, source="rerank", detail=detail)
            results.append(result)
        results.sort(key=lambda s: (-s.score, s.chunk_id))
        return results[: max(1, int(top_k))]


def build_reranker(cfg, llm=None, log=None, idf=None) -> Reranker:
    """``idf`` 是 BM25 的 ``idf(term)``，注入给词法重排做 IDF 加权（None 则退化为等权）。"""

    def lexical() -> LexicalReranker:
        return LexicalReranker(idf=idf)

    if not bool(cfg.get("retrieval.rerank.enabled", False)):
        return NoneReranker()

    provider = str(cfg.get("retrieval.rerank.provider", "auto")).lower()
    model = str(cfg.get("retrieval.rerank.model", "BAAI/bge-reranker-base"))
    device = str(cfg.get("retrieval.rerank.device", "auto"))

    if provider == "none":
        return NoneReranker()
    if provider == "lexical":
        return lexical()
    if provider == "llm":
        reranker = LLMReranker(llm)
        return reranker if reranker.available else lexical()
    if provider == "cross_encoder":
        try:
            reranker = CrossEncoderReranker(model_name=model, device=device)
            reranker.rerank("预热", [], 1)
            return reranker
        except Exception as exc:
            if log:
                log.warning("Cross-Encoder 不可用（%s），降级为词法重排", exc)
            fallback = lexical()
            fallback.fallback_reason = f"cross_encoder 失败：{exc}"
            return fallback

    # auto：优先 Cross-Encoder，否则词法重排
    try:
        import sentence_transformers  # type: ignore  # noqa: F401

        reranker = CrossEncoderReranker(model_name=model, device=device)
        reranker.rerank("预热", [], 1)
        return reranker
    except Exception:
        return lexical()
