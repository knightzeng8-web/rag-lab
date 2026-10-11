"""检索/生成评测指标（纯函数，便于单测与离线复算）。

指标定义（本项目以 **chunk 为检索单元**，gold 标注到 doc 级）：

- ``recall@k``：gold 文档集合中，有多少篇出现在**前 k 个 chunk** 所属文档里。
  衡量「检索有没有把该找的资料捞回来」，是 RAG 的第一优先指标——召回不到，后面全白搭。
- ``precision@k``：前 k 个 chunk 中，属于 gold 文档的比例。衡量候选集的噪声。
- ``MRR``：第一个命中 gold 的 chunk 的排名倒数（1/rank），衡量「首个正确结果的靠前程度」。
- ``nDCG@k``：以 **chunk 为单位、doc 去重** 的二值相关性（doc ∈ gold 记 1）计算：

      DCG@k  = Σ_{i=1..k} rel_i / log2(i + 1)
      IDCG@k = Σ_{i=1..min(|gold|, k)} 1 / log2(i + 1)

  注意两点：① IDCG 的分母只按 min(|gold|, k) 个命中位算，所以 DCG 也必须**按文档去重**，
  否则同一篇文档的多个 chunk 会重复计分，nDCG 会算出 > 1 的荒谬值；
  ② 因此 nDCG@k 惩罚的是「gold 片段排得靠后」，而不像 recall 只关心在不在前 k。
- ``引用精确率``：答案引用里的 doc 落在 gold 里的比例——检测「答案对了但出处编了」。
- ``拒答准确率``：无可答资料的题必须拒答，可答的题不能拒答。

约定：gold 为空的题（unanswerable）所有检索类指标返回 ``None``，由 ``mean()`` 过滤后跳过，
不会被当成 0 分拉低均值。
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence

__all__ = [
    "dedupe",
    "recall_at_k",
    "precision_at_k",
    "reciprocal_rank",
    "dcg_at_k",
    "ndcg_at_k",
    "citation_precision",
    "citation_recall",
    "percentile",
    "mean",
    "aggregate",
]


def dedupe(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def recall_at_k(retrieved_doc_ids: Sequence[str], gold_doc_ids: Sequence[str], k: int) -> Optional[float]:
    gold = set(gold_doc_ids)
    if not gold:
        return None  # 无可答资料，召回率无定义
    # 先截断到前 k 个 chunk，再去重成文档集合：与 precision@k 保持同一候选集口径
    top = set(dedupe(list(retrieved_doc_ids)[: int(k)]))
    return len(gold & top) / float(len(gold))


def precision_at_k(retrieved_doc_ids: Sequence[str], gold_doc_ids: Sequence[str], k: int) -> Optional[float]:
    gold = set(gold_doc_ids)
    if not gold:
        return None
    top = list(retrieved_doc_ids)[: int(k)]
    if not top:
        return 0.0
    return len([d for d in top if d in gold]) / float(len(top))


def reciprocal_rank(retrieved_doc_ids: Sequence[str], gold_doc_ids: Sequence[str]) -> Optional[float]:
    gold = set(gold_doc_ids)
    if not gold:
        return None
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in gold:
            return 1.0 / rank
    return 0.0


def dcg_at_k(relevances: Sequence[float], k: int) -> float:
    total = 0.0
    for index, rel in enumerate(relevances[: int(k)], start=1):
        total += float(rel) / math.log2(index + 1)
    return total


def ndcg_at_k(retrieved_doc_ids: Sequence[str], gold_doc_ids: Sequence[str], k: int) -> Optional[float]:
    gold = set(gold_doc_ids)
    if not gold:
        return None
    k = int(k)
    # 按文档去重后再算增益：同一篇文档的第 2、3 个 chunk 不应重复贡献 DCG
    relevances: List[float] = []
    seen: List[str] = []
    for doc_id in retrieved_doc_ids:
        if len(relevances) >= k:
            break
        if doc_id in seen:
            continue
        seen.append(doc_id)
        relevances.append(1.0 if doc_id in gold else 0.0)
    dcg = dcg_at_k(relevances, k)
    ideal = [1.0] * min(len(gold), k)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return min(1.0, dcg / idcg)  # min 兜底：口径不一致时不至于报出 > 1 的分数


def citation_precision(cited_doc_ids: Sequence[str], gold_doc_ids: Sequence[str]) -> Optional[float]:
    gold = set(gold_doc_ids)
    if not gold:
        return None
    if not cited_doc_ids:
        return 0.0
    return len([d for d in cited_doc_ids if d in gold]) / float(len(cited_doc_ids))


def citation_recall(cited_doc_ids: Sequence[str], gold_doc_ids: Sequence[str]) -> Optional[float]:
    gold = set(gold_doc_ids)
    if not gold:
        return None
    if not cited_doc_ids:
        return 0.0
    return len(gold & set(cited_doc_ids)) / float(len(gold))


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (float(p) / 100.0)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def mean(values: Sequence[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def aggregate(records: Sequence[Dict[str, Any]], ks: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """把逐条记录聚合成一份指标汇总。

    ``ks`` 仅用于调用方自述跑了哪些 k（指标键本身由每条 record 的 ``metrics`` 决定），
    保留该参数是为了接口稳定；每条记录的 ``metrics`` 为扁平字典。
    """
    metrics: Dict[str, Any] = {}
    keys: List[str] = []
    for record in records:
        for key in record.get("metrics", {}):
            if key not in keys:
                keys.append(key)
    for key in keys:
        value = mean([record.get("metrics", {}).get(key) for record in records])
        if value is None:
            continue
        metrics[key] = round(value, 4)

    # 意图正确率：可答的题要答，不可答的题要拒答（衡量「拒答策略」是否过/欠保守）
    refusal_hits = []
    for record in records:
        expected = bool(record.get("unanswerable"))
        actual = bool(record.get("refused"))
        refusal_hits.append(1.0 if expected == actual else 0.0)
    if refusal_hits:
        metrics["refusal_accuracy"] = round(sum(refusal_hits) / len(refusal_hits), 4)

    # 只有真正记录了耗时的记录才参与分位计算，避免把缺失值当作 0 ms 拉低 p50/p95
    latencies = [
        float(record["latency_ms"]["total"])
        for record in records
        if (record.get("latency_ms") or {}).get("total") is not None
    ]
    metrics["latency_p50_ms"] = round(percentile(latencies, 50), 1)
    metrics["latency_p95_ms"] = round(percentile(latencies, 95), 1)

    metrics["items"] = len(records)
    metrics["answerable"] = len([r for r in records if not r.get("unanswerable")])
    metrics["unanswerable"] = len([r for r in records if r.get("unanswerable")])
    return metrics
