"""评测入口：跑评测集 -> 输出逐条明细 + 汇总指标 -> 落盘 JSON / Markdown / CSV。

为什么要有 CSV：面试时最有说服力的不是「我的 RAG 效果不错」，而是一张**可复现的消融表**。

    python -m rag.cli eval --tag baseline
    python -m rag.cli eval --tag +bm25      --set retrieval.weights.vector=0 --set retrieval.weights.bm25=1
    python -m rag.cli eval --tag +rrf       --set retrieval.weights.vector=1 --set retrieval.weights.bm25=1
    python -m rag.cli eval --tag +rewrite   --set retrieval.rewrite.strategy=multi_query
    python -m rag.cli eval --tag +rerank    --set retrieval.rerank.enabled=true
    python -m rag.cli eval --tag +multihop  --set multihop.enabled=true

每跑一次就向 ``reports/ablation.csv`` 追加一行，最后直接拿 CSV 画表。
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..obs import get_logger
from . import metrics as M

__all__ = ["EvalItem", "load_dataset", "evaluate", "format_markdown", "run", "SETTING_KEYS"]

# 写进报告与 CSV 的关键配置项（用于复现实验）
SETTING_KEYS = (
    "chunking.strategy",
    "chunking.max_chars",
    "chunking.overlap_chars",
    "index.backend",
    "embedder.provider",
    "embedder.model",
    "retrieval.top_k",
    "retrieval.top_n_vector",
    "retrieval.top_n_bm25",
    "retrieval.rrf_k",
    "retrieval.weights.vector",
    "retrieval.weights.bm25",
    "retrieval.rewrite.enabled",
    "retrieval.rewrite.strategy",
    "retrieval.rerank.enabled",
    "retrieval.rerank.provider",
    "multihop.enabled",
    "multihop.max_rounds",
    "generation.provider",
    "generation.model",
)


@dataclass
class EvalItem:
    id: str
    question: str
    gold_doc_ids: List[str] = field(default_factory=list)
    gold_answer: str = ""
    difficulty: str = ""
    type: str = ""

    @property
    def unanswerable(self) -> bool:
        return (not self.gold_doc_ids) or str(self.type).lower() in ("unanswerable", "no_answer", "refusal")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "gold_doc_ids": list(self.gold_doc_ids),
            "gold_answer": self.gold_answer,
            "difficulty": self.difficulty,
            "type": self.type,
        }


def load_dataset(path: str) -> List[EvalItem]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"评测集不存在：{path}")
    items: List[EvalItem] = []
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"{path} 第 {line_no} 行不是合法 JSON：{exc}") from exc
                items.append(_to_item(payload, index=line_no))
    else:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        records = payload.get("items", payload) if isinstance(payload, dict) else payload
        for index, record in enumerate(records, start=1):
            items.append(_to_item(record, index=index))
    return items


def _to_item(payload: Dict[str, Any], index: int) -> EvalItem:
    gold = payload.get("gold_doc_ids") or payload.get("gold_docs") or []
    if isinstance(gold, str):
        gold = [gold]
    return EvalItem(
        id=str(payload.get("id") or f"q{index:02d}"),
        question=str(payload.get("question") or "").strip(),
        gold_doc_ids=[str(g) for g in gold],
        gold_answer=str(payload.get("gold_answer") or ""),
        difficulty=str(payload.get("difficulty") or ""),
        type=str(payload.get("type") or ""),
    )


def settings_snapshot(cfg) -> Dict[str, Any]:
    return {key: cfg.get(key) for key in SETTING_KEYS}


def evaluate(
    pipeline,
    items: Sequence[EvalItem],
    ks: Sequence[int] = (1, 3, 5, 10),
    log=None,
    progress: bool = True,
) -> List[Dict[str, Any]]:
    """逐条跑问答并计算指标。检索指标用未截断的 ``ranked`` 列表，故 ``top_k=max(ks)``。"""
    ks = sorted({int(k) for k in ks})
    top_k = max(ks)
    records: List[Dict[str, Any]] = []
    total = len(items)
    for position, item in enumerate(items, start=1):
        if log and progress:
            log.info("[%d/%d] %s：%s", position, total, item.id, item.question[:40])
        answer = pipeline.ask(item.question, top_k=top_k)
        retrieved = list(answer.debug.get("ranked_docs", []) or [])
        cited = list(answer.cited_doc_ids)
        item_metrics: Dict[str, Optional[float]] = {}
        for k in ks:
            item_metrics[f"recall@{k}"] = M.recall_at_k(retrieved, item.gold_doc_ids, k)
            item_metrics[f"precision@{k}"] = M.precision_at_k(retrieved, item.gold_doc_ids, k)
            item_metrics[f"ndcg@{k}"] = M.ndcg_at_k(retrieved, item.gold_doc_ids, k)
        item_metrics["mrr"] = M.reciprocal_rank(retrieved, item.gold_doc_ids)
        item_metrics["citation_precision"] = M.citation_precision(cited, item.gold_doc_ids)
        item_metrics["citation_recall"] = M.citation_recall(cited, item.gold_doc_ids)
        records.append(
            {
                "id": item.id,
                "question": item.question,
                "type": item.type,
                "difficulty": item.difficulty,
                "gold_doc_ids": list(item.gold_doc_ids),
                "gold_answer": item.gold_answer,
                "retrieved_doc_ids": retrieved,
                "cited_doc_ids": cited,
                "num_citations": len(answer.citations),
                "refused": bool(answer.refused),
                "unanswerable": bool(item.unanswerable),
                "answer": answer.text,
                "latency_ms": {k: round(float(v), 3) for k, v in answer.latency_ms.items()},
                "metrics": item_metrics,
            }
        )
    return records


def format_markdown(report: Dict[str, Any]) -> str:
    metrics = report.get("metrics", {})
    lines: List[str] = []
    lines.append(f"# 评测报告：{report.get('tag') or '(untagged)'}")
    lines.append("")
    lines.append(f"- 时间：{report.get('timestamp')}")
    lines.append(f"- 评测集：{report.get('dataset')}")
    lines.append(f"- 题目数：{metrics.get('items')}（可答 {metrics.get('answerable')} / 不可答 {metrics.get('unanswerable')}）")
    lines.append("")
    lines.append("## 关键配置")
    lines.append("")
    lines.append("| 配置项 | 值 |")
    lines.append("| --- | --- |")
    for key, value in (report.get("settings") or {}).items():
        lines.append(f"| `{key}` | {value} |")
    lines.append("")
    lines.append("## 汇总指标")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("| --- | --- |")
    for key, value in metrics.items():
        lines.append(f"| {key} | {value} |")
    lines.append("")
    lines.append("## 逐题明细")
    lines.append("")
    lines.append("| id | type | 难度 | 是否拒答 | 命中文档 | 引用数 | Recall@5 | 总延迟(ms) |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for record in report.get("items", []):
        record_metrics = record.get("metrics", {})
        lines.append(
            "| {id} | {type} | {difficulty} | {refused} | {hit} | {cites} | {r5} | {lat} |".format(
                id=record.get("id"),
                type=record.get("type") or "-",
                difficulty=record.get("difficulty") or "-",
                refused="是" if record.get("refused") else "否",
                hit=", ".join(record.get("retrieved_doc_ids", [])[:2]) or "-",
                cites=record.get("num_citations", 0),
                r5=record_metrics.get("recall@5"),
                lat=record.get("latency_ms", {}).get("total"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def append_csv(report: Dict[str, Any], csv_path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    metrics = dict(report.get("metrics", {}))
    settings = dict(report.get("settings", {}))
    row: Dict[str, Any] = {
        "timestamp": report.get("timestamp"),
        "tag": report.get("tag") or "",
    }
    row.update({f"metric.{k}": v for k, v in metrics.items()})
    row.update({f"setting.{k}": v for k, v in settings.items()})
    exists = os.path.isfile(csv_path)
    fieldnames: List[str] = []
    if exists:
        with open(csv_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            fieldnames = list(header or [])
    for key in row:
        if key not in fieldnames:
            fieldnames.append(key)
    rows: List[Dict[str, Any]] = []
    if exists:
        with open(csv_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = [dict(r) for r in reader]
    rows.append({key: row.get(key, "") for key in fieldnames})
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in rows:
            writer.writerow({key: item.get(key, "") for key in fieldnames})
    return csv_path


def run(
    cfg,
    dataset_path: Optional[str] = None,
    tag: Optional[str] = None,
    pipeline=None,
    log=None,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """执行一次评测，返回 (报告, 落盘路径字典)。"""
    from ..pipeline import RAGPipeline

    log = log or get_logger(level=str(cfg.get("logging.level", "INFO")))
    dataset = dataset_path or cfg.path_of("eval")
    items = load_dataset(dataset)
    max_items = int(cfg.get("eval.max_items", 0) or 0)
    if max_items > 0:
        items = items[:max_items]

    pipeline = pipeline or RAGPipeline.build(config=cfg)
    pipeline.ensure_ready()

    ks = cfg.get("eval.ks", [1, 3, 5, 10]) or [1, 3, 5, 10]
    records = evaluate(pipeline, items, ks=[int(k) for k in ks], log=log)

    # 记录"实际生效的组件"，而不只是配置里写的值。
    # 教训：`--set embedder.provider=bge` 在模型不可用时会静默降级成哈希向量，
    # 结果表里看起来一切正常、数字却是另一套实现的——必须让降级在产物里可见。
    status = pipeline.status()
    components = status.get("components", {}) or {}

    def _name(key: str) -> Any:
        value = components.get(key) or {}
        return value.get("name") if isinstance(value, dict) else value

    settings = settings_snapshot(cfg)
    settings.update(
        {
            "effective.embedder": _name("embedder"),
            "effective.embedder_fallback": (components.get("embedder") or {}).get("fallback_reason"),
            "effective.reranker": _name("reranker"),
            "effective.generator": _name("generator"),
            "effective.device": (status.get("device") or {}).get("selected"),
        }
    )
    if (components.get("embedder") or {}).get("fallback_reason"):
        log.warning(
            "向量模型发生了降级：配置 provider=%s，实际生效 %s（原因：%s）——本次结果的实现与预期不同，"
            "请勿与未降级的实验直接比较",
            cfg.get("embedder.provider"),
            _name("embedder"),
            (components.get("embedder") or {}).get("fallback_reason"),
        )

    report: Dict[str, Any] = {
        "tag": tag or "run",
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "dataset": dataset,
        "settings": settings,
        "components": components,
        "device": status.get("device", {}),
        "index": status.get("index", {}),
        "items": records,
        "metrics": M.aggregate(records, ks=[int(k) for k in ks]),
    }

    out_dir = cfg.resolve(str(cfg.get("eval.out_dir", "reports")))
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_tag = "".join(ch if ch.isalnum() or ch in "-_+" else "_" for ch in (tag or "run"))
    json_path = os.path.join(out_dir, f"eval-{stamp}-{safe_tag}.json")
    md_path = os.path.join(out_dir, f"eval-{stamp}-{safe_tag}.md")
    csv_path = os.path.join(out_dir, "ablation.csv")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(format_markdown(report))
    append_csv(report, csv_path)
    return report, {"json": json_path, "markdown": md_path, "csv": csv_path}
