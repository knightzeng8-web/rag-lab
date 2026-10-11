"""消融实验脚本：一行一条实验，跑完直接生成可贴进复试材料的对比表。

用法（在 rag-lab 目录下）：

    python scripts/ablation.py                # 跑 A~F 六行（全部离线可复现）
    python scripts/ablation.py --only A,C,E   # 只跑指定行
    python scripts/ablation.py --heavy        # 额外跑 G/H（需要 sentence-transformers / API Key）

结果追加到 ``reports/ablation.csv``，并把对比表打印到终端。
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from typing import Dict, List, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "reports", "ablation.csv")

# (标签, 说明, 覆盖项, 是否重依赖)
ROWS: List[Tuple[str, str, List[str], bool]] = [
    (
        "A-baseline",
        "仅向量召回 + 抽取式生成",
        [
            "retrieval.weights.vector=1.0",
            "retrieval.weights.bm25=0.0",
            "retrieval.rewrite.enabled=false",
            "retrieval.rerank.enabled=false",
            "multihop.enabled=false",
        ],
        False,
    ),
    (
        "B-bm25",
        "仅 BM25 召回（验证词面匹配的价值）",
        [
            "retrieval.weights.vector=0.0",
            "retrieval.weights.bm25=1.0",
            "retrieval.rewrite.enabled=false",
            "retrieval.rerank.enabled=false",
            "multihop.enabled=false",
        ],
        False,
    ),
    (
        "C-rrf",
        "向量 + BM25 经 RRF 融合（默认权重 1:1）",
        [
            "retrieval.weights.vector=1.0",
            "retrieval.weights.bm25=1.0",
            "retrieval.rewrite.enabled=false",
            "retrieval.rerank.enabled=false",
            "multihop.enabled=false",
        ],
        False,
    ),
    (
        "D-rewrite",
        "RRF 融合 + 查询改写",
        [
            "retrieval.weights.vector=1.0",
            "retrieval.weights.bm25=1.0",
            "retrieval.rewrite.enabled=true",
            "retrieval.rewrite.strategy=keyword",
            "retrieval.rerank.enabled=false",
            "multihop.enabled=false",
        ],
        False,
    ),
    (
        "E-rerank",
        "RRF 融合 + 改写 + 词法重排",
        [
            "retrieval.weights.vector=1.0",
            "retrieval.weights.bm25=1.0",
            "retrieval.rewrite.enabled=true",
            "retrieval.rewrite.strategy=keyword",
            "retrieval.rerank.enabled=true",
            "retrieval.rerank.provider=lexical",
            "multihop.enabled=false",
        ],
        False,
    ),
    (
        "F-multihop",
        "RRF 融合 + 改写 + 词法重排 + 多跳检索",
        [
            "retrieval.weights.vector=1.0",
            "retrieval.weights.bm25=1.0",
            "retrieval.rewrite.enabled=true",
            "retrieval.rewrite.strategy=keyword",
            "retrieval.rerank.enabled=true",
            "retrieval.rerank.provider=lexical",
            "multihop.enabled=true",
            "multihop.max_rounds=2",
        ],
        False,
    ),
    (
        "G-cross-encoder",
        "在 F 基础上把重排换成 Cross-Encoder（需 sentence-transformers）",
        [
            "retrieval.rerank.enabled=true",
            "retrieval.rerank.provider=cross_encoder",
            "multihop.enabled=true",
        ],
        True,
    ),
    (
        "H-llm",
        "在 F 基础上换成真实 LLM 生成（需 API Key）",
        [
            "generation.provider=openai",
            "retrieval.rerank.enabled=true",
            "retrieval.rerank.provider=lexical",
            "multihop.enabled=true",
        ],
        True,
    ),
    (
        "I-chunking",
        "在 F 基础上把分块换成固定窗口（分块策略消融）",
        [
            "chunking.strategy=fixed",
            "retrieval.rerank.enabled=true",
            "retrieval.rerank.provider=lexical",
        ],
        False,
    ),
    (
        "J-rerank-on-vector",
        "仅向量召回 + 词法重排（验证重排在弱基线上是否正向）",
        [
            "retrieval.weights.vector=1.0",
            "retrieval.weights.bm25=0.0",
            "retrieval.rewrite.enabled=false",
            "retrieval.rerank.enabled=true",
            "retrieval.rerank.provider=lexical",
            "multihop.enabled=false",
        ],
        False,
    ),
    (
        "K-cross-encoder-on-rrf",
        "RRF 融合 + Cross-Encoder 重排（隔离重排本身的收益，不带多跳）",
        [
            "retrieval.weights.vector=1.0",
            "retrieval.weights.bm25=1.0",
            "retrieval.rewrite.enabled=false",
            "retrieval.rerank.enabled=true",
            "retrieval.rerank.provider=cross_encoder",
            "multihop.enabled=false",
        ],
        True,
    ),
]

METRIC_COLUMNS = [
    "metric.recall@1",
    "metric.recall@5",
    "metric.mrr",
    "metric.ndcg@10",
    "metric.citation_precision",
    "metric.refusal_accuracy",
    "metric.latency_p50_ms",
    "metric.latency_p95_ms",
]


def run_row(tag: str, description: str, overrides: Sequence[str]) -> int:
    command = [sys.executable, "-m", "rag.cli", "eval", "--tag", tag, "--log-level", "WARNING"]
    for item in overrides:
        command += ["--set", item]
    print(f"\n=== 运行 {tag}：{description} ===")
    print(" ".join(command))
    result = subprocess.run(command, cwd=ROOT)
    return int(result.returncode or 0)


def print_table() -> None:
    if not os.path.isfile(CSV_PATH):
        print("没有找到 reports/ablation.csv")
        return
    with open(CSV_PATH, "r", encoding="utf-8", newline="") as handle:
        rows: List[Dict[str, str]] = list(csv.DictReader(handle))
    if not rows:
        print("ablation.csv 为空")
        return
    columns = ["tag"] + [c for c in METRIC_COLUMNS if c in rows[0]]
    header = ["实验"] + [c.replace("metric.", "") for c in columns[1:]]
    print("\n## 消融实验对比（追加自 reports/ablation.csv）\n")
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows:
        print("| " + " | ".join([row.get("tag", "")] + [str(row.get(c, "")) for c in columns[1:]]) + " |")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="rag-lab 消融实验批跑脚本")
    parser.add_argument("--only", default="", help="只跑这些标签，逗号分隔，例如 A,C,E")
    parser.add_argument("--heavy", action="store_true", help="包含需要重依赖/API Key 的 G、H 行")
    parser.add_argument("--list", action="store_true", help="只列出所有实验行")
    parser.add_argument(
        "--embedder",
        choices=["", "auto", "hashing", "bge", "openai"],
        default="",
        help="给所有行统一注入 embedder.provider（例如 --embedder bge 得到 BGE 版消融表）",
    )
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="额外注入到每一行的配置覆盖，可重复",
    )
    parser.add_argument("--tag-suffix", default="", help="给每行 tag 加后缀，便于与旧结果区分")
    args = parser.parse_args(argv)

    if args.list:
        for tag, description, _, heavy in ROWS:
            print(f"{tag:16s} {'[重依赖]' if heavy else '        '} {description}")
        return 0

    selected = {item.strip() for item in args.only.split(",") if item.strip()}
    injected: List[str] = list(args.extra)
    if args.embedder:
        injected.append(f"embedder.provider={args.embedder}")
    if injected:
        print(f"注入到每一行的覆盖项：{injected}\n")

    failed: List[str] = []
    for tag, description, overrides, heavy in ROWS:
        short = tag.split("-")[0]
        if selected and short not in selected and tag not in selected:
            continue
        if heavy and not args.heavy:
            print(f"跳过 {tag}（需要 --heavy）")
            continue
        row_tag = f"{tag}{args.tag_suffix}"
        if run_row(row_tag, description, list(overrides) + injected) != 0:
            failed.append(row_tag)

    print_table()
    if failed:
        print(f"\n以下实验返回非零退出码：{', '.join(failed)}（通常是缺少重依赖或 API Key）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
