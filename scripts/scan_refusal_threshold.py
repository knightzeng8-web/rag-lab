"""拒答阈值扫描：离线抽取式生成器的可回答性判据要怎么定。

背景：抽取式生成器靠「查询词覆盖率」判断证据是否足够。阈值定高了会把本可回答的题拒掉
（欠答），定低了会把没有依据的题硬答（过答）——两者在 `refusal_accuracy` 里都算错。
这个小脚本把评测集逐题跑一遍，打印每题的 `best_coverage` / `best_hits`，
再对一组阈值做网格搜索，直接给出「拒答正确数 / 总题数」最高且最稳的那组。

用法：
    python scripts/scan_refusal_threshold.py
    python scripts/scan_refusal_threshold.py --set embedder.provider=bge
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rag.config import load_config, parse_scalar  # noqa: E402
from rag.eval.run_eval import load_dataset  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402

COVERAGE_GRID = (0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
HITS_GRID = (1, 2, 3, 4)


def scan(pipeline: RAGPipeline, dataset: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in load_dataset(dataset):
        answer = pipeline.ask(item.question, top_k=10)
        generator = answer.debug.get("generator", {})
        rows.append(
            {
                "id": item.id,
                "type": item.type or "-",
                "unanswerable": item.unanswerable,
                "coverage": generator.get("best_coverage"),
                "hits": generator.get("hits") or generator.get("best_hits"),
                "refused": bool(answer.refused),
            }
        )
    return rows


def grid_search(rows: List[Dict[str, Any]]) -> List[Tuple[float, int, int, int]]:
    results: List[Tuple[float, int, int, int]] = []
    for coverage in COVERAGE_GRID:
        for hits in HITS_GRID:
            correct = 0
            for row in rows:
                if row["coverage"] is None:
                    predicted_refuse = True  # 一句话都没命中 -> 必然拒答
                else:
                    predicted_refuse = (row["coverage"] < coverage) or ((row["hits"] or 0) < hits)
                if predicted_refuse == row["unanswerable"]:
                    correct += 1
            results.append((coverage, hits, correct, len(rows)))
    results.sort(key=lambda item: (-item[2], item[0], item[1]))
    return results


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="扫描离线抽取式生成器的拒答阈值")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args(argv)

    overrides: Dict[str, Any] = {"generation.provider": "extractive", "logging.level": "ERROR"}
    for pair in args.set:
        if "=" in pair:
            key, value = pair.split("=", 1)
            overrides[key.strip()] = parse_scalar(value)

    cfg = load_config(overrides=overrides)
    pipeline = RAGPipeline.build(config=cfg)
    pipeline.ensure_ready()
    dataset = args.dataset or cfg.path_of("eval")

    rows = scan(pipeline, dataset)
    print(f"评测集：{dataset}（{len(rows)} 题）\n")
    print("| id | type | 不可答 | best_coverage | best_hits | 当前判据下拒答 |")
    print("| --- | --- | --- | --- | --- | --- |")
    for row in sorted(rows, key=lambda r: (-(r["coverage"] or 0.0), r["id"])):
        coverage = "—" if row["coverage"] is None else f"{row['coverage']:.3f}"
        hits = "—" if row["hits"] is None else str(row["hits"])
        print(
            f"| {row['id']} | {row['type']} | {'是' if row['unanswerable'] else '否'} | "
            f"{coverage} | {hits} | {'是' if row['refused'] else '否'} |"
        )

    print("\n阈值网格（拒答正确数）：\n")
    print("| min_coverage | min_hits | 正确 | 准确率 |")
    print("| --- | --- | --- | --- |")
    for coverage, hits, correct, total in grid_search(rows)[:12]:
        print(f"| {coverage:.2f} | {hits} | {correct}/{total} | {correct / total:.3f} |")
    print(
        "\n把这组阈值写进 configs/default.yaml 的 generation.extractive_min_coverage / "
        "extractive_min_hits 即可（也可用 --set 临时覆盖做消融）。"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
