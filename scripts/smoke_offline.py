"""零依赖冒烟自检：只装 numpy 的环境下，验证整条链路是否真的能跑通。

    python scripts/smoke_offline.py

两个阶段：

- **阶段 1（密闭沙箱）**：临时目录里造两篇小语料 + 两题评测集，跑
  建索引 -> 问答（校验引用非空）-> 评测（校验指标落盘）-> 索引复用，
  不依赖仓库里的任何数据，失败的定位最干净。
- **阶段 2（真实语料）**：用仓库自带的 `data/corpus` + `data/eval/qa.jsonl`
  跑一次完整评测，指标表直接打印出来，结果追加到 `reports/ablation.csv`。

任何一步失败都会打印异常堆栈并给出非零退出码，方便当作提交前的自检关卡。
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import traceback
from typing import Callable, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rag.config import Config  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402

PHASE1_CORPUS = {
    "alpha.md": """# 检索算法
## BM25
BM25 是词频类检索算法，参数 k1 控制词频饱和，b 控制文档长度归一化。
它对专有名词和精确数字敏感，但无法处理同义改写。

## RRF 融合
RRF 使用倒数排名融合，常数 k 通常取 60，只使用名次因此无需分数归一化。
""",
    "beta.md": """# 评测方法
## 指标定义
Recall@k 衡量召回率，nDCG@k 衡量排序质量，MRR 衡量首个正确结果的靠前程度。
引用精确率用来检测答案引用的出处是否落在标注文档里。
""",
}

PHASE1_EVAL = [
    {"id": "s1", "question": "RRF 的常数 k 通常取多少？", "gold_doc_ids": ["alpha"], "type": "factual"},
    {"id": "s2", "question": "谁发明了图灵机？", "gold_doc_ids": [], "type": "unanswerable"},
]


class Recorder:
    def __init__(self) -> None:
        self.results: List[Tuple[str, bool, str]] = []

    def check(self, name: str, fn: Callable[[], None]) -> bool:
        print(f"[ .. ] {name}")
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - 自检脚本要兜住一切
            self.results.append((name, False, str(exc)))
            print(f"[FAIL] {name}\n{traceback.format_exc()}")
            return False
        self.results.append((name, True, ""))
        print(f"[ OK ] {name}")
        return True

    def summary(self) -> int:
        print("\n================ 自检结果 ================")
        for name, ok, detail in self.results:
            print(f"{'PASS' if ok else 'FAIL'}  {name}{'' if ok else '  <- ' + detail}")
        failed = [name for name, ok, _ in self.results if not ok]
        print("==========================================")
        if failed:
            print(f"共 {len(self.results)} 项，失败 {len(failed)} 项：{', '.join(failed)}")
            return 1
        print(f"共 {len(self.results)} 项，全部通过。")
        return 0


def phase1(rec: Recorder) -> None:
    # 不用 tempfile.mkdtemp：它按 0o700 建目录，在受限令牌（沙箱 / CI）下
    # 连创建者自己都无法往里写文件（WinError 5）。项目内自建目录则一切正常。
    base = os.path.join(ROOT, ".testtmp")
    os.makedirs(base, exist_ok=True)
    tmp = os.path.join(base, "smoke-" + os.urandom(4).hex())
    os.makedirs(tmp, exist_ok=True)
    print(f"\n--- 阶段 1：密闭沙箱（{tmp}）---")
    corpus_dir = os.path.join(tmp, "corpus")
    os.makedirs(corpus_dir, exist_ok=True)
    for name, text in PHASE1_CORPUS.items():
        with open(os.path.join(corpus_dir, name), "w", encoding="utf-8") as handle:
            handle.write(text)
    dataset_path = os.path.join(tmp, "qa.jsonl")
    with open(dataset_path, "w", encoding="utf-8") as handle:
        for item in PHASE1_EVAL:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    cfg = Config.from_dict(
        {
            "paths": {"corpus": "corpus", "index": "index", "eval": "qa.jsonl", "reports": "reports", "web": "web"},
            "chunking": {"strategy": "structural", "max_chars": 240, "overlap_chars": 60, "min_chars": 20},
            "embedder": {"provider": "hashing", "dim": 256},
            "generation": {"provider": "extractive"},
            "retrieval": {"rerank": {"enabled": True, "provider": "lexical"}},
            "eval": {"out_dir": "reports"},
            "logging": {"level": "ERROR"},
        },
        root=tmp,
    )
    state = {}

    def ingest() -> None:
        pipeline = RAGPipeline(cfg)
        report = pipeline.ingest()
        assert report.docs == 2, f"应加载 2 篇文档，实际 {report.docs}"
        assert report.chunks > 0, "分块数为 0"
        for name in ("vectors.npy", "chunks.jsonl", "bm25.json", "meta.json"):
            assert os.path.isfile(os.path.join(cfg.resolve("index"), name)), f"缺少索引文件 {name}"
        state["pipeline"] = pipeline

    def ask() -> None:
        pipeline = state["pipeline"]
        answer = pipeline.ask("BM25 的 k1 和 b 分别控制什么？")
        assert answer.text.strip(), "答案为空"
        assert not answer.refused, f"该题不应当拒答：{answer.text[:80]}"
        assert answer.citations, "答案没有引用（[n] 解析失败）"
        assert any(c.doc_id == "alpha" for c in answer.citations), "引用未指向正确文档"
        assert "total" in answer.latency_ms, "缺少阶段耗时"
        state["answer"] = answer

    def components() -> None:
        status = state["pipeline"].status()
        components = status["components"]
        assert components["embedder"]["kind"] == "hashing", "离线模式应使用哈希向量"
        assert components["generator"]["kind"] == "extractive", "离线模式应使用抽取式生成器"
        assert status["index"]["chunks"] > 0, "索引为空"

    def reuse() -> None:
        again = RAGPipeline(cfg)
        again.ensure_ready()
        assert len(again.store) == len(state["pipeline"].store), "复用索引时分块数不一致"
        assert len(again.bm25.ids) == len(state["pipeline"].bm25.ids), "复用索引时 BM25 不一致"

    def evaluate() -> None:
        from rag.eval.run_eval import run

        report, paths = run(cfg, dataset_path=dataset_path, tag="smoke-phase1", log=None)
        assert report["metrics"]["items"] == 2, "评测题数不对"
        assert "recall@1" in report["metrics"], "缺少 recall@1 指标"
        assert os.path.isfile(paths["markdown"]), "Markdown 报告未落盘"
        assert os.path.isfile(paths["csv"]), "ablation.csv 未落盘"

    rec.check("阶段1 · 建索引（分块 + 哈希向量 + BM25）", ingest)
    rec.check("阶段1 · 问答闭环（答案 + 引用 + 耗时）", ask)
    rec.check("阶段1 · 组件状态与降级上报", components)
    rec.check("阶段1 · 索引复用（不重建）", reuse)
    rec.check("阶段1 · 评测与报告落盘", evaluate)
    shutil.rmtree(tmp, ignore_errors=True)


def phase2(rec: Recorder) -> None:
    print("\n--- 阶段 2：仓库真实语料 ---")
    corpus = os.path.join(ROOT, "data", "corpus")
    dataset = os.path.join(ROOT, "data", "eval", "qa.jsonl")
    if not os.path.isdir(corpus) or not os.path.isfile(dataset):
        print("跳过：未找到 data/corpus 或 data/eval/qa.jsonl")
        return
    cfg = Config.from_dict(
        {
            "paths": {"corpus": "data/corpus", "index": ".index", "eval": "data/eval/qa.jsonl", "reports": "reports"},
            "generation": {"provider": "extractive"},
            "logging": {"level": "WARNING"},
        },
        root=ROOT,
    )
    state = {}

    def ingest() -> None:
        pipeline = RAGPipeline(cfg)
        report = pipeline.ingest()
        assert report.docs >= 5, f"示例语料应至少 5 篇，实际 {report.docs}"
        state["pipeline"] = pipeline
        print(f"       语料：{report.docs} 篇 -> {report.chunks} 块（{report.elapsed_ms:.0f} ms）")

    def ask() -> None:
        pipeline = state["pipeline"]
        answer = pipeline.ask("混合检索为什么需要 RRF 融合？")
        assert answer.text.strip(), "答案为空"
        print(f"       答案片段：{answer.text.splitlines()[0][:70]}…")
        print(f"       引用：{[(c.doc_id, c.chunk_id) for c in answer.citations]}")

    def evaluate() -> None:
        from rag.eval.run_eval import run

        report, paths = run(cfg, dataset_path=dataset, tag="smoke", log=None)
        metrics = report["metrics"]
        state["report"] = report
        print("\n       | 指标 | 值 |")
        print("       | --- | --- |")
        for key, value in metrics.items():
            print(f"       | {key} | {value} |")
        print(f"\n       报告：{paths['markdown']}")

    rec.check("阶段2 · 用示例语料建索引", ingest)
    rec.check("阶段2 · 示例问题问答", ask)
    rec.check("阶段2 · 12 题评测集跑通并落盘", evaluate)


def main() -> int:
    print("rag-lab 零依赖冒烟自检")
    print(f"Python {sys.version.split()[0]} / 项目根目录 {ROOT}")
    try:
        import numpy  # noqa: F401

        print(f"numpy 已安装：{numpy.__version__}")
    except ImportError:
        print("缺少 numpy：pip install numpy")
        return 2

    rec = Recorder()
    phase1(rec)
    phase2(rec)
    return rec.summary()


if __name__ == "__main__":
    sys.exit(main())
