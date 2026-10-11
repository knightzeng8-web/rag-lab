"""端到端离线测试：只用 numpy + 标准库，验证「索引 -> 混合检索 -> 重排 -> 引用」闭环。

这些用例的价值在于：它们同时是「降级路径可用」的证明——任何一台只装了 numpy 的机器
都能复现同样的结论，面试演示不会因为缺 GPU / 缺 API Key 而翻车。
"""

import json
import os
import unittest

from rag.config import Config
from rag.pipeline import RAGPipeline

try:  # 兼容 python -m unittest discover -s tests 与 -t . 两种调用方式
    from tests.support import TempWorkspace
except ImportError:  # pragma: no cover
    from support import TempWorkspace

ALPHA = """# 检索算法
## BM25
BM25 是词频类检索算法，参数 k1 控制词频饱和，b 控制文档长度归一化。
BM25 对专有名词和精确数字很敏感，但无法处理同义改写。

## RRF 融合
RRF 使用倒数排名融合，常数 k 通常取 60。它只使用排名，因此不需要做分数归一化。
"""

BETA = """# 评测方法
## 指标定义
Recall@k 衡量召回率，nDCG@k 衡量排序质量，MRR 衡量首个正确结果的靠前程度。
引用精确率用来检测答案引用的出处是否落在标注文档里。
"""

GAMMA = """# 图数据库
## 多跳查询
图数据库的多跳查询需要沿路径逐跳展开，并严格控制每一跳的候选数量，否则查询会爆炸。
图数据库的多跳查询通常配合离线索引与在线剪枝一起使用。
"""


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TempWorkspace(prefix="pipeline-")
        root = self.tmp.name
        self.corpus_dir = os.path.join(root, "corpus")
        os.makedirs(self.corpus_dir, exist_ok=True)
        self.alpha_path = os.path.join(self.corpus_dir, "alpha.md")
        self._write(self.alpha_path, ALPHA)
        self._write(os.path.join(self.corpus_dir, "beta.md"), BETA)

        self.cfg = Config.from_dict(
            {
                "paths": {"corpus": "corpus", "index": "index", "reports": "reports", "web": "web"},
                "chunking": {"strategy": "structural", "max_chars": 240, "overlap_chars": 60, "min_chars": 20},
                "embedder": {"provider": "hashing", "dim": 256},
                "generation": {"provider": "extractive", "max_context_chunks": 4},
                "retrieval": {
                    "top_k": 3,
                    "rerank": {"enabled": True, "provider": "lexical"},
                    "rewrite": {"enabled": True, "strategy": "keyword"},
                },
                "logging": {"level": "WARNING"},
            },
            root=root,
        )

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def _write(path: str, text: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)

    # ---- 基本闭环 -----------------------------------------------------------
    def test_ingest_and_ask_offline(self):
        pipeline = RAGPipeline(self.cfg)
        report = pipeline.ingest()
        self.assertEqual(report.docs, 2)
        self.assertGreater(report.chunks, 0)
        for name in ("vectors.npy", "chunks.jsonl", "bm25.json", "meta.json"):
            self.assertTrue(os.path.isfile(os.path.join(self.cfg.resolve("index"), name)), name)

        answer = pipeline.ask("BM25 的 k1 和 b 分别控制什么？")
        self.assertTrue(answer.text)
        self.assertFalse(answer.refused)
        # 检索契约与生成契约分开断言，任一链路变化都能定位到具体环节
        self.assertIn("alpha", answer.debug["ranked_docs"][:3])
        self.assertIn(answer.debug["generator"]["mode"], ("extractive", "extractive_fallback"))
        self.assertTrue(answer.citations, "抽取式答案应当带上引用标记")
        self.assertTrue(any(c.doc_id == "alpha" for c in answer.citations))
        self.assertIn("total", answer.latency_ms)
        self.assertEqual(answer.debug["components"]["embedder"]["kind"], "hashing")

    def test_reranker_is_actually_wired_in(self):
        """回归用例：reranker 曾漏传给 HybridRetriever，导致重排开关与消融行全部失真。"""
        pipeline = RAGPipeline(self.cfg)
        pipeline.ingest()
        answer = pipeline.ask("RRF 的常数 k 取多少？")
        self.assertEqual(answer.debug["reranker"], "lexical-coverage")
        self.assertEqual(
            answer.debug["components"]["reranker"]["name"], answer.debug["reranker"]
        )

    def test_index_reused_on_second_instance(self):
        first = RAGPipeline(self.cfg)
        report = first.ingest()
        second = RAGPipeline(self.cfg)
        second.ensure_ready()
        self.assertEqual(len(second.store), report.chunks)
        self.assertEqual(len(second.bm25.ids), report.chunks)

    def test_unanswerable_question_is_refused(self):
        pipeline = RAGPipeline(self.cfg)
        pipeline.ingest()
        answer = pipeline.ask("请介绍托卡尔丘克的长篇小说《太古和其他的时间》的叙事结构。")
        self.assertTrue(answer.refused or not answer.citations)

    def test_status_reports_components(self):
        pipeline = RAGPipeline(self.cfg)
        pipeline.ingest()
        status = pipeline.status()
        self.assertEqual(status["index"]["chunks"], len(pipeline.store))
        for key in ("embedder", "reranker", "llm", "generator", "chunking", "fusion"):
            self.assertIn(key, status["components"])
        self.assertEqual(status["components"]["chunking"]["strategy"], "structural")
        self.assertEqual(status["index"]["corpus_inputs"], [os.path.abspath(self.corpus_dir)])

    # ---- 索引生命周期回归用例 ----------------------------------------------
    def test_corpus_change_triggers_rebuild(self):
        """语料内容变化后必须重建索引，否则会静默用旧索引回答新语料的问题。"""
        pipeline = RAGPipeline(self.cfg)
        report = pipeline.ingest()

        with open(self.alpha_path, "a", encoding="utf-8") as handle:
            handle.write("\n## 推进剂\n反物质引擎的比冲远高于化学火箭，但工程实现困难。\n")

        second = RAGPipeline(self.cfg)
        second.ensure_ready()
        self.assertGreater(len(second.store), report.chunks, "语料变化后应当重建索引")

        answer = second.ask("反物质引擎的比冲怎么样？")
        self.assertTrue(any("反物质" in item.chunk.text for item in answer.contexts))

    def test_ingest_switches_corpus_and_reassembles(self):
        """回归用例：ingest() 后必须让已装配的 retriever 重新指向新索引。"""
        gamma_dir = os.path.join(self.tmp.name, "corpus2")
        os.makedirs(gamma_dir, exist_ok=True)
        self._write(os.path.join(gamma_dir, "gamma.md"), GAMMA)

        pipeline = RAGPipeline(self.cfg)
        pipeline.ingest()
        pipeline.ingest([gamma_dir])

        self.assertEqual(pipeline.status()["index"]["chunks"], len(pipeline.store))
        self.assertEqual(pipeline.status()["index"]["corpus_inputs"], [os.path.abspath(gamma_dir)])

        answer = pipeline.ask("图数据库的多跳查询要注意什么？")
        self.assertIn("gamma", answer.debug["ranked_docs"])
        self.assertTrue(any(c.doc_id == "gamma" for c in answer.citations))

    # ---- 评测 ---------------------------------------------------------------
    def test_eval_runner_produces_metrics(self):
        dataset_path = os.path.join(self.tmp.name, "qa.jsonl")
        with open(dataset_path, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"id": "q1", "question": "RRF 的常数 k 取多少？", "gold_doc_ids": ["alpha"]},
                    ensure_ascii=False,
                )
                + "\n"
            )
            handle.write(
                json.dumps(
                    {"id": "q2", "question": "谁发明了图灵机？", "gold_doc_ids": [], "type": "unanswerable"},
                    ensure_ascii=False,
                )
                + "\n"
            )

        from rag.eval.run_eval import run

        report, paths = run(self.cfg, dataset_path=dataset_path, tag="unit", log=None)
        self.assertEqual(report["metrics"]["items"], 2)
        self.assertIn("recall@1", report["metrics"])
        self.assertTrue(os.path.isfile(paths["markdown"]))
        self.assertTrue(os.path.isfile(paths["csv"]))


if __name__ == "__main__":
    unittest.main()
