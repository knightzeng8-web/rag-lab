import os
import unittest

from rag.index.bm25 import BM25Index

try:  # 兼容两种 discover 调用方式
    from tests.support import TempWorkspace
except ImportError:  # pragma: no cover
    from support import TempWorkspace

IDS = ["d0", "d1", "d2"]
TEXTS = [
    "向量检索擅长语义相似度，但会漏掉专有名词和精确数字。",
    "BM25 使用 k1 和 b 两个参数：k1 控制词频饱和，b 控制文档长度归一化。",
    "评测指标包括 Recall@k、MRR 与 nDCG@k，用来衡量召回与排序质量。",
]


class BM25Test(unittest.TestCase):
    def setUp(self):
        self.index = BM25Index(k1=1.5, b=0.75).fit(IDS, TEXTS)

    def test_ranking_hits_right_document(self):
        results = self.index.search("k1 和 b 参数是什么", top_n=3)
        self.assertTrue(results)
        scores = {chunk_id: score for chunk_id, score in results}
        # 只断言「相关文档被召回且分数高于无关文档」，不把绝对排名写死
        self.assertIn("d1", scores)
        self.assertGreater(scores["d1"], scores.get("d0", 0.0))
        self.assertGreater(scores["d1"], scores.get("d2", 0.0))

    def test_idf_positive(self):
        self.assertGreater(self.index.idf("bm25"), 0.0)
        self.assertEqual(self.index.idf("不存在的词项xyz"), 0.0)

    def test_no_hit_returns_empty(self):
        # 中文分词是「单字 + 二元组」，所以这里必须挑一个连单字都不重合的查询；
        # 「量子纠缠」这类查询会因为单字「量」而命中（单字召回的特性，不是 bug）
        self.assertEqual(self.index.search("股票期权估值", top_n=3), [])

    def test_stats(self):
        stats = self.index.stats()
        self.assertEqual(stats["docs"], 3)
        self.assertGreater(stats["vocabulary"], 10)
        self.assertGreater(stats["avgdl"], 0)

    def test_save_load_roundtrip(self):
        with TempWorkspace(prefix="bm25-") as tmp:
            self.index.save(tmp)
            self.assertTrue(os.path.isfile(os.path.join(tmp, "bm25.json")))
            loaded = BM25Index.load(tmp)
        self.assertEqual(
            self.index.search("nDCG 排序质量", top_n=3),
            loaded.search("nDCG 排序质量", top_n=3),
        )

    def test_k1_b_from_config(self):
        class FakeCfg:
            def get(self, key, default=None):
                return {"retrieval.bm25.k1": 1.2, "retrieval.bm25.b": 0.6}.get(key, default)

        index = BM25Index.from_config(FakeCfg())
        self.assertAlmostEqual(index.k1, 1.2)
        self.assertAlmostEqual(index.b, 0.6)


if __name__ == "__main__":
    unittest.main()
