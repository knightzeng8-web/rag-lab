import math
import unittest

from rag.eval import metrics as M


class RecallTest(unittest.TestCase):
    def test_recall_at_k(self):
        retrieved = ["g1", "x", "g2"]
        gold = ["g1", "g2"]
        self.assertAlmostEqual(M.recall_at_k(retrieved, gold, 1), 0.5)
        self.assertAlmostEqual(M.recall_at_k(retrieved, gold, 3), 1.0)

    def test_recall_truncates_before_dedupe(self):
        # 回归用例：同一篇文档的多个 chunk 不能把 recall@k 抬高（口径须与 precision@k 一致）
        retrieved = ["A", "A", "A", "A", "A", "B"]
        self.assertAlmostEqual(M.recall_at_k(retrieved, ["B"], 5), 0.0)
        self.assertAlmostEqual(M.recall_at_k(retrieved, ["B"], 6), 1.0)

    def test_recall_undefined_without_gold(self):
        self.assertIsNone(M.recall_at_k(["a"], [], 5))

    def test_precision_at_k_counts_noise(self):
        self.assertAlmostEqual(M.precision_at_k(["g1", "x"], ["g1"], 2), 0.5)

    def test_mrr(self):
        self.assertAlmostEqual(M.reciprocal_rank(["x", "g"], ["g"]), 0.5)
        self.assertEqual(M.reciprocal_rank(["x", "y"], ["g"]), 0.0)


class NDCGTest(unittest.TestCase):
    def test_perfect_ranking_is_one(self):
        self.assertAlmostEqual(M.ndcg_at_k(["g1", "g2"], ["g1", "g2"], 2), 1.0)

    def test_known_value(self):
        # DCG = 1 + 0 + 1/log2(4) = 1.5；IDCG = 1 + 1/log2(3)
        value = M.ndcg_at_k(["g1", "x", "g2"], ["g1", "g2"], 3)
        expected = 1.5 / (1.0 + 1.0 / math.log2(3))
        self.assertAlmostEqual(value, expected, places=6)

    def test_repeated_chunks_of_one_doc_do_not_inflate(self):
        # 同一篇 gold 文档的多个 chunk 进前 k，不应把 nDCG 抬到 1 以上
        value = M.ndcg_at_k(["g", "g", "g"], ["g"], 3)
        self.assertAlmostEqual(value, 1.0, places=6)

    def test_lower_when_gold_is_late(self):
        early = M.ndcg_at_k(["g", "x", "y"], ["g"], 3)
        late = M.ndcg_at_k(["x", "y", "g"], ["g"], 3)
        self.assertGreater(early, late)


class CitationTest(unittest.TestCase):
    def test_citation_precision(self):
        self.assertAlmostEqual(M.citation_precision(["g", "x"], ["g"]), 0.5)

    def test_missing_citation_penalised(self):
        self.assertEqual(M.citation_precision([], ["g"]), 0.0)

    def test_citation_recall(self):
        self.assertAlmostEqual(M.citation_recall(["g"], ["g", "h"]), 0.5)


class AggregateTest(unittest.TestCase):
    def test_aggregate_skips_none_and_adds_refusal_accuracy(self):
        records = [
            {
                "metrics": {"recall@1": 1.0, "mrr": None},
                "refused": False,
                "unanswerable": False,
                "latency_ms": {"total": 10.0},
            },
            {
                "metrics": {"recall@1": 0.0, "mrr": 0.5},
                "refused": True,
                "unanswerable": True,
                "latency_ms": {"total": 30.0},
            },
        ]
        summary = M.aggregate(records, ks=[1])
        self.assertAlmostEqual(summary["recall@1"], 0.5)
        self.assertAlmostEqual(summary["mrr"], 0.5)
        self.assertAlmostEqual(summary["refusal_accuracy"], 1.0)
        # 分位是插值结果，只断言到 0.1 ms，避免插值定义变化造成假失败
        self.assertAlmostEqual(summary["latency_p50_ms"], 20.0, places=1)
        self.assertAlmostEqual(summary["latency_p95_ms"], 29.0, places=1)

    def test_percentile(self):
        self.assertEqual(M.percentile([1, 2, 3], 50), 2.0)
        self.assertEqual(M.percentile([], 95), 0.0)


if __name__ == "__main__":
    unittest.main()
