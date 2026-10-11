import unittest

from rag.retrieve.hybrid import rrf_fuse
from rag.retrieve.rerank import LexicalReranker, NoneReranker
from rag.text import tokenize
from rag.types import Chunk, Scored


def chunk(chunk_id: str, text: str) -> Chunk:
    doc_id = chunk_id.split("#")[0]
    return Chunk(chunk_id=chunk_id, doc_id=doc_id, text=text)


class RRFFusionTest(unittest.TestCase):
    def test_symmetric_lists_tie(self):
        fused = rrf_fuse([[("a", 1.0), ("b", 0.9)], [("b", 5.0), ("a", 1.0)]], k=60)
        scores = {item[0]: item[1] for item in fused}
        self.assertAlmostEqual(scores["a"], scores["b"], places=9)

    def test_weights_change_order(self):
        fused = rrf_fuse([[("a", 1.0), ("b", 0.9)], [("b", 5.0), ("a", 1.0)]], k=60, weights=[1.0, 0.0])
        self.assertEqual(fused[0][0], "a")

    def test_item_ranked_second_in_both_lists_beats_first_in_one(self):
        # 这正是 RRF 相对「取单路第一名」的优势所在
        fused = rrf_fuse([[("A", 1.0), ("B", 0.9)], [("B", 1.0), ("C", 0.5)]], k=60)
        self.assertEqual(fused[0][0], "B")

    def test_detail_records_ranks(self):
        fused = rrf_fuse([[("a", 1.0)], [("a", 0.5)]], k=60, names=["vector", "bm25"])
        self.assertEqual(fused[0][2]["ranks"], {"vector": 1, "bm25": 1})

    def test_empty_input(self):
        self.assertEqual(rrf_fuse([], k=60), [])


class RerankTest(unittest.TestCase):
    def test_none_reranker_truncates(self):
        items = [Scored(chunk("d#0", "x"), 0.5), Scored(chunk("d#1", "y"), 0.4)]
        self.assertEqual(len(NoneReranker().rerank("q", items, 1)), 1)

    def test_lexical_reranker_promotes_term_overlap(self):
        query = "混合检索中的 RRF 融合算法"
        noisy = Scored(chunk("noise#0", "今天天气不错，适合出门旅行和拍照。"), 0.9)
        relevant = Scored(chunk("rel#0", "混合检索使用 RRF 融合算法把多路召回的排名合并。"), 0.1)
        ranked = LexicalReranker().rerank(query, [noisy, relevant], top_k=2)
        self.assertEqual(ranked[0].chunk.chunk_id, "rel#0")
        self.assertIn("lexical", ranked[0].detail)

    def test_lexical_reranker_handles_empty(self):
        self.assertEqual(LexicalReranker().rerank("q", [], top_k=3), [])

    def test_tokenize_used_by_coverage(self):
        self.assertTrue(set(tokenize("RRF 融合")) & set(tokenize("rrf 融合算法")))


if __name__ == "__main__":
    unittest.main()
