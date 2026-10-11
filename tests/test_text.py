import unittest

from rag.text import first_heading, jaccard, keywords, split_sentences, stable_hash, tokenize, truncate


class TokenizeTest(unittest.TestCase):
    def test_cjk_bigrams(self):
        tokens = tokenize("混合检索")
        for expected in ("混", "混合", "合检", "检索"):
            self.assertIn(expected, tokens)

    def test_latin_and_numbers(self):
        tokens = tokenize("BM25 的 k1 与 b 参数")
        self.assertIn("bm25", tokens)
        self.assertIn("k1", tokens)
        self.assertIn("b", tokens)

    def test_empty(self):
        self.assertEqual(tokenize(""), [])


class SentenceTest(unittest.TestCase):
    def test_split_cjk_punctuation(self):
        sentences = split_sentences("第一句。第二句！第三句？")
        self.assertEqual(len(sentences), 3)

    def test_short_fragment_merged(self):
        sentences = split_sentences("这是一个完整的句子。短。")
        self.assertEqual(len(sentences), 1)


class KeywordsTest(unittest.TestCase):
    def test_extracts_technical_term(self):
        picked = keywords("我们应该如何选择 chunk 的大小？", limit=5)
        self.assertTrue(any("chunk" in item for item in picked))

    def test_stopwords_filtered(self):
        picked = keywords("什么是 RRF 融合", limit=5)
        self.assertNotIn("什么", picked)
        self.assertTrue(any("rrf" in item or "融合" in item for item in picked))


class MiscTest(unittest.TestCase):
    def test_truncate_respects_boundary(self):
        text = "甲" * 100 + "。" + "乙" * 100
        out = truncate(text, 150)
        self.assertLessEqual(len(out), 152)
        self.assertTrue(out.endswith("…"))

    def test_truncate_short_text_untouched(self):
        self.assertEqual(truncate("很短", 50), "很短")

    def test_stable_hash_range_and_determinism(self):
        value = stable_hash("abc", 1000)
        self.assertEqual(value, stable_hash("abc", 1000))
        self.assertGreaterEqual(value, 0)
        self.assertLess(value, 1000)

    def test_first_heading(self):
        self.assertEqual(first_heading("# 标题\n正文"), "标题")
        self.assertEqual(first_heading("没有标题"), "")

    def test_jaccard(self):
        self.assertAlmostEqual(jaccard(["a", "b"], ["a", "b"]), 1.0)
        self.assertAlmostEqual(jaccard(["a"], ["b"]), 0.0)


if __name__ == "__main__":
    unittest.main()
