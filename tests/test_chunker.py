import unittest

from rag.ingest.chunker import Chunker
from rag.ingest.loader import Document

MARKDOWN = """# 检索技术
本文介绍混合检索。

## 混合检索
BM25 是词频类检索算法，参数 k1 控制词频饱和，b 控制文档长度归一化。
向量检索擅长语义相似，但对专有名词和数字不敏感。

## 融合
RRF 使用倒数排名融合，常数 k 通常取 60。
多路召回的结果经过 RRF 之后再做一次重排。

# 评测
## 指标
Recall@k 衡量召回率，nDCG@k 衡量排序质量。
"""

LONG_SECTION = "# 长文\n\n" + "\n\n".join(
    f"第 {i} 段：这段文字用来把分块逻辑撑开，讨论检索、重排、评测与上下文压缩之间的关系。" * 3
    for i in range(1, 13)
)


class StructuralChunkingTest(unittest.TestCase):
    def setUp(self):
        self.chunker = Chunker(strategy="structural", max_chars=200, overlap_chars=60, min_chars=20)
        self.doc = Document(doc_id="doc", text=MARKDOWN)

    def test_chunk_ids_are_sequential_and_stable(self):
        first = self.chunker.chunk_document(self.doc)
        second = self.chunker.chunk_document(self.doc)
        self.assertEqual([c.chunk_id for c in first], [f"doc#{i}" for i in range(len(first))])
        self.assertEqual([c.text for c in first], [c.text for c in second])

    def test_heading_metadata_kept(self):
        chunks = self.chunker.chunk_document(self.doc)
        headings = {c.meta["heading"] for c in chunks}
        self.assertIn("检索技术 > 混合检索", headings)
        joined = "\n".join(c.text for c in chunks)
        self.assertIn("检索技术 > 混合检索", joined)  # 标题路径会拼进正文，帮助检索

    def test_sections_do_not_mix(self):
        chunks = self.chunker.chunk_document(self.doc)
        for chunk in chunks:
            if "RRF" in chunk.text:
                self.assertIn("融合", chunk.meta["heading"])


class WindowTest(unittest.TestCase):
    def test_max_chars_respected(self):
        chunker = Chunker(strategy="structural", max_chars=200, overlap_chars=60, min_chars=20)
        chunks = chunker.chunk_document(Document(doc_id="long", text=LONG_SECTION))
        self.assertGreater(len(chunks), 1)
        # 上界 = max_chars（打包时按预算裁剪重叠）+ 长度合并的 1.25 倍余量 + prepend 的标题行
        for chunk in chunks:
            self.assertLessEqual(len(chunk.text), 200 + 60 + 8)

    def test_fixed_strategy_has_overlap(self):
        chunker = Chunker(strategy="fixed", max_chars=200, overlap_chars=80, min_chars=10)
        chunks = chunker.chunk_document(Document(doc_id="long", text=LONG_SECTION))
        self.assertGreaterEqual(len(chunks), 2)
        # 后一块的正文开头应当是前一块尾部的重叠内容（先剥掉 prepend 的标题行）
        body = chunks[1].text.split("\n", 1)[-1]
        self.assertIn(body[:10], chunks[0].text)

    def test_stats(self):
        chunker = Chunker(strategy="structural", max_chars=200, overlap_chars=60, min_chars=20)
        docs = [Document(doc_id="a", text=LONG_SECTION), Document(doc_id="b", text=MARKDOWN)]
        chunks, stats = chunker.chunk_documents(docs)
        self.assertEqual(stats.docs, 2)
        self.assertEqual(stats.chunks, len(chunks))
        self.assertGreater(stats.avg_chars, 0)


if __name__ == "__main__":
    unittest.main()
