import unittest

import numpy as np

from rag.index.embedder import HashingEmbedder, build_embedder
from rag.index.store import NumpyVectorStore
from rag.types import Chunk

try:  # 兼容两种 discover 调用方式
    from tests.support import TempWorkspace
except ImportError:  # pragma: no cover
    from support import TempWorkspace


class HashingEmbedderTest(unittest.TestCase):
    def test_shape_and_normalization(self):
        embedder = HashingEmbedder(dim=128)
        matrix = embedder.encode(["混合检索使用 RRF", "向量召回"])
        self.assertEqual(matrix.shape, (2, 128))
        norms = np.linalg.norm(matrix, axis=1)
        for norm in norms:
            self.assertAlmostEqual(float(norm), 1.0, places=5)

    def test_deterministic(self):
        embedder = HashingEmbedder(dim=64)
        self.assertTrue(np.allclose(embedder.encode(["相同输入"]), embedder.encode(["相同输入"])))

    def test_similar_text_ranks_higher(self):
        embedder = HashingEmbedder(dim=512)
        vectors = embedder.encode(["BM25 使用 k1 和 b 两个参数", "BM25 的 k1 与 b 参数控制什么", "今天天气不错"])
        query = embedder.encode_query("BM25 的 k1 和 b 参数")
        scores = vectors @ query
        self.assertGreater(float(scores[0]), float(scores[2]))
        self.assertGreater(float(scores[1]), float(scores[2]))

    def test_zero_dim_does_not_crash(self):
        # dim 会被当作取模分母；<=0 时必须被夹到 1，而不是 ZeroDivisionError
        embedder = HashingEmbedder(dim=0)
        self.assertEqual(embedder.dim, 1)
        self.assertEqual(embedder.encode(["abc"]).shape, (1, 1))

    def test_empty_input(self):
        embedder = HashingEmbedder(dim=32)
        self.assertEqual(embedder.encode([]).shape, (0, 32))


class BuildEmbedderTest(unittest.TestCase):
    def test_provider_hashing_from_config(self):
        class FakeCfg:
            def get(self, key, default=None):
                values = {"embedder.provider": "hashing", "embedder.dim": 64, "embedder.normalize": True}
                return values.get(key, default)

            def env(self, key, default=""):
                return default

        embedder = build_embedder(FakeCfg())
        self.assertEqual(embedder.kind, "hashing")
        self.assertEqual(embedder.dim, 64)


class StoreTest(unittest.TestCase):
    def test_add_and_search(self):
        embedder = HashingEmbedder(dim=64)
        chunks = [
            Chunk(chunk_id="d#0", doc_id="d", text="BM25 的 k1 控制词频饱和"),
            Chunk(chunk_id="d#1", doc_id="d", text="nDCG 衡量排序质量"),
        ]
        store = NumpyVectorStore(dim=64)
        store.add(chunks, embedder.encode([c.text for c in chunks]))
        hits = store.search(embedder.encode_query("k1 词频饱和"), top_n=2)
        self.assertEqual(hits[0].chunk_id, "d#0")
        self.assertEqual(hits[0].source, "vector")

    def test_add_single_vector_is_reshaped(self):
        # 1-D 输入必须被规整，而不是抛 IndexError
        store = NumpyVectorStore(dim=4)
        store.add([Chunk(chunk_id="d#0", doc_id="d", text="x")], np.ones(4, dtype=np.float32))
        self.assertEqual(len(store), 1)
        self.assertEqual(store.matrix.shape, (1, 4))

    def test_dim_mismatch_raises_clear_error(self):
        store = NumpyVectorStore(dim=4)
        with self.assertRaises(ValueError):
            store.add([Chunk(chunk_id="d#0", doc_id="d", text="x")], np.ones((1, 5), dtype=np.float32))

    def test_save_load_roundtrip(self):
        embedder = HashingEmbedder(dim=32)
        chunks = [Chunk(chunk_id="d#0", doc_id="d", text="混合检索")]
        store = NumpyVectorStore(dim=32)
        store.add(chunks, embedder.encode([c.text for c in chunks]))
        with TempWorkspace(prefix="store-") as tmp:
            store.save(tmp, extra={"note": "unit"})
            loaded = NumpyVectorStore.load(tmp)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded.meta.get("note"), "unit")
        self.assertEqual(loaded.chunks[0].text, "混合检索")


class FaissFallbackTest(unittest.TestCase):
    """faiss 装了就用、没装就回落 numpy —— 两条路径都必须给出同样的 top-1。"""

    def test_search_is_correct_with_or_without_faiss(self):
        from rag.index.store import FaissVectorStore

        store = FaissVectorStore(dim=4)
        chunks = [Chunk(chunk_id=f"d#{i}", doc_id="d", text=str(i)) for i in range(3)]
        store.add(chunks, np.eye(3, 4, dtype=np.float32))
        hits = store.search(np.eye(3, 4, dtype=np.float32)[1], top_n=1)
        self.assertTrue(hits)
        self.assertEqual(hits[0].chunk_id, "d#1")
        self.assertEqual(hits[0].source, "vector")

    def test_empty_store_returns_nothing(self):
        from rag.index.store import FaissVectorStore

        self.assertEqual(FaissVectorStore(dim=4).search(np.ones(4, dtype=np.float32), top_n=3), [])


if __name__ == "__main__":
    unittest.main()
