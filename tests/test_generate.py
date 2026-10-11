"""生成层测试：用**桩 LLM** 覆盖真实 API 才会走到的分支（无需网络、无需 API Key）。

覆盖点：
- 提示词里的资料编号与 ``Citation.index`` 一致（答对但引用错位是 RAG 最典型的幻觉形态）；
- 越界引用编号被丢弃（模型编了 `[7]` 却没有第 7 条资料）；
- LLM 抛异常时自动降级到抽取式，并在 debug 里留下原因；
- 拒答判定与 ``parse_citations`` 的配合。
"""

import unittest

from rag.generate.answer import (
    ExtractiveGenerator,
    LLMGenerator,
    build_generator,
    parse_citations,
)
from rag.generate.prompt import REFUSAL_TEXT, build_user_prompt, is_refusal, render_contexts
from rag.types import Chunk, Scored

try:  # 兼容两种 discover 调用方式
    from tests.support import TempWorkspace  # noqa: F401
except ImportError:  # pragma: no cover
    pass


def make_contexts() -> list:
    specs = [
        ("doc-a", "doc-a#0", "RRF 使用倒数排名融合，常数 k 通常取 60。"),
        ("doc-b", "doc-b#1", "BM25 的 k1 控制词频饱和，b 控制长度归一化。"),
    ]
    return [
        Scored(Chunk(chunk_id=chunk_id, doc_id=doc_id, text=text), 1.0 - index * 0.1, source="fused")
        for index, (doc_id, chunk_id, text) in enumerate(specs)
    ]


class StubLLM:
    """最小可用的 LLM 桩：available/name/complete 三个属性就够。"""

    name = "stub:test"
    available = True

    def __init__(self, reply: str = "", error: Exception = None):
        self.reply = reply
        self.error = error
        self.calls = []

    def complete(self, system, user, temperature=None, max_tokens=None) -> str:
        self.calls.append({"system": system, "user": user})
        if self.error is not None:
            raise self.error
        return self.reply

    def status(self):
        return {"kind": "stub", "name": self.name, "available": True}


class PromptTest(unittest.TestCase):
    def test_context_numbering_matches_citation_index(self):
        contexts = make_contexts()
        block = render_contexts(contexts)
        self.assertTrue(block[0].startswith("[1]"))
        self.assertIn("chunk_id=doc-a#0", block[0])
        self.assertTrue(block[1].startswith("[2]"))

    def test_user_prompt_contains_question_and_context(self):
        prompt = build_user_prompt("k 取多少？", make_contexts())
        self.assertIn("【资料】", prompt)
        self.assertIn("【问题】k 取多少？", prompt)

    def test_is_refusal(self):
        self.assertTrue(is_refusal(REFUSAL_TEXT + "：缺少相关信息。"))
        self.assertTrue(is_refusal(""))
        self.assertFalse(is_refusal("RRF 的常数 k 取 60。[1]"))


class ParseCitationsTest(unittest.TestCase):
    def test_maps_marks_to_contexts(self):
        contexts = make_contexts()
        citations = parse_citations("RRF 的常数 k 取 60[1]，BM25 的 b 控制长度归一化[2]。", contexts)
        self.assertEqual([c.index for c in citations], [1, 2])
        self.assertEqual(citations[0].doc_id, "doc-a")
        self.assertEqual(citations[1].chunk_id, "doc-b#1")
        self.assertTrue(citations[0].quote)

    def test_out_of_range_marks_dropped(self):
        citations = parse_citations("编造的引用[7] 与真实引用[1]。", make_contexts())
        self.assertEqual([c.index for c in citations], [1])

    def test_duplicate_marks_collapsed(self):
        citations = parse_citations("先[1] 再[1] 再[2]。", make_contexts())
        self.assertEqual([c.index for c in citations], [1, 2])

    def test_no_marks(self):
        self.assertEqual(parse_citations("没有任何引用标记。", make_contexts()), [])


class LLMGeneratorTest(unittest.TestCase):
    def test_generates_and_parses_citations(self):
        llm = StubLLM("RRF 的常数 k 通常取 60[1]。BM25 的 b 控制长度归一化[2]。")
        generator = LLMGenerator(llm)
        text, debug = generator.generate("RRF 的 k 是多少？", make_contexts())
        self.assertEqual(debug["mode"], "llm")
        self.assertEqual(len(llm.calls), 1)
        citations = parse_citations(text, make_contexts())
        self.assertEqual([c.doc_id for c in citations], ["doc-a", "doc-b"])

    def test_refusal_from_llm_is_recognised(self):
        llm = StubLLM(f"{REFUSAL_TEXT}：资料里没有提到这个参数。")
        text, debug = LLMGenerator(llm).generate("不存在的问题？", make_contexts())
        self.assertTrue(is_refusal(text))
        self.assertEqual(debug["mode"], "llm")

    def test_llm_error_falls_back_to_extractive(self):
        llm = StubLLM(error=RuntimeError("503 Service Unavailable"))
        text, debug = LLMGenerator(llm).generate("RRF 的 k 是多少？", make_contexts())
        self.assertEqual(debug["mode"], "extractive_fallback")
        self.assertIn("503", debug["llm_error"])
        self.assertTrue(text)

    def test_empty_contexts_never_calls_llm(self):
        llm = StubLLM("不该被调用[1]")
        text, debug = LLMGenerator(llm).generate("无上下文的问题？", [])
        self.assertEqual(debug["mode"], "no_context")
        self.assertEqual(llm.calls, [])
        self.assertTrue(is_refusal(text))


class BuildGeneratorTest(unittest.TestCase):
    class FakeCfg:
        def __init__(self, overrides=None):
            self.values = {"generation.provider": "auto", "generation.min_coverage": None}
            self.values.update(overrides or {})

        def get(self, key, default=None):
            value = self.values.get(key, default)
            return default if value is None else value

    def test_extractive_when_no_llm(self):
        generator = build_generator(self.FakeCfg({"generation.provider": "extractive"}), llm=None)
        self.assertIsInstance(generator, ExtractiveGenerator)

    def test_llm_used_when_available(self):
        generator = build_generator(self.FakeCfg(), llm=StubLLM("x[1]"))
        self.assertIsInstance(generator, LLMGenerator)

    def test_extractive_thresholds_read_from_config(self):
        cfg = self.FakeCfg(
            {
                "generation.provider": "extractive",
                "generation.extractive_min_coverage": 0.42,
                "generation.extractive_min_hits": 5,
                "generation.extractive_max_sentences": 2,
            }
        )
        generator = build_generator(cfg, llm=None)
        self.assertAlmostEqual(generator.min_coverage, 0.42)
        self.assertEqual(generator.min_hits, 5)
        self.assertEqual(generator.max_sentences, 2)


if __name__ == "__main__":
    unittest.main()
