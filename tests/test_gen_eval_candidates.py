"""``scripts/gen_eval_candidates.py`` 的单元测试：**全程离线**，不联网、不需要 API Key。

覆盖点与「为什么」：
- 启发式四条路线必须都能产出候选，并且 ``gold_doc_ids`` 都是真实文档 id——
  评测集里一个不存在的 gold id 会让这道题永远算不中，指标悄悄变低而没人知道原因；
- ``unanswerable`` 题的 gold 必须为空且与语料实词重合度低，否则拒答指标会被算反；
- LLM 回复的解析必须容忍代码块与前后废话：真实模型经常这么回，解析一崩就整批作废；
- ``promote`` 的三道闸门（只取勾选、id 连续、gold 必须存在、问题去重）是评测集的最后防线。

所有临时文件都落在 ``tests/support.py`` 的 ``TempWorkspace``（项目内 ``.testtmp/``），
不用 ``tempfile``：受限令牌下 ``tempfile.mkdtemp()`` 建出的目录连创建者都写不进去。
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:  # 兼容两种 discover 调用方式
    from tests.support import TempWorkspace
except ImportError:  # pragma: no cover
    from support import TempWorkspace


def _load_module():
    """按文件路径加载脚本模块（``scripts`` 不是包，不能直接 import）。"""
    path = os.path.join(ROOT, "scripts", "gen_eval_candidates.py")
    spec = importlib.util.spec_from_file_location("gen_eval_candidates", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # 必须先登记进 sys.modules：dataclass 处理注解时要按 cls.__module__ 反查命名空间，
    # 找不到模块会直接抛 AttributeError: 'NoneType' object has no attribute '__dict__'
    sys.modules["gen_eval_candidates"] = module
    spec.loader.exec_module(module)
    return module


gen = _load_module()

# ---------------------------------------------------------------------------
# 测试语料：4 篇，刻意共享「倒排索引 / rrf / bm25」等高 IDF 术语，
# 让 comparison 与 multihop 两条路线在离线情况下真的有东西可挑。
# ---------------------------------------------------------------------------

DOC_FILES = {
    "alpha.md": """# RAG 概述

## 检索增强生成的动机
大模型的参数化记忆存在知识截止与幻觉问题，检索增强生成在生成前先查外部语料，把非参数化记忆补进来，因此答案可以追溯到具体文档，显著降低凭空编造的比例。

## 能力边界与失败模式
检索增强生成并不能修复检索本身漏掉的证据，如果召回阶段没有命中正确文档，生成阶段无论如何都会答错，所以召回上界决定了整条链路的分数上限。
""",
    "beta.md": """# 混合检索

## 倒排索引与打分
倒排索引把词项映射到包含它的文档列表，bm25 在倒排索引上按词频与文档长度归一化累加分数，因此对专有名词和精确数字非常敏感，但完全无法处理同义改写。

## 向量召回与 rrf 融合
向量召回依赖语义相似度，擅长同义改写却容易漏掉稀有实体；rrf 只使用各路检索的名次而不是分数，因此天然免疫两路分数量纲不一致的问题。
""",
    "gamma.md": """# 分块与索引

## 结构感知切分
结构感知切分按标题层级把文档切成小节，保证一个分块只在讲一件事，标题路径还会被拼进正文，显著改善分块语义模糊的问题。

## 分块对倒排索引的影响
分块越长，bm25 的长度归一化就越难发挥作用，因为单块里混进了多个主题，词频信号被稀释，所以 rrf 融合前通常先把分块长度收敛到几百字。
""",
    "delta.md": """# 重排与评测

## 交叉编码器重排
交叉编码器把查询与候选片段拼在一起送入模型，输出单一相关性分数，判别力明显强于双编码器，但必须逐条前向计算，因此只能用于少量候选，常与 rrf 融合后的候选列表配合使用。

## 评测指标
评测常用 recall@k 与 ndcg@k，前者只看正确文档有没有被召回，后者带位置折损，能同时惩罚把相关文档排太后和把无关文档插前面这两种错误。
""",
}

CORPUS_DOC_IDS = ("alpha", "beta", "gamma", "delta")


def write_corpus(tmp: str) -> str:
    corpus = os.path.join(tmp, "corpus")
    os.makedirs(corpus, exist_ok=True)
    for name, text in DOC_FILES.items():
        with open(os.path.join(corpus, name), "w", encoding="utf-8") as handle:
            handle.write(text)
    return corpus


def load_corpus(corpus: str):
    return gen.load_documents([corpus])


PARSE_ANSWERABLE_JSON = json.dumps(
    [
        {
            "question": "资料里关于「倒排索引」是怎么说的？",
            "gold_doc_ids": ["beta"],
            "gold_answer": "倒排索引把词项映射到包含它的文档列表。",
            "difficulty": "easy",
            "type": "factual",
        },
        {
            "question": "语料里有没有讲过宋代汝窑的开片特征？",
            "gold_doc_ids": [],
            "gold_answer": "语料未涉及瓷器工艺，应拒答。",
            "difficulty": "easy",
            "type": "unanswerable",
        },
    ],
    ensure_ascii=False,
)


class StubLLM:
    """桩 LLM：与 ``tests/test_generate.py`` 的写法一致，只保留被调用到的最小接口。"""

    name = "stub:gen-eval"
    available = True

    def __init__(self, replies=None, error=None):
        self.replies = list(replies or [])
        self.error = error
        self.calls = []

    def complete(self, system, user, temperature=None, max_tokens=None) -> str:
        self.calls.append({"system": system, "user": user, "temperature": temperature})
        if self.error is not None:
            raise self.error
        if not self.replies:
            return PARSE_ANSWERABLE_JSON
        index = min(len(self.calls) - 1, len(self.replies) - 1)
        return self.replies[index]

    def status(self):
        return {"kind": "stub", "name": self.name, "available": True}


# ---------------------------------------------------------------------------
# 1. 启发式生成
# ---------------------------------------------------------------------------


class HeuristicGenerateTest(unittest.TestCase):
    def setUp(self):
        self._tmp = TempWorkspace(prefix="gen-eval-")
        self.tmp = self._tmp.__enter__()
        self.corpus = write_corpus(self.tmp)
        self.docs = load_corpus(self.corpus)
        # load_documents 按文件路径排序，所以顺序是 alpha/beta/delta/gamma
        self.assertEqual([doc.doc_id for doc in self.docs], sorted(CORPUS_DOC_IDS))

    def tearDown(self):
        self._tmp.__exit__(None, None, None)

    def test_corpus_fixture_is_loadable(self):
        self.assertEqual(len(self.docs), 4)
        self.assertTrue(all(doc.text.strip() for doc in self.docs))

    def test_generates_at_least_four_candidates_covering_four_types(self):
        candidates = gen.heuristic_candidates(self.docs, seed=13)
        self.assertGreaterEqual(len(candidates), 4)
        self.assertEqual(
            {c.type for c in candidates},
            {"factual", "comparison", "multihop", "unanswerable"},
        )
        for item in candidates:
            self.assertTrue(item.question.strip())
            self.assertEqual(item.provider, "heuristic")
            self.assertTrue(item.rationale.strip())

    def test_finalize_assigns_ids_and_dedupes(self):
        raw = gen.heuristic_candidates(self.docs, seed=13)
        final = gen._finalize(
            raw,
            docs=self.docs,
            existing_questions=[raw[0].question],  # 模拟「和既有评测集重复」
            types=gen.CANDIDATE_TYPES,
            per_doc=3,
            max_total=0,
        )
        self.assertTrue(final)
        self.assertEqual([c.id for c in final], [f"c{i:03d}" for i in range(1, len(final) + 1)])
        keys = [gen.normalize_question(c.question) for c in final]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertNotIn(gen.normalize_question(raw[0].question), keys)

    def test_finalize_keeps_all_four_types_balanced(self):
        """factual 在原始候选里数量最多，收尾后不能把其它三种题型整批挤掉。"""
        raw = gen.heuristic_candidates(self.docs, seed=13)
        final = gen._finalize(
            raw,
            docs=self.docs,
            existing_questions=[],
            types=gen.CANDIDATE_TYPES,
            per_doc=3,
            max_total=0,
        )
        self.assertEqual(
            {c.type for c in final},
            {"factual", "comparison", "multihop", "unanswerable"},
        )
        self.assertGreaterEqual(len(final), 4)

    def test_seed_is_reproducible(self):
        first = [c.question for c in gen.heuristic_candidates(self.docs, seed=13)]
        second = [c.question for c in gen.heuristic_candidates(self.docs, seed=13)]
        self.assertEqual(first, second)


# ---------------------------------------------------------------------------
# 2. 候选校验
# ---------------------------------------------------------------------------


class ValidationTest(unittest.TestCase):
    def setUp(self):
        self._tmp = TempWorkspace(prefix="gen-eval-validate-")
        self.tmp = self._tmp.__enter__()
        self.corpus = write_corpus(self.tmp)
        self.docs = load_corpus(self.corpus)
        self.valid = {doc.doc_id for doc in self.docs}

    def tearDown(self):
        self._tmp.__exit__(None, None, None)

    def test_all_gold_doc_ids_are_real(self):
        raw = gen.heuristic_candidates(self.docs, seed=13)
        final = gen._finalize(
            raw,
            docs=self.docs,
            existing_questions=[],
            types=gen.CANDIDATE_TYPES,
            per_doc=3,
            max_total=0,
        )
        self.assertTrue(final)
        for item in final:
            for doc_id in item.gold_doc_ids:
                self.assertIn(doc_id, self.valid, f"{item.id} 的 gold 指向了不存在的文档 {doc_id}")

    def test_unanswerable_candidates_have_empty_gold_and_low_presence(self):
        unanswerable = [c for c in gen.heuristic_candidates(self.docs, seed=13) if c.type == "unanswerable"]
        self.assertTrue(unanswerable)
        bm25 = gen.make_bm25(self.docs)
        corpus_text = gen.corpus_text_of(self.docs)
        for item in unanswerable:
            self.assertEqual(item.gold_doc_ids, [])
            # 主题词在题目里被「」包起来，也要能直接从 rationale 里核对判据
            topic = item.rationale.split("「")[1].split("」")[0]
            report = gen.rare_term_report(
                item.question, bm25, topic=topic, corpus_text=corpus_text
            )
            self.assertLess(report["present_share"], report["threshold"])
            self.assertTrue(report["unanswerable"])
            self.assertIn("领域外", item.rationale)

    def test_answerable_candidates_have_higher_presence_than_unanswerable(self):
        """这道断言是「不可回答闸门真的在起作用」的反向证据。"""
        bm25 = gen.make_bm25(self.docs)
        corpus_text = gen.corpus_text_of(self.docs)

        def share(topic: str) -> float:
            question = f"资料里关于「{topic}」有哪些说明？"
            return gen.rare_term_report(
                question, bm25, topic=topic, corpus_text=corpus_text
            )["present_share"]

        # 语料里真正讲过的主题 vs 明确领域外的主题
        self.assertGreater(share("倒排索引"), share("宋代汝窑天青釉的开片特征"))
        self.assertGreater(share("BM25"), share("《源氏物语》成书的年代与作者"))

    def test_validate_rejects_missing_doc_id_and_bad_unanswerable(self):
        ok, reason = gen.validate_candidate(
            {
                "question": "这个文档讲了什么？",
                "gold_doc_ids": ["不存在的文档"],
                "type": "factual",
            },
            self.valid,
        )
        self.assertFalse(ok)
        self.assertIn("不在语料中", reason)

        ok, reason = gen.validate_candidate(
            {"question": "语料里讲过瓷器工艺吗？", "gold_doc_ids": ["alpha"], "type": "unanswerable"},
            self.valid,
        )
        self.assertFalse(ok)
        self.assertIn("必须为空", reason)

        ok, _ = gen.validate_candidate(
            {"question": "资料里关于「倒排索引」是怎么说的？", "gold_doc_ids": ["beta"], "type": "factual"},
            self.valid,
        )
        self.assertTrue(ok)

    def test_validate_rejects_unknown_type_and_empty_question(self):
        ok, reason = gen.validate_candidate({"question": "问题问题问题", "type": "科幻"}, self.valid)
        self.assertFalse(ok)
        self.assertIn("type", reason)
        ok, reason = gen.validate_candidate({"question": "  ", "type": "factual"}, self.valid)
        self.assertFalse(ok)
        self.assertIn("为空", reason)


# ---------------------------------------------------------------------------
# 2b. 「不可回答」判据（回归测试）：语料变大后旧判据整体失效，这里锁住新判据
# ---------------------------------------------------------------------------

# 一份「技术词恰好撞车」的小语料：故意包含「工艺」「控制」「要点」「选择」「机制」
# 这类通用二元组，以及「月饼」「糖浆」「转化」这些会出现在领域外主题里的词。
# 这正是把旧判据打回来的场景：长问题里总能撞上几个通用词。
OVERLAP_CORPUS = {
    "tech-a.md": """# 索引构建

## 分块工艺与要点
分块工艺的参数选择需要控制分块长度与重叠，要点是让每个分块只讲一件事，
因为分块长度会直接影响长度归一化的效果与检索的召回上界，选择参数时要一并考虑。

## 转化的控制机制
索引构建的转化流程需要控制批大小，机制上要避免内存峰值过高，
选择批量写入的要点是把工艺参数记录进索引元数据，便于复现与排查。
""",
    "tech-b.md": """# 检索融合

## 召回选择的机制
融合时选择权重需要控制各路的量纲差异，机制上可以用倒数排名融合规避分数不可比的问题，
要点是只使用名次而不是原始分数，避免某一路的分数尺度主导最终排序。

## 评测要点
评测的选择与转化需要控制评测集规模，要点是保证每题都能在语料里找到依据，
否则拒答指标会被污染，工艺参数再调也看不出真实差异。
""",
}


class UnanswerableGateTest(unittest.TestCase):
    """判据必须做对的两件事：不误杀、不放过。"""

    def setUp(self):
        self._tmp = TempWorkspace(prefix="gen-eval-gate-")
        self.tmp = self._tmp.__enter__()
        corpus = os.path.join(self.tmp, "corpus")
        os.makedirs(corpus, exist_ok=True)
        for name, text in OVERLAP_CORPUS.items():
            _write(corpus, name, text)
        self.corpus = corpus
        self.docs = load_corpus(corpus)
        self.bm25 = gen.make_bm25(self.docs)
        self.corpus_text = gen.corpus_text_of(self.docs)

    def tearDown(self):
        self._tmp.__exit__(None, None, None)

    def test_topics_are_offsets_from_the_corpus_topics(self):
        """主题池应该以叙述性事实/专有名词为主，而不是容易撞词的技术表述。"""
        self.assertGreaterEqual(len(gen.OUT_OF_DOMAIN_TOPICS), 15)
        self.assertLessEqual(len(gen.OUT_OF_DOMAIN_TOPICS), 25)
        # 旧版清单里的技术性表述必须已经清掉，否则闸门会持续误杀
        joined = "".join(gen.OUT_OF_DOMAIN_TOPICS)
        for banned in ("月饼糖浆", "高血压", "马拉松", "甲醛", "变速箱", "糖原"):
            self.assertNotIn(banned, joined)

    def test_technical_topic_absent_from_corpus_is_not_dropped(self):
        """构造语料含「月饼/糖浆/工艺」等词，但关键术语缺席的问题仍必须产出 unanswerable。"""
        keywords = ["《清明上河图》现藏于哪家博物馆", "珠穆朗玛峰 2020 年公布的最新雪面高程"]
        candidates = gen._unanswerable_candidates(
            self.docs, self.bm25, seed=13, topics=keywords, target=8
        ) 
        self.assertEqual(len(candidates), 2, "关键术语缺席的题目不应被覆盖率噪声误杀")
        for item in candidates:
            self.assertEqual(item.gold_doc_ids, [])
            self.assertEqual(item.type, "unanswerable")

    def test_common_words_do_not_rescue_an_out_of_domain_question(self):
        """问题里堆满语料里的通用词，但关键术语完全缺席 —— 不能被误判成可回答。"""
        topic = "马孔多小镇的建立者"
        question = "资料里关于「马孔多小镇的建立者」有哪些说明，要点和控制机制是什么？"
        report = gen.rare_term_report(
            question, self.bm25, topic=topic, corpus_text=self.corpus_text
        )
        # 语料里的通用词（「要点」「控制」「机制」）确实撞上了，覆盖率不为 0，
        # 但关键术语「马孔」在语料里毫无踪迹，出现次数必须是 0
        counts = {row["term"]: row["count"] for row in report["all_terms"]}
        self.assertIn("马孔", counts)
        self.assertEqual(counts["马孔"], 0)
        self.assertGreater(report["coverage"], 0.0)
        self.assertLess(report["present_share"], report["threshold"])
        self.assertTrue(report["unanswerable"])

    def test_in_domain_question_is_not_dropped(self):
        """反向场景：语料里真的讲过的主题不能被误判成领域外。"""
        topic = "分块工艺的参数选择"
        question = "资料里关于「分块工艺的参数选择」有哪些说明？"
        report = gen.rare_term_report(
            question, self.bm25, topic=topic, corpus_text=self.corpus_text
        )
        self.assertFalse(report["unanswerable"])
        self.assertTrue(report["relevant"])
        self.assertGreater(report["present_share"], report["threshold"])

    def test_report_exposes_droppable_statistics(self):
        """被丢弃的候选必须能解释：关键词 + 出现次数 + df + 命中率都要在报告里。"""
        topic = "马孔多小镇"
        question = "资料里关于「马孔多小镇」有哪些说明？"
        report = gen.rare_term_report(
            question, self.bm25, topic=topic, corpus_text=self.corpus_text
        )
        self.assertTrue(report["terms"])
        for row in report["terms"]:
            for key in ("term", "count", "df", "idf", "absent", "present"):
                self.assertIn(key, row)
        line = gen.format_rare_terms(report)
        self.assertIn("实词命中率", line)
        self.assertIn("马孔", line)

    def test_topic_terms_are_extracted_from_topic_not_template(self):
        """模板在拼接处产生的假二元组（「料里」「些说」）绝不能进入关键术语。"""
        terms = gen.term_frequency_rank("资料里关于「马孔多小镇」有哪些说明？", self.bm25, "马孔多小镇")
        self.assertTrue(terms)
        self.assertIn("马孔", terms)
        self.assertIn("小镇", terms)
        for noise in ("料里", "些说", "有哪", "里关", "于「"):
            self.assertNotIn(noise, terms)

    def test_subsumed_substring_terms_are_filtered(self):
        """有更长词条作证时，中文碎片必须被剔除；拉丁标识符不受影响。"""
        self.assertEqual(gen._subsumed_terms(["倒排", "倒排索", "索引"]), {"倒排"})
        # 拉丁 token（如 bge-small-zh-v1.5 的碎片）不参与该判据，避免误删真实标识符
        self.assertEqual(gen._subsumed_terms(["bge", "bge-small", "small"]), set())
        # 真实中文 token 里没有更长的容器（tokenize 只产出单字与二元组），
        # 所以「倒排」「索引」都必须保留——碎片判据不能把正常词一起误杀
        terms = gen._content_terms("倒排索引")
        self.assertIn("索引", terms)
        self.assertIn("倒排", terms)

    def test_boundary_test_rejects_cross_word_fragments(self):
        """跨词边界的接缝词（「块策」）在全文里从不落在词边界上，必须被拒绝。"""
        text = "分块 策略 直接影响召回。分块 策略 的要点是保持语义完整。"
        # 「块策」总是被前后汉字夹住 -> 自由次数 0
        self.assertEqual(gen.term_free_occurrences("块策", text)[0], 0)
        self.assertFalse(gen.term_has_boundary("块策", text))
        # 「分块」「策略」都以标点或空格收尾 -> 有词边界
        self.assertTrue(gen.term_has_boundary("分块", text))
        self.assertTrue(gen.term_has_boundary("策略", text))
        # 拉丁术语不参与该判据
        self.assertTrue(gen.term_has_boundary("bm25", "分块策略"))

    def test_adaptive_thresholds_grow_slowly_and_are_capped(self):
        small = gen.adaptive_presence_threshold(10)
        large = gen.adaptive_presence_threshold(10000)
        self.assertLessEqual(small, large)
        self.assertLessEqual(large, gen.PRESENCE_SHARE_CAP)
        self.assertLessEqual(gen.adaptive_coverage_threshold(10000), gen.COVERAGE_CAP)
        # 48 篇语料的实测量级：约 0.46，必须落在合理区间内
        self.assertTrue(0.4 <= gen.adaptive_presence_threshold(48) <= 0.5)

    def test_coverage_gate_is_off_by_default(self):
        """兜底覆盖率默认关闭：它会把领域外题目整批误杀。"""
        report = gen.rare_term_report(
            "资料里关于「马孔多小镇」有哪些说明？",
            self.bm25,
            topic="马孔多小镇",
            corpus_text=self.corpus_text,
        )
        self.assertEqual(report["coverage_threshold"], 0.0)
        self.assertFalse(report["coverage_enabled"])
        # 显式打开（阈值极低）时才允许它否决判定
        strict = gen.rare_term_report(
            "资料里关于「马孔多小镇」有哪些说明？",
            self.bm25,
            topic="马孔多小镇",
            corpus_text=self.corpus_text,
            coverage_threshold=0.01,
        )
        self.assertTrue(strict["coverage_enabled"])
        self.assertFalse(strict["unanswerable"])


# ---------------------------------------------------------------------------
# 3. LLM JSON 解析
# ---------------------------------------------------------------------------


class ParseLLMJsonTest(unittest.TestCase):
    def test_plain_json(self):
        items = gen.parse_llm_json(PARSE_ANSWERABLE_JSON, valid_doc_ids=["beta", "alpha"])
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["type"], "factual")
        self.assertEqual(items[0]["gold_doc_ids"], ["beta"])
        self.assertEqual(items[1]["type"], "unanswerable")
        self.assertEqual(items[1]["gold_doc_ids"], [])

    def test_fenced_json_block(self):
        text = "```json\n" + PARSE_ANSWERABLE_JSON + "\n```"
        items = gen.parse_llm_json(text, valid_doc_ids=["beta"])
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["question"], "资料里关于「倒排索引」是怎么说的？")

    def test_json_with_surrounding_chatter(self):
        text = (
            "好的，下面是我出的题目，请查收：\n\n"
            + PARSE_ANSWERABLE_JSON
            + "\n\n如果需要更多题目我可以继续。"
        )
        items = gen.parse_llm_json(text, valid_doc_ids=["beta"])
        self.assertEqual(len(items), 2)

    def test_invalid_json_raises_readable_error(self):
        with self.assertRaises(ValueError) as ctx:
            gen.parse_llm_json("这不是 JSON，只是一段解释文字。", valid_doc_ids=["beta"])
        self.assertIn("JSON", str(ctx.exception))

    def test_unanswerable_gold_is_forced_empty_and_unknown_ids_dropped(self):
        text = json.dumps(
            [
                {
                    "question": "语料外的问题？",
                    "gold_doc_ids": ["beta"],
                    "gold_answer": "",
                    "difficulty": "hard",
                    "type": "unanswerable",
                },
                {
                    "question": "编造 id 的题？",
                    "gold_doc_ids": ["beta", "从未出现的文档"],
                    "gold_answer": "",
                    "difficulty": "不存在的难度",
                    "type": "比较",
                },
            ],
            ensure_ascii=False,
        )
        items = gen.parse_llm_json(text, valid_doc_ids=["beta"])
        self.assertEqual(items[0]["gold_doc_ids"], [])
        self.assertEqual(items[1]["gold_doc_ids"], ["beta"])
        self.assertEqual(items[1]["difficulty"], "medium")
        self.assertEqual(items[1]["type"], "factual")  # 非法 type 回落到 factual

    def test_wrapped_object_is_accepted(self):
        text = json.dumps({"questions": json.loads(PARSE_ANSWERABLE_JSON)}, ensure_ascii=False)
        self.assertEqual(len(gen.parse_llm_json(text, valid_doc_ids=["beta"])), 2)


class LLMCandidateBuildTest(unittest.TestCase):
    def setUp(self):
        self._tmp = TempWorkspace(prefix="gen-eval-llm-")
        self.tmp = self._tmp.__enter__()
        self.corpus = write_corpus(self.tmp)
        self.docs = load_corpus(self.corpus)

    def tearDown(self):
        self._tmp.__exit__(None, None, None)

    def test_prompt_contains_doc_id_and_strict_rules(self):
        system, user = gen.build_llm_prompt([self.docs[1]], per_doc=2, types=["factual", "multihop"])
        self.assertIn("JSON", system)
        self.assertIn("[beta]", user)
        self.assertIn("倒排索引", user)
        self.assertIn("unanswerable", user)
        self.assertIn("factual, multihop", user)

    def test_snippet_respects_char_budget(self):
        snippet = gen.doc_snippet(self.docs[0], max_chars=300)
        self.assertLessEqual(len(snippet), 320)
        self.assertTrue(snippet)

    def test_build_candidates_from_stub_llm(self):
        llm = StubLLM([PARSE_ANSWERABLE_JSON])
        candidates = gen.build_llm_candidates(self.docs, llm, per_doc=2, max_docs=1)
        self.assertTrue(candidates)
        self.assertEqual(candidates[0].provider, "llm")
        self.assertIn(candidates[0].source, CORPUS_DOC_IDS)
        self.assertTrue(candidates[0].rationale)
        self.assertEqual(len(llm.calls), 1)

    def test_retry_once_then_succeed(self):
        llm = StubLLM(["抱歉，我先解释一下思路。", PARSE_ANSWERABLE_JSON])
        candidates = gen.build_llm_candidates(self.docs, llm, per_doc=1, max_docs=1)
        self.assertTrue(candidates)
        self.assertEqual(len(llm.calls), 2)  # 第一次失败后重试
        self.assertIn("只输出 JSON 数组", llm.calls[1]["user"])

    def test_retry_fails_twice_and_skips_without_crash(self):
        llm = StubLLM(["不是 JSON", "仍然不是 JSON"])
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            candidates = gen.build_llm_candidates(self.docs, llm, per_doc=1, max_docs=1)
        self.assertEqual(candidates, [])
        self.assertEqual(len(llm.calls), 2)
        self.assertIn("跳过文档", stderr.getvalue())

    def test_llm_exception_is_reported_not_raised(self):
        llm = StubLLM(error=RuntimeError("503 Service Unavailable"))
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            candidates = gen.build_llm_candidates(self.docs, llm, per_doc=1, max_docs=1)
        self.assertEqual(candidates, [])
        self.assertIn("503", stderr.getvalue())


# ---------------------------------------------------------------------------
# 4. promote：解析勾选 + 写回评测集
# ---------------------------------------------------------------------------


class ReviewParseTest(unittest.TestCase):
    def setUp(self):
        self._tmp = TempWorkspace(prefix="gen-eval-review-")
        self.tmp = self._tmp.__enter__()

    def tearDown(self):
        self._tmp.__exit__(None, None, None)

    def test_render_and_parse_roundtrip(self):
        candidates = [
            gen.Candidate(
                id="c001",
                question="资料里关于「倒排索引」是怎么说的？",
                gold_doc_ids=["beta"],
                type="factual",
                rationale="答案在 beta 的小节里",
            ),
            gen.Candidate(
                id="c002",
                question="语料里讲过宋代汝窑的开片特征吗？",
                gold_doc_ids=[],
                type="unanswerable",
                rationale="属领域外主题",
            ),
        ]
        text = gen.render_review_md(candidates)
        for line in text.splitlines():
            if line.startswith("- [ ] |"):
                self.assertEqual(line.count("|"), 7)  # 7 列之间没有竖线残留
        rows = [line for line in text.splitlines() if line.startswith("- [ ]")]
        self.assertEqual(len(rows), 2)
        self.assertIn("不可回答的题", text)
        self.assertEqual(gen.parse_review_file(_write(self.tmp, "rev.md", text), only_checked=True), [])
        self.assertEqual(
            gen.parse_review_file(_write(self.tmp, "rev2.md", text)), ["c001", "c002"]
        )

    def test_only_checked_rows_are_returned(self):
        text = "\n".join(
            [
                "# 复核表",
                "| 勾选 | id | type | difficulty | question | gold_doc_ids | 备注 |",
                "| --- | --- | --- | --- | --- | --- | --- |",
                "- [ ] | c001 | factual | easy | 未勾选题 | beta | 备注 |",
                "- [x] | c002 | multihop | hard | 已勾选题 | beta, gamma | 备注 |",
                "- [X] | c003 | comparison | medium | 大写 X 也算勾选 | beta, delta | 备注 |",
                "- [ ] | c004 | unanswerable | easy | 未勾选题二 | （空） | 备注 |",
                "",
            ]
        )
        path = _write(self.tmp, "rev.md", text)
        self.assertEqual(gen.parse_review_file(path, only_checked=True), ["c002", "c003"])
        # only_checked=False 时会把未勾选行也收进来（诊断场景），但 promote 永远用 True
        self.assertEqual(gen.parse_review_file(path, only_checked=False), ["c001", "c002", "c003", "c004"])

    def test_missing_review_file_raises_readable_error(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            gen.parse_review_file(os.path.join(self.tmp, "不存在.md"))
        self.assertIn("复核表不存在", str(ctx.exception))


class PromoteTest(unittest.TestCase):
    EXISTING = [
        {
            "id": "q01",
            "question": "RAG 的全称是什么？",
            "gold_doc_ids": ["alpha"],
            "gold_answer": "检索增强生成。",
            "difficulty": "easy",
            "type": "factual",
        },
        {
            "id": "q12",
            "question": "混合检索里向量与 bm25 各有什么强项？",
            "gold_doc_ids": ["beta"],
            "gold_answer": "一个擅长语义，一个擅长精确匹配。",
            "difficulty": "hard",
            "type": "comparison",
        },
    ]
    CANDIDATES = [
        {
            "id": "c001",
            "question": "资料里关于「倒排索引」是怎么说的？",
            "gold_doc_ids": ["beta"],
            "gold_answer": "倒排索引把词项映射到文档列表。",
            "difficulty": "easy",
            "type": "factual",
            "provider": "heuristic",
            "status": "pending",
        },
        {
            "id": "c002",
            "question": "语料里讲过宋代汝窑的开片特征吗？",
            "gold_doc_ids": [],
            "gold_answer": "语料未涉及，应拒答。",
            "difficulty": "easy",
            "type": "unanswerable",
            "provider": "heuristic",
            "status": "pending",
        },
        {
            "id": "c003",
            "question": "这个 gold 文档并不存在？",
            "gold_doc_ids": ["幽灵文档"],
            "gold_answer": "",
            "difficulty": "medium",
            "type": "factual",
            "provider": "llm",
            "status": "pending",
        },
        {
            "id": "c004",
            "question": "RAG 的全称是什么？",  # 与既有 q01 重复
            "gold_doc_ids": ["alpha"],
            "gold_answer": "检索增强生成。",
            "difficulty": "easy",
            "type": "factual",
            "provider": "llm",
            "status": "pending",
        },
        {
            "id": "c005",
            "question": "未被勾选的题，不应该被写入。",
            "gold_doc_ids": ["alpha"],
            "gold_answer": "",
            "difficulty": "easy",
            "type": "factual",
            "provider": "heuristic",
            "status": "pending",
        },
    ]

    def test_only_checked_are_promoted_with_continuous_ids(self):
        result = gen.promote_candidates(
            self.CANDIDATES,
            ["c001", "c002", "c003", "c004"],
            self.EXISTING,
            valid_doc_ids={"alpha", "beta", "gamma", "delta"},
        )
        ids = [row["id"] for row in result.accepted]
        self.assertEqual(ids[0], "q13")  # 既有最大编号是 q12
        self.assertTrue(all(len(i) == 3 and i.startswith("q") for i in ids))
        questions = [row["question"] for row in result.accepted]
        self.assertIn("资料里关于「倒排索引」是怎么说的？", questions)
        self.assertIn("语料里讲过宋代汝窑的开片特征吗？", questions)
        self.assertNotIn("未被勾选的题，不应该被写入。", questions)
        self.assertNotIn("RAG 的全称是什么？", questions)  # 重复题被跳过
        self.assertEqual([row["gold_doc_ids"] for row in result.accepted if row["type"] == "unanswerable"], [[]])

        self.assertEqual([cid for cid, _ in result.rejected], ["c003"])
        self.assertIn("不存在", result.rejected[0][1])
        self.assertIn("幽灵文档", result.rejected[0][1])
        self.assertEqual([cid for cid, _ in result.skipped], ["c004"])

    def test_next_qa_id_formats(self):
        self.assertEqual(gen.next_qa_id(["q01", "q12"]), "q13")
        self.assertEqual(gen.next_qa_id([]), "q01")
        self.assertEqual(gen.next_qa_id(["q1", "q9"]), "q10")
        self.assertEqual(gen.next_qa_id(["自定义", "q05"]), "q06")

    def test_promote_candidates_is_pure(self):
        snapshot = json.dumps(self.EXISTING, ensure_ascii=False)
        gen.promote_candidates(self.CANDIDATES, ["c001"], self.EXISTING, valid_doc_ids={"beta"})
        self.assertEqual(json.dumps(self.EXISTING, ensure_ascii=False), snapshot)


class PromoteCommandTest(unittest.TestCase):
    def setUp(self):
        self._tmp = TempWorkspace(prefix="gen-eval-promote-")
        self.tmp = self._tmp.__enter__()
        self.corpus = write_corpus(self.tmp)
        self.qa = os.path.join(self.tmp, "qa.jsonl")
        self.candidates = os.path.join(self.tmp, "candidates.jsonl")
        self.review = os.path.join(self.tmp, "candidates_review.md")
        gen.write_jsonl(self.qa, PromoteTest.EXISTING)
        gen.write_jsonl(self.candidates, PromoteTest.CANDIDATES)
        rows = []
        checked = {"c001", "c002", "c003", "c004"}
        for item in PromoteTest.CANDIDATES:
            mark = "x" if item["id"] in checked else " "
            rows.append(
                f"- [{mark}] | {item['id']} | {item['type']} | {item['difficulty']} | "
                f"{item['question']} | {', '.join(item['gold_doc_ids']) or '（空）'} | 备注 |"
            )
        _write(self.tmp, "candidates_review.md", "\n".join([gen.render_review_md([]).split("## 勾选方式")[0], "## 勾选", *rows, ""]))

    def tearDown(self):
        self._tmp.__exit__(None, None, None)

    def test_dry_run_writes_nothing(self):
        before = os.path.getsize(self.qa)
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = gen.main(
                [
                    "promote",
                    "--qa",
                    self.qa,
                    "--candidates",
                    self.candidates,
                    "--review",
                    self.review,
                    "--corpus",
                    self.corpus,
                    "--only-checked",
                    "--dry-run",
                ]
            )
        self.assertEqual(code, 0)
        self.assertEqual(os.path.getsize(self.qa), before)
        self.assertIn("[dry-run]", stdout.getvalue())
        self.assertIn("拒绝 c003", stderr.getvalue())
        self.assertIn("跳过 c004", stderr.getvalue())

    def test_real_promote_appends_and_reports_stats(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = gen.main(
                [
                    "promote",
                    "--qa",
                    self.qa,
                    "--candidates",
                    self.candidates,
                    "--review",
                    self.review,
                    "--corpus",
                    self.corpus,
                    "--only-checked",
                ]
            )
        self.assertEqual(code, 0, stderr.getvalue())
        rows = gen.load_candidates(self.qa)
        self.assertEqual(len(rows), len(PromoteTest.EXISTING) + 2)
        self.assertEqual([row["id"] for row in rows[-2:]], ["q13", "q14"])
        self.assertIn("各 type 分布", stdout.getvalue())
        # 追加后仍应是合法评测集（能被 run_eval 的 loader 读出来）
        self.assertEqual(len(gen.load_qa_items(self.qa)), 4)

    def test_no_checked_rows_returns_failure(self):
        empty = _write(self.tmp, "empty_review.md", "# 空复核表\n")
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = gen.main(
                [
                    "promote",
                    "--qa",
                    self.qa,
                    "--candidates",
                    self.candidates,
                    "--review",
                    empty,
                    "--corpus",
                    self.corpus,
                    "--only-checked",
                ]
            )
        self.assertEqual(code, 1)
        self.assertIn("没有被勾选", stderr.getvalue())


# ---------------------------------------------------------------------------
# 5. generate 子命令：无 key 时的路线选择
# ---------------------------------------------------------------------------


class GenerateCommandTest(unittest.TestCase):
    def setUp(self):
        self._tmp = TempWorkspace(prefix="gen-eval-cmd-")
        self.tmp = self._tmp.__enter__()
        self.corpus = write_corpus(self.tmp)
        self.qa = os.path.join(self.tmp, "qa.jsonl")
        gen.write_jsonl(self.qa, PromoteTest.EXISTING)
        self.out = os.path.join(self.tmp, "candidates.jsonl")
        self.review = os.path.join(self.tmp, "candidates_review.md")

    def tearDown(self):
        self._tmp.__exit__(None, None, None)

    def _run(self, extra):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = gen.main(
                [
                    "generate",
                    "--corpus",
                    self.corpus,
                    "--out",
                    self.out,
                    "--review",
                    self.review,
                    "--seed",
                    "13",
                ]
                + extra
            )
        return code, stdout.getvalue(), stderr.getvalue()

    def test_auto_falls_back_to_heuristic_without_key(self):
        """build_llm 返回 None（没有 key）时，auto 必须走启发式并**明确提示原因**。"""
        with mock.patch.object(gen, "build_llm", return_value=None):
            code, stdout, stderr = self._run(["--provider", "auto", "--api-key-env", "RAGLAB_NO_SUCH_KEY"])
        self.assertEqual(code, 0, stderr)
        self.assertIn("未检测到环境变量 RAGLAB_NO_SUCH_KEY", stderr)
        self.assertIn("启发式", stdout)
        rows = gen.load_candidates(self.out)
        self.assertGreaterEqual(len(rows), 4)
        self.assertTrue(all(row["provider"] == "heuristic" for row in rows))
        self.assertTrue(all(row["status"] == "pending" for row in rows))
        self.assertTrue(os.path.isfile(self.review))
        with open(self.review, "r", encoding="utf-8") as handle:
            review_text = handle.read()
        self.assertGreaterEqual(review_text.count("- [ ] |"), 4)
        # 候选与既有评测集不能重复
        existing = {gen.normalize_question(row["question"]) for row in gen.load_qa_items(self.qa)}
        for row in rows:
            self.assertNotIn(gen.normalize_question(row["question"]), existing)

    def test_llm_provider_without_key_still_produces_heuristic_output(self):
        with mock.patch.object(gen, "build_llm", return_value=None):
            code, stdout, stderr = self._run(["--provider", "llm", "--api-key-env", "RAGLAB_NO_SUCH_KEY"])
        self.assertEqual(code, 0, stderr)
        self.assertTrue(os.path.isfile(self.out))

    def test_dry_run_does_not_create_files(self):
        with mock.patch.object(gen, "build_llm", return_value=None):
            code, stdout, _ = self._run(["--provider", "heuristic", "--dry-run"])
        self.assertEqual(code, 0)
        self.assertIn("[dry-run]", stdout)
        self.assertFalse(os.path.exists(self.out))
        self.assertFalse(os.path.exists(self.review))

    def test_generate_with_stub_llm_goes_llm_route(self):
        with mock.patch.object(gen, "build_llm", return_value=StubLLM([PARSE_ANSWERABLE_JSON])):
            code, stdout, stderr = self._run(["--provider", "auto", "--api-key-env", "RAGLAB_NO_SUCH_KEY"])
        self.assertEqual(code, 0, stderr)
        self.assertIn("LLM", stdout)
        rows = gen.load_candidates(self.out)
        self.assertTrue(rows)
        self.assertEqual({row["provider"] for row in rows}, {"llm"})

    def test_existing_review_checkmarks_are_preserved(self):
        _write(self.tmp, "candidates_review.md", "# 旧表\n- [x] | c001 | factual | easy | 旧题 | beta | 备注 |\n")
        with mock.patch.object(gen, "build_llm", return_value=None):
            code, _, stderr = self._run(["--provider", "heuristic"])
        self.assertEqual(code, 0, stderr)
        with open(self.review, "r", encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("- [x] | c001 | factual | easy | 旧题 | beta | 备注 |", text)

    def test_parse_types_validates_input(self):
        self.assertEqual(gen.parse_types("factual,multihop"), ["factual", "multihop"])
        self.assertEqual(gen.parse_types(""), list(gen.CANDIDATE_TYPES))
        with self.assertRaises(ValueError) as ctx:
            gen.parse_types("factual,科幻")
        self.assertIn("未知类型", str(ctx.exception))

    def test_stats_command_runs(self):
        with mock.patch.object(gen, "build_llm", return_value=None):
            self._run(["--provider", "heuristic"])
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = gen.main(["stats", "--qa", self.qa, "--candidates", self.out])
        self.assertEqual(code, 0)
        self.assertIn("评测集", stdout.getvalue())
        self.assertIn("候选", stdout.getvalue())
        self.assertIn("下一个可用编号：q13", stdout.getvalue())

    def test_main_without_command_prints_help(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = gen.main([])
        self.assertEqual(code, 1)
        self.assertIn("generate", stdout.getvalue())

    def test_help_lists_required_flags(self):
        import contextlib

        for command, flags in (
            ("generate", ["--corpus", "--out", "--review", "--per-doc", "--max-docs", "--provider", "--model",
                          "--api-base", "--api-key-env", "--types", "--seed"]),
            ("promote", ["--review", "--qa", "--only-checked", "--dry-run"]),
            ("stats", ["--qa", "--candidates"]),
        ):
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer), self.assertRaises(SystemExit):
                gen.main([command, "--help"])
            text = buffer.getvalue()
            for flag in flags:
                self.assertIn(flag, text, f"{command} --help 缺少 {flag}")


def _write(directory: str, name: str, text: str) -> str:
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


if __name__ == "__main__":
    unittest.main()
