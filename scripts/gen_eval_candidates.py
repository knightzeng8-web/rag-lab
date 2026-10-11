"""半自动生成评测集候选问题，并把人工筛选结果回填进 ``data/eval/qa.jsonl``。

为什么需要这个脚本：``data/eval/qa.jsonl`` 目前只有 12 题，而 12 题的评测集上
任何一个指标都可能有 ±8% 的抖动（一道题就是 1/12），拿它比较消融实验等于掷骰子。
评测集必须扩，但「人工从语料里想题」既慢又容易只想到自己会答的那类题；
所以这里的定位是**半自动**：

1. ``generate``：先按「LLM 优先、启发式兜底」产出候选，同时落盘一份可勾选的
   Markdown 复核表。为什么要兜底：评测集生成的脚本必须能在没有 API Key 的
   环境里跑出可用的结果，否则「扩评测集」这件事会被环境卡住。
2. 人工在复核表里勾选（``- [x]``）并改错——**这一步不能省**，因为 LLM 生成的
   ``unanswerable`` 题有相当比例其实语料里答得出来，直接入库会污染拒答指标。
3. ``promote``：把勾选项按新 id 追加进评测集，并做三道闸门
   （问题去重 / gold 文档必须真实存在 / 编号连续）。

设计上刻意把「生成 / 解析 / 校验 / 写回」拆成纯函数：
整条链路里唯一无法离线验证的部分就是 LLM 的网络调用，其余全部可单测。

用法::

    python scripts/gen_eval_candidates.py generate --corpus data/corpus
    python scripts/gen_eval_candidates.py promote  --only-checked --dry-run
    python scripts/gen_eval_candidates.py stats
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# 让脚本既能 `python scripts/xxx.py` 直接跑，也能被测试模块按路径加载
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rag.config import load_config  # noqa: E402
from rag.eval.run_eval import load_dataset  # noqa: E402
from rag.generate.llm import OpenAICompatLLM, build_llm  # noqa: E402
from rag.index.bm25 import BM25Index  # noqa: E402
from rag.ingest.chunker import _sections  # noqa: E402
from rag.ingest.loader import Document, load_documents  # noqa: E402
from rag.text import (  # noqa: E402
    _CJK_RUN,  # noqa: PLC2701 - 复用项目内同一套 CJK 切分口径，避免词表与分词不一致
    normalize_ws,
    split_sentences,
    tokenize,
    truncate,
)

__all__ = [
    "CANDIDATE_TYPES",
    "DIFFICULTIES",
    "OUT_OF_DOMAIN_TOPICS",
    "PRESENCE_MIN_COUNT",
    "PRESENCE_SHARE_CAP",
    "PRESENCE_SHARE_THRESHOLD",
    "RARE_TOP_K",
    "Candidate",
    "adaptive_coverage_threshold",
    "build_llm_candidates",
    "build_llm_prompt",
    "cmd_generate",
    "cmd_promote",
    "cmd_stats",
    "corpus_doc_ids",
    "corpus_hit_count",
    "corpus_text_of",
    "corpus_vocabulary",
    "corpus_words",
    "doc_snippet",
    "format_rare_terms",
    "heuristic_candidates",
    "is_good_term",
    "is_standalone_term",
    "load_candidates",
    "load_qa_items",
    "main",
    "make_bm25",
    "missing_idf",
    "next_qa_id",
    "normalize_question",
    "overlap_ratio",
    "parse_llm_json",
    "parse_review_file",
    "parse_types",
    "promote_candidates",
    "rare_term_report",
    "render_review_md",
    "select_llm_docs",
    "structured_terms",
    "term_frequency_rank",
    "term_free_occurrences",
    "term_has_boundary",
    "token_overlap",
    "validate_candidate",
    "weighted_overlap",
    "write_candidates",
    "write_jsonl",
]
# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

CANDIDATE_TYPES: Tuple[str, ...] = ("factual", "comparison", "multihop", "unanswerable")
DIFFICULTIES: Tuple[str, ...] = ("easy", "medium", "hard")

# 这些单字/套话几乎是语料里必然存在的词，若把它们算进「不可回答」的覆盖率，
# 任何中文问题都会被判成「和语料相关」，这道闸门就失效了 —— 所以对不可回答题
# 只统计实词（长度 >= 2 的 token），并额外排除题目模板自带的骨架词。
_QUESTION_SKELETON = {
    "资料", "关于", "怎么", "如何", "什么", "哪些", "都有", "说法", "介绍",
    "说明", "综合", "结合", "对比", "两者", "区别", "差异", "需要", "请问",
    "文档", "内容", "怎样", "为何", "为什么",
}

# ---------------------------------------------------------------------------
# 「不可回答」判据的参数
#
# 判据的演进（两次都被真实语料打回来，记在这里避免再走回头路）：
#
# v1：只看「IDF 加权覆盖率」（问题实词的 IDF 有多少落在语料里），阈值 0.3。
#     5 篇手写笔记时够用；语料扩到 48 篇（含 43 篇中文维基条目）后整体失效——
#     长问题里总会撞上一堆通用二元组（"控制""选择""工艺""机制"），覆盖率被
#     稳定抬到 0.4~0.6，**一条 unanswerable 都产不出来**。
#
# v2（当前）：回到「最关键的术语在语料里到底有没有支撑」，并且把
#     **「词表里出现过」与「真的在讲」区分开**——这是关键的一步。
#     中文分词是「单字 + 重叠二元组」，所以几乎任何短语都会撞上几个恰好出现的
#     二元组（「与后」「料里」「数分」）。用「出现过就算数」（df >= 1）去判，
#     领域外题目会被大面积误判成可答题。
#     实测：真正在讲某个话题的词在语料里会出现很多次，而巧合撞上的二元组通常
#     只出现 1 次。于是判据变成一条**出现次数**的线：
#         topic 的实词中，在语料全文里出现 >= PRESENCE_MIN_COUNT 次的比例
#         低于阈值 → 语料答不了。
#     48 篇语料上的实测分离度：领域内问题（BM25 参数 / 倒排索引 / RRF / 精确率
#     与召回率 / Transformer 自注意力）落在 0.50~0.88；领域外问题（红楼梦成书、
#     珠峰高程、清明上河图、登月坐标、世界杯比分）落在 0.00~0.40。
#     通用词仍会贡献少量噪声（「作者」「年代」这类词在任何技术语料里都有），
#     所以阈值取在两者中间，并保留覆盖率作为日志里的辅助说明。
# ---------------------------------------------------------------------------

# 参与判据的「关键术语」个数：太少容易被单个巧合术语带偏，太多会把通用词算进来
RARE_TOP_K = 5
# 一个术语算「语料里真的在讲它」所需的最小出现次数（全文子串计数）
PRESENCE_MIN_COUNT = 2
# 「实词命中率」阈值：低于它判定为「语料答不了」；随语料规模轻微放宽
PRESENCE_SHARE_THRESHOLD = 0.45
PRESENCE_SHARE_CAP = 0.60
# 覆盖率上限的基准值与放宽上限（见 adaptive_coverage_threshold）
#
# 注意：覆盖率**默认不参与判定**（`DEFAULT_COVERAGE_LIMIT = 0.0` 表示关闭），只作为
# 日志里的辅助统计量输出。原因是它在这里必然很低——分母是「主题实词的 IDF 总和」，
# 而中文主题里大量二元组本就不在语料词表内，算出来的覆盖率会被这些词主导，
# 从而总是低于任何有意义的阈值（实测 48 篇语料上每个主题都是 0.57），
# 一开就会把领域外题目全部误杀。需要更严格时可以用 `--unanswerable-coverage` 显式打开。
COVERAGE_BASE = 0.55
COVERAGE_CAP = 0.75
DEFAULT_COVERAGE_LIMIT = 0.0
# 生成多少条 unanswerable 就够（候选总量有限，再多也只是重复题型）
UNANSWERABLE_TARGET = 8

# 领域外主题清单。两条挑选原则（都是被真实语料打回来之后总结的）：
# 1. **以叙述性事实与专有名词为主**：文学作品的成书年代与作者、历史事件的时间地点、
#    体育赛事结果、地理地标、影视音乐作品与获奖、考古发现等。
#    早期版本用的是「月饼糖浆转化工艺」「高血压合并痛风用药」这类**技术性表述**，
#    它们反而容易和通用技术语料撞上「工艺」「控制」「机制」「选择」等二元组，
#    于是被「覆盖率」判据成批误判成可答题。
# 2. 每条都要能被「专有名词缺席」判据识别：关键实体在通用 ML/NLP 语料里必然查不到，
#    这一点由 `_unanswerable_candidates` 在运行期用 BM25 的 df 再筛一遍（见 rare_term_report）。
OUT_OF_DOMAIN_TOPICS: Tuple[str, ...] = (
    "《红楼梦》前八十回与后四十回的作者归属之争",
    "《百年孤独》中马孔多小镇的建立者是谁",
    "哥伦布首次抵达美洲的年份与登陆地点",
    "第一届现代奥林匹克运动会的举办城市与年份",
    "2022 年卡塔尔世界杯决赛的比分与进球球员",
    "珠穆朗玛峰 2020 年公布的最新雪面高程",
    "敦煌莫高窟第 220 窟壁画的创作年代",
    "《清明上河图》现藏于哪家博物馆",
    "贝多芬第九交响曲首演时的指挥与地点",
    "科幻电影《银翼杀手》的导演与上映年份",
    "《千与千寻》获得的国际电影节奖项",
    "哥白尼《天体运行论》的出版年份",
    "阿波罗 11 号登月舱降落点的月面坐标",
    "《蒙娜丽莎》在 1911 年被盗案的经过",
    "日本战国时代桶狭间之战的交战双方",
    "丝绸之路沿线楼兰古城的废弃原因",
    "复活节岛摩艾石像的雕刻年代",
    "《马拉喀什条约》的签署年份与缔约国数量",
    "古罗马斗兽场建成时的皇帝是谁",
    "亚马逊河与尼罗河长度之争的测量结论",
    "《源氏物语》成书的年代与作者",
    "玛雅文明长计历纪元的结束日期",
)

_UNANSWERABLE_TEMPLATES: Tuple[str, ...] = (
    "资料里关于「{topic}」有哪些说明？",
    "请问「{topic}」在语料中是怎么描述的？",
    "关于「{topic}」，资料给出的关键结论是什么？",
    "{topic}的主要流程和注意事项是什么？",
)

_FACTUAL_TEMPLATES: Tuple[str, ...] = (
    "资料里关于「{topic}」是怎么说的？",
    "根据语料，「{topic}」的关键要点是什么？",
    "「{topic}」在资料中是如何定义的？",
)

_COMPARISON_TEMPLATE = "资料中「{a}」与「{b}」对「{term}」的说法有什么差异？"
_MULTIHOP_TEMPLATE = (
    "「{doc_a}」提到了「{term}」，而「{doc_b}」解释了「{term}」的做法，"
    "这与「{doc_c}」的内容结合起来说明了什么？"
)

_LLM_SYSTEM_PROMPT = (
    "你是 RAG 评测集出题专家。你只依据用户给出的【资料】出题，"
    "绝不使用资料之外的知识，也绝不编造资料里没有的文档 id。"
    "你的输出必须是合法的 JSON 数组，不能包含解释、前后缀或 Markdown 代码块。"
)

_LLM_USER_PROMPT = """请为下面的【资料】出 {per_doc} 道中文评测题，覆盖这些类型：{types}。

【资料】
{materials}

【出题硬性要求】
1. 每道题必须**仅凭上面的资料就能回答**，不允许问需要外部知识、常识推测或时效的问题。
2. `gold_doc_ids` 只能填资料里出现过的文档 id（方括号里的 id，如 [{first_id}]），
   通常 1~3 个；只能填 id 本身，不要加解释。
3. `type=unanswerable` 的题，`gold_doc_ids` 必须是空数组 `[]`，
   且问题必须**确实无法由这些资料回答**（问资料里完全没涉及的领域）。
4. `gold_answer` 写一段可以据资料核对的中文参考答案（不可回答的题写「资料未涉及」类说明）。
5. `difficulty` 只能取 easy / medium / hard；`type` 只能取 {types}。
6. 同一批题目之间不要重复，也不要用同一个问法反复问同一个点。

【输出格式】只输出一个 JSON 数组，不要任何解释文字、不要 Markdown 代码块：
[{{"question": "问题", "gold_doc_ids": ["doc-id"], "gold_answer": "参考答案",
  "difficulty": "easy", "type": "factual"}}]
"""


def _stderr(message: str) -> None:
    """统一的中文提示出口；不抛异常，避免把「提示」升级成「失败」。"""
    print(f"[gen_eval_candidates] {message}", file=sys.stderr)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    """一条候选题目；``rationale`` 记「为什么这题能答/不能答」，是人工复核的主要依据。"""

    id: str = ""
    question: str = ""
    gold_doc_ids: List[str] = field(default_factory=list)
    gold_answer: str = ""
    difficulty: str = "medium"
    type: str = "factual"
    source: str = "heuristic"
    provider: str = "heuristic"
    status: str = "pending"
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "gold_doc_ids": list(self.gold_doc_ids),
            "gold_answer": self.gold_answer,
            "difficulty": self.difficulty,
            "type": self.type,
            "source": self.source,
            "provider": self.provider,
            "status": self.status,
            "rationale": self.rationale,
        }


@dataclass
class PromoteResult:
    """``promote_candidates`` 的结果：写回的行 + 被拒绝的行（带中文原因）。"""

    accepted: List[Dict[str, Any]] = field(default_factory=list)
    rejected: List[Tuple[str, str]] = field(default_factory=list)
    skipped: List[Tuple[str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 基础工具（纯函数，便于单测）
# ---------------------------------------------------------------------------


def normalize_question(text: str) -> str:
    """归一化用于去重：只看字符与数字，忽略标点/大小写/空白差异。

    为什么要这么激进：`「X」是怎么说的？` 与 `X 是怎么说的` 是同一个问题，
    如果按原始文本比对，评测集里会堆积大量近重复题，把指标权重悄悄带偏。
    """
    text = (text or "").lower()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)


def token_overlap(text: str) -> List[str]:
    """取出「参与相关性判断的 token」：排除模板套话与单字。

    单字必须排除：中文分词会产出大量单字（的/是/在…），它们与语料几乎必然重合，
    留着会让「不可回答性」这道闸门的区分度降到 0。
    """
    picked: List[str] = []
    for token in tokenize(text or ""):
        if token in _QUESTION_SKELETON:
            continue
        if len(token) == 1 and not token.isdigit():
            continue
        picked.append(token)
    return picked


def overlap_ratio(question: str, vocab: Iterable[str], ignore: str = "") -> float:
    """问题实词里有多大比例出现在语料词表中（0.0~1.0）。

    ``ignore`` 里出现的 token 会被排除出分母——用来剔除「题目模板自己带来的词」
    （如「资料/关于/说法」），否则模板写得越长，分母越大，比率就越失真。
    """
    ignored: Set[str] = set()
    if ignore:
        ignored = {token for token in tokenize(ignore) if len(token) >= 2}
    tokens = [token for token in token_overlap(question) if token not in ignored]
    if not tokens:
        return 0.0
    known: Set[str] = set(vocab)
    hits = sum(1 for token in tokens if token in known)
    return hits / float(len(tokens))


def corpus_hit_count(question: str, vocab: Iterable[str], ignore: str = "") -> int:
    """问题实词中命中语料词表的 token 数（诊断用）。"""
    ignored: Set[str] = set()
    if ignore:
        ignored = {token for token in tokenize(ignore) if len(token) >= 2}
    known: Set[str] = set(vocab)
    return sum(1 for token in token_overlap(question) if token not in ignored and token in known)


def term_free_occurrences(term: str, text: str) -> Tuple[int, int]:
    """统计术语在语料里「有多少次出现在词边界上」（返回 ``(自由次数, 总次数)``）。

    这是目前唯一能机械识别「跨词边界假词」的信号：中文分词产出重叠二元组，
    「分块策略」里的「块策」客观存在于文本中，但它每一次出现都被前后汉字夹住 ——
    它从来不是一个词，只是两个词的接缝。
    实测（48 篇语料）：「块策」自由次数 0/3（必须丢），
    「网络」90/279、「策略」17/23、「意力」18/102（保留；误杀的代价更高）。
    """
    if not term or not text:
        return 0, 0
    size = len(term)
    free = 0
    total = 0
    start = text.find(term)
    while start != -1:
        total += 1
        left_ok = start == 0 or not _is_cjk_char(text[start - 1])
        end = start + size
        right_ok = end >= len(text) or not _is_cjk_char(text[end])
        if left_ok or right_ok:
            free += 1
        start = text.find(term, start + 1)
    return free, total


def term_has_boundary(term: str, text: str, min_free: int = 1) -> bool:
    """术语是否至少有一次出现在词边界上（拉丁/数字术语不参与该判据）。"""
    if not _cjk_only(term):
        return True
    return term_free_occurrences(term, text)[0] >= max(1, int(min_free))


def corpus_words(text: str, min_len: int = 4, min_free: int = 1) -> Set[str]:
    """语料里「被写成独立单元」的长 CJK 串（长度 >= min_len 的连续 CJK run）。

    与 ``structured_terms`` 配合使用：标题覆盖「作者明确命名的小节」，
    这里覆盖「作者成串写出的长短语」。短片段（二元组）刻意不单独收——
    它们正是跨词边界假词（「块策」「意力」）的来源；确实需要二元组时，
    由 ``structured_terms`` 从标题里取，并用 ``term_has_boundary`` 复核。
    """
    words: Set[str] = set()
    for run in _CJK_RUN.findall(text or ""):
        if len(run) >= int(min_len):
            for size in range(int(min_len), len(run) + 1):
                for start in range(0, len(run) - size + 1):
                    words.add(run[start : start + size])
    return {word for word in words if term_has_boundary(word, text, min_free=min_free)}


def is_standalone_term(term: str, words: Set[str]) -> bool:
    """术语是否出现在「结构化的位置」（标题/小节标题/独立长词条）上。

    为什么不用「是不是语料的子串」：中文分词产出的是重叠二元组，
    「分块策略」里的「块策」、「注意力机制」里的「意力」都**是**语料子串，
    用子串判定等于没判。有效的信号是「它有没有被当作一个单元写出来」——
    小节标题、文档标题，或者作者成串写出的长短语。
    """
    if not _cjk_only(term):
        return True
    return term in words


# 题面（comparison / multihop）里会写进术语，假术语比事实题更刺眼，
# 所以这两条路线只采纳「出现在结构化位置」的术语
_DOC_TERM_HINT = "结构化位置（文档标题 / 小节标题 / 独立长词条）"


def structured_terms(docs: Sequence[Document], corpus_text: str = "") -> Set[str]:
    """收集语料里「被当作一个单元写出来」的术语，供 comparison / multihop 挑词。

    做法：把每个标题单独分词，再在同一标题内部剔除被更长词条包含的重叠二元组。
    为什么必须逐标题做而不是全局做——中文 tokenize 对整段 CJK run 只产出
    「单字 + 重叠二元组」，所以「分块策略」必然带来跨词边界的假词「块策」、
    「注意力机制」带来「意力」。这些假词**同样会出现在标题的分词结果里**，
    光看「是否出现在标题中」是挡不住的；只有在标题内部比较长短、
    才能认出「块策 ⊂ 分块策略」，从而把碎片剔掉。

    传了 ``corpus_text`` 时还会用 ``term_has_boundary`` 再筛一遍：
    标题分词本身也会产出跨词边界的接缝词。另外两类来源一并收进来：
    拉丁标识符（分词不会切错）、长度 >= 4 的独立 CJK 串（作者成串写出的长短语）。
    """
    terms: Set[str] = set()
    for doc in docs:
        for title in [doc.title] + _heading_terms(doc):
            # 关键：用**未截断**的 token 集合当碎片判据的证人。
            # 一旦先把长 token（「注意力机」）按长度上限丢掉，
            # 就再也证明不了「意力 ⊂ 注意力」，假词会照样流出去。
            raw = [
                token
                for token in tokenize(title or "")
                if len(token) >= 2 and _HAS_MEANING.search(token)
            ]
            subsumed = _subsumed_terms(raw)
            for token in raw:
                if token in subsumed or not is_good_term(token):
                    continue
                if corpus_text and not term_has_boundary(token, corpus_text, min_free=2):
                    continue
                terms.add(token)
    return terms


def missing_idf(bm25: BM25Index) -> float:
    """语料里完全没出现过的词应当占的权重。

    这是整套判据里最容易做错的一步：如果给未命中词一个近似 0 的权重，
    那么「覆盖率」就退化成「问题里有多少个词恰好和语料撞上」——而中文二元组
    撞上通用词的概率很高，领域外主题会被大面积误判成「语料其实讲过」。
    这里按 IDF 的定义把 df=0 外推，得到与「从未出现的词」相称的高权重。
    """
    n = max(1, len(bm25.ids))
    return math.log(1.0 + (n + 0.5) / 0.5)


def weighted_overlap(question: str, bm25: BM25Index, ignore: str = "") -> float:
    """按 IDF 加权的语料覆盖率（0.0~1.0）：分母是问题实词的 IDF 总和。

    - 命中语料的词按真实 IDF（越稀有越重要）计分；
    - 语料里根本没有的词按 ``missing_idf`` 计分（权重最大）；
    - 于是「覆盖率」真正回答的是：**这道题谈的东西在语料里的分量占多少**，
      而不是「凑巧撞上了几个通用二元组」。
    """
    ignored: Set[str] = set()
    if ignore:
        ignored = {token for token in tokenize(ignore) if len(token) >= 2}
    tokens = [token for token in token_overlap(question) if token not in ignored]
    if not tokens:
        return 0.0
    absent = missing_idf(bm25)
    total = 0.0
    hit = 0.0
    for token in tokens:
        if token in bm25.postings:
            score = bm25.idf(token)
            total += score
            hit += score
        else:
            total += absent
    return hit / total if total > 0 else 0.0


def term_frequency_rank(question: str, bm25: BM25Index, topic: str = "") -> List[str]:
    """把问题拆成「有信息量的实词」并按 BM25 的 IDF 降序排列（最关键的术语在前）。

    这是「稀有词缺席」判据的输入：只看最关键的几个术语在语料里有没有支撑，
    而不看被一堆通用词稀释之后的平均覆盖率。

    ``topic`` 是题目里真正要问的主题文本。**必须**从它而不是从整道题里抽术语：
    题目模板会在拼接处产出假二元组（「与后」「料里」「些说」），而且模板文字会和
    主题拼成更长的 CJK 串，让 ``_subsumed_terms`` 把真正的主题词当成碎片剔掉。
    """
    if topic:
        # 主题单独抽词：模板不再参与分词，主题也不会被模板干扰
        terms = [term for term in _content_terms(topic) if is_good_term(term)]
    else:
        terms = [term for term in _content_terms(question) if is_good_term(term)]
    terms.sort(key=lambda term: (-bm25.idf(term), -len(term), term))
    return terms


def adaptive_presence_threshold(num_docs: int) -> float:
    """「实词命中率」阈值：随语料规模轻微放宽，封顶在 ``PRESENCE_SHARE_CAP``。

    语料越大、领域越杂，「一个中文短语恰好撞上几次通用二元组」的概率就越高，
    命中率这个量的绝对水平会整体上移；固定阈值在 5 篇语料上合适、
    在 1000 篇语料上就会把领域外题目全部误杀。这里只做很慢的对数放宽。
    """
    if num_docs <= 0:
        return PRESENCE_SHARE_THRESHOLD
    drift = 0.02 * math.log10(max(10, num_docs) / 10.0)
    return min(PRESENCE_SHARE_CAP, PRESENCE_SHARE_THRESHOLD + drift)


def adaptive_coverage_threshold(num_docs: int) -> float:
    """覆盖率兜底阈值：随语料规模缓慢放宽，并封顶在 ``COVERAGE_CAP``。

    为什么会随规模漂移：语料越大、领域越杂，「任何一个中文问题都能撞上几个
    通用二元组」的概率就越高，覆盖率这个量的绝对水平整体上移。固定阈值在
    5 篇语料上合适、在 500 篇语料上就会把领域外题目全部误杀。
    这里用对数放宽（不是线性），保证漂移很慢：48 篇约 0.58，1000 篇也只到 0.70。
    """
    if num_docs <= 0:
        return COVERAGE_BASE
    drift = 0.03 * math.log10(max(10, num_docs) / 10.0)
    return min(COVERAGE_CAP, COVERAGE_BASE + drift)


def corpus_text_of(docs: Sequence[Document]) -> str:
    """把所有文档正文拼成一段文本，用于「这个术语在语料里出现几次」的计数。"""
    return "\n".join(doc.text for doc in docs)


def rare_term_report(
    question: str,
    bm25: BM25Index,
    top_k: int = RARE_TOP_K,
    min_count: int = PRESENCE_MIN_COUNT,
    threshold: Optional[float] = None,
    topic: str = "",
    corpus_text: str = "",
    coverage_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """给出「这条问题在语料里有没有关键支撑」的可解释判据。

    返回的 ``terms`` 列出主题的关键实词、它们在语料全文里的出现次数、文档频次与
    IDF —— 这是判断阈值是否合理时唯一要看的东西（「这条题被丢，是因为哪几个词
    在语料里根本没被讲」）。

    判据（两条，``unanswerable`` 要求两条同时成立）：
    1. **出现次数**：主题实词中在语料里出现 >= ``min_count`` 次的比例（``present_share``）
       低于 ``threshold``。这条是主判据——它把「词表里出现过」与「真的在讲」区分开，
       因为巧合撞上的二元组通常只出现 1 次。
    2. **覆盖率兜底**：IDF 加权覆盖率（``weighted_overlap``）低于按语料规模自适应的
       上限。用于挡住「恰好撞上一个高频术语」的漏网之鱼。

    ``topic`` 必须传：题目模板在拼接处会产出假二元组，且模板文字会与主题拼成更长的
    CJK 串、让真正的主题词被当成碎片剔掉（详见 ``term_frequency_rank``）。
    """
    terms = term_frequency_rank(question, bm25, topic)
    if not terms:
        # 抽不出任何实词：无法判断，保守地当成「答得出来」，避免污染拒答指标
        return {
            "terms": [],
            "all_terms": [],
            "present_share": 1.0,
            "present_count": 0,
            "total_terms": 0,
            "threshold": threshold if threshold is not None else adaptive_presence_threshold(len(bm25.ids)),
            "coverage": 1.0,
            "coverage_threshold": (
                DEFAULT_COVERAGE_LIMIT if coverage_threshold is None else float(coverage_threshold)
            ),
            "coverage_enabled": False,
            "unanswerable": False,
            "relevant": True,
        }

    counts = {term: corpus_text.count(term) for term in terms} if corpus_text else {}
    if not corpus_text:
        # 没给全文时退化成「文档频次 >= 2」：信息略少但语义等价（确实在两篇里被讲到）
        counts = {term: len(bm25.postings.get(term, [])) for term in terms}

    rows: List[Dict[str, Any]] = []
    for term in terms[: max(1, int(top_k))]:
        count = counts.get(term, 0)
        rows.append(
            {
                "term": term,
                "count": count,
                "df": len(bm25.postings.get(term, [])),
                "idf": round(bm25.idf(term), 2),
                "absent": count == 0,
                "present": count >= min_count,
            }
        )
    # 全量术语统计（不只 top-k）：日志与单测需要能查到某个具体术语的出现次数
    all_terms = [
        {
            "term": term,
            "count": counts.get(term, 0),
            "df": len(bm25.postings.get(term, [])),
            "idf": round(bm25.idf(term), 2),
        }
        for term in terms
    ]
    present = sum(1 for term in terms if counts.get(term, 0) >= min_count)
    share = present / float(len(terms))
    share_limit = adaptive_presence_threshold(len(bm25.ids)) if threshold is None else float(threshold)
    coverage = weighted_overlap(question, bm25, ignore=topic)
    # 覆盖率兜底默认关闭（0.0）：显式传入正数才启用，见 DEFAULT_COVERAGE_LIMIT 的说明
    coverage_limit = (
        DEFAULT_COVERAGE_LIMIT
        if coverage_threshold is None
        else float(coverage_threshold)
    )
    # 出现次数太低 → 语料没在讲这件事；top-k 术语全都弱 → 更确定的信号
    weak_terms = bool(rows) and all(item["count"] < min_count for item in rows)
    unanswerable = share < share_limit or weak_terms
    if coverage_limit > 0.0 and coverage >= coverage_limit:
        unanswerable = False  # 覆盖率兜底：显式启用时，覆盖率高就不算领域外
    return {
        "terms": rows,
        "all_terms": all_terms,
        "present_share": share,
        "present_count": present,
        "total_terms": len(terms),
        "threshold": share_limit,
        "coverage": coverage,
        "coverage_threshold": coverage_limit,
        "coverage_enabled": coverage_limit > 0.0,
        "unanswerable": unanswerable,
        # 「关键术语在语料里都有支撑」= 不是领域外题目
        "relevant": present > 0 and share >= share_limit,
    }


def format_rare_terms(report: Dict[str, Any]) -> str:
    """把判据统计量压成一行，用于日志（便于人工复核阈值）。"""
    parts = []
    for item in report.get("terms", [])[:5]:
        flag = "缺席" if item["absent"] else f"出现{item['count']}次"
        parts.append(f"{item['term']}({flag}, df={item['df']}, idf={item['idf']})")
    coverage_note = (
        f"，覆盖率={report.get('coverage', 0.0):.2f}<上限{report.get('coverage_threshold', 0.0):.2f}"
        if report.get("coverage_enabled")
        else f"，覆盖率={report.get('coverage', 0.0):.2f}(未启用)"
    )
    return (
        f"关键词={'/'.join(parts) or '无'}；"
        f"实词命中率={report.get('present_count', 0)}/{report.get('total_terms', 0)}"
        f"={report.get('present_share', 0.0):.2f}<阈值{report.get('threshold', 0.0):.2f}"
        + coverage_note
    )


def corpus_vocabulary(docs: Sequence[Document], max_df_ratio: float = 0.5) -> Set[str]:
    """语料词表：只保留「不是几乎每篇都出现」的 token。

    全量 tokenize 会把「的/是/一」这类单字也算进词表，让任何中文问题都显得「相关」；
    这里顺手把出现在超过一半文档里的 token 剔掉，只留有一点区分度的词。
    """
    df: Dict[str, int] = {}
    for doc in docs:
        for token in set(tokenize(doc.text)):
            df[token] = df.get(token, 0) + 1
    limit = max(1, int(len(docs) * max_df_ratio))
    return {token for token, count in df.items() if count <= limit}


def make_bm25(docs: Sequence[Document]) -> BM25Index:
    """用文档正文建一个 BM25，仅用来取 IDF（词的稀有度）。"""
    return BM25Index().fit([doc.doc_id for doc in docs], [doc.text for doc in docs])


def select_llm_docs(docs: Sequence[Document], max_docs: int, seed: int) -> List[Document]:
    """挑进入 LLM 提示词的文档：按长度排序，长的（信息量大）优先，同长度按 doc_id 稳定排序。"""
    ordered = sorted(docs, key=lambda d: (-len(d.text), d.doc_id))
    picked = ordered[: max(1, int(max_docs))]
    picked.sort(key=lambda d: d.doc_id)  # 输出顺序与 doc_id 一致，便于人看
    return picked


def doc_snippet(doc: Document, max_chars: int = 1200, max_sections: int = 3) -> str:
    """取一篇文档的代表性片段：按标题分节的前 N 节，总长不超过 max_chars。

    为什么不用首 N 个字符：技术文档的开头常常是标题 + 导语，几乎不含可出题的细节，
    按「节」取能覆盖不同的语义单元，出出来的题才不会全挤在同一个知识点上。
    """
    budget = max(200, int(max_chars))
    parts: List[str] = []
    used = 0
    sections = _sections(doc.text)
    if not sections:
        # 没有任何标题的纯文本：退化成按段落取
        for block in [b for b in doc.text.split("\n\n") if b.strip()]:
            room = budget - used
            if room <= 0:
                break
            chunk = truncate(block.strip(), room, "")
            parts.append(chunk)
            used += len(chunk) + 1
            if len(parts) >= max_sections:
                break
    else:
        for section in sections[: max(1, int(max_sections))]:
            room = budget - used
            if room <= 0:
                break
            lines = section.body.split("\n")
            body = "\n".join(lines[1:]).strip() if lines else ""
            if not body:
                body = section.body.strip()
            head = f"### {section.heading}" if section.heading else "### (正文)"
            text = f"{head}\n{body}"
            parts.append(truncate(text, room, "…"))
            used += len(parts[-1]) + 2
    return "\n\n".join(parts).strip()


def render_materials(docs: Sequence[Document], max_chars: int = 1200) -> str:
    """把片段拼成提示词里的资料块。

    必须把 doc_id 明确写进资料里：``gold_doc_ids`` 要求「资料里出现过的 id」，
    如果模型看不到 id，它只能靠猜，产出的候选会被校验闸门整批丢掉。
    """
    blocks: List[str] = []
    for doc in docs:
        snippet = doc_snippet(doc, max_chars=max_chars)
        if not snippet:
            continue
        blocks.append(f"[{doc.doc_id}] 标题：{doc.title}\n{snippet}")
    return "\n\n".join(blocks)


def load_qa_items(path: str) -> List[Dict[str, Any]]:
    """读既有评测集（``load_dataset`` 已做字段兜底），解析失败给出中文提示。"""
    if not os.path.isfile(path):
        return []
    try:
        return [item.to_dict() for item in load_dataset(path)]
    except Exception as exc:
        raise RuntimeError(f"评测集 {path} 解析失败（请检查 JSONL 每行是否合法）：{exc}") from exc


def load_candidates(path: str) -> List[Dict[str, Any]]:
    """读候选文件；空行与 ``#`` 注释行跳过，坏行报错并指出行号。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"候选文件不存在：{path}（先跑 generate 子命令）")
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path} 第 {line_no} 行不是合法 JSON：{exc}") from exc
            if not isinstance(payload, dict):
                raise RuntimeError(f"{path} 第 {line_no} 行必须是 JSON 对象")
            records.append(payload)
    return records


def write_jsonl(path: str, records: Iterable[Dict[str, Any]], dry_run: bool = False) -> int:
    """写 JSONL；``dry_run`` 只统计不落盘，便于先看结果再决定。"""
    rows = list(records)
    if dry_run:
        return len(rows)
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def append_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> int:
    rows = list(records)
    if not rows:
        return 0
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def write_candidates(path: str, candidates: Sequence[Candidate], dry_run: bool = False) -> int:
    return write_jsonl(path, [c.to_dict() for c in candidates], dry_run=dry_run)


# ---------------------------------------------------------------------------
# 校验 / 去重
# ---------------------------------------------------------------------------


def corpus_doc_ids(corpus: Any) -> Set[str]:
    """接受「目录/文件路径」「Document 列表」「doc_id 列表」三种形态，统一成 id 集合。"""
    if corpus is None:
        return set()
    if isinstance(corpus, (str, os.PathLike)):
        docs = load_documents([str(corpus)])
        return {doc.doc_id for doc in docs}
    ids: Set[str] = set()
    for item in corpus:
        if isinstance(item, Document):
            ids.add(item.doc_id)
        elif isinstance(item, dict):
            ids.add(str(item.get("doc_id") or item.get("id") or ""))
        else:
            ids.add(str(item))
    ids.discard("")
    return ids


def _coerce_doc_id_list(raw: Any) -> List[str]:
    if raw is None or raw == "":
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, (list, tuple, set)):
        return [str(x).strip() for x in raw if str(x).strip()]
    return [str(raw)]


def validate_candidate(candidate: Any, valid_doc_ids: Set[str]) -> Tuple[bool, str]:
    """校验一条候选；返回 ``(是否通过, 中文原因)``，原因同时用于 rationale 与拒绝日志。

    三道闸门分别对应三种会污染评测集的错误：
    问题为空（指标分母里混进空题）、gold 指向不存在的文档（永远不可能被召回）、
    类型非法的 unanswerable / 答案不一致（拒答指标被算反）。
    """
    if isinstance(candidate, dict):
        getter = candidate.get
    else:  # Candidate
        getter = lambda key, default=None: getattr(candidate, key, default)  # noqa: E731

    question = str(getter("question") or "").strip()
    if not question:
        return False, "问题为空，跳过"
    kind = str(getter("type") or "factual").strip().lower()
    if kind not in CANDIDATE_TYPES:
        return False, f"type={kind!r} 不在允许集合 {CANDIDATE_TYPES} 中"
    gold = _coerce_doc_id_list(getter("gold_doc_ids"))
    if len(question) < 6:
        return False, f"问题过短（{len(question)} 字），无法作为评测题"
    if kind == "unanswerable":
        if gold:
            return False, f"unanswerable 题的 gold_doc_ids 必须为空，实际为 {gold}"
    else:
        if not gold:
            return False, f"{kind} 题必须给出至少一个 gold_doc_ids"
    missing = [doc_id for doc_id in gold if doc_id not in valid_doc_ids]
    if missing:
        return False, "gold_doc_ids 不在语料中：" + ", ".join(missing)
    return True, "题目可仅凭语料回答" if kind != "unanswerable" else "题目应无法由本语料回答"


def dedupe_candidates(
    items: Sequence[Candidate],
    existing_questions: Iterable[str] = (),
    limit: int = 0,
) -> List[Candidate]:
    """按归一化问题文本去重，并排除既有评测集里的问题；同时收紧同一文档的重复出题。"""
    seen: Set[str] = set()
    for question in existing_questions:
        key = normalize_question(question)
        if key:
            seen.add(key)
    kept: List[Candidate] = []
    for item in items:
        key = normalize_question(item.question)
        if not key or key in seen:
            continue
        seen.add(key)
        kept.append(item)
        if limit and len(kept) >= limit:
            break
    return kept


def _balanced_pick(
    items: Sequence[Candidate],
    per_doc: int,
    max_total: int,
    existing_questions: Iterable[str],
) -> List[Candidate]:
    """按题型轮转挑选，再按文档限流、最后去重。

    为什么必须轮转而不是「排好序取前 N」：启发式一次能产出几十条 factual，
    如果按分数排序截断，comparison / multihop / unanswerable 会被整批挤掉，
    候选文件就只剩一种题型——评测集的题型分布也就无从谈起。
    """
    buckets: Dict[str, List[Candidate]] = {kind: [] for kind in CANDIDATE_TYPES}
    for item in items:
        buckets.setdefault(item.type, []).append(item)

    counted: Dict[str, int] = {}
    ordered: List[Candidate] = []
    index = 0
    while True:
        added = False
        for kind in CANDIDATE_TYPES:
            bucket = buckets.get(kind) or []
            if index >= len(bucket):
                continue
            item = bucket[index]
            added = True
            key = item.gold_doc_ids[0] if item.gold_doc_ids else ""
            counted[key] = counted.get(key, 0) + 1
            if key and counted[key] > max(1, per_doc):
                continue  # 同一篇文档出题过多，跳过这一条
            ordered.append(item)
            if max_total and len(ordered) >= max_total:
                return ordered
        if not added:
            break
        index += 1
    # 编号交给调用方：先按题型稳定排序，再去重
    ordered.sort(key=lambda c: (CANDIDATE_TYPES.index(c.type), c.question))
    return dedupe_candidates(ordered, existing_questions=existing_questions, limit=max_total)


def _finalize(
    raw: Sequence[Candidate],
    docs: Sequence[Document],
    existing_questions: Iterable[str],
    types: Sequence[str],
    per_doc: int,
    max_total: int,
) -> List[Candidate]:
    """公共收尾：类型过滤 -> 校验 -> 按题型轮转挑选 -> 去重（含既有评测集）-> 编号。"""
    valid_ids = {doc.doc_id for doc in docs}
    allowed = {t.lower() for t in types}
    filtered: List[Candidate] = []
    for item in raw:
        if item.type not in allowed:
            continue
        ok, reason = validate_candidate(item, valid_ids)
        if not ok:
            _stderr(f"丢弃候选（{reason}）：{truncate(item.question, 40, '…')}")
            continue
        if not item.rationale:
            item.rationale = reason
        filtered.append(item)
    unique = _balanced_pick(
        filtered,
        per_doc=per_doc,
        max_total=max_total,
        existing_questions=existing_questions,
    )
    for index, item in enumerate(unique, start=1):
        item.id = f"c{index:03d}"
    return unique


# ---------------------------------------------------------------------------
# LLM JSON 解析
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _json_payload(text: str) -> str:
    """从可能带代码块/前后废话的回复里抠出 JSON 文本。"""
    if not text:
        return ""
    match = _FENCE_RE.search(text)
    if match and match.group(1).strip():
        return match.group(1).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text.strip()


def parse_llm_json(
    text: str,
    valid_doc_ids: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """把 LLM 回复解析成候选 dict 列表。

    为什么不用正则硬抠字段：模型偶尔会把某道题的字段写歪，用 ``json.loads`` 逐项校验
    可以做到「坏一条丢一条」，而不是整批作废（一次调用可能值几毛钱和几十秒）。
    """
    valid: Optional[Set[str]] = {str(x) for x in valid_doc_ids} if valid_doc_ids is not None else None
    payload = _json_payload(text)
    if not payload:
        raise ValueError("LLM 回复为空，无法解析出 JSON")
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM 回复不是合法 JSON：{exc}；原文片段：{truncate(text, 120, '…')}") from exc

    if isinstance(data, dict):
        # 容忍 {"questions": [...]} / {"items": [...]} 这类结构
        for key in ("questions", "items", "candidates", "data", "results"):
            value = data.get(key)
            if isinstance(value, list):
                data = value
                break
        else:
            data = [data]
    if not isinstance(data, list):
        raise ValueError(f"LLM 回复的 JSON 顶层应为数组，实际是 {type(data).__name__}")

    items: List[Dict[str, Any]] = []
    for record in data:
        if not isinstance(record, dict):
            continue
        question = str(record.get("question") or record.get("q") or "").strip()
        if not question:
            continue
        kind = str(record.get("type") or "factual").strip().lower()
        if kind not in CANDIDATE_TYPES:
            kind = "factual"
        gold = _coerce_doc_id_list(record.get("gold_doc_ids"))
        if kind == "unanswerable":
            gold = []
        if valid is not None:
            gold = [doc_id for doc_id in gold if doc_id in valid]
        difficulty = str(record.get("difficulty") or "medium").strip().lower()
        if difficulty not in DIFFICULTIES:
            difficulty = "medium"
        items.append(
            {
                "question": question,
                "gold_doc_ids": gold,
                "gold_answer": str(
                    record.get("gold_answer") or record.get("answer") or ""
                ).strip(),
                "difficulty": difficulty,
                "type": kind,
            }
        )
    return items


def build_llm_prompt(
    docs: Sequence[Document],
    per_doc: int = 3,
    types: Sequence[str] = CANDIDATE_TYPES,
    max_chars: int = 1200,
) -> Tuple[str, str]:
    """构造 ``(system, user)`` 提示词；单独成函数是为了能离线断言提示词内容。"""
    materials = render_materials(docs, max_chars=max_chars)
    if not materials:
        raise ValueError("没有可用于出题的资料片段（语料为空或全部被截断成空串）")
    first_id = docs[0].doc_id if docs else "doc-id"
    user = _LLM_USER_PROMPT.format(
        per_doc=max(1, int(per_doc)),
        types=", ".join(types),
        materials=materials,
        first_id=first_id,
    )
    return _LLM_SYSTEM_PROMPT, user


def build_llm_candidates(
    docs: Sequence[Document],
    llm: Any,
    per_doc: int = 3,
    types: Sequence[str] = CANDIDATE_TYPES,
    max_docs: int = 40,
    max_chars: int = 1200,
    seed: int = 13,
    sample_chars: int = 300,
) -> List[Candidate]:
    """LLM 路线：逐篇出题，解析失败重试一次，再失败就跳过该篇（并给出中文提示）。"""
    valid_ids = {doc.doc_id for doc in docs}
    picked = select_llm_docs(docs, max_docs=max_docs, seed=seed)
    candidates: List[Candidate] = []
    for position, doc in enumerate(picked, start=1):
        system, user = build_llm_prompt([doc], per_doc=per_doc, types=types, max_chars=max_chars)
        first = normalize_ws(doc_display_text(doc, sample_chars))
        records: List[Dict[str, Any]] = []
        last_error = ""
        for attempt in range(2):  # 第一次失败再试一次：模型偶尔会在长资料上「忘记」输出 JSON
            prompt_text = user if attempt == 0 else f"{user}\n\n注意：上一次回复不是合法 JSON，请只输出 JSON 数组。"
            try:
                reply = llm.complete(system, prompt_text, temperature=0.4, max_tokens=1600)
            except Exception as exc:  # 网络/限流等问题：记录后跳过，不让整批任务崩掉
                last_error = f"调用失败：{exc}"
                _stderr(f"第 {position}/{len(picked)} 篇（{doc.doc_id}）LLM {last_error}")
                continue
            try:
                records = parse_llm_json(reply, valid_doc_ids=valid_ids)
                if records:
                    break
                last_error = "解析成功但没有任何有效题目"
            except ValueError as exc:
                last_error = str(exc)
                _stderr(f"第 {position}/{len(picked)} 篇（{doc.doc_id}）解析失败（第 {attempt + 1} 次）：{exc}")
        if not records:
            _stderr(f"跳过文档 {doc.doc_id}：{last_error or '未产出候选'}")
            continue
        for record in records:
            candidates.append(
                Candidate(
                    question=record["question"],
                    gold_doc_ids=list(record["gold_doc_ids"]) or [doc.doc_id],
                    gold_answer=record["gold_answer"],
                    difficulty=record["difficulty"],
                    type=record["type"],
                    source=doc.doc_id,
                    provider="llm",
                    rationale=(
                        f"由 {doc.doc_id} 的片段生成，原文摘录：{truncate(first, 60, '…')}"
                    ),
                )
            )
    return candidates


def doc_display_text(doc: Document, sample_chars: int = 300) -> str:
    """给 rationale 用的原文摘录（去掉换行，避免破坏 Markdown 表格）。"""
    body = re.sub(r"^#{1,6}\s+", "", doc.text.strip(), flags=re.MULTILINE)
    return re.sub(r"\s+", " ", body)[: max(20, int(sample_chars))]


# ---------------------------------------------------------------------------
# 启发式路线（完全离线）
# ---------------------------------------------------------------------------


_HAS_MEANING = re.compile(r"[a-z\u4e00-\u9fff]", re.IGNORECASE)

# 这些词在中文里通用到没有区分度：拿它们去问「两篇的说法有什么差异」，
# 得到的只会是一道无法评分的题。宁可少出几道，也不要把复核表塞满噪声。
_LOW_INFO_TERMS = {
    "例子", "示例", "举例", "方法", "技术", "方式", "问题", "使用", "包括", "存在",
    "具有", "进行", "提供", "采用", "獲得", "获得", "计算", "得到", "產生", "产生",
    "一個", "一个", "這個", "这个", "可以", "需要", "所有", "對於", "对于", "由於",
    "由于", "然而", "因此", "例如", "其他", "不同", "相同", "一般", "主要", "以上",
    "以下", "名字", "名称", "名稱", "定義", "定义",
}


def _latin_only(term: str) -> bool:
    return bool(term) and all(ord(ch) < 128 for ch in term)


def _cjk_only(term: str) -> bool:
    return bool(term) and all(0x3400 <= ord(ch) <= 0x9FFF for ch in term)


def _is_cjk_char(ch: str) -> bool:
    return 0x3400 <= ord(ch) <= 0x9FFF


def is_good_term(term: str) -> bool:
    """术语级的 token 才算「有信息量」：拉丁词至少 3 字符，中文避免通用词。"""
    if len(term) < 2:
        return False
    if term in _QUESTION_SKELETON or term in _LOW_INFO_TERMS:
        return False
    if _latin_only(term):
        return len(term) >= 3
    if _cjk_only(term):
        return len(term) >= 2
    return False


def _subsumed_terms(terms: Iterable[str]) -> Set[str]:
    """找出「只是同一篇里更长词条碎片」的中文 token。

    判据：一个 CJK 片段若出现在更长的词条内部，就是碎片。
    为什么保留这个检查（尽管当前 tokenizer 下它基本是空操作）：
    ``rag.text.tokenize`` 对 CJK run 只产出「单字 + 重叠二元组」，所以中文里几乎
    不存在能被它命中的容器；但分词实现一旦改为真正的词典分词，这里就会立刻生效，
    避免「召回率」被切成「回率」这类碎片继续流向出题环节。
    拉丁 token（``bge`` vs ``bge-small``）刻意不参与该判据，避免误删真实标识符。
    """
    terms = sorted({term for term in terms if term}, key=len)
    subsumed: Set[str] = set()
    for index, short in enumerate(terms):
        if not _cjk_only(short):
            continue
        for longer in terms[index + 1 :]:
            if len(longer) > len(short) and short in longer:
                subsumed.add(short)
                break
    return subsumed


def _content_terms(text: str, max_len: int = 4) -> Dict[str, int]:
    """从正文里抽「像术语」的高频 token：长度 2~max_len、含字母或汉字、且不是碎片。

    实现上先不做长度上限地收集候选，再剔除碎片，最后才按 ``max_len`` 截断——
    顺序不能反：碎片判据（``_subsumed_terms``）需要**更长的词条**当证据，
    如果先把长词条丢掉，就再也证明不了「倒排」只是「倒排索引」的头部碎片。
    """
    counts: Dict[str, int] = {}
    for token in tokenize(text):
        if len(token) < 2:
            continue
        if token.isdigit() or not _HAS_MEANING.search(token):
            continue
        if _latin_only(token) and len(token) < 3:
            continue
        if token in _QUESTION_SKELETON:
            continue
        counts[token] = counts.get(token, 0) + 1
    for fragment in _subsumed_terms(counts):
        counts.pop(fragment, None)
    return {term: count for term, count in counts.items() if 2 <= len(term) <= int(max_len)}


def _term_doc_freq(docs: Sequence[Document]) -> Dict[str, Dict[str, int]]:
    """token -> {doc_id: 该 token 在这篇里出现的次数}（只留低频、可读的实词）。"""
    total = max(1, len(docs))
    limit = max(1, int(total * 0.8))
    df: Dict[str, Set[str]] = {}
    table: Dict[str, Dict[str, int]] = {}
    for doc in docs:
        for token, count in _content_terms(doc.text).items():
            df.setdefault(token, set()).add(doc.doc_id)
            table.setdefault(token, {})[doc.doc_id] = count
    return {token: table[token] for token, docs_ in df.items() if len(docs_) <= limit}


def _heading_terms(doc: Document) -> List[str]:
    """文档的小节标题，去掉 Markdown 记号与层级前缀（``A > B`` 取最后一段）。"""
    titles: List[str] = []
    for section in _sections(doc.text):
        title = section.heading.split(">")[-1].strip()
        title = re.sub(r"^#{1,6}\s*", "", title).strip()
        if 2 <= len(title) <= 24:
            titles.append(title)
    return titles


def _generic_heading_terms(docs: Sequence[Document]) -> Set[str]:
    """跨文档重复出现的标题词（如维基条目的「定義 / 概述 / 示例」）。

    为什么必须识别：这类标题在语料里到处都是，拿它出题会得到「资料里关于「定義」
    是怎么说的？」这种既无区分度、又几乎无法评分的题——它会把人工复核的时间全耗掉。
    """
    appearances: Dict[str, Set[str]] = {}
    for doc in docs:
        for heading in _heading_terms(doc):
            for term in tokenize(heading):
                if len(term) >= 2:
                    appearances.setdefault(term, set()).add(doc.doc_id)
    return {term for term, owners in appearances.items() if len(owners) >= 2}


def _heading_scores(
    docs: Sequence[Document],
    bm25: BM25Index,
    generic: Set[str],
) -> Dict[str, List[Tuple[float, str]]]:
    """按「标题里实词的平均 IDF」给每篇文档的小节标题打分，用于挑更具体的标题出题。

    标题只是「话题的入口」，越具体的标题出出来的题越接近真实提问；
    全由通用词构成且很短的标题（如「定義」）则直接不参与出题。
    """
    scores: Dict[str, List[Tuple[float, str]]] = {}
    for doc in docs:
        rated: List[Tuple[float, str]] = []
        for title in _heading_terms(doc):
            terms = [t for t in tokenize(title) if len(t) >= 2 and t not in generic]
            if not terms:
                continue  # 纯通用标题：出了也没法评分
            if len(title) <= 4 and len(terms) < 2 and title in generic:
                continue
            average = sum(bm25.idf(term) for term in terms) / float(len(terms))
            rated.append((average, title))
        rated.sort(key=lambda item: (-item[0], item[1]))
        scores[doc.doc_id] = rated
    return scores


def _first_sentence(text: str, limit: int = 120) -> str:
    """取第一句像样的句子作为 gold_answer 的种子（人工复核时再改）。"""
    for sentence in split_sentences(text):
        if len(sentence) >= 8:
            return truncate(sentence, limit, "…")
    return truncate(text, limit, "…")


def _factual_candidates(
    docs: Sequence[Document],
    bm25: BM25Index,
    per_doc: int,
    rng: random.Random,
    heading_scores: Optional[Dict[str, List[Tuple[float, str]]]] = None,
) -> List[Candidate]:
    """factual：用「具体的小节标题」或「高 IDF 实词」问 X 是怎么说的，gold = 该文档。"""
    out: List[Candidate] = []
    scores = heading_scores or {}
    for doc in docs:
        topics: List[Tuple[str, str]] = []
        for _, title in scores.get(doc.doc_id, []):
            topics.append((title, f"小节标题「{title}」"))
        ranked = sorted(_content_terms(doc.text), key=lambda token: (-bm25.idf(token), token))
        for token in ranked[:4]:
            topics.append((token, f"高 IDF 术语「{token}」(idf={bm25.idf(token):.2f})"))
        seen: Set[str] = set()
        picked: List[Tuple[str, str]] = []
        for topic, why in topics:
            key = normalize_question(topic)
            if key in seen:
                continue
            seen.add(key)
            picked.append((topic, why))
        for topic, why in picked[: max(1, per_doc)]:
            question = _FACTUAL_TEMPLATES[rng.randrange(len(_FACTUAL_TEMPLATES))].format(topic=topic)
            out.append(
                Candidate(
                    question=question,
                    gold_doc_ids=[doc.doc_id],
                    gold_answer=_first_sentence(doc.text),
                    difficulty="easy" if len(topic) <= 6 else "medium",
                    type="factual",
                    source=doc.doc_id,
                    provider="heuristic",
                    rationale=f"答案在 {doc.doc_id} 的{why}所在小节里，可直接摘录。",
                )
            )
    return out


def _comparison_candidates(
    docs: Sequence[Document],
    bm25: BM25Index,
    term_df: Dict[str, Dict[str, int]],
    words: Optional[Set[str]] = None,
) -> List[Candidate]:
    """comparison：两篇文档共享某个高 IDF 术语时问「说法差异」，gold = 两篇。

    排序把「两篇各自的出现次数」放在 IDF 之前：共享一个稀但只是顺带一提的术语
    （例如两篇都出现过一次「美元」）问出来的差异是伪问题，双边的词频才是
    「两篇都真的在讲这件事」的证据。
    ``words`` 非空时只采纳「以独立词形态出现在语料里」的术语，用来挡掉重叠二元组
    造出来的假词（「意力」「块策」）。
    """
    shared: List[Tuple[str, str, str, int]] = []
    for term, by_doc in term_df.items():
        if not is_good_term(term) or bm25.idf(term) <= 0:
            continue
        if words is not None and not is_standalone_term(term, words):
            continue
        owners = sorted(by_doc)
        if len(owners) < 2:
            continue
        for i in range(len(owners)):
            for j in range(i + 1, len(owners)):
                left, right = owners[i], owners[j]
                weight = min(by_doc[left], by_doc[right])
                if weight < 1:
                    continue
                shared.append((term, left, right, weight))
    shared.sort(
        key=lambda item: (
            -item[3],
            -bm25.idf(item[0]),
            -max(term_df[item[0]].values()),
            item[1],
            item[2],
        )
    )
    out: List[Candidate] = []
    used_pairs: Set[Tuple[str, str]] = set()
    used_terms: Set[str] = set()
    for term, left, right, _weight in shared:
        pair = (left, right)
        if pair in used_pairs or term in used_terms:
            continue  # 同一对文档 / 同一个术语只出一题，避免整张表都是同一组对比
        used_pairs.add(pair)
        used_terms.add(term)
        counts = term_df.get(term, {})
        out.append(
            Candidate(
                question=_COMPARISON_TEMPLATE.format(a=left, b=right, term=term),
                gold_doc_ids=[left, right],
                gold_answer=f"需要同时读 {left} 与 {right} 中关于「{term}」的段落才能比较。",
                difficulty="hard" if bm25.idf(term) >= 2.0 else "medium",
                type="comparison",
                source=left,
                provider="heuristic",
                rationale=(
                    f"「{term}」在 {left} 出现 {counts.get(left, 0)} 次、"
                    f"{right} 出现 {counts.get(right, 0)} 次（idf={bm25.idf(term):.2f}），"
                    f"差异必须对照两篇才能说清。"
                ),
            )
        )
        if len(out) >= 6:
            break
    return out


def _multihop_candidates(
    docs: Sequence[Document],
    bm25: BM25Index,
    term_df: Dict[str, Dict[str, int]],
    words: Optional[Set[str]] = None,
) -> List[Candidate]:
    """multihop：A 提到术语 t、B 解释 t、C 承接 → 构造跨文档综合题，gold >= 2 篇。

    「B 解释 t」的判据是 t 在 B 里出现多次（高词频），「A 只提到 t」则是低词频：
    这个不对称正是「单看一篇答不全」的机械证据，也是人工复核时最容易验证的一条。
    ``words`` 用来挡掉重叠二元组造出来的假术语——多跳题会把术语写进题干，
    假术语（「块策」「意力」）比在比较题里更刺眼。
    """
    if len(docs) < 3:
        return []
    doc_ids = [doc.doc_id for doc in docs]
    captions = {doc.doc_id: (doc.title or doc.doc_id) for doc in docs}
    bridge_terms: List[Tuple[str, str, str]] = []
    for term, by_doc in term_df.items():
        if not is_good_term(term) or len(by_doc) < 2 or bm25.idf(term) <= 0:
            continue
        if words is not None and not is_standalone_term(term, words):
            continue
        bridge = max(by_doc, key=lambda doc_id: (by_doc[doc_id], doc_id))
        mention = min(by_doc, key=lambda doc_id: (by_doc[doc_id], doc_id))
        if bridge == mention or by_doc[bridge] < 2:
            continue
        third = next((doc_id for doc_id in doc_ids if doc_id not in (bridge, mention)), "")
        if not third:
            continue
        bridge_terms.append((term, mention, bridge))
    bridge_terms.sort(key=lambda item: (-bm25.idf(item[0]), item[1], item[2], item[0]))
    out: List[Candidate] = []
    used: Set[str] = set()
    for term, mention, bridge in bridge_terms:
        if term in used:
            continue
        used.add(term)
        third = next((doc_id for doc_id in doc_ids if doc_id not in (bridge, mention)), "")
        by_doc = term_df.get(term, {})
        out.append(
            Candidate(
                question=_MULTIHOP_TEMPLATE.format(
                    doc_a=captions[mention],
                    doc_b=captions[bridge],
                    doc_c=captions[third],
                    term=term,
                ),
                gold_doc_ids=sorted({mention, bridge, third}),
                gold_answer=(
                    f"需要把 {mention}（提到「{term}」）、{bridge}（解释「{term}」）"
                    f"与 {third} 三处内容串起来才能完整回答。"
                ),
                difficulty="hard",
                type="multihop",
                source=mention,
                provider="heuristic",
                rationale=(
                    f"「{term}」的线索分散在 {mention}（{by_doc.get(mention, 0)} 次）与 "
                    f"{bridge}（{by_doc.get(bridge, 0)} 次），单看任意一篇都答不全。"
                ),
            )
        )
        if len(out) >= 6:
            break
    return out


def _unanswerable_candidates(
    docs: Sequence[Document],
    bm25: BM25Index,
    seed: int,
    coverage_threshold: Optional[float] = None,
    presence_threshold: Optional[float] = None,
    top_k: int = RARE_TOP_K,
    min_count: int = PRESENCE_MIN_COUNT,
    topics: Optional[Sequence[str]] = None,
    target: int = UNANSWERABLE_TARGET,
    corpus_text: Optional[str] = None,
) -> List[Candidate]:
    """unanswerable：从领域外主题池里挑出**经判据确认语料答不了**的题目。

    为什么不是「生成完再过滤」：主题池是手工维护的，语料会变；
    唯一可靠的锚点是用当前语料实测判据（``rare_term_report``），
    所以这里对每个主题先算判据、只保留确认领域外的，再按「实词命中率」由低到高
    优先采纳（命中率越低，题目越不可能被语料答出来）。
    """
    rng = random.Random(seed + 17)
    pool = list(topics if topics is not None else OUT_OF_DOMAIN_TOPICS)
    rng.shuffle(pool)
    text = corpus_text if corpus_text is not None else corpus_text_of(docs)
    scored: List[Tuple[float, str, str, Dict[str, Any]]] = []
    for topic in pool:
        template = _UNANSWERABLE_TEMPLATES[rng.randrange(len(_UNANSWERABLE_TEMPLATES))]
        question = template.format(topic=topic)
        report = rare_term_report(
            question,
            bm25,
            top_k=top_k,
            min_count=min_count,
            threshold=presence_threshold,
            topic=topic,
            corpus_text=text,
            coverage_threshold=coverage_threshold,
        )
        if not report["unanswerable"]:
            # 被丢弃时必须打印原因与关键统计量：否则无法判断阈值是否定得不合理
            reason = "关键术语在语料里有支撑" if report.get("relevant") else "覆盖率超过上限"
            _stderr(f"丢弃领域外候选：{topic}｜{format_rare_terms(report)}｜原因={reason}")
            continue
        scored.append((report["present_share"], topic, question, report))

    scored.sort(key=lambda item: (item[0], item[3]["coverage"], item[1]))
    out: List[Candidate] = []
    for _share, topic, question, report in scored[: max(1, int(target))]:
        out.append(
            Candidate(
                question=question,
                gold_doc_ids=[],
                gold_answer=(
                    f"语料只覆盖 RAG 工程与信息检索/机器学习的基础概念，"
                    f"未涉及「{topic}」，正确行为是拒答。"
                ),
                difficulty="easy" if len(topic) <= 14 else "medium",
                type="unanswerable",
                source="heuristic",
                provider="heuristic",
                rationale=(
                    f"主题「{topic}」属领域外：关键实词 "
                    + "、".join(
                        f"{item['term']}(出现{item['count']}次/df={item['df']})"
                        for item in report["terms"][:4]
                    )
                    + f" 中没有几个在语料里被真正讲到"
                    f"（实词命中率 {report['present_share']:.2f} < {report['threshold']:.2f}，"
                    f"覆盖率 {report['coverage']:.2f} < {report['coverage_threshold']:.2f}）。"
                ),
            )
        )
    if not out:
        _stderr(
            "没有产出任何 unanswerable 候选：领域外主题池与当前语料的关键词重合过多，"
            "请补充更「远离语料」的主题，或放宽 --unanswerable-coverage 上限"
        )
    return out


def heuristic_candidates(
    docs: Sequence[Document],
    types: Sequence[str] = CANDIDATE_TYPES,
    per_doc: int = 3,
    seed: int = 13,
    coverage_threshold: Optional[float] = None,
    presence_threshold: Optional[float] = None,
    rare_top_k: int = RARE_TOP_K,
    presence_min_count: int = PRESENCE_MIN_COUNT,
    topics: Optional[Sequence[str]] = None,
) -> List[Candidate]:
    """启发式路线总入口（离线）：四条规则各产出若干候选，未编号、未去重之外的收尾交给调用方。

    ``coverage_threshold=None`` 表示按语料规模自适应（见 ``adaptive_presence_threshold``）。
    """
    if not docs:
        return []
    rng = random.Random(seed)
    bm25 = make_bm25(docs)
    term_df = _term_doc_freq(docs)
    generic = _generic_heading_terms(docs)
    # comparison / multihop 会把术语写进题干，只采纳「被当作单元写出来」的词
    corpus_text = corpus_text_of(docs)
    words = structured_terms(docs, corpus_text) | corpus_words(corpus_text)
    wanted = {t.lower() for t in types}
    out: List[Candidate] = []
    if "factual" in wanted:
        out.extend(
            _factual_candidates(
                docs, bm25, per_doc, rng, heading_scores=_heading_scores(docs, bm25, generic)
            )
        )
    if "comparison" in wanted:
        out.extend(_comparison_candidates(docs, bm25, term_df, words=words))
    if "multihop" in wanted:
        out.extend(_multihop_candidates(docs, bm25, term_df, words=words))
    if "unanswerable" in wanted:
        out.extend(
            _unanswerable_candidates(
                docs,
                bm25,
                seed,
                coverage_threshold=coverage_threshold,
                presence_threshold=presence_threshold,
                top_k=rare_top_k,
                min_count=presence_min_count,
                topics=topics,
            )
        )
    return out


# ---------------------------------------------------------------------------
# 复核表（Markdown）
# ---------------------------------------------------------------------------


def _cell(text: str, limit: int = 90) -> str:
    """Markdown 表格单元格：管道符与换行必须消毒，否则整张表会错位。"""
    value = truncate(re.sub(r"\s+", " ", str(text or "")).strip(), limit, "…")
    return value.replace("|", "/").replace("\n", " ")


def render_review_md(candidates: Sequence[Candidate], source_path: str = "") -> str:
    """渲染人工复核表；每行以 ``- [ ]`` 开头，方便在编辑器里直接勾选。

    表头里的来源路径**只写相对项目根的路径**：复核表会进版本库，写绝对路径会把本机
    用户名与目录结构一起提交上去。
    """
    shown = source_path or "data/eval/candidates.jsonl"
    try:
        relative = os.path.relpath(shown, ROOT)
        if not relative.startswith(".."):
            shown = relative.replace(os.sep, "/")
    except ValueError:  # 不同盘符时 relpath 会抛错，直接用原值
        pass
    lines: List[str] = [
        "# 评测集候选复核表",
        "",
        f"- 候选来源：`{shown}`（本表由 generate 子命令生成）",
        f"- 候选数量：{len(candidates)}",
        "",
        "## 筛选规则（勾选前请逐条确认）",
        "",
        "1. **不可回答的题**：必须确认语料里确实没有对应内容；只要某篇文档能答，",
        "   就必须改成可答题并补上 `gold_doc_ids`，否则拒答指标会被算反。",
        "   机械判据只能提供线索：备注里的「实词命中率」越高、越说明语料其实讲过这个主题。",
        "2. **多跳题**：必须真的需要两篇以上文档；如果单看一篇就能答，降级成 factual。",
        "3. **gold_doc_ids**：必须与语料里真实存在的文档 id 完全一致（`promote` 会拒绝不存在的 id）。",
        "4. **答案可核对**：`gold_answer` 必须能在语料里找到依据；有疑问的题直接不勾。",
        "5. **重复**：与 `data/eval/qa.jsonl` 已有问题重复的题不要勾（`promote` 会自动跳过）。",
        "6. **术语类题目**（comparison / multihop 的「X」）：中文分词不支持词典切分，",
        "   偶尔会把长词切成「意力」（来自「注意力」）这类接缝碎片；脚本已经用",
        "   「跨词边界检测 + 标题内碎片判据」过滤掉大部分，但**勾选前请人工把术语改成",
        "   语料里真实存在的说法**，改不动就直接不勾。",
        "",
        "## 勾选方式",
        "",
        "把要采纳的行首 `- [ ]` 改成 `- [x]`，然后执行：",
        "",
        "```powershell",
        "python scripts/gen_eval_candidates.py promote --only-checked --dry-run   # 先预览",
        "python scripts/gen_eval_candidates.py promote --only-checked             # 真正写回",
        "```",
        "",
        f"| 勾选 | id | type | difficulty | question | gold_doc_ids | 备注 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in candidates:
        gold = ", ".join(item.gold_doc_ids) if item.gold_doc_ids else "（空）"
        lines.append(
            "- [ ] | {id} | {type} | {difficulty} | {question} | {gold} | {note} |".format(
                id=item.id,
                type=item.type,
                difficulty=item.difficulty,
                question=_cell(item.question, 80),
                gold=gold,
                note=_cell(item.rationale, 70),
            )
        )
    lines.append("")
    return "\n".join(lines)


_REVIEW_ROW = re.compile(r"^\s*-\s*\[( |x|X)\]\s*\|")
_REVIEW_CHECKED = re.compile(r"^\s*-\s*\[(x|X)\]\s*\|")


def parse_review_file(path: str, only_checked: bool = False) -> List[str]:
    """解析复核表，返回候选 id 列表（保持文件顺序）。

    ``only_checked=True`` 只收 ``- [x]`` 的行；``False`` 时把所有数据行都收进来——
    后者是给 ``stats`` 之类的诊断场景用的（想看看「有多少候选还没勾」），
    ``promote`` 永远走 ``only_checked=True``，绝不能让未复核的题混进评测集。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"复核表不存在：{path}（先生成候选并人工勾选）")
    ids: List[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not _REVIEW_ROW.match(line):
                continue
            if only_checked and not _REVIEW_CHECKED.match(line):
                continue
            cells = [cell.strip() for cell in line.strip().rstrip("|").split("|")]
            # 结构固定为：`- [x]` | id | type | difficulty | question | gold | 备注
            if len(cells) < 2:
                continue
            candidate_id = cells[1]
            if candidate_id and candidate_id not in ids:
                ids.append(candidate_id)
    return ids


def _marked_ids(path: str) -> List[str]:
    """复核表里所有被勾选的 id（用于「文件存在时不覆盖」的合并判断）。"""
    if not os.path.isfile(path):
        return []
    return parse_review_file(path, only_checked=True)


# ---------------------------------------------------------------------------
# 写回评测集
# ---------------------------------------------------------------------------


_QA_ID_RE = re.compile(r"^q(\d+)$", re.IGNORECASE)


def next_qa_id(existing_ids: Iterable[str]) -> str:
    """从既有编号继续（``q01..q12`` -> ``q13``），并保持两位数字格式。"""
    max_n = 0
    width = 2
    for raw in existing_ids:
        match = _QA_ID_RE.match(str(raw).strip())
        if not match:
            continue
        digits = match.group(1)
        max_n = max(max_n, int(digits))
        width = max(width, len(digits))
    return f"q{max_n + 1:0{width}d}"


def promote_candidates(
    candidates: Sequence[Dict[str, Any]],
    marked_ids: Sequence[str],
    existing: Sequence[Dict[str, Any]],
    valid_doc_ids: Optional[Set[str]] = None,
    only_checked: bool = True,
) -> PromoteResult:
    """把勾选的候选整理成可直接追加进评测集的行（去重 + gold 校验 + 编号）。"""
    result = PromoteResult()
    by_id = {str(item.get("id") or ""): item for item in candidates}
    seen = {normalize_question(str(item.get("question") or "")) for item in existing}
    next_id = next_qa_id([str(item.get("id") or "") for item in existing])
    next_n = int(next_id[1:])
    width = len(next_id) - 1
    valid = set(valid_doc_ids) if valid_doc_ids is not None else None

    for candidate_id in marked_ids:
        item = by_id.get(str(candidate_id))
        if item is None:
            result.rejected.append((str(candidate_id), "复核表里的 id 在 candidates.jsonl 中找不到"))
            continue
        question = str(item.get("question") or "").strip()
        if not question:
            result.rejected.append((str(candidate_id), "候选问题为空"))
            continue
        gold = _coerce_doc_id_list(item.get("gold_doc_ids"))
        if valid is not None:
            missing = [doc_id for doc_id in gold if doc_id not in valid]
            if missing:
                result.rejected.append(
                    (str(candidate_id), "gold_doc_ids 在语料中不存在：" + ", ".join(missing))
                )
                continue
        ok, reason = validate_candidate(item, valid if valid is not None else set(gold))
        if not ok:
            result.rejected.append((str(candidate_id), reason))
            continue
        key = normalize_question(question)
        if key in seen:
            result.skipped.append((str(candidate_id), f"与既有/已选问题重复：{truncate(question, 30, '…')}"))
            continue
        seen.add(key)
        result.accepted.append(
            {
                "id": f"q{next_n:0{width}d}",
                "question": question,
                "gold_doc_ids": list(gold),
                "gold_answer": str(item.get("gold_answer") or "").strip(),
                "difficulty": str(item.get("difficulty") or "medium").strip() or "medium",
                "type": str(item.get("type") or "factual").strip() or "factual",
            }
        )
        next_n += 1
    return result


def type_distribution(items: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    dist: Dict[str, int] = {}
    for item in items:
        kind = str(item.get("type") or "(未标注)")
        dist[kind] = dist.get(kind, 0) + 1
    return dict(sorted(dist.items(), key=lambda kv: (-kv[1], kv[0])))


# ---------------------------------------------------------------------------
# 子命令
# ---------------------------------------------------------------------------


def parse_types(raw: str) -> List[str]:
    """``--types factual,multihop`` -> 规范化列表；顺序始终按 CANDIDATE_TYPES。"""
    if raw is None:
        return list(CANDIDATE_TYPES)
    wanted = {part.strip().lower() for part in str(raw).split(",") if part.strip()}
    unknown = sorted(wanted - set(CANDIDATE_TYPES))
    if unknown:
        raise ValueError(f"--types 中含有未知类型 {unknown}，可选值：{', '.join(CANDIDATE_TYPES)}")
    picked = [t for t in CANDIDATE_TYPES if t in wanted]
    return picked or list(CANDIDATE_TYPES)


def cmd_generate(args: argparse.Namespace) -> int:
    types = parse_types(args.types)
    cfg = load_config()
    corpus = args.corpus or cfg.path_of("corpus")
    out = args.out or os.path.join(cfg.path_of("eval").rsplit(os.sep, 1)[0], "candidates.jsonl")
    review = args.review or os.path.join(os.path.dirname(out), "candidates_review.md")

    docs = load_documents([corpus])
    if not docs:
        _stderr(f"没有从 {corpus} 读到任何文档，请检查 --corpus 是否指向真实语料目录")
        return 2
    existing = load_qa_items(cfg.path_of("eval"))
    existing_questions = [str(item.get("question") or "") for item in existing]
    print(f"语料 {corpus}：{len(docs)} 篇文档；既有评测集 {len(existing)} 题")

    llm: Optional[Any] = None
    provider = str(args.provider or "auto").lower()
    if provider in ("llm", "auto"):
        llm = _build_llm_for_generate(args, provider, cfg)
    chosen = provider
    if provider == "auto":
        chosen = "llm" if llm is not None else "heuristic"

    picked = select_llm_docs(docs, max_docs=int(args.max_docs), seed=int(args.seed))
    # 启发式路线同样受 --max-docs 限制：语料一大，IDF 与两两配对的代价会迅速上升，
    # 而「先出一批候选给人勾」本身也不需要一次扫完整个语料
    heuristic_docs = sorted(docs, key=lambda d: d.doc_id)[: max(1, int(args.max_docs))]
    gate_kwargs = {
        "coverage_threshold": args.unanswerable_coverage,
        "presence_threshold": args.presence_threshold,
        "rare_top_k": int(args.rare_top_k),
    }
    if args.coverage_adaptive and args.unanswerable_coverage is None:
        gate_kwargs["coverage_threshold"] = adaptive_coverage_threshold(len(heuristic_docs))
    coverage_show = (
        float(gate_kwargs["coverage_threshold"])
        if gate_kwargs.get("coverage_threshold")
        else DEFAULT_COVERAGE_LIMIT
    )
    presence_show = (
        adaptive_presence_threshold(len(heuristic_docs))
        if args.presence_threshold is None
        else float(args.presence_threshold)
    )
    if chosen == "llm" and llm is not None:
        print(f"路线：LLM（model={getattr(llm, 'name', args.model)}，逐篇出题，共 {len(picked)} 篇）")
        raw = build_llm_candidates(
            docs,
            llm,
            per_doc=int(args.per_doc),
            types=types,
            max_docs=int(args.max_docs),
            seed=int(args.seed),
        )
        if not raw:
            _stderr("LLM 路线没有产出有效候选，已自动退回启发式路线")
            chosen = "heuristic"
            raw = heuristic_candidates(
                heuristic_docs,
                types=types,
                per_doc=int(args.per_doc),
                seed=int(args.seed),
                **gate_kwargs,
            )
    else:
        if provider == "auto":
            _stderr(
                f"未检测到环境变量 {args.api_key_env}，已退回启发式路线"
                f"（要跑 LLM 路线请先设置 {args.api_key_env}）"
            )
        else:
            _stderr(f"--provider={provider}，按启发式路线生成（离线）")
        print(
            f"路线：启发式（离线，provider={chosen}，{len(heuristic_docs)} 篇文档）；"
            f"不可回答判据：实词在语料里出现 >= {PRESENCE_MIN_COUNT} 次才算「语料在讲」，"
            f"命中率须 < {presence_show:.2f}（top-{int(args.rare_top_k)} 术语全弱时直接判定）"
            + (
                f"，兜底覆盖率上限 {coverage_show:.2f}"
                if coverage_show > 0
                else "；兜底覆盖率已关闭（--coverage-adaptive 或 --unanswerable-coverage 可开启）"
            )
        )
        raw = heuristic_candidates(
            heuristic_docs,
            types=types,
            per_doc=int(args.per_doc),
            seed=int(args.seed),
            **gate_kwargs,
        )

    candidates = _finalize(
        raw,
        docs=docs,
        existing_questions=existing_questions,
        types=types,
        per_doc=int(args.per_doc),
        max_total=max(8, min(48, len(docs) * max(1, int(args.per_doc)))),
    )
    if not candidates:
        _stderr("没有生成任何候选：请检查语料内容是否足够（每个小节建议 >= 80 字）")
        return 3

    review_text = render_review_md(candidates, source_path=out)
    if args.dry_run:
        print(f"[dry-run] 将写入 {len(candidates)} 条候选到 {out}")
        print(f"[dry-run] 将写入复核表 {review}（{len(review_text.splitlines())} 行）")
        for item in candidates[:5]:
            print(f"  {item.id} [{item.type}] {item.question}")
        kept = len(_marked_ids(review))
        if kept:
            print(f"[dry-run] 复核表已存在，其中的 {kept} 行勾选会被保留（不会被覆盖）")
        return 0

    write_candidates(out, candidates, dry_run=False)
    merged = _merge_review_text(review, review_text)
    os.makedirs(os.path.dirname(os.path.abspath(review)), exist_ok=True)
    with open(review, "w", encoding="utf-8") as handle:
        handle.write(merged)
    print(f"已写入候选：{out}（{len(candidates)} 条）")
    print(f"已更新复核表：{review}")
    print("下一步：在复核表里把采纳的候选改成 `- [x]`，再执行")
    print("  python scripts/gen_eval_candidates.py promote --only-checked --dry-run")
    return 0


def _build_llm_for_generate(args: argparse.Namespace, provider: str, cfg: Any) -> Optional[Any]:
    """构造 LLM：优先用专用环境变量 + 参数覆盖，其次回落到配置里的 generation.*。"""
    api_key = os.environ.get(args.api_key_env, "").strip()
    if api_key:
        return OpenAICompatLLM(
            model=str(args.model),
            api_base=str(args.api_base),
            api_key=api_key,
            temperature=0.4,
            max_tokens=1600,
            timeout=120,
        )
    try:
        fallback = build_llm(cfg)
    except Exception as exc:  # 配置异常不应该把「生成候选」整件事弄崩
        _stderr(f"读取 generation.* 配置失败（{exc}），按无 LLM 处理")
        return None
    if fallback is not None and provider == "llm":
        _stderr(
            f"未检测到 {args.api_key_env}，改用配置里的 generation.api_key_env"
            f"（{cfg.get('generation.api_key_env')}）对应的 LLM"
        )
    return fallback


def _merge_review_text(path: str, fresh_text: str) -> str:
    """复核表已存在时不能直接覆盖——人勾过的内容丢了就得重勾一遍。

    做法：新生成的表照写，但把旧表里已勾选的行（``- [x]``）按 id 覆盖回去。
    """
    if not os.path.isfile(path):
        return fresh_text
    try:
        with open(path, "r", encoding="utf-8") as handle:
            old_lines = handle.readlines()
    except OSError as exc:
        _stderr(f"读取旧复核表 {path} 失败（{exc}），本次将覆盖它")
        return fresh_text
    checked: Dict[str, str] = {}
    for line in old_lines:
        if not _REVIEW_CHECKED.match(line):
            continue
        cells = [cell.strip() for cell in line.strip().rstrip("|").split("|")]
        if len(cells) >= 2 and cells[1]:
            checked[cells[1]] = line if line.endswith("\n") else line + "\n"
    if not checked:
        return fresh_text
    merged: List[str] = []
    for line in fresh_text.splitlines(keepends=True):
        if _REVIEW_ROW.match(line):
            cells = [cell.strip() for cell in line.strip().rstrip("|").split("|")]
            if len(cells) >= 2 and cells[1] in checked:
                merged.append(checked[cells[1]])
                continue
        merged.append(line)
    return "".join(merged)


def cmd_promote(args: argparse.Namespace) -> int:
    cfg = load_config()
    review = args.review or os.path.join(os.path.dirname(cfg.path_of("eval")), "candidates_review.md")
    qa = args.qa or cfg.path_of("eval")
    candidates_path = args.candidates or os.path.join(os.path.dirname(qa), "candidates.jsonl")
    corpus = args.corpus or cfg.path_of("corpus")

    marked = parse_review_file(review, only_checked=bool(args.only_checked))
    if not marked:
        _stderr(f"复核表 {review} 里没有被勾选（`- [x]`）的候选；请先勾选再执行 promote")
        return 1
    candidates = load_candidates(candidates_path)
    existing = load_qa_items(qa)
    valid_ids = corpus_doc_ids(corpus)
    if not valid_ids:
        _stderr(f"语料目录 {corpus} 里没有读到任何文档，无法校验 gold_doc_ids")
        return 2

    result = promote_candidates(
        candidates,
        marked,
        existing,
        valid_doc_ids=valid_ids,
        only_checked=bool(args.only_checked),
    )
    for candidate_id, reason in result.rejected:
        _stderr(f"拒绝 {candidate_id}：{reason}")
    for candidate_id, reason in result.skipped:
        _stderr(f"跳过 {candidate_id}：{reason}")

    if not result.accepted:
        print(f"没有可追加的题目（勾选 {len(marked)} 条：拒绝 {len(result.rejected)}，跳过 {len(result.skipped)}）")
        return 0

    if args.dry_run:
        print(f"[dry-run] 将向 {qa} 追加 {len(result.accepted)} 题：")
        for row in result.accepted:
            print("  " + json.dumps(row, ensure_ascii=False))
    else:
        append_jsonl(qa, result.accepted)
        print(f"已向 {qa} 追加 {len(result.accepted)} 题")

    merged = list(existing) + list(result.accepted)
    dist = type_distribution(merged)
    print(
        f"统计：评测集共 {len(merged)} 题"
        f"（新增 {len(result.accepted)}，拒绝 {len(result.rejected)}，跳过 {len(result.skipped)}）"
    )
    print("各 type 分布：" + "，".join(f"{key} {value}" for key, value in dist.items()))
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    cfg = load_config()
    qa = args.qa or cfg.path_of("eval")
    candidates_path = args.candidates or os.path.join(os.path.dirname(qa), "candidates.jsonl")

    existing = load_qa_items(qa)
    print(f"评测集 {qa}：{len(existing)} 题")
    for key, value in type_distribution(existing).items():
        print(f"  - {key}: {value}")
    ids = [str(item.get("id") or "") for item in existing]
    print(f"  下一个可用编号：{next_qa_id(ids)}")

    if os.path.isfile(candidates_path):
        candidates = load_candidates(candidates_path)
        print(f"候选 {candidates_path}：{len(candidates)} 条")
        for key, value in type_distribution(candidates).items():
            print(f"  - {key}: {value}")
        providers = {}
        for item in candidates:
            key = str(item.get("provider") or "(未知)")
            providers[key] = providers.get(key, 0) + 1
        print("  来源：" + "，".join(f"{k} {v}" for k, v in sorted(providers.items())))
    else:
        print(f"候选 {candidates_path}：不存在（先执行 generate 子命令）")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gen_eval_candidates.py",
        description="半自动生成评测集候选问题，并把人工筛选结果回填进 data/eval/qa.jsonl",
    )
    sub = parser.add_subparsers(dest="command", metavar="command")

    gen = sub.add_parser("generate", help="生成候选问题 + 人工复核表")
    gen.add_argument("--corpus", default=None, help="语料目录或文件（默认取配置 paths.corpus）")
    gen.add_argument("--out", default=None, help="候选输出文件（默认 data/eval/candidates.jsonl）")
    gen.add_argument("--review", default=None, help="复核表输出文件（默认 data/eval/candidates_review.md）")
    gen.add_argument("--per-doc", type=int, default=3, help="每篇文档最多出几题（默认 3）")
    gen.add_argument("--max-docs", type=int, default=40, help="最多处理多少篇文档（默认 40）")
    gen.add_argument(
        "--provider",
        default="auto",
        choices=("auto", "llm", "heuristic"),
        help="生成路线：auto 有 key 走 LLM，否则启发式",
    )
    gen.add_argument("--model", default="deepseek-chat", help="LLM 模型名（默认 deepseek-chat）")
    gen.add_argument(
        "--api-base",
        default="https://api.deepseek.com/v1",
        help="OpenAI 兼容的 api_base（默认 https://api.deepseek.com/v1）",
    )
    gen.add_argument("--api-key-env", default="DEEPSEEK_API_KEY", help="API Key 所在的环境变量名")
    gen.add_argument(
        "--types",
        default=",".join(CANDIDATE_TYPES),
        help="要生成的题型，逗号分隔（默认全部四种）",
    )
    gen.add_argument("--seed", type=int, default=13, help="随机种子，保证可复现（默认 13）")
    gen.add_argument(
        "--rare-top-k",
        type=int,
        default=RARE_TOP_K,
        help=(
            "不可回答判据里「关键术语」的个数 k（默认 %(default)s）：日志会打印这些术语"
            "在语料里的出现次数，便于核对判据；top-k 术语全部很弱时直接判定为不可回答"
        ),
    )
    gen.add_argument(
        "--presence-threshold",
        type=float,
        default=None,
        help=(
            "不可回答主判据：主题实词中「在语料里出现 >= %d 次」的比例低于该阈值即判为"
            "语料答不了。默认按语料规模自适应（48 篇约 %.2f，上限 %.2f）；调小它 = 更容易"
            "判定为不可回答" % (PRESENCE_MIN_COUNT, adaptive_presence_threshold(48), PRESENCE_SHARE_CAP)
        ),
    )
    gen.add_argument(
        "--unanswerable-coverage",
        type=float,
        default=None,
        help=(
            "不可回答判据的兜底覆盖率上限：IDF 加权覆盖率高于它就不算领域外。"
            "默认关闭（0，只作为日志里的参考值输出），因为覆盖率会被主题里的"
            "大量生僻二元组压低、一开就会误杀领域外题目；"
            "需要更严格时可显式给一个值，例如 0.2（可用 --coverage-adaptive 查看自适应值）"
        ),
    )
    gen.add_argument(
        "--coverage-adaptive",
        action="store_true",
        help=(
            "启用按语料规模自适应的覆盖率上限（48 篇语料约 %.2f，上限 %.2f）作为兜底判据"
            % (adaptive_coverage_threshold(48), COVERAGE_CAP)
        ),
    )
    gen.add_argument("--dry-run", action="store_true", help="只打印将要写入的内容，不落盘")

    pro = sub.add_parser("promote", help="把复核表里勾选的候选追加进评测集")
    pro.add_argument("--review", default=None, help="复核表路径（默认 data/eval/candidates_review.md）")
    pro.add_argument("--qa", default=None, help="评测集路径（默认配置 paths.eval）")
    pro.add_argument("--candidates", default=None, help="候选文件路径（默认 data/eval/candidates.jsonl）")
    pro.add_argument("--corpus", default=None, help="用于校验 gold_doc_ids 的语料（默认 paths.corpus）")
    pro.add_argument("--only-checked", action="store_true", help="只取勾选（- [x]）的候选")
    pro.add_argument("--dry-run", action="store_true", help="只打印将要追加的内容")

    st = sub.add_parser("stats", help="打印评测集与候选的题型分布")
    st.add_argument("--qa", default=None, help="评测集路径（默认配置 paths.eval）")
    st.add_argument("--candidates", default=None, help="候选文件路径（默认 data/eval/candidates.jsonl）")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    handlers: Dict[str, Callable[[argparse.Namespace], int]] = {
        "generate": cmd_generate,
        "promote": cmd_promote,
        "stats": cmd_stats,
    }
    try:
        return handlers[args.command](args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        _stderr(f"{args.command} 失败：{exc}")
        return 2
    except KeyboardInterrupt:  # pragma: no cover - 人工中断
        _stderr("已被用户中断")
        return 130


if __name__ == "__main__":
    sys.exit(main())
