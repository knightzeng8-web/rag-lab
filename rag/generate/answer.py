"""生成层：生成式回答（LLM）与离线抽取式回答（兜底），以及引用解析。

两类生成器共用一个接口 ``generate(question, contexts) -> (text, debug)``：

- ``LLMGenerator``：把编号后的资料塞进提示词，要求逐句带 [n] 引用；
- ``ExtractiveGenerator``：无 API Key / 无网络时的兜底。按「查询词覆盖率 × 片段排名先验」
  给片段内的句子打分，拼出带引用的答案。它的质量明显低于生成式回答，但保证了：
  ① 全链路可离线演示；② 评测里的引用准确率仍有意义；③ 它是判断“错在检索还是错在生成”
     的良好对照——如果抽取式都能答对，说明检索没问题，问题出在生成。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..text import jaccard, split_sentences, tokenize, truncate
from ..types import Citation, Scored
from .llm import LLM
from .prompt import DEFAULT_SYSTEM_PROMPT, REFUSAL_TEXT, build_user_prompt, is_refusal

__all__ = [
    "Generator",
    "LLMGenerator",
    "ExtractiveGenerator",
    "build_generator",
    "limit_contexts",
    "parse_citations",
    "best_quote",
]

_MARK = re.compile(r"[\[\u3010](\d{1,2})[\]\u3011]")
_MARKUP_PREFIXES = ("#", "|", "- ", "* ", "+ ", "> ", "```", "~~~")


def _strip_heading(text: str, heading: str) -> str:
    """剥掉分块时拼在正文前面的标题路径（它不是可读的答案句子）。"""
    if not heading:
        return text
    body = text.lstrip()
    if body.startswith(heading):
        return body[len(heading) :].lstrip()
    return body


def _is_markup_fragment(sentence: str) -> bool:
    """过滤 Markdown 结构行：表格、列表符号、代码围栏、标题路径。"""
    if sentence.startswith(_MARKUP_PREFIXES):
        return True
    if sentence.count("|") >= 2:
        return True
    if " > " in sentence and len(sentence.split(" > ")[-1]) < 30:
        return True
    return False


class Generator:
    kind = "base"
    name = "base"

    def generate(self, question: str, contexts: Sequence[Scored]) -> Tuple[str, Dict[str, Any]]:
        raise NotImplementedError

    def status(self) -> Dict[str, Any]:
        return {"kind": self.kind, "name": self.name}


class LLMGenerator(Generator):
    kind = "llm"

    def __init__(self, llm: LLM, system_prompt: Optional[str] = None, max_context_chars: int = 4000):
        self.llm = llm
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.max_context_chars = int(max_context_chars)
        self.name = f"llm:{getattr(llm, 'name', 'unknown')}"
        self._fallback = ExtractiveGenerator()

    def generate(self, question: str, contexts: Sequence[Scored]) -> Tuple[str, Dict[str, Any]]:
        if not contexts:
            return f"{REFUSAL_TEXT}：知识库中没有召回任何相关片段。", {"mode": "no_context"}
        user = build_user_prompt(question, contexts, max_chars=self.max_context_chars)
        try:
            text = self.llm.complete(system=self.system_prompt, user=user)
        except Exception as exc:
            text, debug = self._fallback.generate(question, contexts)
            debug["llm_error"] = str(exc)[:200]
            debug["mode"] = "extractive_fallback"
            return text, debug
        if not text:
            text, debug = self._fallback.generate(question, contexts)
            debug["llm_error"] = "LLM 返回空内容"
            return text, debug
        return text, {"mode": "llm", "model": self.name}


class ExtractiveGenerator(Generator):
    kind = "extractive"
    name = "extractive-offline"

    def __init__(self, max_sentences: int = 3, min_coverage: float = 0.30, min_hits: int = 3):
        self.max_sentences = int(max_sentences)
        self.min_coverage = float(min_coverage)
        self.min_hits = int(min_hits)

    def generate(self, question: str, contexts: Sequence[Scored]) -> Tuple[str, Dict[str, Any]]:
        if not contexts:
            return f"{REFUSAL_TEXT}：知识库中没有召回任何相关片段。", {"mode": "no_context"}

        # 查询词过滤：丢掉单个 CJK 字（噪声大），但要保留单个拉丁字母/数字——
        # 像 `b`、`k`、`n` 这种单字母术语在技术语料里是很强的信号（如 "k1 和 b 分别控制什么"）
        terms = [t for t in tokenize(question) if len(t) > 1 or t.isascii()] or tokenize(question)
        term_set = set(terms)
        scored: List[Dict[str, Any]] = []
        for index, item in enumerate(contexts, start=1):
            # 分块时我们把「标题路径」拼在了正文前面（帮助检索），但它不是答案句子，先剥掉；
            # 表格行、列表符号、代码围栏同理，都不该出现在答案里
            paragraph = _strip_heading(item.chunk.text, item.chunk.heading)
            for sentence in split_sentences(paragraph):
                sentence = sentence.strip()
                if len(sentence) < 8 or _is_markup_fragment(sentence):
                    continue
                sentence_terms = set(tokenize(sentence))
                hits = len(term_set & sentence_terms)
                if not hits:
                    continue
                coverage = hits / max(1, len(term_set))
                length_bonus = min(1.0, len(sentence) / 100.0)
                # 排名先验：抽取式答案偏向最高分片段。衰减越小，越容易把排名靠后的
                # 片段也拉进答案，从而引入 gold 之外的出处（实测会拉低引用精确率）
                rank_prior = 1.0 / (1.0 + 0.35 * (index - 1))
                scored.append(
                    {
                        "score": coverage * (0.6 + 0.4 * length_bonus) * rank_prior,
                        "index": index,
                        "sentence": sentence,
                        "coverage": coverage,
                        "hits": hits,
                    }
                )

        if not scored:
            return (
                f"{REFUSAL_TEXT}：检索到的片段与问题没有词面重合，无法给出有依据的答案。",
                {"mode": "extractive", "reason": "no_term_overlap"},
            )

        scored.sort(key=lambda item: (-item["score"], item["index"]))
        # 拒答判据只用「查询词覆盖率」，不用那个混了长度与排名先验的合成分数：
        #   覆盖率低 = 没有任何一句话真正在讲这件事（例如问 llama.cpp 却只命中「参数/配置」）。
        #   合成分数会把「排名第一但只是词面沾边」的句子推过阈值，实测导致不可回答的题被作答。
        best_coverage = max(float(item["coverage"]) for item in scored)
        best_hits = max(int(item["hits"]) for item in scored)
        if best_coverage < self.min_coverage or best_hits < self.min_hits:
            failed = []
            if best_coverage < self.min_coverage:
                failed.append(f"最高覆盖率 {best_coverage:.2f} < {self.min_coverage:.2f}")
            if best_hits < self.min_hits:
                failed.append(f"最高命中词数 {best_hits} < {self.min_hits}")
            return (
                f"{REFUSAL_TEXT}：检索结果的词面重合度不足（{'；'.join(failed)}），证据不足以作答。",
                {
                    "mode": "extractive",
                    "reason": "low_coverage",
                    "best_coverage": round(best_coverage, 4),
                    "best_hits": best_hits,
                },
            )

        picked: List[Dict[str, Any]] = []
        for candidate in scored:
            if len(picked) >= self.max_sentences:
                break
            if any(
                jaccard(tokenize(candidate["sentence"]), tokenize(chosen["sentence"])) > 0.75
                for chosen in picked
            ):
                continue
            picked.append(candidate)

        body = " ".join(f"{item['sentence']}[{item['index']}]" for item in picked)
        text = (
            f"根据知识库资料：{body}\n\n"
            "（离线抽取式答案：以上是检索命中原句的拼接，未经过生成模型改写。"
            "配置 LLM API Key 后会自动切换为生成式回答。）"
        )
        return text, {
            "mode": "extractive",
            "sentences": len(picked),
            "best_coverage": round(best_coverage, 4),
            "best_hits": best_hits,
            "sources": sorted({item["index"] for item in picked}),
        }


def limit_contexts(contexts: Sequence[Scored], cfg) -> List[Scored]:
    keep = int(cfg.get("generation.max_context_chunks", 6))
    return list(contexts)[: max(1, keep)]


def build_generator(cfg, llm: Optional[LLM], log=None) -> Generator:
    provider = str(cfg.get("generation.provider", "auto")).lower()

    def extractive() -> ExtractiveGenerator:
        return ExtractiveGenerator(
            max_sentences=int(cfg.get("generation.extractive_max_sentences", 3)),
            min_coverage=float(cfg.get("generation.extractive_min_coverage", 0.30)),
            min_hits=int(cfg.get("generation.extractive_min_hits", 3)),
        )

    if provider in ("extractive", "offline", "none"):
        return extractive()
    if llm is not None and getattr(llm, "available", False):
        return LLMGenerator(
            llm,
            system_prompt=cfg.get("generation.system_prompt"),
            max_context_chars=int(cfg.get("generation.max_context_chars", 4000)),
        )
    if log:
        log.info("未检测到可用 LLM，使用离线抽取式生成器（extractive-offline）")
    return extractive()


def best_quote(chunk_text: str, reference_terms: set, limit: int = 240) -> str:
    """从片段中挑与答案最相关的一句作为引用原文，便于人工核验。"""
    sentences = split_sentences(chunk_text) or [chunk_text]
    if reference_terms:
        best = max(
            sentences,
            key=lambda s: (len(set(tokenize(s)) & reference_terms), -abs(len(s) - 120)),
        )
    else:
        best = sentences[0]
    return truncate(best, limit)


def parse_citations(
    text: str,
    contexts: Sequence[Scored],
    quote_chars: int = 240,
) -> List[Citation]:
    """把答案里的 [n] 映射回具体片段；越界编号会被丢弃（说明模型编了引用）。"""
    citations: List[Citation] = []
    if not text:
        return citations
    answer_terms = set(tokenize(_MARK.sub(" ", text)))
    seen = set()
    for match in _MARK.finditer(text):
        index = int(match.group(1))
        if index in seen or index < 1 or index > len(contexts):
            continue
        seen.add(index)
        item = contexts[index - 1]
        citations.append(
            Citation(
                index=index,
                doc_id=item.chunk.doc_id,
                chunk_id=item.chunk.chunk_id,
                quote=best_quote(item.chunk.text, answer_terms, limit=quote_chars),
                score=float(item.score),
            )
        )
    return citations


__all__ += ["is_refusal"]
