"""轻量文本工具：分词、句切分、关键词抽取、哈希。

中文场景不引入 jieba，用「CJK 单字 + 二元组」+「拉丁词」的混合切分：
- 对 BM25 来说，二元组等价于隐式的 n-gram 词表，无需词典即可获得可用的检索质量；
- 对哈希向量模型来说，同一套 token 直接复用，保证两条召回链路共享同一套信号。
"""

from __future__ import annotations

import hashlib
import re
from typing import Dict, Iterable, List, Sequence, Set

__all__ = [
    "CJK_RANGE",
    "normalize_ws",
    "tokenize",
    "split_sentences",
    "keywords",
    "truncate",
    "stable_hash",
    "first_heading",
    "jaccard",
]

CJK_RANGE = "\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff"

_CJK_RUN = re.compile(f"[{CJK_RANGE}]+")
_LATIN = re.compile(r"[a-zA-Z0-9_]+(?:[.\-][a-zA-Z0-9_]+)*")
_SENT_SPLIT = re.compile(r"(?<=[\u3002\uff01\uff1f\uff1b!?;])|\n+")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.*)$", re.MULTILINE)
_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "to", "in", "is", "are", "for", "on", "with", "that", "this",
    "什么", "怎么", "如何", "为什么", "是否", "可以", "我们", "他们", "以及", "并且", "但是", "因为",
    "所以", "如果", "一个", "这个", "那个", "这样", "非常", "就是", "还是", "然后", "而且", "需要",
}


def normalize_ws(text: str) -> str:
    """压缩连续空白，去掉零宽字符，保留段落换行。"""
    if not text:
        return ""
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    """混合分词：拉丁词 + CJK 单字 + CJK 二元组（全部小写）。"""
    if not text:
        return []
    lowered = text.lower()
    tokens: List[str] = [m.group(0) for m in _LATIN.finditer(lowered)]
    for match in _CJK_RUN.finditer(lowered):
        run = match.group(0)
        if len(run) == 1:
            tokens.append(run)
            continue
        for i in range(len(run)):
            tokens.append(run[i])
            if i + 1 < len(run):
                tokens.append(run[i : i + 2])
    return tokens


def split_sentences(text: str, min_len: int = 4) -> List[str]:
    """中英混排的粗粒度句切分；过短的碎片会被合并回上一句。"""
    if not text:
        return []
    parts = [p.strip() for p in _SENT_SPLIT.split(text)]
    sentences: List[str] = []
    for part in parts:
        if not part:
            continue
        if sentences and len(part) < min_len:
            sentences[-1] = f"{sentences[-1]} {part}".strip()
        else:
            sentences.append(part)
    return sentences


def keywords(
    text: str,
    limit: int = 8,
    df: Dict[str, int] | None = None,
    total_docs: int = 0,
) -> List[str]:
    """抽取关键词用于查询改写。

    有 ``df``/``total_docs``（来自 BM25 索引）时按 IDF 排序，否则退化为
    「长度优先 + 去停用词」的启发式规则，保证离线也能工作。
    """
    if not text:
        return []
    counts: Dict[str, int] = {}
    for token in tokenize(text):
        if token in _STOPWORDS:
            continue
        if len(token) == 1 and not token.isdigit():
            continue  # 单字噪声大，除非是数字
        counts[token] = counts.get(token, 0) + 1

    def weight(token: str) -> float:
        tf = counts[token]
        if df and total_docs:
            import math

            d = df.get(token, 0)
            idf = math.log(1.0 + (total_docs - d + 0.5) / (d + 0.5))
        else:
            idf = 1.0 + 0.3 * len(token)
        return tf * idf * (1.0 + 0.2 * len(token))

    ranked = sorted(counts, key=lambda t: (-weight(t), -len(t), t))
    picked: List[str] = []
    for token in ranked:
        if any(token in p or p in token for p in picked):
            continue
        picked.append(token)
        if len(picked) >= limit:
            break
    return picked


def truncate(text: str, limit: int = 240, ellipsis: str = "…") -> str:
    if not text:
        return ""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    for sep in ("。", "！", "？", "；", ".", "!", "?", ";", "\n"):
        pos = cut.rfind(sep)
        if pos >= limit * 0.5:
            return cut[: pos + 1] + ellipsis
    return cut + ellipsis


def stable_hash(value: str, mod: int) -> int:
    """跨进程稳定的哈希（不用内置 hash，避免 PYTHONHASHSEED 影响可复现性）。"""
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % mod


def first_heading(text: str) -> str:
    match = _HEADING.search(text or "")
    return match.group(1).strip() if match else ""


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa: Set[str] = set(a)
    sb: Set[str] = set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / float(len(sa | sb))
