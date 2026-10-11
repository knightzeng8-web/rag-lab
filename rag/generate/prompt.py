"""提示词与上下文拼装。

RAG 的生成质量一半取决于提示词里对「引用」和「拒答」的约束是否足够硬：

1. 只允许依据资料回答 —— 抑制参数化知识带来的幻觉（幻觉在 RAG 里最典型的形态是
   「答案对了但出处是编的」，所以引用必须可核验）；
2. 每句结论必须带 [n] 编号 —— 让答案变成可审计的结构化输出，引用解析器才能工作；
3. 允许并鼓励说「资料不足」 —— 拒答率是评测指标之一，宁可少答也不能编；
4. 上下文按编号 + doc_id + chunk_id 组织 —— 便于生成时引用，也便于排查“错在检索还是错在生成”。
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from ..types import Scored

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "REFUSAL_TEXT",
    "build_context_block",
    "build_user_prompt",
    "render_contexts",
    "is_refusal",
]

REFUSAL_TEXT = "资料不足，无法回答"

DEFAULT_SYSTEM_PROMPT = """你是一个严谨的知识库问答助手，只能依据给定的【资料】回答用户问题。

规则：
1. 每一句结论后面必须标注来源编号，格式为 [1]、[2]，编号对应资料片段的序号，可以并列如 [1][3]。
2. 如果资料不足以回答问题，直接回答“资料不足，无法回答”，并用一句话说明缺少什么信息；不要用你自己的知识补充。
3. 不要编造资料中没有的数字、名称、引文；不要把资料里的内容说成“通常认为”“众所周知”。
4. 如果多个资料互相矛盾，指出矛盾并分别标注来源。
5. 用中文回答，先给结论，再给依据；不要重复问题，不要输出与问题无关的客套话。
"""


def render_contexts(contexts: Sequence[Scored], max_chars: int = 4000) -> List[str]:
    """把候选片段渲染成 ``[i] (doc_id=..., chunk_id=...) 正文`` 的字符串列表，并做总长度截断。"""
    rendered: List[str] = []
    used = 0
    for index, item in enumerate(contexts, start=1):
        header = f"[{index}] (doc_id={item.chunk.doc_id}, chunk_id={item.chunk.chunk_id}"
        heading = item.chunk.heading
        if heading:
            header += f", heading={heading}"
        header += ")"
        text = item.chunk.text.strip()
        block = f"{header}\n{text}"
        if used + len(block) > max_chars and rendered:
            break
        rendered.append(block)
        used += len(block)
    return rendered


def build_context_block(contexts: Sequence[Scored], max_chars: int = 4000) -> str:
    return "\n\n".join(render_contexts(contexts, max_chars=max_chars))


def build_user_prompt(question: str, contexts: Sequence[Scored], max_chars: int = 4000) -> str:
    block = build_context_block(contexts, max_chars=max_chars)
    if not block:
        block = "（没有任何可用资料）"
    return f"【资料】\n{block}\n\n【问题】{question}\n\n【回答】"


def is_refusal(text: str) -> bool:
    normalized = (text or "").strip()
    if not normalized:
        return True
    head = normalized[:40]
    markers = (REFUSAL_TEXT, "资料不足", "无法回答", "没有相关资料", "不足以回答")
    return any(marker in head for marker in markers)
