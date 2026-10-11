"""生成层：LLM 客户端、提示词、生成器与引用解析。"""

from .answer import (
    ExtractiveGenerator,
    Generator,
    LLMGenerator,
    best_quote,
    build_generator,
    limit_contexts,
    parse_citations,
)
from .llm import LLM, OpenAICompatLLM, build_llm, llm_status
from .prompt import DEFAULT_SYSTEM_PROMPT, REFUSAL_TEXT, build_context_block, build_user_prompt, is_refusal

__all__ = [
    "LLM",
    "OpenAICompatLLM",
    "build_llm",
    "llm_status",
    "DEFAULT_SYSTEM_PROMPT",
    "REFUSAL_TEXT",
    "build_context_block",
    "build_user_prompt",
    "is_refusal",
    "Generator",
    "LLMGenerator",
    "ExtractiveGenerator",
    "build_generator",
    "limit_contexts",
    "parse_citations",
    "best_quote",
]
