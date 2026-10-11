"""把 rag-lab 暴露成编辑器（Cursor / VSCode + Copilot）可调用的 MCP 工具。

注意：这里用 PEP 562 的模块级 ``__getattr__`` **惰性导出**，而不是直接 ``from .server import``。
原因是 ``python -m rag.mcp.server`` 会先导入本包：若本包在导入时就加载了 ``server`` 模块，
Python 会警告「module found in sys.modules prior to execution」，既吵又可能引起顺序问题。
"""

from __future__ import annotations

from typing import Any

__all__ = ["RagMcpServer", "PROTOCOL_VERSION", "TOOLS", "main"]


def __getattr__(name: str) -> Any:  # pragma: no cover - 一行惰性转发
    if name in __all__:
        from . import server

        return getattr(server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
