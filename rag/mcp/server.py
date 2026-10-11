"""极简 MCP（Model Context Protocol）服务器：把 rag-lab 变成编辑器能调用的工具。

为什么自己实现：MCP 的 stdio 传输就是「**一行一个 JSON-RPC 2.0 消息**」，不需要任何依赖。
自己实现的好处是"RAG 作为一个工具被 AI 助手调用"这件事可以被讲得很具体：协议怎么握手、
工具有哪些、参数怎么校验、错误怎么回。接进 Cursor / VSCode 后，你在编辑器里就能直接问
自己的知识库，而不用切窗口跑命令行。

协议要点（踩过才知道的部分）：
- **stdout 只能出现协议消息**，一行一个 JSON；任何日志都必须走 stderr，否则客户端解析崩溃；
- 客户端的 `notifications/initialized` **不需要回复**（通知没有 id）；
- 工具执行失败不应该返回 JSON-RPC 错误，而应返回 `isError: true` 的 content，
  这样模型能看到失败原因并自己决定下一步；
- 参数错误返回 -32602、未知方法返回 -32601、解析失败返回 -32700。

工具：
- `kb_stats()`                    索引与生效组件（含 device），模型用它确认"现在连的是什么"
- `kb_search(query, top_k)`       只检索，返回片段与分数（不调用 LLM，不需要 Key）
- `kb_answer(question, top_k)`    完整问答，带引用（走配置的生成器：真实 LLM 或离线抽取式）
- `kb_ingest(paths)`              重建索引（默认禁用，需要 --allow-ingest 显式打开）
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Sequence

from ..config import load_config, parse_scalar
from ..obs import get_logger
from ..pipeline import RAGPipeline
from ..types import Answer, RetrieveResult

__all__ = ["RagMcpServer", "TOOLS", "PROTOCOL_VERSION", "main"]

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "rag-lab"
SERVER_VERSION = "0.1.0"

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "kb_stats",
        "description": "查看知识库索引状态与当前生效组件（向量模型、重排器、生成器、设备、分块数）。",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "kb_search",
        "description": "在本地知识库里做混合检索（向量 + BM25 + RRF，可选重排），返回片段与分数，不调用大模型。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索查询，中文或英文均可"},
                "top_k": {"type": "integer", "description": "返回几条，默认 5", "minimum": 1, "maximum": 50},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "kb_answer",
        "description": "对知识库提问并返回带引用编号的答案（引用可追溯到 doc_id / chunk_id / 原文片段）。资料不足时会明确拒答。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "用户问题"},
                "top_k": {"type": "integer", "description": "送入生成器的片段数，默认 5", "minimum": 1, "maximum": 20},
            },
            "required": ["question"],
            "additionalProperties": False,
        },
    },
    {
        "name": "kb_ingest",
        "description": "重建知识库索引（全量）。默认禁用，需要启动时加 --allow-ingest。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "语料目录或文件；省略则用配置里的 paths.corpus",
                }
            },
            "additionalProperties": False,
        },
    },
]


class ToolError(Exception):
    """工具执行失败（会以 isError 形式回给模型，而不是协议错误）。"""


class RagMcpServer:
    def __init__(self, cfg=None, allow_ingest: bool = False, log=None):
        self.cfg = cfg or load_config()
        self.allow_ingest = bool(allow_ingest)
        self.log = log or get_logger(
            name="rag.mcp", level=str((cfg or load_config()).get("logging.level", "INFO"))
        )
        self._pipeline: Optional[RAGPipeline] = None

    # ------------------------------------------------------------------ #
    # 惰性构建：编辑器启动 MCP 服务器时可能只是想列一下工具，
    # 不应该为此加载 BGE 与整份索引（那会拖慢编辑器启动）。
    # ------------------------------------------------------------------ #
    @property
    def pipeline(self) -> RAGPipeline:
        if self._pipeline is None:
            self._pipeline = RAGPipeline.build(config=self.cfg)
            self._pipeline.ensure_ready()
        return self._pipeline

    # ------------------------------------------------------------------ #
    # 工具实现
    # ------------------------------------------------------------------ #
    def _tool_stats(self) -> str:
        status = self.pipeline.status()
        lines = [
            f"索引：{status.get('index', {}).get('chunks', 0)} 个分块 / "
            f"{status.get('index', {}).get('docs', 0)} 篇文档 / 后端 {status.get('index', {}).get('backend')}",
            f"向量模型：{(status.get('components', {}).get('embedder') or {}).get('name')}",
            f"重排器：{(status.get('components', {}).get('reranker') or {}).get('name')}",
            f"生成器：{(status.get('components', {}).get('generator') or {}).get('name')}",
            f"设备：{(status.get('device') or {}).get('selected')}"
            f"（cuda={((status.get('device') or {}).get('cuda_available'))}）",
        ]
        return "\n".join(lines)

    @staticmethod
    def _format_hits(result: RetrieveResult, top_k: int) -> str:
        ranked = result.ranked[: max(1, int(top_k))]
        if not ranked:
            return "检索没有返回任何片段（索引可能为空，先运行 ingest）。"
        blocks = [f"查询改写后的检索式：{' | '.join(result.queries)}"]
        for index, item in enumerate(ranked, start=1):
            text = item.chunk.text.strip().replace("\n", " ")
            blocks.append(
                f"[{index}] {item.chunk.doc_id} / {item.chunk.chunk_id}"
                f"（score={item.score:.4f}，source={item.source}）\n    {text[:400]}"
            )
        return "\n".join(blocks)

    @staticmethod
    def _format_answer(answer: Answer) -> str:
        lines = [answer.text.strip()]
        if answer.citations:
            lines.append("")
            lines.append("引用：")
            for cite in answer.citations:
                lines.append(f"  [{cite.index}] {cite.doc_id} / {cite.chunk_id}：{cite.quote[:160]}")
        lines.append("")
        latency = answer.latency_ms
        lines.append(
            "耗时：rewrite {:.0f}ms / retrieve {:.0f}ms / rerank {:.0f}ms / generate {:.0f}ms / total {:.0f}ms".format(
                latency.get("rewrite", 0.0),
                latency.get("retrieve", 0.0),
                latency.get("rerank", 0.0),
                latency.get("generate", 0.0),
                latency.get("total", 0.0),
            )
        )
        lines.append(f"是否拒答：{'是' if answer.refused else '否'}")
        return "\n".join(lines)

    def _tool_search(self, arguments: Dict[str, Any]) -> str:
        query = str(arguments.get("query") or "").strip()
        if not query:
            raise ToolError("缺少必填参数 query")
        top_k = int(arguments.get("top_k") or 5)
        result = self.pipeline.retrieve(query, top_k=top_k)
        return self._format_hits(result, top_k)

    def _tool_answer(self, arguments: Dict[str, Any]) -> str:
        question = str(arguments.get("question") or arguments.get("query") or "").strip()
        if not question:
            raise ToolError("缺少必填参数 question")
        top_k = int(arguments.get("top_k") or 5)
        answer = self.pipeline.ask(question, top_k=top_k)
        return self._format_answer(answer)

    def _tool_ingest(self, arguments: Dict[str, Any]) -> str:
        if not self.allow_ingest:
            raise ToolError("kb_ingest 已禁用：请用 `python -m rag.mcp.server --allow-ingest` 启动")
        paths = arguments.get("paths") or None
        if paths is not None and not isinstance(paths, list):
            raise ToolError("paths 必须是字符串数组")
        report = self.pipeline.ingest([str(p) for p in paths] if paths else None)
        self.pipeline.ensure_ready()
        return report.summary()

    # ------------------------------------------------------------------ #
    # JSON-RPC
    # ------------------------------------------------------------------ #
    def _call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        name = str(params.get("name") or "")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise ValueError("arguments 必须是对象")
        handlers = {
            "kb_stats": lambda _args: self._tool_stats(),
            "kb_search": self._tool_search,
            "kb_answer": self._tool_answer,
            "kb_ingest": self._tool_ingest,
        }
        handler = handlers.get(name)
        if handler is None:
            return {
                "content": [{"type": "text", "text": f"未知工具：{name}"}],
                "isError": True,
            }
        try:
            text = handler(arguments)
        except ToolError as exc:
            return {"content": [{"type": "text", "text": str(exc)}], "isError": True}
        except Exception as exc:  # noqa: BLE001 - 工具内部错误也要让模型看到
            self.log.error("工具 %s 执行失败：%s", name, exc)
            return {
                "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
                "isError": True,
            }
        return {"content": [{"type": "text", "text": text}], "isError": False}

    @staticmethod
    def _error(msg_id: Any, code: int, message: str) -> Dict[str, Any]:
        return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}

    def handle_message(self, message: Any) -> Optional[Dict[str, Any]]:
        """处理一条消息；通知（没有 id）返回 None。"""
        if not isinstance(message, dict) or "method" not in message:
            return self._error(None, -32600, "不是合法的 JSON-RPC 请求")
        method = str(message.get("method"))
        msg_id = message.get("id")
        is_notification = msg_id is None

        if method.startswith("notifications/"):
            self.log.debug("收到通知：%s", method)
            return None
        try:
            if method == "initialize":
                result: Dict[str, Any] = {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                }
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": TOOLS}
            elif method == "tools/call":
                params = message.get("params") or {}
                if not isinstance(params, dict) or not params.get("name"):
                    return self._error(msg_id, -32602, "tools/call 需要 name 参数")
                result = self._call_tool(params)
            else:
                return self._error(msg_id, -32601, f"未支持的方法：{method}")
        except Exception as exc:  # noqa: BLE001
            self.log.error("处理 %s 失败：%s", method, exc)
            return self._error(msg_id, -32603, f"内部错误：{exc}")

        if is_notification:
            return None
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    # ------------------------------------------------------------------ #
    def serve_stdio(self, stdin=None, stdout=None) -> int:
        stdin = stdin or sys.stdin
        stdout = stdout or sys.stdout
        self.log.info("%s MCP 服务器已启动（stdio，等待客户端消息）", SERVER_NAME)
        for raw in stdin:
            line = raw.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError as exc:
                self._write(stdout, self._error(None, -32700, f"JSON 解析失败：{exc}"))
                continue
            response = self.handle_message(message)
            if response is not None:
                self._write(stdout, response)
        self.log.info("stdin 关闭，MCP 服务器退出")
        return 0

    @staticmethod
    def _write(stdout, payload: Dict[str, Any]) -> None:
        # 一行一个 JSON-RPC 消息；不换行会在客户端侧造成解析阻塞
        stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
        stdout.flush()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="rag-lab MCP 服务器（stdio）")
    parser.add_argument("--config", default=None, help="配置文件路径（默认 configs/default.yaml）")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--root", default=None)
    parser.add_argument("--allow-ingest", action="store_true", help="允许通过 MCP 重建索引")
    parser.add_argument("--log-level", default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    overrides: Dict[str, Any] = {}
    for pair in args.set:
        if "=" in pair:
            key, value = pair.split("=", 1)
            overrides[key.strip()] = parse_scalar(value)

    cfg = load_config(path=args.config, overrides=overrides, root=args.root)
    log = get_logger(
        name="rag.mcp",
        level=str(args.log_level or cfg.get("logging.level", "INFO")),
    )
    server = RagMcpServer(cfg=cfg, allow_ingest=args.allow_ingest, log=log)
    return server.serve_stdio()


if __name__ == "__main__":
    sys.exit(main())
