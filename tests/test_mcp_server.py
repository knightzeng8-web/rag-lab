"""MCP 服务器测试：协议握手、工具调用、错误码、以及「stdout 只能有协议消息」这条硬约束。

为什么要专门测 stdout 纯净性：MCP 客户端是**按行解析 stdout** 的，任何一行日志或警告
都会让客户端解析失败并断开连接。这类问题在集成时才爆炸，所以要在单元测试里锁死。
"""

import io
import json
import os
import unittest

from rag.config import Config
from rag.mcp.server import PROTOCOL_VERSION, TOOLS, RagMcpServer

try:  # 兼容两种 discover 调用方式
    from tests.support import TempWorkspace
except ImportError:  # pragma: no cover
    from support import TempWorkspace

ALPHA = """# 检索算法
## BM25
BM25 是词频类检索算法，参数 k1 控制词频饱和，b 控制文档长度归一化。

## RRF
RRF 使用倒数排名融合，常数 k 通常取 60。
"""


class McpServerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TempWorkspace(prefix="mcp-")
        root = self.tmp.name
        self.corpus_dir = os.path.join(root, "corpus")
        os.makedirs(self.corpus_dir, exist_ok=True)
        with open(os.path.join(self.corpus_dir, "alpha.md"), "w", encoding="utf-8") as handle:
            handle.write(ALPHA)
        self.cfg = Config.from_dict(
            {
                "paths": {"corpus": "corpus", "index": "index", "reports": "reports", "web": "web"},
                "chunking": {"strategy": "structural", "max_chars": 240, "overlap_chars": 60, "min_chars": 20},
                "embedder": {"provider": "hashing", "dim": 128},
                "generation": {"provider": "extractive", "extractive_min_coverage": 0.1, "extractive_min_hits": 2},
                "logging": {"level": "ERROR"},
            },
            root=root,
        )
        self.server = RagMcpServer(cfg=self.cfg)

    def tearDown(self):
        self.tmp.cleanup()

    def call(self, method, params=None, msg_id=1):
        message = {"jsonrpc": "2.0", "id": msg_id, "method": method}
        if params is not None:
            message["params"] = params
        return self.server.handle_message(message)

    # ---- 协议 ---------------------------------------------------------------
    def test_initialize(self):
        response = self.call("initialize", {"protocolVersion": PROTOCOL_VERSION})
        result = response["result"]
        self.assertEqual(result["protocolVersion"], PROTOCOL_VERSION)
        self.assertEqual(result["serverInfo"]["name"], "rag-lab")
        self.assertIn("tools", result["capabilities"])

    def test_tools_list(self):
        response = self.call("tools/list")
        tools = response["result"]["tools"]
        self.assertEqual(len(tools), 4)
        names = {tool["name"] for tool in tools}
        self.assertEqual(names, {"kb_stats", "kb_search", "kb_answer", "kb_ingest"})
        for tool in tools:
            self.assertTrue(tool["description"])
            self.assertEqual(tool["inputSchema"]["type"], "object")

    def test_notification_gets_no_reply(self):
        self.assertIsNone(self.server.handle_message({"jsonrpc": "2.0", "method": "notifications/initialized"}))

    def test_ping(self):
        self.assertEqual(self.call("ping")["result"], {})

    def test_unknown_method(self):
        response = self.call("resources/list", msg_id=7)
        self.assertEqual(response["error"]["code"], -32601)
        self.assertEqual(response["id"], 7)

    def test_invalid_request(self):
        self.assertEqual(self.server.handle_message("不是对象")["error"]["code"], -32600)

    # ---- 工具 ---------------------------------------------------------------
    def test_stats_tool(self):
        response = self.call("tools/call", {"name": "kb_stats", "arguments": {}})
        result = response["result"]
        self.assertFalse(result["isError"])
        text = result["content"][0]["text"]
        self.assertIn("分块", text)
        self.assertIn("hashing", text)

    def test_search_tool(self):
        response = self.call("tools/call", {"name": "kb_search", "arguments": {"query": "k1 控制什么", "top_k": 3}})
        text = response["result"]["content"][0]["text"]
        self.assertFalse(response["result"]["isError"])
        self.assertIn("alpha#", text)  # chunk_id 形如 alpha#0
        self.assertIn("score=", text)

    def test_answer_tool_has_citations(self):
        response = self.call("tools/call", {"name": "kb_answer", "arguments": {"question": "RRF 的常数 k 取多少？"}})
        result = response["result"]
        self.assertFalse(result["isError"])
        text = result["content"][0]["text"]
        self.assertIn("引用", text)
        self.assertIn("alpha#", text)
        self.assertIn("是否拒答", text)

    def test_missing_required_argument(self):
        response = self.call("tools/call", {"name": "kb_search", "arguments": {}})
        result = response["result"]
        self.assertTrue(result["isError"])
        self.assertIn("query", result["content"][0]["text"])

    def test_unknown_tool(self):
        response = self.call("tools/call", {"name": "kb_delete_all", "arguments": {}})
        self.assertTrue(response["result"]["isError"])

    def test_tools_call_without_name(self):
        response = self.call("tools/call", {"arguments": {}})
        self.assertEqual(response["error"]["code"], -32602)

    def test_ingest_disabled_by_default(self):
        response = self.call("tools/call", {"name": "kb_ingest", "arguments": {}})
        result = response["result"]
        self.assertTrue(result["isError"])
        self.assertIn("--allow-ingest", result["content"][0]["text"])

    # ---- stdio 传输 ---------------------------------------------------------
    def test_stdio_output_is_protocol_only(self):
        """stdout 里每一行都必须是合法 JSON-RPC；混进任何日志都会让客户端断连。"""
        stdin = io.StringIO(
            "\n".join(
                [
                    json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
                    json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
                    json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
                    "",
                ]
            )
        )
        stdout = io.StringIO()
        self.server.serve_stdio(stdin=stdin, stdout=stdout)
        lines = [line for line in stdout.getvalue().splitlines() if line.strip()]
        # 通知不产生回复，所以正好 2 行
        self.assertEqual(len(lines), 2)
        payloads = [json.loads(line) for line in lines]
        self.assertEqual([payload["id"] for payload in payloads], [1, 2])
        self.assertNotIn("\n", payloads[0]["result"]["protocolVersion"])

    def test_stdio_reports_parse_error_and_keeps_going(self):
        stdin = io.StringIO("\n".join(["{这不是 JSON", json.dumps({"jsonrpc": "2.0", "id": 3, "method": "ping"})]))
        stdout = io.StringIO()
        self.server.serve_stdio(stdin=stdin, stdout=stdout)
        payloads = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
        self.assertEqual(len(payloads), 2)
        self.assertEqual(payloads[0]["error"]["code"], -32700)
        self.assertEqual(payloads[1]["id"], 3)


if __name__ == "__main__":
    unittest.main()
