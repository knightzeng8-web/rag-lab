"""零依赖 HTTP 服务（标准库 http.server）。

为什么要有一个不依赖 FastAPI 的实现：
- 面试演示最怕「装环境装了十分钟」，只要 Python 能跑就能开服务；
- 它是 ``rag.api.main``（FastAPI 版）的兜底，两者接口完全一致，便于说明「接口优先、框架可换」。

接口：
    GET  /health            -> 组件与索引状态
    GET  /status            -> 同上（详细）
    POST /query             -> {"question": "...", "top_k": 5} -> Answer.to_dict()
    POST /ingest            -> {"paths": ["data/corpus"]} -> IngestReport（总是全量重建）
    GET  /                  -> web/index.html
    GET  /<静态文件>         -> web/ 目录内的文件
"""

from __future__ import annotations

import json
import mimetypes
import os
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple
from urllib.parse import unquote, urlparse

from ..obs import get_logger
from ..pipeline import RAGPipeline

__all__ = ["make_handler", "serve", "build_pipeline"]

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
MAX_BODY = 1 * 1024 * 1024


def build_pipeline(cfg) -> RAGPipeline:
    pipeline = RAGPipeline.build(config=cfg)
    pipeline.ensure_ready()
    return pipeline


def make_handler(pipeline: RAGPipeline, web_dir: str, log):
    class Handler(BaseHTTPRequestHandler):
        server_version = "rag-lab/0.1"
        protocol_version = "HTTP/1.1"

        # ---- 工具 -----------------------------------------------------------
        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
            log.info("%s - %s", self.address_string(), fmt % args)

        def _cors(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

        def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path: str, status: int = 200) -> None:
            if not os.path.isfile(path):
                self._send_json({"error": f"文件不存在：{path}"}, status=404)
                return
            with open(path, "rb") as handle:
                body = handle.read()
            content_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
            if content_type.startswith("text/") or content_type in ("application/javascript",):
                content_type += "; charset=utf-8"
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return None, "Content-Length 非法"
            if length < 0:
                # read(-N) 会一直读到 EOF，等于把连接挂住
                return None, "Content-Length 不能为负"
            if length > MAX_BODY:
                # 不读 body 就必须关连接，否则剩余字节会被当成下一个请求解析
                self.close_connection = True
                return None, "请求体过大"
            raw = self.rfile.read(length) if length else b"{}"
            if not raw:
                return {}, None
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                return None, f"请求体不是合法 JSON：{exc}"
            if not isinstance(payload, dict):
                return None, "请求体必须是 JSON 对象"
            return payload, None

        # ---- 路由 -----------------------------------------------------------
        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            path = unquote(urlparse(self.path).path)
            try:
                if path in ("/health", "/status"):
                    self._send_json(pipeline.status())
                    return
                if path in ("/", "/index.html"):
                    self._send_file(os.path.join(web_dir, "index.html"))
                    return
                # 静态资源：限制在 web 目录内，避免路径穿越。
                # 必须比较 root + sep，否则同前缀的兄弟目录（web-backup/）也能读到
                candidate = os.path.abspath(os.path.join(web_dir, path.lstrip("/")))
                root = os.path.abspath(web_dir)
                if candidate.startswith(root + os.sep) and os.path.isfile(candidate):
                    self._send_file(candidate)
                    return
                self._send_json({"error": f"未知路径：{path}"}, status=404)
            except Exception as exc:  # pragma: no cover - 运行时错误
                # 没有这层兜底，异常会直接冒泡到 socketserver，客户端只能看到 "Failed to fetch"
                log.error("处理 %s 失败：%s\n%s", path, exc, traceback.format_exc())
                self._send_json({"error": str(exc)}, status=500)

        def do_POST(self) -> None:  # noqa: N802
            path = unquote(urlparse(self.path).path)
            payload, error = self._read_json()
            if error:
                self._send_json({"error": error}, status=400)
                return
            assert payload is not None
            try:
                if path == "/query":
                    question = str(payload.get("question") or payload.get("q") or "").strip()
                    if not question:
                        self._send_json({"error": "缺少 question 字段"}, status=400)
                        return
                    top_k = payload.get("top_k")
                    answer = pipeline.ask(question, top_k=int(top_k) if top_k else None)
                    self._send_json(answer.to_dict())
                    return
                if path == "/ingest":
                    paths = payload.get("paths") or [pipeline.cfg.path_of("corpus")]
                    if isinstance(paths, str):
                        paths = [paths]
                    # ingest 永远是全量重建；它内部会把 _built 置 False，
                    # 因此这里用非 force 的 ensure_ready 即可按新索引重新装配组件
                    report = pipeline.ingest([str(p) for p in paths])
                    pipeline.ensure_ready()
                    self._send_json(report.to_dict())
                    return
                self._send_json({"error": f"未知路径：{path}"}, status=404)
            except Exception as exc:  # pragma: no cover - 运行时错误
                log.error("处理 %s 失败：%s\n%s", path, exc, traceback.format_exc())
                self._send_json({"error": str(exc)}, status=500)

    return Handler


def serve(
    cfg,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    log=None,
) -> None:
    log = log or get_logger(level=str(cfg.get("logging.level", "INFO")))
    pipeline = build_pipeline(cfg)
    web_dir = cfg.path_of("web")
    handler = make_handler(pipeline, web_dir, log)
    httpd = ThreadingHTTPServer((host, int(port)), handler)
    status = pipeline.status()
    log.info(
        "服务已启动：http://%s:%d  （索引 %d 块 / 向量模型 %s / 重排 %s / 生成 %s）",
        host,
        int(port),
        status.get("index", {}).get("chunks", 0),
        status.get("components", {}).get("embedder", {}).get("name"),
        status.get("components", {}).get("reranker", {}).get("name"),
        status.get("components", {}).get("generator", {}).get("name"),
    )
    log.info("前端页面：http://%s:%d/  （若未打开，请确认 %s 存在）", host, int(port), web_dir)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - 人工中断
        log.info("收到中断信号，正在关闭服务")
    finally:
        httpd.server_close()
