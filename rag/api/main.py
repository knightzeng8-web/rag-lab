"""FastAPI 版服务（可选依赖）。接口与 ``rag.api.simple_server`` 完全一致。

安装：``pip install fastapi uvicorn``；
启动：``python -m rag.cli serve --server fastapi``。

面试时可以把两份实现放在一起讲：同一套 pipeline，一份零依赖、一份带自动文档与校验，
说明「业务逻辑与传输层解耦」这件事是被真正落地过的，而不只是写在简历上。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from ..obs import get_logger
from ..pipeline import RAGPipeline

__all__ = ["create_app", "run"]


def create_app(cfg, log=None):  # pragma: no cover - 需要 fastapi
    try:
        from fastapi import Body, FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import FileResponse
    except ImportError as exc:
        raise RuntimeError("未安装 fastapi：pip install fastapi uvicorn") from exc

    log = log or get_logger(level=str(cfg.get("logging.level", "INFO")))
    pipeline = RAGPipeline.build(config=cfg)
    pipeline.ensure_ready()

    app = FastAPI(title="rag-lab", version="0.1.0", description="领域知识库问答（混合检索 + 重排 + 引用溯源）")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return pipeline.status()

    @app.get("/status")
    def status() -> Dict[str, Any]:
        return pipeline.status()

    @app.post("/query")
    def query(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        question = str(payload.get("question") or payload.get("q") or "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="缺少 question 字段")
        top_k = payload.get("top_k")
        answer = pipeline.ask(question, top_k=int(top_k) if top_k else None)
        return answer.to_dict()

    @app.post("/ingest")
    def ingest(payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
        paths: List[str] = payload.get("paths") or [cfg.path_of("corpus")]
        if isinstance(paths, str):
            paths = [paths]
        # 全量重建；ingest 内部已重置装配状态，无需 force_rebuild
        report = pipeline.ingest([str(p) for p in paths])
        pipeline.ensure_ready()
        return report.to_dict()

    web_index = os.path.join(cfg.path_of("web"), "index.html")

    @app.get("/")
    def index() -> Any:
        if not os.path.isfile(web_index):
            return {"message": "未找到 web/index.html", "status": pipeline.status()}
        return FileResponse(web_index)

    log.info("FastAPI 应用已创建（索引 %d 块）", len(pipeline.store) if pipeline.store else 0)
    return app


def run(cfg, host: str = "127.0.0.1", port: int = 8000, log=None) -> None:  # pragma: no cover
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("未安装 uvicorn：pip install uvicorn") from exc

    app = create_app(cfg, log=log)
    uvicorn.run(app, host=host, port=int(port), log_level="info")
