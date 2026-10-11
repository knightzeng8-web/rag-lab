"""日志与计时。故意做得简单，但对「可观测」这件事给出明确落点：

- 每次检索/生成都会在 ``Answer.latency_ms`` 里留阶段耗时，API 与前端都能看到分解；
- ``StageTimer`` 用上下文管理器写阶段耗时，避免到处手写 perf_counter；
- 日志里统一带 ``[rag]`` 前缀，方便把 CLI / API 输出与其它输出区分开。
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Any, Dict, Optional

__all__ = ["get_logger", "configure_logging", "StageTimer"]


def get_logger(name: str = "rag", level: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, str(level or "INFO").upper(), logging.INFO))
    logger.propagate = False
    return logger


def configure_logging(level: str = "INFO", name: str = "rag") -> logging.Logger:
    return get_logger(name=name, level=level)


class StageTimer:
    """``with StageTimer(bucket, "retrieve"): ...`` 把耗时写进 ``bucket['retrieve']``。"""

    def __init__(self, bucket: Dict[str, float], stage: str):
        self.bucket = bucket
        self.stage = stage
        self.started = 0.0

    def __enter__(self) -> "StageTimer":
        self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        elapsed = (time.perf_counter() - self.started) * 1000.0
        self.bucket[self.stage] = self.bucket.get(self.stage, 0.0) + elapsed
        return False
