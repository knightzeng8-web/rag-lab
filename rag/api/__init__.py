"""服务层：零依赖实现 + FastAPI 可选实现。"""

from .simple_server import build_pipeline, make_handler, serve

__all__ = ["build_pipeline", "make_handler", "serve"]
