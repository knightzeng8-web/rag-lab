"""向量模型：统一接口 + 三种实现 + 自动降级。

- ``HashingEmbedder``：零依赖离线实现。用「CJK 二元组 + 拉丁词」做特征哈希（signed hashing trick，
  sublinear tf 加权），1024 维 L2 归一化。质量不如神经向量，但保证任何机器上都能跑通全链路、
  且结果确定可复现（评测结论不会因为随机性漂移）。
- ``BGEEmbedder``：sentence-transformers + BAAI/bge-small-zh-v1.5，中文检索的性价比之选。
- ``OpenAIEmbedder``：任何 OpenAI 兼容的 /v1/embeddings 服务（不依赖 openai SDK，纯 urllib）。

``provider=auto`` 的选择顺序：bge -> openai（有 Key 时）-> hashing。
任何降级都会写进日志和 ``status()``，面试演示时可以现场展示「拔掉依赖系统仍然工作」。
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..device import resolve_device
from ..text import stable_hash, tokenize

__all__ = [
    "Embedder",
    "HashingEmbedder",
    "BGEEmbedder",
    "OpenAIEmbedder",
    "build_embedder",
    "l2_normalize",
]


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class Embedder:
    """向量模型接口。子类只需实现 ``_encode_batch``。"""

    kind = "base"
    name = "base"
    available = True
    fallback_reason: Optional[str] = None

    def __init__(self, dim: int = 0, normalize: bool = True):
        self.dim = int(dim)
        self.normalize = bool(normalize)

    # ---- 子类实现 -----------------------------------------------------------
    def _encode_batch(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - 抽象
        raise NotImplementedError

    # ---- 对外 ---------------------------------------------------------------
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        matrix = np.asarray(self._encode_batch(texts), dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if self.dim and matrix.shape[1] != self.dim:
            self.dim = int(matrix.shape[1])
        return l2_normalize(matrix) if self.normalize else matrix

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def status(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "dim": self.dim,
            "available": bool(self.available),
            "fallback_reason": self.fallback_reason,
        }


class HashingEmbedder(Embedder):
    """离线确定性向量：signed hashing trick + sublinear tf。"""

    kind = "hashing"

    def __init__(self, dim: int = 1024, normalize: bool = True):
        # dim 会被用作取模的分母，必须 >= 1，否则 stable_hash 直接 ZeroDivisionError
        dim = max(1, int(dim))
        super().__init__(dim=dim, normalize=normalize)
        self.name = f"hashing-{dim}"

    def _encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: Dict[str, int] = {}
            for token in tokenize(text):
                counts[token] = counts.get(token, 0) + 1
            for token, tf in counts.items():
                index = stable_hash(token, self.dim)
                sign = 1.0 if stable_hash("s|" + token, 2) == 0 else -1.0
                matrix[row, index] += (1.0 + math.log(tf)) * sign
        return matrix


class BGEEmbedder(Embedder):
    """sentence-transformers 向量模型（默认 bge-small-zh-v1.5，CPU 上也可用）。"""

    kind = "bge"

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", device: str = "auto", batch_size: int = 32):
        super().__init__(dim=0, normalize=True)
        self.model_name = model_name
        # auto -> 有 CUDA/MPS 就用；名字里不带设备，避免 CPU/GPU 切换被误判成"换了模型"而重建索引
        self.device = resolve_device(device)
        self.batch_size = int(batch_size)
        self.name = f"bge:{model_name}"
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise RuntimeError("未安装 sentence-transformers：pip install sentence-transformers") from exc
        self._model = SentenceTransformer(self.model_name, device=self.device)
        try:
            self.dim = int(self._model.get_sentence_embedding_dimension())
        except Exception:  # pragma: no cover - 版本差异
            self.dim = 0
        return self._model

    def _encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        model = self._ensure_model()
        vectors = model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


class OpenAIEmbedder(Embedder):
    """任何 OpenAI 兼容的 /v1/embeddings 服务。"""

    kind = "openai"

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_base: str = "https://api.openai.com/v1",
        api_key: str = "",
        timeout: int = 60,
        batch_size: int = 64,
    ):
        super().__init__(dim=0, normalize=True)
        self.model_name = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.timeout = int(timeout)
        self.batch_size = int(batch_size)
        self.name = f"openai:{model}"

    def _encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        vectors: List[List[float]] = []
        texts = list(texts)
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            payload = json.dumps({"model": self.model_name, "input": batch}).encode("utf-8")
            request = urllib.request.Request(
                f"{self.api_base}/embeddings",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:  # pragma: no cover - 需要网络
                detail = exc.read().decode("utf-8", errors="ignore")[:400]
                raise RuntimeError(f"embedding 接口返回 {exc.code}：{detail}") from exc
            except Exception as exc:  # pragma: no cover - 需要网络
                raise RuntimeError(f"embedding 接口调用失败：{exc}") from exc
            for item in sorted(body.get("data", []), key=lambda d: d.get("index", 0)):
                vectors.append(list(item.get("embedding", [])))
        return np.asarray(vectors, dtype=np.float32)


def _st_importable() -> bool:
    try:
        import sentence_transformers  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def build_embedder(cfg, log=None) -> Embedder:
    """按配置构建向量模型，失败自动降级到离线哈希向量。"""
    provider = str(cfg.get("embedder.provider", "auto")).lower()
    dim = int(cfg.get("embedder.dim", 1024))
    normalize = bool(cfg.get("embedder.normalize", True))
    api_key = cfg.env(str(cfg.get("embedder.api_key_env", "OPENAI_API_KEY")), "")

    if provider in ("hashing", "hash", "offline"):
        return HashingEmbedder(dim=dim, normalize=normalize)

    if provider == "bge":
        try:
            embedder = BGEEmbedder(
                model_name=str(cfg.get("embedder.model", "BAAI/bge-small-zh-v1.5")),
                device=str(cfg.get("embedder.device", "auto")),
                batch_size=int(cfg.get("embedder.batch_size", 32)),
            )
            embedder.encode(["预热"])
            return embedder
        except Exception as exc:
            if log:
                log.warning("BGE 向量不可用（%s），降级为离线哈希向量", exc)
            fallback = HashingEmbedder(dim=dim, normalize=normalize)
            fallback.fallback_reason = f"bge 失败：{exc}"
            return fallback

    if provider == "openai":
        if not api_key:
            if log:
                log.warning("缺少 API Key（%s），降级为离线哈希向量", cfg.get("embedder.api_key_env"))
            fallback = HashingEmbedder(dim=dim, normalize=normalize)
            fallback.fallback_reason = "缺少 embedding API Key"
            return fallback
        embedder = OpenAIEmbedder(
            model=str(cfg.get("embedder.api_model", "text-embedding-3-small")),
            api_base=str(cfg.get("embedder.api_base", "https://api.openai.com/v1")),
            api_key=api_key,
            timeout=int(cfg.get("embedder.timeout", 60)),
        )
        try:
            embedder.encode(["预热"])
            return embedder
        except Exception as exc:
            if log:
                log.warning("OpenAI embedding 不可用（%s），降级为离线哈希向量", exc)
            fallback = HashingEmbedder(dim=dim, normalize=normalize)
            fallback.fallback_reason = f"openai embedding 失败：{exc}"
            return fallback

    # auto：优先本地神经向量，其次在线 embedding，最后离线哈希
    if _st_importable():
        try:
            embedder = BGEEmbedder(
                model_name=str(cfg.get("embedder.model", "BAAI/bge-small-zh-v1.5")),
                device=str(cfg.get("embedder.device", "auto")),
                batch_size=int(cfg.get("embedder.batch_size", 32)),
            )
            embedder.encode(["预热"])
            return embedder
        except Exception as exc:
            if log:
                log.warning("auto：BGE 加载失败（%s），继续尝试其它后端", exc)
    if api_key:
        try:
            embedder = OpenAIEmbedder(
                model=str(cfg.get("embedder.api_model", "text-embedding-3-small")),
                api_base=str(cfg.get("embedder.api_base", "https://api.openai.com/v1")),
                api_key=api_key,
                timeout=int(cfg.get("embedder.timeout", 60)),
            )
            embedder.encode(["预热"])
            return embedder
        except Exception as exc:
            if log:
                log.warning("auto：OpenAI embedding 不可用（%s），降级为离线哈希向量", exc)

    fallback = HashingEmbedder(dim=dim, normalize=normalize)
    fallback.fallback_reason = "auto：未检测到可用神经向量后端"
    if log:
        log.info("使用离线哈希向量 hashing-%d（无需下载模型）", dim)
    return fallback
