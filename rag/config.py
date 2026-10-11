"""配置系统：默认值 <- 配置文件 <- 命令行 --set 覆盖。

优先级（后者覆盖前者）：
1. ``DEFAULTS``（本文件内置，保证任何情况下都能跑）
2. 配置文件（YAML 或 JSON，见 configs/default.yaml）
3. 命令行 ``--set a.b.c=value``（消融实验靠它切换组件，不必复制一堆配置文件）

所有相对路径都相对于 ``Config.root``（默认=项目根目录，即包含 pyproject.toml 的那层），
这样无论从哪个工作目录启动，行为都一致。
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = [
    "DEFAULTS",
    "Config",
    "load_config",
    "default_config",
    "find_project_root",
    "parse_scalar",
    "deep_set",
    "deep_merge",
]

DEFAULTS: Dict[str, Any] = {
    "project": "rag-lab",
    "paths": {
        "corpus": "data/corpus",
        "index": ".index",
        "eval": "data/eval/qa.jsonl",
        "reports": "reports",
        "web": "web",
    },
    "chunking": {
        # structural: 按 Markdown 标题 + 段落切分（推荐，保留语义边界）
        # fixed:      纯固定窗口 + 重叠
        # recursive:  段落 -> 句子 -> 硬切 的递归回退
        "strategy": "structural",
        "max_chars": 512,
        "overlap_chars": 96,
        "min_chars": 80,
    },
    "index": {
        # auto: 有 faiss 用 faiss，否则 numpy 暴力检索（语料 < 10 万块时完全够用）
        "backend": "auto",
        # 语料文件指纹变化时自动重建索引（换了向量模型或分块参数也一定会重建）
        "auto_rebuild": True,
    },
    "embedder": {
        # auto: 有 sentence-transformers 用 BGE，否则用内置哈希向量（离线可跑）
        "provider": "auto",
        "dim": 1024,
        "model": "BAAI/bge-small-zh-v1.5",
        # auto: 有可用 CUDA/MPS 就用，否则 CPU（装了 CUDA torch 却写死 cpu 会让显卡白装）
        "device": "auto",
        "batch_size": 32,
        "normalize": True,
        "api_base": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "api_model": "text-embedding-3-small",
        "timeout": 60,
    },
    "retrieval": {
        "top_k": 5,
        "top_n_vector": 20,
        "top_n_bm25": 20,
        "rrf_k": 60,
        "weights": {"vector": 1.0, "bm25": 1.0},
        "bm25": {"k1": 1.5, "b": 0.75},
        "rewrite": {
            # identity | keyword | multi_query | hyde | auto
            # auto = 有 LLM 时 multi_query，无 LLM 时 identity（不改写）。
            # 不要让 auto 在无 LLM 时退化成 keyword：实测 keyword 压缩是负收益
            # （BGE 基线 Recall@1 0.90→0.75、nDCG@10 1.000→0.920），需要时显式指定。
            "enabled": True,
            "strategy": "auto",
            "num_queries": 3,
            "max_keywords": 8,
        },
        "rerank": {
            # auto | none | lexical | cross_encoder | llm
            "enabled": False,
            "provider": "auto",
            "model": "BAAI/bge-reranker-base",
            "top_n": 20,
            "device": "auto",
        },
    },
    "generation": {
        # auto: 有 API Key 用真实 LLM，否则用抽取式兜底生成器
        "provider": "auto",
        "model": "gpt-4o-mini",
        "api_base": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "temperature": 0.2,
        "max_tokens": 800,
        "timeout": 90,
        "max_context_chunks": 6,
        "max_context_chars": 4000,
        "quote_chars": 240,
        "allow_refusal": True,
        # 离线抽取式生成器的拒答判据：单句查询词覆盖率与命中词数。
        # 这组值由 scripts/scan_refusal_threshold.py 在 12 题示例评测集上网格扫得，
        # 实测 12/12 全对（0.10/1 与 0.10/2 并列最优，取更严的 2）。
        # 但余量很小、必须如实记录：可回答题里最弱的 q08 覆盖率 0.100（**恰好压在阈值上**），
        # 不可回答题里最强的 q12 是 0.087，只差 0.013。评测集扩到 60 题以上后必须重扫。
        "extractive_max_sentences": 3,
        "extractive_min_coverage": 0.10,
        "extractive_min_hits": 2,
        "system_prompt": None,
    },
    "multihop": {
        "enabled": False,
        "max_rounds": 2,
        "bridge_terms": 3,
    },
    "eval": {
        "ks": [1, 3, 5, 10],
        "max_items": 0,
        "out_dir": "reports",
    },
    "logging": {"level": "INFO"},
}

_SENSITIVE_HINTS = ("key", "token", "secret", "password", "authorization")


def find_project_root(start: Optional[str] = None) -> str:
    """向上查找项目根目录（含 pyproject.toml 或 configs 目录）。"""
    candidates: List[Path] = []
    if start:
        candidates.append(Path(start).resolve())
    candidates.append(Path(__file__).resolve().parent.parent)
    candidates.append(Path.cwd())
    for base in candidates:
        node = base if base.is_dir() else base.parent
        for parent in [node, *node.parents]:
            if (parent / "pyproject.toml").is_file() or (parent / "configs").is_dir():
                return str(parent)
    return str(Path.cwd())


def parse_scalar(value: str) -> Any:
    """把命令行字符串解析成 bool / None / int / float / str / 内联 JSON。"""
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    lowered = stripped.lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if lowered in ("none", "null", ""):
        return None
    if (stripped.startswith("[") and stripped.endswith("]")) or (
        stripped.startswith("{") and stripped.endswith("}")
    ):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        pass
    return stripped


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def deep_set(target: Dict[str, Any], dotted: str, value: Any) -> None:
    """``deep_set(cfg, "retrieval.rerank.enabled", True)``。"""
    parts = [p for p in dotted.split(".") if p]
    if not parts:
        raise ValueError("覆盖项为空")
    node = target
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[parts[-1]] = value


def _mask(key: str, value: Any) -> Any:
    if isinstance(value, str) and value and any(h in key.lower() for h in _SENSITIVE_HINTS):
        return value[:3] + "***" if len(value) > 3 else "***"
    return value


def _read_config_file(path: str) -> Dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - 取决于环境
            raise RuntimeError(
                "读取 YAML 配置需要 pyyaml：pip install pyyaml（或把配置改成 JSON）"
            ) from exc
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise RuntimeError(f"配置文件顶层必须是映射（dict）：{path}")
    return data


class Config:
    """不可变语义的配置包装：读取用 ``get``/``section``，路径用 ``resolve``。"""

    def __init__(self, raw: Dict[str, Any], path: Optional[str] = None, root: Optional[str] = None):
        self.raw = raw
        self.path = path
        self.root = root or find_project_root()

    # ---- 读取 ----------------------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node if node is not None else default

    def section(self, name: str) -> Dict[str, Any]:
        value = self.raw.get(name)
        return dict(value) if isinstance(value, dict) else {}

    def merged(self) -> Dict[str, Any]:
        return copy.deepcopy(self.raw)

    def resolve(self, path: str) -> str:
        p = Path(path)
        if not p.is_absolute():
            p = Path(self.root) / p
        return str(p)

    def path_of(self, key: str) -> str:
        """取 paths.* 下的相对路径并解析成绝对路径。"""
        return self.resolve(str(self.get(f"paths.{key}", "")))

    def env(self, key: str, default: str = "") -> str:
        return os.environ.get(key, default)

    def describe(self, mask: bool = True) -> str:
        def walk(node: Any, prefix: str = "") -> List[str]:
            lines: List[str] = []
            if isinstance(node, dict):
                for key in sorted(node):
                    lines.extend(walk(node[key], f"{prefix}{key}."))
            else:
                key = prefix.rstrip(".")
                shown = _mask(key, node) if mask else node
                lines.append(f"{key} = {shown!r}")
            return lines

        header = f"# config file: {self.path or '(built-in defaults)'}\n# root: {self.root}"
        return header + "\n" + "\n".join(walk(self.raw))

    @classmethod
    def from_dict(
        cls,
        data: Optional[Dict[str, Any]] = None,
        path: Optional[str] = None,
        root: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "Config":
        merged = deep_merge(DEFAULTS, data or {})
        for dotted, value in (overrides or {}).items():
            deep_set(merged, dotted, value)
        return cls(merged, path=path, root=root or find_project_root())

    def with_overrides(self, overrides: Optional[Dict[str, Any]]) -> "Config":
        return Config.from_dict(self.raw, path=self.path, root=self.root, overrides=overrides)


def default_config(root: Optional[str] = None) -> Config:
    return Config.from_dict({}, root=root)


def load_config(
    path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    root: Optional[str] = None,
) -> Config:
    """加载配置。``path=None`` 时按顺序尝试 configs/default.yaml / .yml / .json。"""
    root = root or find_project_root()
    resolved: Optional[str] = None
    if path:
        candidate = path if os.path.isabs(path) else os.path.join(root, path)
        if not os.path.isfile(candidate):
            raise FileNotFoundError(f"配置文件不存在：{candidate}")
        resolved = candidate
    else:
        for name in ("configs/default.yaml", "configs/default.yml", "configs/default.json"):
            candidate = os.path.join(root, name)
            if os.path.isfile(candidate):
                resolved = candidate
                break

    data: Dict[str, Any] = {}
    if resolved:
        try:
            data = _read_config_file(resolved)
        except Exception as exc:
            if path:
                # 显式 --config 指定的文件解析失败必须报错：静默退回默认值会让
                # 「配置没生效」伪装成「实验跑成功了」，比直接失败危险得多
                raise
            # 自动发现的 configs/default.yaml 解析失败，最常见的原因是没装 pyyaml。
            # 此时退回内置 DEFAULTS（与 default.yaml 内容一致），
            # 保证「只装 numpy 就能跑通 CLI」这条主线不被配置解析拖死。
            print(
                f"[rag] 读取配置文件 {resolved} 失败（{exc}），本次使用内置默认配置。",
                file=sys.stderr,
            )
            data = {}
    return Config.from_dict(data, path=resolved, root=root, overrides=overrides)
