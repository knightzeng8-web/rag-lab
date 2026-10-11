"""LLM 客户端：只实现 OpenAI 兼容的 /chat/completions，不依赖 openai SDK。

为什么不用官方 SDK：一是减少依赖体积（本项目「零依赖也能跑」的卖点），二是面试演示时
只要对方给一个 base_url + api_key 就能接上（本地 vLLM / Ollama / one-api 都能用），
三是纯标准库实现让「网络调用、超时、重试、错误处理」这些工程细节变得可见、可讲。

``provider=auto`` 时没有 API Key 就返回 ``None``，由上层切换到抽取式兜底生成器。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

__all__ = ["LLM", "OpenAICompatLLM", "build_llm", "llm_status"]


class LLM:
    kind = "null"
    name = "null"
    available = False
    fallback_reason: Optional[str] = None

    def complete(
        self,
        system: str,
        user: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:  # pragma: no cover - 抽象
        raise NotImplementedError

    def status(self) -> Dict[str, Any]:
        return {"kind": self.kind, "name": self.name, "available": bool(self.available)}


class OpenAICompatLLM(LLM):
    kind = "openai_compat"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_base: str = "https://api.openai.com/v1",
        api_key: str = "",
        temperature: float = 0.2,
        max_tokens: int = 800,
        timeout: int = 90,
        retries: int = 2,
    ):
        self.model_name = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout = int(timeout)
        self.retries = int(retries)
        self.name = f"openai:{model}"
        self.available = bool(api_key)

    def complete(
        self,
        system: str,
        user: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature if temperature is None else float(temperature),
            "max_tokens": self.max_tokens if max_tokens is None else int(max_tokens),
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_error: Optional[Exception] = None
        for attempt in range(max(1, self.retries)):
            request = urllib.request.Request(
                f"{self.api_base}/chat/completions",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                choices = data.get("choices") or []
                if not choices:
                    raise RuntimeError(f"接口未返回 choices：{str(data)[:200]}")
                message = choices[0].get("message") or {}
                content = message.get("content")
                if isinstance(content, list):  # 部分服务返回分段内容
                    content = "".join(
                        part.get("text", "") for part in content if isinstance(part, dict)
                    )
                return str(content or "").strip()
            except urllib.error.HTTPError as exc:  # pragma: no cover - 需要网络
                detail = exc.read().decode("utf-8", errors="ignore")[:300]
                last_error = RuntimeError(f"LLM 返回 {exc.code}：{detail}")
                if exc.code in (400, 401, 403, 404):
                    break
            except Exception as exc:  # pragma: no cover - 需要网络
                last_error = exc
            time.sleep(0.6 * (attempt + 1))
        raise RuntimeError(f"LLM 调用失败：{last_error}")


def build_llm(cfg, log=None) -> Optional[LLM]:
    provider = str(cfg.get("generation.provider", "auto")).lower()
    if provider in ("extractive", "offline", "none"):
        return None
    api_key = cfg.env(str(cfg.get("generation.api_key_env", "OPENAI_API_KEY")), "")
    if not api_key:
        if provider in ("openai", "llm"):
            if log:
                log.warning(
                    "generation.provider=%s 但环境变量 %s 为空，改用抽取式兜底生成器",
                    provider,
                    cfg.get("generation.api_key_env"),
                )
        return None
    return OpenAICompatLLM(
        model=str(cfg.get("generation.model", "gpt-4o-mini")),
        api_base=str(cfg.get("generation.api_base", "https://api.openai.com/v1")),
        api_key=api_key,
        temperature=float(cfg.get("generation.temperature", 0.2)),
        max_tokens=int(cfg.get("generation.max_tokens", 800)),
        timeout=int(cfg.get("generation.timeout", 90)),
    )


def llm_status(llm: Optional[LLM]) -> Dict[str, Any]:
    if llm is None:
        return {
            "kind": "extractive",
            "name": "extractive-offline",
            "available": False,
            "fallback_reason": "未配置可用的 LLM API Key",
        }
    return llm.status()
