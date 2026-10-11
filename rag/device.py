"""计算设备选择。

为什么需要这一层：`sentence-transformers` / `CrossEncoder` 都接受 ``device`` 参数，但配置里
写死 ``cpu`` 会让装了 CUDA 版 torch 的机器白装显卡。这里把 ``auto`` 解析成实际可用的设备，
并把结论暴露给 ``/health``——演示时"我现在跑在 GPU 上"是可以现场验证的，而不是口头声称。

依赖缺失（没装 torch）时一律回落 ``cpu``，保证离线路径不受影响。
"""

from __future__ import annotations

import os
from typing import Any, Dict

__all__ = ["resolve_device", "device_report"]

VALID_DEVICES = ("auto", "cpu", "cuda", "mps")


def resolve_device(preferred: str = "auto") -> str:
    """把 ``auto`` 解析成实际可用设备；显式指定则原样返回（可用性交给上层报错）。"""
    preferred = str(preferred or "auto").strip().lower()
    if preferred and preferred != "auto":
        return preferred
    try:
        import torch  # type: ignore
    except Exception:
        return "cpu"
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    try:  # Apple Silicon
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def device_report() -> Dict[str, Any]:
    """给 ``/health`` 用的设备报告：不导入 torch 也能安全调用。"""
    report: Dict[str, Any] = {
        "selected": resolve_device("auto"),
        "torch": False,
        "cuda_available": False,
        "gpu": None,
        "reason": "",
    }
    try:
        import torch  # type: ignore
    except Exception as exc:
        report["reason"] = f"未安装 torch（{type(exc).__name__}），使用 CPU"
        return report
    report["torch"] = True
    report["torch_version"] = getattr(torch, "__version__", "")
    try:
        report["cuda_available"] = bool(torch.cuda.is_available())
        if report["cuda_available"]:
            report["gpu"] = torch.cuda.get_device_name(0)
            report["reason"] = "检测到可用 CUDA 设备"
        else:
            report["reason"] = "torch 已安装但未检测到 CUDA（可能是 CPU 版 torch）"
    except Exception as exc:  # pragma: no cover - 驱动异常
        report["reason"] = f"CUDA 检测失败：{exc}"
    report["env_cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return report
