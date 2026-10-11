"""测试公共设施。

为什么不用 ``tempfile``：
``tempfile.mkdtemp()`` 以 ``0o700`` 创建目录，CPython 在 Windows 上会据此构造一个
只授权「当前用户」的 DACL；在受限令牌（沙箱、CI 低权限账户、部分容器）下，连创建者
自己都无法在该目录里再建子目录或写文件（``WinError 5`` / ``Errno 13``）。
本项目改成在项目内自建临时目录：不碰系统 TEMP、不受这类 ACL 限制，
测试在容器与受限环境里也能跑。
"""

from __future__ import annotations

import os
import shutil
from typing import Optional
from uuid import uuid4

__all__ = ["PROJECT_ROOT", "TempWorkspace"]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ROOT = os.environ.get("RAGLAB_TEST_TMP") or os.path.join(PROJECT_ROOT, ".testtmp")


class TempWorkspace:
    """临时工作目录（接口与 ``tempfile.TemporaryDirectory`` 对齐，便于替换）。"""

    def __init__(self, prefix: str = "case-", root: Optional[str] = None):
        base = root or DEFAULT_ROOT
        os.makedirs(base, exist_ok=True)
        self.name = os.path.join(base, f"{prefix}{uuid4().hex[:8]}")
        os.makedirs(self.name, exist_ok=True)

    def cleanup(self) -> None:
        shutil.rmtree(self.name, ignore_errors=True)

    def __enter__(self) -> str:
        return self.name

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.cleanup()
        return False
