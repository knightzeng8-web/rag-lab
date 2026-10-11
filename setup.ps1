<#
rag-lab 依赖安装脚本 —— 请在**你自己的** PowerShell 窗口里运行（管理员不必要，但也可以）。

关键点：venv 默认建在 **D:\rag-lab-venv**，也就是 DSH 文件沙箱**管不到**的目录。
原因：DSH 会给工作区（以及它派生的临时目录）打上一条
`Everyone Deny DeleteSubdirectoriesAndFiles` 的 ACL，pip 解包时需要重命名/删除临时文件，
在这种目录里必然报 `[Errno 13] Permission denied: ...whl.metadata`。
把 venv 放到沙箱之外，pip 就一切正常：既不需要 takeown，也不需要改任何一条 ACL。

用法：
    cd C:\Users\zeng\Documents\deepseek-harness\default-workspace\rag-lab
    powershell -ExecutionPolicy Bypass -File .\setup.ps1                       # CPU 版 torch（约 250MB，推荐先跑）
    powershell -ExecutionPolicy Bypass -File .\setup.ps1 -Torch cuda           # CUDA 版 torch（约 2.5GB，用上 RTX 4070）
    powershell -ExecutionPolicy Bypass -File .\setup.ps1 -Mirror               # pip 走清华源
    powershell -ExecutionPolicy Bypass -File .\setup.ps1 -VenvDir C:\raglab-venv

装完自检（两条都走项目里的 run.ps1，它会自动找到上面这个 venv）：
    .\run.ps1 -m unittest discover -s tests
    .\run.ps1 scripts/smoke_offline.py
#>
param(
    [ValidateSet("cpu", "cuda")] [string] $Torch = "cpu",
    [string] $VenvDir = "D:\rag-lab-venv",
    [switch] $Mirror,
    [switch] $SkipTests
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root

$BasePython = "C:\Users\zeng\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"
if (-not (Test-Path $BasePython)) {
    throw "找不到基础 Python：$BasePython（PATH 上的 python 是 Microsoft Store 占位符，不能用）"
}

# pip 缓存也放到项目外，避免再被沙箱 ACL 绊到
$env:PIP_CACHE_DIR = Join-Path $env:LOCALAPPDATA "raglab-pip-cache"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
New-Item -ItemType Directory -Force -Path $env:PIP_CACHE_DIR | Out-Null

# 中文输出在 GBK 控制台下会乱码：让 Python 以 UTF-8 写、控制台以 UTF-8 解码
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

if ($Mirror) {
    $env:PIP_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"
}

Write-Host "== 0/5 检查 venv 位置 ==" -ForegroundColor Cyan
if (-not (Test-Path (Split-Path $VenvDir -Qualifier))) {
    Write-Host "    $VenvDir 所在盘不存在，退回项目内 .venv" -ForegroundColor Yellow
    $VenvDir = Join-Path $Root ".venv"
}
Write-Host "    venv 目录：$VenvDir" -ForegroundColor Green

Write-Host "== 1/5 创建 venv ==" -ForegroundColor Cyan
$vpy = Join-Path $VenvDir "Scripts\python.exe"
$pipDir = Join-Path $VenvDir "Lib\site-packages\pip"
if ((-not (Test-Path $vpy)) -or (-not (Test-Path $pipDir))) {
    if (Test-Path $VenvDir) { Remove-Item -Recurse -Force $VenvDir -ErrorAction SilentlyContinue }
    & $BasePython -m venv $VenvDir
}
if (-not (Test-Path $vpy)) { throw "venv 创建失败：$vpy" }
& $vpy --version
& $vpy -m pip install -q --upgrade pip

Write-Host "== 2/5 核心依赖（numpy / pyyaml / pytest）==" -ForegroundColor Cyan
& $vpy -m pip install -q numpy pyyaml pytest

Write-Host "== 3/5 服务与更多语料格式 ==" -ForegroundColor Cyan
& $vpy -m pip install -q fastapi "uvicorn[standard]" pypdf python-docx

Write-Host "== 4/5 向量与重排 ==" -ForegroundColor Cyan
if ($Torch -eq "cuda") {
    Write-Host "    安装 CUDA 版 torch（约 2.5GB，请耐心等待）" -ForegroundColor Yellow
    & $vpy -m pip install torch --index-url https://download.pytorch.org/whl/cu124
} else {
    Write-Host "    安装 CPU 版 torch（约 250MB；本项目语料很小，CPU 完全够用）" -ForegroundColor Yellow
    & $vpy -m pip install torch --index-url https://download.pytorch.org/whl/cpu
}
# 顺序有讲究：torch 先装好，pip 就不会再用 PyPI 上的 CUDA 轮子（2.5GB）覆盖它
& $vpy -m pip install -q sentence-transformers faiss-cpu

Write-Host "== 5/5 校验 ==" -ForegroundColor Cyan
& $vpy -c @"
import importlib
for name in ('numpy', 'yaml', 'torch', 'sentence_transformers', 'faiss', 'fastapi', 'pypdf', 'docx'):
    try:
        mod = importlib.import_module(name)
        print(f'  OK   {name:22s} {getattr(mod, "__version__", "")}')
    except Exception as exc:
        print(f'  FAIL {name:22s} {exc}')
try:
    import torch
    print(f'  torch.cuda.is_available() = {torch.cuda.is_available()}')
except Exception:
    pass
"@

if (-not $SkipTests) {
    $env:HF_HOME = Join-Path $Root ".hf-cache"
    Write-Host "`n== 单元测试 ==" -ForegroundColor Cyan
    & $vpy -m unittest discover -s tests
    Write-Host "`n== 离线冒烟自检 ==" -ForegroundColor Cyan
    & $vpy scripts/smoke_offline.py
}

Write-Host "`n完成。日常命令统一走 .\run.ps1（自动找 venv、自动设置 HF_HOME）：" -ForegroundColor Green
Write-Host "  .\run.ps1 -m rag.cli ingest data/corpus --set embedder.provider=bge"
Write-Host "  .\run.ps1 -m rag.cli ask `"混合检索为什么要用 RRF 融合？`" --debug"
Write-Host "  .\run.ps1 -m rag.cli eval --tag baseline"
Write-Host "  .\run.ps1 -m rag.cli serve --port 8000"
