<#
rag-lab 命令包装器：自动找到依赖 venv 并运行，同时把 HF 缓存指到项目内。

venv 查找顺序：
    1. 环境变量 RAGLAB_VENV
    2. D:\rag-lab-venv            （setup.ps1 的默认位置，在 DSH 沙箱之外）
    3. 项目内 .venv

用法（如果执行策略拦住了 `.\run.ps1`，前面加 `powershell -ExecutionPolicy Bypass -File`）：
    .\run.ps1 -m unittest discover -s tests
    .\run.ps1 -m rag.cli ask "RRF 的常数 k 取多少？"
    .\run.ps1 scripts/smoke_offline.py
    .\run.ps1 -c "import torch; print(torch.cuda.is_available())"

实现要点（两个踩过的坑，写在这里避免再犯）：
1. **不声明 param()**：PowerShell 参数名支持前缀匹配，`param([string[]] $CommandArgs)` 会把
   python 的 `-c` 当成 `-CommandArgs` 吃掉；改用 `$args` 才能原样透传。
2. **管道结果必须用 @() 包住**：单元素管道会退化成标量字符串，`$x[0]` 取到的是"第一个字符"，
   于是 `D:\rag-lab-venv` 被截成 `D`，拼出 `D:\Scripts\python.exe` 这种荒谬路径。
#>

$Root = $PSScriptRoot
$CommandArgs = @($args)

if ($CommandArgs.Count -eq 0) {
    Write-Host "用法：.\run.ps1 <python 参数…>      例如 .\run.ps1 -m rag.cli info" -ForegroundColor Yellow
    exit 1
}

# 每个候选都用 @() 包成数组，避免单元素管道退化成字符串
$candidates = @(
    @(
        $env:RAGLAB_VENV,
        "D:\rag-lab-venv",
        (Join-Path $Root ".venv")
    ) | Where-Object { $_ -and (Test-Path (Join-Path $_ "Scripts\python.exe")) }
)

if ($candidates.Count -eq 0) {
    Write-Host "未找到依赖 venv，请先运行：" -ForegroundColor Red
    Write-Host "  powershell -ExecutionPolicy Bypass -File .\setup.ps1" -ForegroundColor Red
    exit 1
}
$venv = @($candidates)[0]
$vpy = Join-Path $venv "Scripts\python.exe"

# 模型权重写在项目内方便整体迁移；同时禁止往 venv 写 .pyc（沙箱外不可写，省掉无害告警）
$env:HF_HOME = Join-Path $Root ".hf-cache"
$env:PYTHONDONTWRITEBYTECODE = "1"

# 中文输出在 GBK 控制台下会乱码：让 Python 以 UTF-8 写、控制台以 UTF-8 解码
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

# 只有项目缓存里还没有模型时才建议镜像（已有缓存时避免多余的联网检查）
if (-not $env:HF_ENDPOINT -and -not (Test-Path (Join-Path $env:HF_HOME "hub\models--BAAI--bge-small-zh-v1.5"))) {
    $env:HF_ENDPOINT = "https://hf-mirror.com"
}

Set-Location $Root
Write-Host "[run] $vpy $($CommandArgs -join ' ')" -ForegroundColor DarkGray
& $vpy @CommandArgs
exit $LASTEXITCODE
