# PowerShell 脚本语法自检（纯 ASCII，避免自身也踩编码坑）
# 用法: powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\check_ps_syntax.ps1 -Files .\setup.ps1,.\run.ps1
param([string[]] $Files)

# 允许 -Files a.ps1,b.ps1 这种写法（powershell -File 会把整串当成一个参数）
$Files = $Files | ForEach-Object { $_ -split ',' } | Where-Object { $_ }
if (-not $Files -or $Files.Count -eq 0) {
    $Files = @((Join-Path $PSScriptRoot "..\setup.ps1"), (Join-Path $PSScriptRoot "..\run.ps1"))
}

$failed = 0
foreach ($f in $Files) {
    if (-not (Test-Path $f)) { Write-Output "MISSING $f"; $failed++; continue }
    $tokens = $null
    $errors = $null
    [System.Management.Automation.Language.Parser]::ParseFile((Resolve-Path $f).Path, [ref]$tokens, [ref]$errors) | Out-Null
    if ($errors -and $errors.Count -gt 0) {
        $failed++
        Write-Output "SYNTAX-ERROR $f"
        foreach ($e in $errors) {
            Write-Output ("  line {0}: {1}" -f $e.Extent.StartLineNumber, $e.Message)
        }
    } else {
        Write-Output ("OK  {0}  (PowerShell {1})" -f $f, $PSVersionTable.PSVersion.ToString())
    }
}

if ($failed -gt 0) { exit 1 }
exit 0
