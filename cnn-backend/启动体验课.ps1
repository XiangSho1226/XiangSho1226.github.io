$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonExe = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    $candidate = Join-Path $env:LOCALAPPDATA 'Programs/Python/Python312/python.exe'
    if (Test-Path -LiteralPath $candidate) { & $candidate -m venv .venv }
    elseif (Get-Command py -ErrorAction SilentlyContinue) { py -3 -m venv .venv }
    else { python -m venv .venv }
    if ($LASTEXITCODE -ne 0) { throw '请先安装 Python 3.10 或以上版本，再运行此脚本。' }
}
& $pythonExe -c 'import importlib.util, sys; sys.exit(0 if all(importlib.util.find_spec(n) is not None for n in ("torch", "fastapi", "uvicorn", "sklearn", "numpy")) else 1)'
if ($LASTEXITCODE -ne 0) {
    Write-Host '首次启动需要下载依赖，PyTorch CPU 安装包较大，请耐心等待。'
    & $pythonExe -m pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) { throw 'PyTorch 安装失败，请检查网络后重试。' }
    & $pythonExe -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw '依赖安装失败，请检查网络后重试。' }
}
Write-Host '请在浏览器打开 http://127.0.0.1:8000/ ，保持本窗口开启。'
& $pythonExe main.py
