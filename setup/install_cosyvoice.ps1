# =============================================================================
#  install_cosyvoice.ps1 —— 可选：装本地 TTS（CosyVoice2）
#
#  为什么是"可选"：
#    CosyVoice2 要用它自己仓库里的代码（不是纯 pip 包），还要 pynini / WeTextProcessing
#    这类在 Windows 上出了名难装的依赖。如果你只想先把整条链路跑通，
#    **用 pyttsx3（Windows 自带语音）或者干脆用 API 的 TTS 服务就够了**。
#
#  什么时候值得装：
#    面试要演示"本地全链路"、或者你确实需要一个自然的中文音色。
#
#  用法：
#      powershell -ExecutionPolicy Bypass -File setup\install_cosyvoice.ps1
#
#  成功后 config.yaml 改成：
#      tts:
#        backend: cosyvoice
#        cosyvoice:
#          model_path: models/CosyVoice2-0.5B
# =============================================================================
param(
    [string]$Dir = "third_party\CosyVoice",
    [switch]$SkipDeps
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$venvPy = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Write-Host "[X] 没找到 .venv\Scripts\python.exe，先跑 setup\install_windows.ps1" -ForegroundColor Red
    exit 1
}

Write-Host "=== 1) 检查 git ===" -ForegroundColor Cyan
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "[X] 没装 git。装一个：https://git-scm.com/download/win" -ForegroundColor Red
    exit 1
}

Write-Host "=== 2) clone CosyVoice ===" -ForegroundColor Cyan
if (Test-Path (Join-Path $Dir ".git")) {
    Write-Host "已存在，拉最新代码..." -ForegroundColor DarkGray
    Push-Location $Dir; git pull; Pop-Location
} else {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Dir) | Out-Null
    git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git $Dir
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[X] clone 失败。国内网络可以试镜像：https://gitee.com/mirrors/CosyVoice" -ForegroundColor Red
        exit 1
    }
}

Write-Host "=== 3) 装依赖 ===" -ForegroundColor Cyan
if (-not $SkipDeps) {
    $req = Join-Path $Dir "requirements.txt"
    if (Test-Path $req) {
        & $venvPy -m pip install -r $req -i https://pypi.tuna.tsinghua.edu.cn/simple
    }

    # WeTextProcessing / pynini 在 Windows 上经常要单独处理：
    #   先试 conda 装（最省事），失败就提示用户这步可以跳过（只影响数字/单位的口语化）
    Write-Host "      尝试装 WeTextProcessing（失败可跳过，只影响数字读法）..." -ForegroundColor Yellow
    & $venvPy -m pip install WeTextProcessing -i https://pypi.tuna.tsinghua.edu.cn/simple
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[!] WeTextProcessing 装失败 —— 不影响 CosyVoice 基本合成，先忽略。" -ForegroundColor Yellow
        Write-Host "    真想装的话，最稳的办法是用 conda： conda install -c conda-forge pynini" -ForegroundColor DarkGray
    }

    Push-Location $Dir
    & $venvPy -m pip install -e .
    Pop-Location
}

Write-Host "=== 4) 下模型 ===" -ForegroundColor Cyan
& $venvPy setup\download_models.py --only tts

Write-Host ""
Write-Host "=== 5) 自检 ===" -ForegroundColor Cyan
Push-Location $root
& $venvPy -c @"
import sys, os
sys.path.insert(0, os.path.join('third_party', 'CosyVoice'))
sys.path.insert(0, os.path.join('third_party', 'CosyVoice', 'third_party', 'Matcha-TTS'))
try:
    from cosyvoice.cli.cosyvoice import CosyVoice2
    print('[OK] 能 import CosyVoice2')
except Exception as e:
    print('[X] import 失败：', type(e).__name__, e)
    print('    多数是 PYTHONPATH 问题：CosyVoice 需要把 third_party/Matcha-TTS 也加进路径。')
p = os.path.join('models', 'CosyVoice2-0.5B')
print('[OK] 模型目录存在' if os.path.isdir(p) else '[!] 模型目录还没有：' + p)
"@
Pop-Location

Write-Host ""
Write-Host "完成。把 config.yaml 的 tts.backend 改成 cosyvoice，model_path 指向 models/CosyVoice2-0.5B。" -ForegroundColor Green
Write-Host "提示：src/tts.py 里的 CosyVoiceTTS 会自动把 third_party/CosyVoice 加进 sys.path，" -ForegroundColor DarkGray
Write-Host "      所以不用手动设 PYTHONPATH。" -ForegroundColor DarkGray
