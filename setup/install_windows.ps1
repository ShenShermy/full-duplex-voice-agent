# =============================================================================
#  install_windows.ps1 —— 一键在 Windows 上把环境装好
#
#  用法（在 voice_agent_demo 目录下）：
#      powershell -ExecutionPolicy Bypass -File setup\install_windows.ps1
#
#  它做五件事：
#      1. 找到合适的 Python（3.10~3.12）
#      2. 建虚拟环境 .venv（不污染系统 Python）
#      3. 装 torch（CUDA 12.1 轮子，4060 Ti 用这个）+ 其它依赖
#      4. 装核心依赖（numpy，够跑 sim 模式）
#      5. 跑一遍 setup/verify_env.py 自检
#
#  参数：
#      -Cpu            只装 CPU 版 torch（没显卡 / 只想先跑通逻辑）
#      -SkipGpu        跳过 requirements-gpu.txt（只跑 sim）
#      -Mirror tuna    用清华源（默认）；传 none 用官方源
# =============================================================================
param(
    [switch]$Cpu,
    [switch]$SkipGpu,
    [string]$Mirror = "tuna"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host ""
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "  voice_agent_demo 环境安装" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "项目目录: $root"

# ---------------------------------------------------------------- 1. 找 Python
function Find-Python {
    # 优先用 py 启动器挑 3.10~3.12：3.13 上 torch/funasr 的轮子还不齐
    foreach ($v in @("3.12", "3.11", "3.10")) {
        try {
            $p = & py "-$v" -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $p) { return @{ Exe = "py"; Args = @("-$v"); Path = $p.Trim() } }
        } catch { }
    }
    foreach ($cand in @("python", "python3")) {
        try {
            $p = & $cand -c "import sys; print('%d.%d' % sys.version_info[:2]); print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $p) {
                $lines = @($p)
                if ($lines[0] -match "3\.(10|11|12)") {
                    return @{ Exe = $cand; Args = @(); Path = $lines[1].Trim() }
                }
            }
        } catch { }
    }
    return $null
}

$py = Find-Python
if (-not $py) {
    Write-Host "[X] 没找到 Python 3.10~3.12。" -ForegroundColor Red
    Write-Host "    去 https://www.python.org/downloads/windows/ 装 3.11 或 3.12，"
    Write-Host "    安装时勾上 'Add python.exe to PATH'，然后重开终端再来一次。"
    exit 1
}
Write-Host "[1/5] Python: $($py.Path)" -ForegroundColor Green

# ---------------------------------------------------------------- 2. 建 venv
$venv = Join-Path $root ".venv"
if (-not (Test-Path $venv)) {
    Write-Host "[2/5] 创建虚拟环境 .venv ..." -ForegroundColor Yellow
    if ($py.Exe -eq "py") { & py $py.Args[0] -m venv $venv } else { & $py.Exe -m venv $venv }
    if ($LASTEXITCODE -ne 0) { Write-Host "[X] 建虚拟环境失败" -ForegroundColor Red; exit 1 }
} else {
    Write-Host "[2/5] 复用已有虚拟环境 .venv" -ForegroundColor Green
}

$vpy = Join-Path $venv "Scripts\python.exe"
if (-not (Test-Path $vpy)) { Write-Host "[X] 找不到 $vpy" -ForegroundColor Red; exit 1 }

# ---------------------------------------------------------------- 3. 镜像
$indexArgs = @()
if ($Mirror -eq "tuna") {
    $indexArgs = @("-i", "https://pypi.tuna.tsinghua.edu.cn/simple")
    Write-Host "      使用清华镜像（下载快）。要换官方源: -Mirror none" -ForegroundColor DarkGray
}

Write-Host "[3/5] 升级 pip ..." -ForegroundColor Yellow
& $vpy -m pip install --upgrade pip @indexArgs
if ($LASTEXITCODE -ne 0) { Write-Host "[!] 升级 pip 失败，继续尝试" -ForegroundColor Yellow }

# ---------------------------------------------------------------- 4. torch
if (-not $SkipGpu) {
    if ($Cpu) {
        Write-Host "[4/5] 安装 CPU 版 torch ..." -ForegroundColor Yellow
        & $vpy -m pip install torch torchaudio
    } else {
        Write-Host "[4/5] 安装 torch 2.5.1 + CUDA 12.1（约 2.5 GB，耐心等）..." -ForegroundColor Yellow
        & $vpy -m pip install torch==2.5.1 torchaudio==2.5.1 `
            --index-url https://download.pytorch.org/whl/cu121
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[!] CUDA 版装失败，退回 CPU 版（先能跑通再说）" -ForegroundColor Yellow
            & $vpy -m pip install torch torchaudio @indexArgs
        }
    }

    Write-Host "      安装 requirements-gpu.txt ..." -ForegroundColor Yellow
    & $vpy -m pip install -r requirements-gpu.txt @indexArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[!] 部分包装失败（常见：bitsandbytes / comtypes）。" -ForegroundColor Yellow
        Write-Host "    sim 模式和 API 模式不受影响，可以继续。细节见 docs/04_排错与调参.md"
    }
} else {
    Write-Host "[4/5] 跳过 GPU 依赖（-SkipGpu）" -ForegroundColor DarkGray
}

Write-Host "      安装核心依赖 ..." -ForegroundColor Yellow
& $vpy -m pip install -r requirements-core.txt @indexArgs

# ---------------------------------------------------------------- 5. 自检
Write-Host "[5/5] 环境自检 ..." -ForegroundColor Yellow
& $vpy setup\verify_env.py

Write-Host ""
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "  装完了。下一步：" -ForegroundColor Cyan
Write-Host ""
Write-Host "  # 1) 先跑纯模拟，确认逻辑没问题（不需要模型/麦克风）"
Write-Host "  .\.venv\Scripts\python.exe run.py --backend mock"
Write-Host ""
Write-Host "  # 2) 再跑真麦克风 + 假模型，调 VAD 阈值"
Write-Host "  .\.venv\Scripts\python.exe run.py --mode mic --backend mock"
Write-Host ""
Write-Host "  # 3) 最后接真模型（先设 API key 最简单）"
Write-Host '  $env:DEEPSEEK_API_KEY = "sk-xxxx"'
Write-Host "  .\.venv\Scripts\python.exe run.py --mode mic --backend real"
Write-Host "==============================================" -ForegroundColor Cyan
