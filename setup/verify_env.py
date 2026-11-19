# -*- coding: utf-8 -*-
"""
verify_env.py —— 环境体检

在你笔记本上装完之后，**第一时间**跑这个，而不是直接跑 run.py。
它会逐项告诉你：缺什么、能不能用、下一步该干什么。

用法（项目根目录）：
    python setup/verify_env.py

设计说明（面试可以讲）：
    部署类项目的最大痛点是"装完了但不知道能不能跑"。所以把检查做成一个脚本，
    每一项都有明确的 ✅/⚠️/❌ 和修复建议 —— 这就是可运维性。
"""
import importlib
import os
import platform
import sys

OK, WARN, BAD = "✅", "⚠️ ", "❌"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def line(title):
    print("\n" + "-" * 68)
    print(title)
    print("-" * 68)


def _read_config_raw():
    """
    读 config.yaml，返回 (字典, 用的是哪个解析器)。

    策略和 src/config.py 完全一致：能用 PyYAML 就用，装不上就退回内置 miniyaml。
    所以在没装 PyYAML 的机器上，这里也不会误报"配置文件读不到"。
    """
    path = os.path.join(ROOT, "config.yaml")
    try:
        import yaml  # type: ignore
        with open(path, encoding="utf-8") as f:
            return (yaml.safe_load(f) or {}), "PyYAML"
    except ImportError:
        pass
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from src import miniyaml  # 内置兜底，零依赖
    with open(path, encoding="utf-8") as f:
        return (miniyaml.safe_load(f.read()) or {}), "miniyaml（内置，零依赖）"


def check_py():
    line("1. Python 解释器")
    v = sys.version_info
    print(f"   版本：{platform.python_version()}   ({sys.executable})")
    if v < (3, 10):
        print(f"   {BAD} 需要 3.10 及以上（3.10 是 torch / funasr 生态最稳的版本）")
        return False
    if v >= (3, 13):
        print(f"   {WARN}3.13+ 上 torch / funasr 的 wheel 可能还不全，建议用 3.10 或 3.11")
    else:
        print(f"   {OK} 版本合适")
    return True


def check_pkgs():
    """逐个 import 关键包，并给出装法。"""
    line("2. Python 依赖包")
    # (import 名, pip 名, 是否必需, 说明)
    # 注意：yaml **不是**必需——config.py 里有内置的 miniyaml 兜底（src/miniyaml.py），
    # 装不上 PyYAML 也能读配置。所以这里标 False，只提示。
    pkgs = [
        ("yaml", "PyYAML", False, "读 config.yaml（缺了用内置 miniyaml）"),
        ("numpy", "numpy", True, "全链路音频数组"),
        ("sounddevice", "sounddevice", False, "麦克风/扬声器（mic 模式必需）"),
        ("funasr", "funasr", False, "中文 ASR"),
        ("modelscope", "modelscope", False, "模型下载"),
        ("openai", "openai", False, "LLM（连 API 或本地服务）"),
        ("pyttsx3", "pyttsx3", False, "Windows 语音合成（零下载）"),
        ("torch", "torch", False, "深度学习运行时"),
        ("transformers", "transformers", False, "本地 LLM（可选）"),
        ("faster_whisper", "faster-whisper", False, "备选 ASR"),
        ("cosyvoice", "cosyvoice", False, "CosyVoice TTS（要单独装，见 setup/install_cosyvoice.ps1）"),
    ]
    missing_required, missing_optional = [], []
    for mod, pip_name, required, why in pkgs:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            print(f"   {OK} {mod:<16} {ver:<12} {why}")
        except Exception:
            tag = BAD if required else WARN
            print(f"   {tag} {mod:<16} {'未安装':<12} {why}")
            (missing_required if required else missing_optional).append(pip_name)

    if missing_required:
        print(f"\n   {BAD} 必需包缺失，先跑： python -m pip install -r requirements-core.txt")
    if missing_optional:
        print(f"\n   {WARN}可选包缺失（sim 模式不需要，跑真模型才要）：")
        print("       python -m pip install -r requirements-gpu.txt")
    return not missing_required


def check_torch_cuda():
    line("3. GPU / CUDA")
    try:
        import torch
    except Exception:
        print(f"   {WARN}没装 torch，跳过 GPU 检查（只跑 sim 模式可以不管）")
        return
    print(f"   torch：{torch.__version__}")
    if "+cu" not in torch.__version__:
        print(f"   {BAD} 这是 CPU 版 torch！重装：")
        print("       pip uninstall -y torch torchaudio")
        print("       pip install torch==2.5.1 torchaudio==2.5.1 "
              "--index-url https://download.pytorch.org/whl/cu121")
        return
    if not torch.cuda.is_available():
        print(f"   {BAD} torch 看不到 CUDA。常见原因：")
        print("       ① 显卡驱动太旧 → 更新 NVIDIA 驱动")
        print("       ② 装的是 CPU 版 torch（看版本号里有没有 +cu121）")
        return
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"   {OK} CUDA {torch.version.cuda} 可用")
    print(f"   {OK} 显卡：{name}，显存 {total:.1f} GB")
    if total < 7.5:
        print(f"   {WARN}显存偏小：建议 LLM 走 API，本地只留 ASR + TTS（见 docs/04）")


def check_audio():
    line("4. 音频设备")
    try:
        import sounddevice as sd
    except Exception:
        print(f"   {WARN}没装 sounddevice，mic 模式不可用（sim 模式不受影响）")
        return
    try:
        ins = sd.query_devices(kind="input")
        outs = sd.query_devices(kind="output")
        print(f"   {OK} 默认输入：{ins['name']}")
        print(f"   {OK} 默认输出：{outs['name']}")
    except Exception as e:
        print(f"   {BAD} 查不到默认设备：{e}")
        print("       插上耳机/麦克风，或在 Windows 设置里选好默认设备")


def check_models():
    line("5. 模型文件")
    cfg = {}
    try:
        cfg, _how = _read_config_raw()
    except Exception as e:
        print(f"   {WARN}读不到 config.yaml：{e}")

    cv_path = os.path.join(ROOT, ((cfg.get("tts") or {}).get("cosyvoice") or {}).get(
        "model_path", "pretrained_models/CosyVoice2-0.5B"))
    hf_path = os.path.join(ROOT, ((cfg.get("llm") or {}).get("hf_local") or {}).get(
        "model_path", "models/Qwen2.5-7B-Instruct-AWQ"))

    def look(label, path, hint):
        if os.path.isdir(path):
            n = sum(len(fs) for _r, _d, fs in os.walk(path))
            print(f"   {OK} {label}：{path}（{n} 个文件）")
        else:
            print(f"   {WARN}{label} 不存在：{path}")
            print(f"       {hint}")

    look("CosyVoice 模型", cv_path, "下载：python setup/download_models.py --cosyvoice")
    look("本地 LLM 模型", hf_path, "下载：python setup/download_models.py --llm（可选，也可用 API）")
    print(f"\n   提示：FunASR 模型首次运行会自动下载到 ~/.cache/modelscope，不需要手动准备。")


def check_config():
    line("6. 配置文件")
    path = os.path.join(ROOT, "config.yaml")
    if not os.path.exists(path):
        print(f"   {BAD} 找不到 {path}")
        return
    try:
        cfg, how = _read_config_raw()
        asr = (cfg.get("asr") or {}).get("backend")
        llm = (cfg.get("llm") or {}).get("backend")
        tts = (cfg.get("tts") or {}).get("backend")
        mode = (cfg.get("run") or {}).get("mode")
        print(f"   {OK} config.yaml 可读（解析器：{how}）")
        print(f"       run.mode={mode}  asr={asr}  llm={llm}  tts={tts}")
        if llm == "openai_compat":
            env = ((cfg.get("llm") or {}).get("openai_compat") or {}).get("api_key_env", "DEEPSEEK_API_KEY")
            if os.environ.get(env):
                print(f"   {OK} 环境变量 {env} 已设置")
            else:
                print(f"   {BAD} 环境变量 {env} 没设置（llm.backend=openai_compat 时会报错）")
                print(f'       PowerShell： $env:{env}="你的key"')
    except Exception as e:
        print(f"   {BAD} 解析失败：{e}")


def main():
    print("=" * 68)
    print(" 语音 Agent Demo —— 环境体检")
    print(f" 项目目录：{ROOT}")
    print("=" * 68)
    ok = check_py()
    ok = check_pkgs() and ok
    check_torch_cuda()
    check_audio()
    check_models()
    check_config()

    line("结论")
    if ok:
        print(f"   {OK} 基础环境没问题。接下来：")
        print("       ① 先验证代码：   python run.py --backend mock")
        print("       ② 想听声音：     config.yaml 里 tts.backend 改 pyttsx3")
        print("       ③ 用真实麦克风： config.yaml 里 run.mode 改 mic，然后 python run.py")
    else:
        print(f"   {BAD} 先补齐上面标记为 ❌ 的项，再继续。")
    print()


if __name__ == "__main__":
    main()
