# -*- coding: utf-8 -*-
"""
download_models.py —— 一键下载本地模型

【为什么单独写一个脚本】
模型都不小（FunASR 约 1 GB、CosyVoice2 约 1.5 GB、Qwen2.5-7B-AWQ 约 4.5 GB），
而且下载源有三种（ModelScope / HuggingFace / 模型自带下载）。
把它做成一个带参数的脚本，比在 README 里写一堆命令可靠得多：
少打错一个字，省半小时。

【用法】
    python setup/download_models.py --list              # 看有哪些模型、多大
    python setup/download_models.py --only asr          # 只下 ASR（最小，1 GB）
    python setup/download_models.py --only asr,tts      # ASR + TTS（推荐，约 2.5 GB）
    python setup/download_models.py --only llm          # 再下 7B（4.5 GB，8G 显存很紧）
    python setup/download_models.py --all               # 全下

    python setup/download_models.py --source hf         # 用 HuggingFace（需要能上外网）

【下完之后】
    模型会放在项目根目录的 models/ 下，config.yaml 里对应的 model_path 已经指好了：
        models/paraformer-zh
        models/CosyVoice2-0.5B
        models/Qwen2.5-7B-Instruct-AWQ
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")

# 每项：key -> (说明, 大约大小, 目标目录名, modelscope id, huggingface id)
CATALOG = {
    "asr": (
        "FunASR Paraformer-zh（中文识别主力）",
        "~840 MB",
        "paraformer-zh",
        "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "funasr/paraformer-zh",
    ),
    "vad": (
        "FunASR FSMN-VAD（语音端点检测，比能量 VAD 准）",
        "~2 MB",
        "fsmn-vad",
        "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        None,
    ),
    "punc": (
        "FunASR CT-Punc（自动加标点，切句要靠它）",
        "~290 MB",
        "ct-punc",
        "iic/punc_ct-transformer_cn-en-common-vocab471067-large",
        None,
    ),
    "tts": (
        "CosyVoice2-0.5B（中文语音合成，支持流式）",
        "~1.6 GB",
        "CosyVoice2-0.5B",
        "iic/CosyVoice2-0.5B",
        "FunAudioLLM/CosyVoice2-0.5B",
    ),
    "llm": (
        "Qwen2.5-7B-Instruct-AWQ（4bit 量化，约 5 GB 显存）",
        "~4.6 GB",
        "Qwen2.5-7B-Instruct-AWQ",
        "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "Qwen/Qwen2.5-7B-Instruct-AWQ",
    ),
}


def show_list() -> None:
    print("\n可用模型：\n")
    print(f"  {'key':<6}{'大小':<12}{'目录':<28}说明")
    print("  " + "-" * 84)
    for key, (desc, size, dirname, _ms, _hf) in CATALOG.items():
        print(f"  {key:<6}{size:<12}{dirname:<28}{desc}")
    print(f"\n下载目录：{MODELS_DIR}")
    print("建议顺序：--only asr → asr,tts → 都跑通了再加 llm\n")


def ensure_modelscope():
    try:
        import modelscope  # noqa: F401
        return True
    except ImportError:
        print("[!] 没装 modelscope，正在安装：pip install modelscope")
        rc = os.system(f'"{sys.executable}" -m pip install modelscope -q')
        if rc != 0:
            print("[X] 装 modelscope 失败，请手动执行： pip install modelscope")
            return False
        return True


def download_one(key: str, source: str) -> bool:
    desc, size, dirname, ms_id, hf_id = CATALOG[key]
    target = os.path.join(MODELS_DIR, dirname)

    if os.path.isdir(target) and os.listdir(target):
        print(f"[=] {key} 已存在，跳过：{target}")
        return True

    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"\n[{key}] {desc}（{size}）")
    print(f"      → {target}")

    if source == "hf":
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print("[X] 需要先装： pip install huggingface_hub")
            return False
        if not hf_id:
            print(f"[!] {key} 在 HuggingFace 上没有对应的仓库，请改用 --source ms")
            return False
        snapshot_download(repo_id=hf_id, local_dir=target, local_dir_use_symlinks=False)
        return True

    # 默认走 ModelScope（国内快，不用梯子）
    if not ensure_modelscope():
        return False
    from modelscope import snapshot_download  # type: ignore

    snapshot_download(ms_id, local_dir=target)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="下载 voice_agent_demo 需要的本地模型")
    ap.add_argument("--only", default="", help="逗号分隔的 key，例如 asr,tts")
    ap.add_argument("--all", action="store_true", help="全部下载")
    ap.add_argument("--list", action="store_true", help="列出可下载的模型")
    ap.add_argument("--source", default="ms", choices=["ms", "hf"],
                    help="ms=ModelScope（默认，国内快） / hf=HuggingFace")
    args = ap.parse_args()

    if args.list:
        show_list()
        return 0

    if args.all:
        keys = list(CATALOG.keys())
    elif args.only:
        keys = [k.strip() for k in args.only.split(",") if k.strip()]
    else:
        show_list()
        print("没指定要下什么。用 --only asr,tts 或 --all")
        return 1

    bad = [k for k in keys if k not in CATALOG]
    if bad:
        print(f"[X] 不认识的 key：{bad}")
        show_list()
        return 1

    ok = True
    for k in keys:
        try:
            if not download_one(k, args.source):
                ok = False
        except Exception as e:
            ok = False
            print(f"[X] {k} 下载失败：{type(e).__name__}: {e}")
            print("    常见原因：网络不通 / 磁盘空间不足 / 模型 id 变了。")
            print("    可以换源试试：--source hf")

    print("\n" + "=" * 60)
    if ok:
        print("全部完成。接着把 config.yaml 里对应的 backend 改成真模型，例如：")
        print("    asr.backend: funasr")
        print("    tts.backend: cosyvoice     # 需要先跑 setup/install_cosyvoice.ps1")
        print("    llm.backend: hf_local      # 或者 openai_compat 走 API")
    else:
        print("有模型没下成，往上翻日志看是哪一步。")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
