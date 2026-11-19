# -*- coding: utf-8 -*-
"""
main.py —— 程序入口：把各个零件组装起来，跑起来

【为什么组装代码要单独放一个文件】
因为每个零件（VAD/ASR/LLM/TTS）都不知道彼此的存在，只有这里知道"谁接谁"。
好处：
  1) 换后端只改 config.yaml，这个文件一行不动
  2) 想单独测某个零件，可以绕过这里直接 new 一个
  3) 面试讲架构时，你指着这个文件说"依赖注入都在这一处"，很清楚

跑法（在项目根目录）：
    python run.py                 # 按 config.yaml 跑（默认 sim 模式）
    python run.py --mode mic      # 用真实麦克风 + 扬声器
    python run.py --list-devices  # 先看声卡编号
    python run.py --max-turns 3   # 只跑 3 轮
"""
from __future__ import annotations

import argparse
import os
import sys

from .agent import VoiceAgent
from .asr import build_asr
from .audio_io import SimSpeaker, SoundDeviceMic, SoundDeviceSpeaker, ScriptedMic, list_devices
from .config import load_config
from .llm import build_llm
from .tracer import Tracer, summarize
from .tts import build_tts
from .vad import build_vad


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Full-duplex 语音 Agent Demo（级联：VAD→ASR→LLM→TTS）")
    p.add_argument("--config", default="config.yaml", help="配置文件路径（默认 config.yaml）")
    p.add_argument("--mode", choices=["sim", "mic"], default=None,
                   help="覆盖 config.yaml 里的 run.mode")
    p.add_argument("--max-turns", type=int, default=None, help="覆盖最大轮数")
    p.add_argument("--backend", choices=["mock", "real"], default=None,
                   help="快捷覆盖：mock=全假模型（只验证流程）；real=按 config 用真模型")
    p.add_argument("--list-devices", action="store_true", help="列出音频设备后退出")
    p.add_argument("--no-report", action="store_true", help="退出时不打延迟报告")
    return p.parse_args(argv)


def _ensure_utf8_stdout() -> None:
    """Windows 控制台默认 GBK，日志里的 ⚡ / ✅ 会抛 UnicodeEncodeError。

    run.py 里已经处理过一次；这里再兜一次，保证 `python -m src.main` 这种跑法也不会崩。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv=None) -> int:
    _ensure_utf8_stdout()
    args = parse_args(argv)

    if args.list_devices:
        list_devices()
        return 0

    if not os.path.exists(args.config):
        print(f"找不到配置文件 {args.config}，请在项目根目录运行（或 --config 指定路径）")
        return 2

    cfg = load_config(args.config)
    if args.mode:
        cfg.run.mode = "mic" if args.mode == "mic" else "sim"
    if args.max_turns is not None:
        cfg.run.max_turns = args.max_turns

    # --backend mock：把所有后端强制成假模型，用来先确认"链路是通的"
    if args.backend == "mock":
        cfg.asr.backend = "mock"
        cfg.llm.backend = "mock"
        cfg.tts.backend = "silent"

    sr = cfg.audio.sample_rate
    block = cfg.audio.block_samples

    # ---------------------------------------------------------------- 造零件
    vad = build_vad(cfg, sr)
    tracer = Tracer(cfg.run.events_path)

    if cfg.run.mode == "sim":
        mic = ScriptedMic(sample_rate=sr, block_samples=block, gain=cfg.audio.mic_gain)
    else:
        mic = SoundDeviceMic(sample_rate=sr, block_samples=block, gain=cfg.audio.mic_gain,
                             device=cfg.audio.mic_device)

    if cfg.run.mode == "sim":
        # sim 模式不真的放声（避免吵到人），但按真实时间推进 → 延迟数据仍然有效
        speaker = SimSpeaker(sample_rate=sr, block_samples=block,
                             volume=cfg.audio.speaker_volume, realtime=True)
    else:
        speaker = SoundDeviceSpeaker(sample_rate=sr, block_samples=block,
                                     volume=cfg.audio.speaker_volume,
                                     device=cfg.audio.speaker_device)

    asr = build_asr(cfg)
    llm = build_llm(cfg)
    tts = build_tts(cfg)

    print("=" * 70)
    print(" 语音 Agent Demo —— 级联管线（VAD → ASR → LLM → TTS → 可打断播放）")
    print("=" * 70)
    print(f" 模式      : {cfg.run.mode}（{'真实麦克风/扬声器' if cfg.run.mode == 'mic' else '脚本化音源，不发声'}）")
    print(f" 模型      : ASR={asr.name}  LLM={llm.name}  TTS={tts.name}  VAD={getattr(vad, 'name', '?')}")
    print(f" 打断      : {'开启' if cfg.barge_in.enabled else '关闭'}"
          f"（判据 {cfg.barge_in.min_speech_ms} ms 连续语音）")
    print(f" 上下文    : 最近 {cfg.run.history_turns} 轮 / {cfg.run.history_chars} 字，超了自动截断")
    print(f" 埋点文件  : {os.path.abspath(cfg.run.events_path)}")
    print("=" * 70)

    if cfg.run.mode == "mic" and tts.name == "silent":
        print("\n提示：当前 TTS 是 silent（不出声）。想听到声音改 config.yaml：")
        print("      tts:\n        backend: pyttsx3     # 零下载，Windows 自带\n")

    agent = VoiceAgent(cfg=cfg, tracer=tracer, vad=vad, mic=mic,
                       speaker=speaker, asr=asr, llm=llm, tts=tts)

    code = 0
    try:
        agent.run()
    except Exception as e:
        import traceback
        print(f"\n[main] 运行出错：{e}")
        traceback.print_exc()
        code = 1
    finally:
        tracer.close()
        for obj in (asr, llm, tts):
            try:
                obj.close()
            except Exception:
                pass

        # ---------------------------------------------------- 退出时打延迟报告
        if cfg.run.report_on_exit and not args.no_report:
            path = cfg.run.events_path
            if os.path.exists(path):
                print("\n" + "=" * 70)
                print(" 延迟报告")
                print("=" * 70)
                try:
                    from tools.latency import load, report  # type: ignore
                    turns = load(path)
                    md, _rows = report(turns)
                    print(md)
                except Exception:
                    # tools 不在 path 里时退回内置的轻量汇总
                    print(summarize(path))
                print()

    return code


if __name__ == "__main__":
    sys.exit(main())
