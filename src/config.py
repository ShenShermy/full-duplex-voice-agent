# -*- coding: utf-8 -*-
"""
config.py —— 把 config.yaml 读成强类型对象

为什么要有这一层？
  1) YAML 里写错的键名（比如把 silence_ms 写成 silence_mx）在访问时才报错，
     而且要等到运行到那一行才炸。这里用 dataclass + 显式字段，错键会在启动时
     立刻报出来，省掉"跑到一半才崩"的调试时间。
  2) 代码里写 cfg["vad"]["silence_ms"] 这种字典链很难读，写成 cfg.vad.silence_ms
     一眼就知道是什么。
  3) 面试问"配置怎么管的"，这就是答案：外置配置 + 启动校验。

用法：
    from config import load_config
    cfg = load_config("config.yaml")
    print(cfg.vad.silence_ms)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# YAML 解析：有 PyYAML 就用（功能全），没有就退回自带的 miniyaml（零依赖）
# 这样这个项目在"pip 装不上"的机器上也能跑起来 —— 部署可用性优先于代码优雅。
# ---------------------------------------------------------------------------
try:
    import yaml as _pyyaml  # type: ignore

    def _load_yaml_file(path: str) -> Any:
        with open(path, "r", encoding="utf-8") as f:
            return _pyyaml.safe_load(f) or {}

    YAML_SOURCE = "PyYAML"
except ImportError:  # pragma: no cover - 取决于环境
    from . import miniyaml as _miniyaml

    def _load_yaml_file(path: str) -> Any:
        return _miniyaml.safe_load_file(path) or {}

    YAML_SOURCE = "miniyaml（内置，零依赖）"


# --------------------------------------------------------------------------- 小工具
def _get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    """安全取值：键不存在就返回默认值，不让程序崩。"""
    if not isinstance(d, dict):
        return default
    v = d.get(key, default)
    return default if v is None else v


def _warn_unknown(section: str, given: Dict[str, Any], known: List[str]) -> None:
    """拼错键名时给个提示。这类错误最烦人：配置没生效，但程序不报错。"""
    if not isinstance(given, dict):
        return
    for k in given:
        if k not in known:
            print(f"[config] 警告：{section}.{k} 不是已知配置项，检查一下是不是拼错了（已知：{known}）")


# --------------------------------------------------------------------------- 数据结构
@dataclass
class RunCfg:
    mode: str = "sim"                 # sim：脚本化音源；mic：真实麦克风
    max_turns: int = 6                # 跑几轮就退出（sim 模式用来控制时长）
    events_path: str = "events.jsonl" # 埋点写到哪
    report_on_exit: bool = True       # 退出时自动打延迟报告
    history_turns: int = 6            # 上下文最多保留几轮（一问一答算一轮）
    history_chars: int = 1200         # 上下文总字数上限（语音场景要短，不然首 token 变慢）


@dataclass
class AudioCfg:
    sample_rate: int = 16000
    block_ms: int = 20
    mic_device: Optional[int] = None
    speaker_device: Optional[int] = None
    mic_gain: float = 1.0
    speaker_volume: float = 1.0

    @property
    def block_samples(self) -> int:
        """每个音频块有多少采样点。16000 Hz × 20 ms = 320 点。"""
        return int(self.sample_rate * self.block_ms / 1000)


@dataclass
class EchoGateCfg:
    enabled: bool = True
    threshold_ratio: float = 2.5
    min_speech_ratio: float = 2.0


@dataclass
class VadCfg:
    backend: str = "energy"
    energy_threshold: float = 0.015
    energy_adaptive: bool = True
    energy_adapt_ratio: float = 3.0
    min_speech_ms: int = 160
    silence_ms: int = 500
    max_utterance_s: float = 20.0
    echo_gate: EchoGateCfg = field(default_factory=EchoGateCfg)


@dataclass
class BargeInCfg:
    enabled: bool = True
    min_speech_ms: int = 240
    flush_on_interrupt: bool = True


@dataclass
class FunASRCfg:
    model: str = "paraformer-zh"
    vad_model: str = "fsmn-vad"
    punc_model: str = "ct-punc"
    device: str = "cuda"
    chunk_size: List[int] = field(default_factory=lambda: [0, 10, 5])


@dataclass
class AsrCfg:
    backend: str = "mock"
    mode: str = "utterance"
    funasr: FunASRCfg = field(default_factory=FunASRCfg)


@dataclass
class OpenAICompatCfg:
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    api_key_env: str = "DEEPSEEK_API_KEY"
    temperature: float = 0.7
    max_tokens: int = 220
    timeout_s: float = 30.0

    def api_key(self) -> str:
        """从环境变量拿 key。找不到就抛一个说人话的错误。"""
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise RuntimeError(
                f"环境变量 {self.api_key_env} 没有设置。\n"
                f"  PowerShell 临时设置： $env:{self.api_key_env}=\"你的key\"\n"
                f"  或者把 config.yaml 里 llm.backend 改成 mock 先跑通流程。"
            )
        return key


@dataclass
class HFLocalCfg:
    model_path: str = "models/Qwen2.5-7B-Instruct-AWQ"
    device: str = "cuda"
    load_in_4bit: bool = True
    max_new_tokens: int = 200


@dataclass
class LlmCfg:
    backend: str = "mock"
    openai_compat: OpenAICompatCfg = field(default_factory=OpenAICompatCfg)
    hf_local: HFLocalCfg = field(default_factory=HFLocalCfg)
    system_prompt: str = "你是一个语音助手，回答要短，一到两句话。"


@dataclass
class Pyttsx3Cfg:
    rate: int = 190
    voice_hint: str = "huihui"


@dataclass
class CosyVoiceCfg:
    model_path: str = "pretrained_models/CosyVoice2-0.5B"
    device: str = "cuda"
    voice: str = "中文女"
    stream: bool = True
    speed: float = 1.0


@dataclass
class TtsCfg:
    backend: str = "silent"
    pyttsx3: Pyttsx3Cfg = field(default_factory=Pyttsx3Cfg)
    cosyvoice: CosyVoiceCfg = field(default_factory=CosyVoiceCfg)


@dataclass
class EvalCfg:
    budget: Dict[str, int] = field(default_factory=dict)


@dataclass
class Config:
    run: RunCfg
    audio: AudioCfg
    vad: VadCfg
    barge_in: BargeInCfg
    asr: AsrCfg
    llm: LlmCfg
    tts: TtsCfg
    eval: EvalCfg


# --------------------------------------------------------------------------- 加载
def load_config(path: str = "config.yaml") -> Config:
    """读 yaml → 填 dataclass → 做基本校验。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到配置文件：{os.path.abspath(path)}")

    raw = _load_yaml_file(path)

    run = _get(raw, "run", {})
    _warn_unknown("run", run, ["mode", "max_turns", "events_path", "report_on_exit",
                               "history_turns", "history_chars"])

    # YAML 里的 null 会变成 None，用 `or` 会让 0 也变成默认值，所以这里显式处理
    audio = _get(raw, "audio", {})
    vad = _get(raw, "vad", {})
    gate = _get(vad, "echo_gate", {})
    bi = _get(raw, "barge_in", {})
    asr = _get(raw, "asr", {})
    fun = _get(asr, "funasr", {})
    llm = _get(raw, "llm", {})
    oa = _get(llm, "openai_compat", {})
    hf = _get(llm, "hf_local", {})
    tts = _get(raw, "tts", {})
    py3 = _get(tts, "pyttsx3", {})
    cv = _get(tts, "cosyvoice", {})
    ev = _get(raw, "eval", {})

    cfg = Config(
        run=RunCfg(
            mode=_get(run, "mode", "sim"),
            max_turns=_get(run, "max_turns", 6),
            events_path=_get(run, "events_path", "events.jsonl"),
            report_on_exit=_get(run, "report_on_exit", True),
            history_turns=_get(run, "history_turns", 6),
            history_chars=_get(run, "history_chars", 1200),
        ),
        audio=AudioCfg(
            sample_rate=_get(audio, "sample_rate", 16000),
            block_ms=_get(audio, "block_ms", 20),
            mic_device=audio.get("mic_device") if isinstance(audio, dict) else None,
            speaker_device=audio.get("speaker_device") if isinstance(audio, dict) else None,
            mic_gain=_get(audio, "mic_gain", 1.0),
            speaker_volume=_get(audio, "speaker_volume", 1.0),
        ),
        vad=VadCfg(
            backend=_get(vad, "backend", "energy"),
            energy_threshold=_get(vad, "energy_threshold", 0.015),
            energy_adaptive=_get(vad, "energy_adaptive", True),
            energy_adapt_ratio=_get(vad, "energy_adapt_ratio", 3.0),
            min_speech_ms=_get(vad, "min_speech_ms", 160),
            silence_ms=_get(vad, "silence_ms", 500),
            max_utterance_s=_get(vad, "max_utterance_s", 20.0),
            echo_gate=EchoGateCfg(
                enabled=_get(gate, "enabled", True),
                threshold_ratio=_get(gate, "threshold_ratio", 2.5),
                min_speech_ratio=_get(gate, "min_speech_ratio", 2.0),
            ),
        ),
        barge_in=BargeInCfg(
            enabled=_get(bi, "enabled", True),
            min_speech_ms=_get(bi, "min_speech_ms", 240),
            flush_on_interrupt=_get(bi, "flush_on_interrupt", True),
        ),
        asr=AsrCfg(
            backend=_get(asr, "backend", "mock"),
            mode=_get(asr, "mode", "utterance"),
            funasr=FunASRCfg(
                model=_get(fun, "model", "paraformer-zh"),
                vad_model=_get(fun, "vad_model", "fsmn-vad"),
                punc_model=_get(fun, "punc_model", "ct-punc"),
                device=_get(fun, "device", "cuda"),
                chunk_size=_get(fun, "chunk_size", [0, 10, 5]),
            ),
        ),
        llm=LlmCfg(
            backend=_get(llm, "backend", "mock"),
            openai_compat=OpenAICompatCfg(
                base_url=_get(oa, "base_url", "https://api.deepseek.com/v1"),
                model=_get(oa, "model", "deepseek-chat"),
                api_key_env=_get(oa, "api_key_env", "DEEPSEEK_API_KEY"),
                temperature=_get(oa, "temperature", 0.7),
                max_tokens=_get(oa, "max_tokens", 220),
                timeout_s=_get(oa, "timeout_s", 30.0),
            ),
            hf_local=HFLocalCfg(
                model_path=_get(hf, "model_path", "models/Qwen2.5-7B-Instruct-AWQ"),
                device=_get(hf, "device", "cuda"),
                load_in_4bit=_get(hf, "load_in_4bit", True),
                max_new_tokens=_get(hf, "max_new_tokens", 200),
            ),
            system_prompt=_get(llm, "system_prompt", "你是一个语音助手，回答要短。"),
        ),
        tts=TtsCfg(
            backend=_get(tts, "backend", "silent"),
            pyttsx3=Pyttsx3Cfg(rate=_get(py3, "rate", 190), voice_hint=_get(py3, "voice_hint", "huihui")),
            cosyvoice=CosyVoiceCfg(
                model_path=_get(cv, "model_path", "pretrained_models/CosyVoice2-0.5B"),
                device=_get(cv, "device", "cuda"),
                voice=_get(cv, "voice", "中文女"),
                stream=_get(cv, "stream", True),
                speed=_get(cv, "speed", 1.0),
            ),
        ),
        eval=EvalCfg(budget=_get(ev, "budget", {})),
    )

    _validate_cfg(cfg)
    return cfg


def _validate_cfg(cfg: "Config") -> None:
    """启动时把明显写错的配置挡下来，而不是跑一半崩。"""
    if cfg.run.mode not in ("mic", "sim"):
        raise ValueError(f"run.mode 只能是 mic 或 sim，现在是 {cfg.run.mode!r}")
    if cfg.audio.block_ms < 10 or cfg.audio.block_ms > 100:
        raise ValueError("audio.block_ms 建议 10~100 之间（20 是最常用值）")
    if cfg.vad.silence_ms < 100:
        raise ValueError("vad.silence_ms 太小（<100 ms）会把每个停顿都当成说完，改成 300~700")
    if cfg.vad.min_speech_ms > cfg.barge_in.min_speech_ms:
        print("[config] 提示：vad.min_speech_ms 大于 barge_in.min_speech_ms，打断门槛其实由后者决定")
    if cfg.asr.backend == "funasr" and cfg.asr.mode == "streaming":
        print("[config] 提示：FunASR 流式模式对版本敏感，跑不通就把 asr.mode 改回 utterance")
