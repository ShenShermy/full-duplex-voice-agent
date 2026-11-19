# -*- coding: utf-8 -*-
"""
asr.py —— 语音转文字（ASR）

【两种模式，一定要理解它们的差别，面试必问】
  utterance（整段）: 等 VAD 判定"说完了"，把整段 2~3 秒音频丢给模型。
                    优点：稳、准、实现简单；缺点：延迟 = 音频时长的一部分 + 推理时间。
                    实际延迟通常 150~350 ms。
  streaming（流式） : 每 600 ms 就把当前缓存送去识别，边说边出字。
                    优点：能更早开始生成回复；缺点：接口对版本敏感、需要处理
                    "识别结果会反复改写"的问题（热词回退）。

本项目默认 utterance：先把系统跑通比追极致延迟重要，而 JD 里"流式音频"这个点
你可以在面试时讲清两者的取舍——这比嘴上说"我做过流式"更有说服力。

【兜底设计】
  funasr 装不上 / 模型没下载 → 自动降级到 mock，程序继续跑，只是识别结果是假的。
  这种"优雅降级"是工程项目的基本素养，也是你面试可以讲的一个点。
"""
from __future__ import annotations

import re
from typing import List, Optional

import numpy as np


class BaseASR:
    """ASR 统一接口：吃 int16 波形，吐文字。"""

    name = "base"

    def transcribe(self, pcm: np.ndarray, sample_rate: int = 16000) -> str:
        raise NotImplementedError

    def close(self) -> None:
        pass


# =============================================================================
# 0. 兜底：假识别
# =============================================================================
class MockASR(BaseASR):
    """
    sim 模式用。按顺序返回预设句子，让你不用麦克风也能把整条链路跑起来。

    注意：它假装"听"到了内容——真实系统里这是 ASR 的输出位置，
    所以埋点（asr_final）在它这里打，时间也是真的。
    """

    name = "mock"

    DEFAULT_UTTERANCES: List[str] = [
        "你好，你能介绍一下自己吗？",
        "等一下，先说说你的延迟是怎么测的。",
        "如果我说到一半就插话，你会怎么处理？",
        "那你的上下文会不会乱掉？",
        "好，谢谢，就这样。",
        "再说说你现在还在学什么。",
    ]

    def __init__(self, utterances: Optional[List[str]] = None, delay_ms: int = 60):
        self._texts = utterances or list(self.DEFAULT_UTTERANCES)
        self._i = 0
        self._delay_ms = delay_ms   # 模拟推理耗时，让延迟数据别太"假"

    def transcribe(self, pcm: np.ndarray, sample_rate: int = 16000) -> str:
        import time
        # 按音频长度模拟一点真实耗时（每 1 秒音频约 30 ms，GPU 大致量级）
        time.sleep(self._delay_ms / 1000.0 + len(pcm) / sample_rate * 0.03)
        text = self._texts[self._i % len(self._texts)]
        self._i += 1
        return text


# =============================================================================
# 1. 主力：FunASR（阿里，中文识别最强的一档）
# =============================================================================
class FunASRBackend(BaseASR):
    """
    FunASR 的 Python 封装。

    安装（GPU 版）：
        pip install funasr modelscope
        # 依赖 torch（见 requirements-gpu.txt）
    首次运行会自动从 ModelScope 下载模型（paraformer-zh 约 1 GB），
    下载路径默认在 ~/.cache/modelscope，想放项目里就设环境变量 MODELSCOPE_CACHE。

    关键 API（记住这两行，面试被问"你怎么接的"能直接说出来）：
        from funasr import AutoModel
        model = AutoModel(model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc")
        res = model.generate(input=wav_float32, batch_size_s=60)
        text = res[0]["text"]
    """

    name = "funasr"

    def __init__(self, model: str = "paraformer-zh", vad_model: str = "fsmn-vad",
                 punc_model: str = "ct-punc", device: str = "cuda"):
        try:
            from funasr import AutoModel  # type: ignore
        except Exception as e:  # pragma: no cover - 环境相关
            raise RuntimeError(
                "没装 funasr。运行： pip install funasr modelscope\n"
                "或者把 config.yaml 的 asr.backend 改成 mock 先跑通流程。"
            ) from e

        self.sample_rate = 16000
        kwargs = dict(model=model, device=device, disable_update=True, disable_pbar=True)
        # vad_model / punc_model 传空字符串表示不用（本项目 VAD 自己做了）
        if vad_model:
            kwargs["vad_model"] = vad_model
        if punc_model:
            kwargs["punc_model"] = punc_model
        self._model = AutoModel(**kwargs)

    def transcribe(self, pcm: np.ndarray, sample_rate: int = 16000) -> str:
        x = pcm.astype(np.float32)
        if np.max(np.abs(x)) > 1.5:
            x = x / 32768.0        # FunASR 要 float32 [-1,1]，不是 int16
        if len(x) < sample_rate * 0.1:   # 太短的音频直接丢掉，避免模型报错
            return ""
        res = self._model.generate(input=x, batch_size_s=60)
        if not res:
            return ""
        text = res[0].get("text", "") or ""
        return _clean(text)


# =============================================================================
# 2. 备选：faster-whisper（英文/多语种，部署最省心）
# =============================================================================
class FasterWhisperASR(BaseASR):
    """
    faster-whisper 是 CTranslate2 版 Whisper，比原版快 3~4 倍，pip 直接装。
    中文可用但不如 Paraformer；英文场景非常稳。作为"能跑起来的第二选择"保留。

        pip install faster-whisper
        model = WhisperModel("small", device="cuda", compute_type="float16")
        segments, info = model.transcribe(audio_float32)
    """

    name = "faster-whisper"

    def __init__(self, model_size: str = "small", device: str = "cuda", compute_type: str = "float16"):
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("没装 faster-whisper。 pip install faster-whisper") from e
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, pcm: np.ndarray, sample_rate: int = 16000) -> str:
        x = pcm.astype(np.float32)
        if np.max(np.abs(x)) > 1.5:
            x = x / 32768.0
        segments, _info = self._model.transcribe(x, language="zh", beam_size=1, vad_filter=False)
        return _clean("".join(s.text for s in segments))


# =============================================================================
# 工具
# =============================================================================
def _clean(text: str) -> str:
    """去掉 FunASR 常见的特殊标记（<unk>、[笑声] 之类）和多余空格。"""
    text = re.sub(r"<[^>]*>", "", text)
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_asr(cfg) -> BaseASR:
    """根据配置造 ASR。任何后端初始化失败都降级到 mock，保证程序能跑。"""
    backend = cfg.asr.backend.lower()
    try:
        if backend == "funasr":
            f = cfg.asr.funasr
            print(f"[asr] 加载 FunASR 模型 {f.model}（device={f.device}），首次运行会下载模型…")
            return FunASRBackend(model=f.model, vad_model=f.vad_model,
                                 punc_model=f.punc_model, device=f.device)
        if backend in ("faster-whisper", "whisper"):
            return FasterWhisperASR()
        if backend == "mock":
            return MockASR()
    except Exception as e:
        print(f"[asr] 初始化 {backend} 失败，降级到 mock。原因：{e}")
        return MockASR()
    raise ValueError(f"未知的 asr.backend：{cfg.asr.backend}（可选 mock / funasr / faster-whisper）")
