# -*- coding: utf-8 -*-
"""
tts.py —— 文字转语音（TTS）

【为什么按"句子"合成，而不是整段】
LLM 流式输出是一段一段的。如果等整段生成完再合成，首字延迟会增加 1~2 秒。
正确做法是：一边收 LLM 的 token，一边按句号/问号切句；一句凑好就立刻送去合成；
合成出来的音频立刻进播放队列。
这样"首帧音频时间" ≈ LLM 首 token + 第一句合成时间，而不是整段生成时间。

【三个后端，按"能跑通"优先级排列】
  silent   : 不发声，按字数估算时长，产出等长的静音 PCM。
             作用是把整条链路的时序跑通 —— 没有声卡、没有模型也能验证延迟埋点。
  pyttsx3  : Windows 自带 SAPI，pip 装完就能出声，零模型下载。演示够用。
  cosyvoice: 真正的神经 TTS，音色自然，是 JD 里那种产品的做法。
             需要装 CosyVoice 仓库 + 下载 0.5B 模型（约 2 GB），见 setup/install_cosyvoice.ps1

【统一约定】
  synth() 产出 int16 单声道 numpy 数组，采样率由 self.sample_rate 声明；
  上层（agent）负责把不同采样率重采样到声卡采样率。
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
import wave
from typing import Iterator, Optional

import numpy as np

# 项目根目录（src 的上一层），定位 third_party/CosyVoice 用
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# 采样率转换（不引入 scipy / librosa，pip 依赖越少越不容易装崩）
# =============================================================================
def resample_int16(pcm: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """
    线性插值重采样。语音场景够用（20k→16k、24k→16k 都只是缩一点带宽）。

    面试点：为什么不用 scipy.signal.resample_poly？
      —— 那个质量更好（带抗混叠滤波），但多一个依赖。工程上先保证装得上。
      生产环境建议改成 resample_poly，或用 48k 统一采样。
    """
    if src_rate == dst_rate or len(pcm) == 0:
        return pcm
    x = pcm.astype(np.float32)
    n_out = int(round(len(x) * dst_rate / src_rate))
    if n_out <= 0:
        return np.zeros(0, dtype=np.int16)
    # 在"源索引空间"上取等距点，再线性插值
    idx = np.linspace(0, len(x) - 1, n_out, dtype=np.float32)
    i0 = np.floor(idx).astype(np.int32)
    i1 = np.minimum(i0 + 1, len(x) - 1)
    w = idx - i0
    out = x[i0] * (1.0 - w) + x[i1] * w
    return np.clip(out, -32768, 32767).astype(np.int16)


def _fade_edges(pcm: np.ndarray, ms: int = 8, sample_rate: int = 16000) -> np.ndarray:
    """
    给每个音频片段首尾加极短的淡入淡出。

    为什么需要：直接拼接两段波形，边界会有一个突变，听感上是"咔"的一声。
    这是真实语音系统里必做的小细节，面试提到会很加分。
    """
    n = min(int(sample_rate * ms / 1000), len(pcm) // 2)
    if n <= 1:
        return pcm
    x = pcm.astype(np.float32)
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    x[:n] *= ramp
    x[-n:] *= ramp[::-1]
    return x.astype(np.int16)


# =============================================================================
# 统一接口
# =============================================================================
class BaseTTS:
    name = "base"
    sample_rate = 16000

    def synth(self, text: str) -> Iterator[np.ndarray]:
        """把一句话合成为若干 int16 音频块（流式 yield，块大小自定）。"""
        raise NotImplementedError

    def close(self) -> None:
        pass


# =============================================================================
# 0. 兜底：静音（只为把时序跑通）
# =============================================================================
class SilentTTS(BaseTTS):
    """
    按字数估算这句话该说多久，然后产出等长的静音。

    估算公式用实测经验值：中文自然语速约 5.5 字/秒（播客偏慢约 4.5，客服偏快约 6.5）。
    加 120 ms 固定开销模拟推理启动。

    它存在的意义不是"假"，而是让 sim 模式在**没有声卡、没有模型**的机器上
    也能产出有意义的端到端延迟数字（因为 SimSpeaker 是按真实时间播放的）。
    """

    name = "silent"

    def __init__(self, sample_rate: int = 16000, chars_per_sec: float = 5.5, chunk_ms: int = 100):
        self.sample_rate = sample_rate
        self._cps = chars_per_sec
        self._chunk = int(sample_rate * chunk_ms / 1000)

    def synth(self, text: str) -> Iterator[np.ndarray]:
        n_chars = max(1, len(text.strip()))
        duration = n_chars / self._cps + 0.12
        total = int(duration * self.sample_rate)
        pos = 0
        while pos < total:
            n = min(self._chunk, total - pos)
            # 不是全零：叠一点点极低幅度的噪声，方便你确认"确实是这段在播"
            yield np.zeros(n, dtype=np.int16)
            pos += n


# =============================================================================
# 1. 零下载：Windows SAPI（pyttsx3）
# =============================================================================
class Pyttsx3TTS(BaseTTS):
    """
    用 Windows 自带的语音引擎（SAPI5）合成。

    实现思路（面试可以这么讲）：
      pyttsx3 只能"直接播到扬声器"或"存成文件"，没有流式接口。
      所以这里：合成 → 写临时 WAV → 读回来 → 分块 yield。
      代价是每句多一次磁盘 IO（几毫秒，可忽略），换来的是零模型下载、零显存占用。

    中文音色：Windows 10/11 中文系统一般自带 "Huihui"（微软慧慧）。
    如果没装，说话会是英文腔调读中文 —— 去 设置→时间和语言→语音 里加一个中文语音包。
    """

    name = "pyttsx3"
    sample_rate = 22050     # SAPI 默认输出，读 WAV 时以文件为准

    def __init__(self, rate: int = 190, voice_hint: str = "huihui", chunk_ms: int = 100):
        self._rate = rate
        self._voice_hint = voice_hint.lower()
        self._chunk_ms = chunk_ms
        self._engine = None

    def _ensure_engine(self):
        """懒加载 + 每次重新初始化。SAPI 在同一个 engine 上多次 save_to_file 会出问题。"""
        import pyttsx3  # type: ignore
        engine = pyttsx3.init()
        engine.setProperty("rate", self._rate)
        for v in engine.getProperty("voices"):
            blob = f"{getattr(v, 'name', '')} {getattr(v, 'id', '')}".lower()
            if self._voice_hint and self._voice_hint in blob:
                engine.setProperty("voice", v.id)
                break
        return engine

    def synth(self, text: str) -> Iterator[np.ndarray]:
        engine = self._ensure_engine()
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        path = tmp.name
        tmp.close()
        try:
            engine.save_to_file(text, path)
            engine.runAndWait()
            with wave.open(path, "rb") as w:
                sr = w.getframerate()
                nch = w.getnchannels()
                raw = w.readframes(w.getnframes())
            pcm = np.frombuffer(raw, dtype=np.int16)
            if nch > 1:                      # 多声道取左声道
                pcm = pcm[::nch]
            self.sample_rate = sr
            pcm = _fade_edges(pcm, sample_rate=sr)
            step = max(1, int(sr * self._chunk_ms / 1000))
            for i in range(0, len(pcm), step):
                yield pcm[i:i + step]
        finally:
            try:
                os.remove(path)
            except OSError:
                pass


# =============================================================================
# 2. 正式方案：CosyVoice2（神经 TTS）
# =============================================================================
class CosyVoiceTTS(BaseTTS):
    """
    CosyVoice2-0.5B：阿里 FunAudioLLM 出品，中文自然度接近商用，0.5B 显存占用小。

    安装（一次性，较麻烦，见 setup/install_cosyvoice.ps1）：
        git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
        cd CosyVoice && pip install -r requirements.txt
        modelscope download --model iic/CosyVoice2-0.5B --local_dir pretrained_models/CosyVoice2-0.5B

    关键 API：
        from cosyvoice.cli.cosyvoice import CosyVoice2
        model = CosyVoice2("pretrained_models/CosyVoice2-0.5B", load_jit=False, fp16=False)
        for out in model.inference_sft(text, "中文女", stream=True):
            pcm = out["tts_speech"]      # torch.Tensor [1, T], float32, 24 kHz

    stream=True 的意义：CosyVoice 是"边说边生成"的（chunk 级流式），
    所以第一块音频出来得比整体合成完早很多 —— 这正是降低首字延迟的关键。
    """

    name = "cosyvoice"
    sample_rate = 24000     # CosyVoice2 输出 24 kHz

    def __init__(self, model_path: str, device: str = "cuda", voice: str = "中文女",
                 stream: bool = True, load_jit: bool = False):
        # CosyVoice 不是纯 pip 包：它要自己仓库里的代码 + 它内部的 Matcha-TTS 子模块。
        # 这里自动把 third_party/CosyVoice 和它的 Matcha-TTS 加进 sys.path，
        # 用户就不用自己设 PYTHONPATH 了（少一个"跑不起来"的理由）。
        _repo = os.path.join(ROOT, "third_party", "CosyVoice")
        for _p in (_repo, os.path.join(_repo, "third_party", "Matcha-TTS")):
            if os.path.isdir(_p) and _p not in sys.path:
                sys.path.insert(0, _p)

        try:
            from cosyvoice.cli.cosyvoice import CosyVoice2  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "没找到 cosyvoice 包。先跑 setup/install_cosyvoice.ps1，\n"
                "或把 config.yaml 的 tts.backend 改成 pyttsx3 / silent。"
            ) from e

        if not os.path.isdir(model_path):
            raise RuntimeError(f"模型目录不存在：{model_path}（先下载 CosyVoice2-0.5B）")

        print(f"[tts] 加载 CosyVoice2：{model_path}（device={device}）…")
        self._model = CosyVoice2(model_path, load_jit=load_jit, fp16=(device == "cuda"))
        self._voice = voice
        self._stream = stream

    def synth(self, text: str) -> Iterator[np.ndarray]:
        for out in self._model.inference_sft(text, self._voice, stream=self._stream):
            speech = out.get("tts_speech")
            if speech is None:
                continue
            # torch.Tensor [1, T] float32 in [-1, 1] → int16
            arr = speech.squeeze(0).detach().cpu().numpy()
            pcm = np.clip(arr * 32767.0, -32768, 32767).astype(np.int16)
            if len(pcm):
                yield pcm


# =============================================================================
# 工厂
# =============================================================================
def build_tts(cfg) -> BaseTTS:
    backend = cfg.tts.backend.lower()
    sr = cfg.audio.sample_rate
    try:
        if backend == "silent":
            return SilentTTS(sample_rate=sr)
        if backend == "pyttsx3":
            p = cfg.tts.pyttsx3
            return Pyttsx3TTS(rate=p.rate, voice_hint=p.voice_hint)
        if backend == "cosyvoice":
            c = cfg.tts.cosyvoice
            return CosyVoiceTTS(model_path=c.model_path, device=c.device,
                                voice=c.voice, stream=c.stream)
    except Exception as e:
        print(f"[tts] 初始化 {backend} 失败，降级到 silent。原因：{e}")
        return SilentTTS(sample_rate=sr)
    raise ValueError(f"未知的 tts.backend：{cfg.tts.backend}（可选 silent / pyttsx3 / cosyvoice）")


# =============================================================================
# 工具：把任意 TTS 输出对齐到声卡采样率
# =============================================================================
def synth_to_speaker_rate(tts: BaseTTS, text: str, target_rate: int) -> Iterator[np.ndarray]:
    """
    包装一层：TTS 输出什么采样率都行，这里统一转成声卡采样率再交给播放器。

    为什么要单独抽出来：CosyVoice 是 24 kHz，SAPI 是 22.05 kHz，声卡是我们配置的 16 kHz。
    不做统一，播放出来就是快放/慢放（音调都不对）。
    """
    for chunk in tts.synth(text):
        yield resample_int16(chunk, tts.sample_rate, target_rate)
