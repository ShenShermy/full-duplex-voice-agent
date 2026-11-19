# -*- coding: utf-8 -*-
"""
vad.py —— 语音活动检测（Voice Activity Detection）

【它在整个系统里干什么】
麦克风每秒吐 50 个音频块。VAD 负责回答两个问题：
  1. 现在有人在说话吗？        → 用来判断"要不要开始录音"
  2. 这个人说完了吗？          → 用来判断"可以送去识别了"（这个叫 endpointing）

第二个问题才是语音 Agent 难的地方。判早了：用户还没说完就被打断思路；
判晚了：用户干等，体验像在跟一个反应迟钝的人讲话。

【两种实现，都留着】
  EnergyVAD  ：算音量（RMS）超过阈值就算说话。零依赖、一定能跑、嘈杂环境很笨。
  SileroVAD  ：小神经网络，抗噪强很多，但要 pip install silero-vad。
先用 energy 跑通，再换 silero 看差别——这个对比本身就是很好的面试素材。

【状态机】—— 理解这个就理解了 VAD
    SILENCE ──(连续 speech_frames 帧超阈值)──> SPEAKING
    SPEAKING ──(连续 silence 帧超阈值)──────> SILENCE
  关键：不是"一帧超阈值就算说话"，必须连续若干帧，否则咳嗽、敲键盘都会触发。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# --------------------------------------------------------------------------- 事件
@dataclass
class VadEvent:
    """
    VAD 一次判定的结果。

    kind:
        "start"  —— 用户开始说话了（已经满足 min_speech_ms）
        "end"    —— 用户说完了（静音已超过 silence_ms）
        None     —— 状态没变，什么都不用做

    at:
        事件真正发生的时刻（perf_counter）。注意是**判定成立的时刻**，
        不是"这一帧被处理完的时刻"，两者能差一帧（20 ms）。
    """
    kind: Optional[str]
    at: float


class BaseVAD:
    """VAD 统一接口。换实现只要保证 process() 的语义一致。"""

    def __init__(self, sample_rate: int, block_ms: int,
                 min_speech_ms: int, silence_ms: int):
        self.sample_rate = sample_rate
        self.block_ms = block_ms
        self.min_speech_ms = min_speech_ms
        self.silence_ms = silence_ms

        # 把"毫秒"换算成"多少帧"，状态机里只数帧数
        self._need_speech_frames = max(1, int(round(min_speech_ms / block_ms)))
        self._need_silence_frames = max(1, int(round(silence_ms / block_ms)))

        # 状态
        self._in_speech = False
        self._speech_run = 0      # 连续"像语音"的帧数
        self._silence_run = 0     # 连续"像静音"的帧数

        # 回声门控：机器人正在说话时把阈值抬高，避免被自己的声音触发
        self._echo_mode = False

    # ------------------------------------------------------------------ 开关
    def set_echo_mode(self, on: bool) -> None:
        """
        播放期间调用 set_echo_mode(True)。
        更彻底的做法是 AEC（回声消除），但那需要声卡级别的参考信号，
        Python 里做不好；抬高阈值 + 戴耳机是性价比最高的方案。
        """
        self._echo_mode = on

    # ------------------------------------------------------------------ 主流程
    def process(self, block: np.ndarray, at: float) -> VadEvent:
        """
        喂一帧音频，问它"现在什么情况"。

        block: int16 或 float32 的单声道音频，长度 = block_samples
        at   : 这一帧的采集时刻
        """
        is_speech = self.is_speech(block)

        if is_speech:
            self._speech_run += 1
            self._silence_run = 0
        else:
            self._silence_run += 1
            self._speech_run = 0

        # ---- 状态迁移 ----
        if not self._in_speech:
            # 需要连续 N 帧语音才算真的开始（防误触发）
            need = self._need_speech_frames
            if self._echo_mode:
                need = int(need * self.echo_min_speech_ratio())
            if self._speech_run >= need:
                self._in_speech = True
                return VadEvent("start", at)
            return VadEvent(None, at)

        # 已经在说话：判断有没有说完
        if self._silence_run >= self._need_silence_frames:
            self._in_speech = False
            return VadEvent("end", at)

        return VadEvent(None, at)

    # ------------------------------------------------------------------ 子类实现
    def is_speech(self, block: np.ndarray) -> bool:
        raise NotImplementedError

    def echo_min_speech_ratio(self) -> float:
        """播放期间"开始说话"的门槛倍数，子类可覆盖。"""
        return 2.0


# --------------------------------------------------------------------------- 能量 VAD
class EnergyVAD(BaseVAD):
    """
    音量阈值法。够用、够快、零依赖。

    原理：把一帧音频转成 float [-1, 1]，算均方根 RMS，超过阈值就算有声音。
    难点在阈值：安静房间 0.01 就够，嘈杂环境 0.05 都可能全是噪声。

    解法：自适应噪声底（energy_adaptive）
      持续记录"最近最安静的那批帧"的能量，把阈值设成 噪声底 × 倍数。
      这样换环境不用改配置。
    """

    def __init__(self, sample_rate: int, block_ms: int, min_speech_ms: int, silence_ms: int,
                 threshold: float = 0.015, adaptive: bool = True, adapt_ratio: float = 3.0,
                 echo_threshold_ratio: float = 2.5):
        super().__init__(sample_rate, block_ms, min_speech_ms, silence_ms)
        self.base_threshold = threshold
        self.adaptive = adaptive
        self.adapt_ratio = adapt_ratio
        self.echo_threshold_ratio = echo_threshold_ratio

        self._noise_floor = threshold   # 噪声底的估计值
        self._adapt_alpha = 0.02        # 噪声底的更新速度（越小越稳、越大越快跟上环境）

    def echo_min_speech_ratio(self) -> float:
        return 2.0

    # ------------------------------------------------------------------
    @staticmethod
    def rms(block: np.ndarray) -> float:
        """均方根：一句话的音量大小。int16 先归一化到 [-1,1]，float 直接用。"""
        x = np.asarray(block, dtype=np.float32)
        # int16 的取值范围是 ±32767，超出 [-1,1] 说明传进来的是整型样本，需要归一化
        if float(np.max(np.abs(x))) > 1.5:
            x = x / 32768.0
        return float(np.sqrt(np.mean(x * x) + 1e-12))

    def is_speech(self, block: np.ndarray) -> bool:
        r = self.rms(block)

        # ---- 自适应噪声底 ----
        # 只在"这一帧看起来像背景音"的时候更新，否则会出现一个致命的自激振荡：
        #   人一说话 → 帧能量很高 → 噪声底被抬高 → 阈值超过人的音量 → 永远判不出语音
        # （这个 bug 在测 barge-in 时真的踩到了：机器人说话时用户插话，噪声底从 0.0001
        #   一路爬到 0.13，阈值 3 倍就是 0.39，任何正常人声都越不过去。）
        # 判据：只有比当前噪声底大不了多少的帧才算"背景"，才允许把它往上带。
        if self.adaptive and not self._in_speech:
            if r < self._noise_floor * 2.0:
                self._noise_floor = (1 - self._adapt_alpha) * self._noise_floor + self._adapt_alpha * r
            # 兜底下限：防止长时间绝对安静导致噪声底趋近 0，阈值过敏感
            self._noise_floor = max(self._noise_floor, 1e-4)
            # 兜底上限：噪声底最多到基础阈值的 3 倍，再高一定是被自己的语音污染了
            self._noise_floor = min(self._noise_floor, self.base_threshold * 3.0)

        thr = self.base_threshold
        if self.adaptive:
            thr = max(self._noise_floor * self.adapt_ratio, self.base_threshold * 0.5)
        if self._echo_mode:
            thr *= self.echo_threshold_ratio    # 播放期间抬高门槛，压住自己的回声

        return r > thr


# --------------------------------------------------------------------------- Silero VAD
class SileroVAD(BaseVAD):
    """
    Silero VAD：一个很小的神经网络（约 2 MB），专门判断"这段音频像不像人说话"。
    抗噪比能量法强很多，代价是要装包、要跑一点点推理。

    安装：pip install silero-vad
    用法（官方推荐）：
        from silero_vad import load_silero_vad, VADIterator
        model = load_silero_vad()
        it = VADIterator(model, sampling_rate=16000)
    这里不用 VADIterator，而是自己管状态机，保持和 EnergyVAD 完全一致的接口——
    这样"换 VAD"对上层代码零侵入。
    """

    def __init__(self, sample_rate: int, block_ms: int, min_speech_ms: int, silence_ms: int,
                 echo_threshold_ratio: float = 2.5, threshold: float = 0.5):
        super().__init__(sample_rate, block_ms, min_speech_ms, silence_ms)
        self.echo_threshold_ratio = echo_threshold_ratio
        self.base_threshold = threshold
        try:
            from silero_vad import load_silero_vad  # type: ignore
        except Exception as e:  # pragma: no cover - 环境相关
            raise RuntimeError(
                "没装 silero-vad。运行： pip install silero-vad\n"
                "或者把 config.yaml 里的 vad.backend 改回 energy。"
            ) from e
        self._model = load_silero_vad()
        self._model.reset_states()

    def echo_min_speech_ratio(self) -> float:
        return 2.0

    def is_speech(self, block: np.ndarray) -> bool:
        import torch  # silero 依赖 torch

        x = block.astype(np.float32)
        if np.max(np.abs(x)) > 1.5:
            x = x / 32768.0
        tensor = torch.from_numpy(x)
        prob = float(self._model(tensor, self.sample_rate).item())
        thr = self.base_threshold
        if self._echo_mode:
            # 播放期间不只抬阈值，还要求概率更确定，双保险
            thr = min(0.95, thr * 1.3)
        return prob > thr


# --------------------------------------------------------------------------- 工厂
def build_vad(cfg, sample_rate: int) -> BaseVAD:
    """根据配置造一个 VAD。所有可选后端在这里汇总，方便扩展。"""
    common = dict(sample_rate=sample_rate, block_ms=cfg.audio.block_ms,
                  min_speech_ms=cfg.vad.min_speech_ms, silence_ms=cfg.vad.silence_ms)
    backend = cfg.vad.backend.lower()

    if backend == "energy":
        return EnergyVAD(
            threshold=cfg.vad.energy_threshold,
            adaptive=cfg.vad.energy_adaptive,
            adapt_ratio=cfg.vad.energy_adapt_ratio,
            echo_threshold_ratio=cfg.vad.echo_gate.threshold_ratio,
            **common,
        )
    if backend == "silero":
        return SileroVAD(echo_threshold_ratio=cfg.vad.echo_gate.threshold_ratio, **common)

    raise ValueError(f"未知的 vad.backend：{cfg.vad.backend}（可选 energy / silero）")
