# -*- coding: utf-8 -*-
"""
audio_io.py —— 麦克风采集、扬声器播放、以及"没有硬件时"的模拟替身

【这一层的两个职责】
  采集（Mic）  ：把声卡的连续音频流切成固定长度的小块（每块 20 ms = 320 点），
                 交给上层一拍一拍地处理。为什么切块？因为 VAD、打断判断、
                 延迟统计都需要"边收边处理"，不能等一整段录完。
  播放（Speaker）：把 TTS 生成的音频排队送到声卡；**关键是能被随时清空**——
                 用户插话时要立刻停下来，这就是打断能力。

【为什么要写 sim 版本】
  1. 你现在这台机器可能没有麦克风权限/没装模型，sim 模式让你立刻跑通整条链路。
  2. 面试现场演示不了麦克风时，sim 也能跑（虽然没声音，但延迟表是真的）。
  3. 单元测试友好：脚本化音源可以精确构造"第 4 秒用户插话"这种场景。

【sim 的声音是什么】
  一段 220 Hz 的正弦波（模拟人声的能量），静音段是接近零的噪声。
  能量 VAD 能稳定分辨这两种块，所以整条逻辑（VAD→ASR→LLM→TTS→播放→打断）
  和真实情况完全一致，只是声音不是人说的。
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, Iterator, List, Optional, Tuple

import numpy as np


# =============================================================================
# 播放：一个可以被"立刻清空"的音频队列
# =============================================================================
@dataclass
class AudioChunk:
    """
    播放队列里的一段音频。

    pcm      : int16 单声道 numpy 数组
    seg_id   : 属于第几句话（一句话可能被切成多个 chunk）
    text     : 这句话的文本（打断后要写回对话历史，所以跟着音频走）
    index    : 这句话的第几个 chunk（0 开始）
    is_last  : 是不是这句话的最后一个 chunk
    """
    pcm: np.ndarray
    seg_id: int
    text: str
    index: int
    is_last: bool


@dataclass
class SegProgress:
    """一句话的播放进度——打断后决定"用户到底听到了多少"。"""
    seg_id: int
    text: str
    played_samples: int = 0
    total_samples: int = 0

    @property
    def fraction(self) -> float:
        if self.total_samples <= 0:
            return 0.0
        return min(1.0, self.played_samples / self.total_samples)


class BaseSpeaker:
    """
    播放器统一接口。

    on_start / on_end 是回调：
      on_start 第一帧音频真正送给声卡时调用（用于埋点 bot_speech_start）
      on_end   队列空了、这一轮说完了调用（用于埋点 bot_speech_end）
    用回调而不是直接 import tracer，是为了让音频层不依赖别的模块（解耦）。
    """

    def __init__(self, sample_rate: int, block_samples: int, volume: float = 1.0,
                 on_start: Optional[Callable[[], None]] = None,
                 on_end: Optional[Callable[[], None]] = None):
        self.sample_rate = sample_rate
        self.block_samples = block_samples
        self.volume = volume
        self.on_start = on_start
        self.on_end = on_end

        self._q: "queue.Queue[AudioChunk]" = queue.Queue()
        self._lock = threading.Lock()
        self._gen = 0                    # 打断代数：每次 flush +1，播放线程看到变化就丢弃
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._playing = False            # 是否正在出声
        self._segs: Dict[int, SegProgress] = {}
        self._cur_seg: Optional[int] = None

        # 打断时播放线程的"已停下"回执：flush() 会等它，用来测真实的打断延迟
        self._interrupted = threading.Event()
        self.last_flush_ms: float = 0.0

    # ------------------------------------------------------------------ 生命周期
    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, name="speaker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------ 入队
    def enqueue(self, chunk: AudioChunk) -> None:
        with self._lock:
            if chunk.seg_id not in self._segs:
                self._segs[chunk.seg_id] = SegProgress(chunk.seg_id, chunk.text)
            self._segs[chunk.seg_id].total_samples += len(chunk.pcm)
        self._q.put(chunk)

    def mark_seg_end(self, seg_id: int) -> None:
        """告诉播放器"这句话已经没有后续 chunk 了"，用于统计总长度。"""
        pass

    # ------------------------------------------------------------------ 打断
    def flush(self) -> List[Tuple[int, str, float]]:
        """
        【打断的核心】清空所有还没播的音频，返回"用户实际听到了什么"。

        返回：[ (seg_id, text, 播放比例), ... ]

        为什么必须返回这个？因为被打断的那句话如果原样写进对话历史，
        模型下一轮会以为自己说过用户没听到的内容 —— 表现就是"答非所问"。
        这是语音 Agent 最容易翻车、也最能体现工程功力的细节。

        另外：flush 会**等播放线程确认停下**（最多 300 ms），并把真实耗时记在
        self.last_flush_ms 里。因为"我发了停止指令"和"声音真的没了"是两回事，
        埋点要记后者，否则测出来的打断延迟永远是 0。
        """
        was_active = self._playing or not self._q.empty()
        t0 = time.perf_counter()

        with self._lock:
            self._interrupted.clear()
            self._gen += 1                       # 让正在写声卡的那一块立刻作废
            drained: List[AudioChunk] = []
            while True:                           # 清空队列
                try:
                    drained.append(self._q.get_nowait())
                except queue.Empty:
                    break
            played = []
            for seg_id, prog in self._segs.items():
                if prog.played_samples > 0:
                    played.append((seg_id, prog.text, prog.fraction))
            # 清空全部进度记录：每一次 flush 只汇报一次，避免下一轮重复统计
            self._segs = {}
        self._playing = False

        # 等播放线程发现 gen 变了、真的不再写声卡
        if was_active:
            self._interrupted.wait(timeout=0.3)
        self.last_flush_ms = (time.perf_counter() - t0) * 1000.0

        # 被打断也属于"本轮语音结束"，这里补一次 on_end，
        # 否则状态机会永远停在 BOT_SPEAKING，echo 门控也一直开着。
        if was_active and self.on_end:
            self.on_end()
        return played

    # ------------------------------------------------------------------ 状态
    @property
    def is_playing(self) -> bool:
        """正在出声 = 上一块还在写 或 队列里还有东西。"""
        return self._playing or not self._q.empty()

    # ------------------------------------------------------------------ 播放线程
    def _run(self) -> None:
        raise NotImplementedError

    # 子类只需要实现"怎么把一段 pcm 真正送出去"
    def _write(self, pcm: np.ndarray, gen: int) -> bool:
        """返回 True 表示写成功，False 表示被 flush 打断、应立即停止。"""
        raise NotImplementedError

    # ------------------------------------------------------------------ 通用播放循环
    def _pump(self) -> None:
        """
        所有播放器共用的节奏：
          取 chunk → 拆成 20 ms 小片 → 一片一片写 → 每片检查有没有被 flush
        拆小片的意义：flush 的响应时间 = 一个小片的时长（20 ms），
        如果整块 2 秒一起写，打断就会延迟 2 秒，体验直接崩。
        """
        while self._running:
            try:
                chunk = self._q.get(timeout=0.05)
            except queue.Empty:
                if self._playing:
                    self._playing = False
                    if self.on_end:
                        self.on_end()
                continue

            self._playing = True
            # 一轮回答只在第一块音频真正播出时上报一次 bot_speech_start
            if self.on_start and not self._started_this_round:
                self._started_this_round = True
                self.on_start()

            gen = self._gen
            seg = self._segs.get(chunk.seg_id)
            for i in range(0, len(chunk.pcm), self.block_samples):
                if not self._running or gen != self._gen:
                    # 被 flush 作废了：告诉 flush "我已经真的停下来"
                    self._interrupted.set()
                    break
                piece = chunk.pcm[i:i + self.block_samples]
                ok = self._write(piece, gen)
                if ok and seg is not None:
                    # 只有"真的写进去了"的样本才算用户听到了，
                    # 否则打断时算出的播放比例会偏大，写进历史的内容就多了
                    seg.played_samples += len(piece)
                if not ok:
                    # _write 返回 False = 这一片被 flush 作废了
                    self._interrupted.set()
                    break

            # 这句话播完了 → 允许下一句重新触发 on_start（新一轮回答）
            if chunk.is_last:
                self._started_this_round = False

    _started_this_round = False


class SoundDeviceSpeaker(BaseSpeaker):
    """
    真实扬声器。用 sounddevice 的 OutputStream。
    装包：pip install sounddevice  （Windows 上会自带 PortAudio，不需要额外编译）
    """

    def __init__(self, *args, device: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self._stream = None

    def start(self) -> None:
        import sounddevice as sd
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.block_samples,
            device=self.device,
        )
        self._stream.start()
        super().start()

    def stop(self) -> None:
        super().stop()
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass

    def _write(self, pcm: np.ndarray, gen: int) -> bool:
        if gen != self._gen:
            return False
        data = pcm.astype(np.float32) * self.volume
        data = np.clip(data, -32768, 32767).astype(np.int16)
        try:
            self._stream.write(data)   # 阻塞写，写完这一小块才返回
        except Exception:
            return False
        return gen == self._gen

    def _run(self) -> None:
        self._pump()


class SimSpeaker(BaseSpeaker):
    """
    模拟扬声器：不发声，但按真实时间等待，让延迟数据仍然有意义。

    注意：它**真的 sleep**（不是跳过），所以 sim 模式下测出来的时间轴和
    真实播放是一致的。想快速跑逻辑可以把 realtime=False（此时延迟数字不可信）。
    """

    def __init__(self, *args, realtime: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.realtime = realtime
        self._next_t = time.perf_counter()

    def _write(self, pcm: np.ndarray, gen: int) -> bool:
        if gen != self._gen:
            return False
        if self.realtime:
            dur = len(pcm) / self.sample_rate
            now = time.perf_counter()
            # 累加式节奏：保证总时长准确，不会因为调度抖动越跑越慢
            self._next_t = max(self._next_t, now) + dur
            delay = self._next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        return gen == self._gen

    def _run(self) -> None:
        self._pump()


# =============================================================================
# 采集：把麦克风切成 20 ms 的小块
# =============================================================================
class BaseMic:
    """麦克风统一接口：blocks() 是个生成器，一直往外吐 int16 音频块。"""

    def __init__(self, sample_rate: int, block_samples: int, gain: float = 1.0):
        self.sample_rate = sample_rate
        self.block_samples = block_samples
        self.gain = gain
        self._stop = threading.Event()

    def blocks(self) -> Iterator[np.ndarray]:
        raise NotImplementedError

    def stop(self) -> None:
        self._stop.set()

    # 统一的增益处理：麦克风太小声时不用改系统设置
    def _apply_gain(self, data: np.ndarray) -> np.ndarray:
        if abs(self.gain - 1.0) < 1e-6:
            return data
        x = data.astype(np.float32) * self.gain
        return np.clip(x, -32768, 32767).astype(np.int16)


class SoundDeviceMic(BaseMic):
    """
    真实麦克风。用 InputStream 的回调模式把数据推进队列，
    再在主线程里按块取出来——这样录音和 VAD 计算互不阻塞。
    """

    def __init__(self, *args, device: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self._q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=200)
        self._stream = None

    def open(self) -> None:
        import sounddevice as sd

        def callback(indata, frames, time_info, status):
            # 回调里不能做重活，只做拷贝和入队
            if self._stop.is_set():
                return
            mono = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
            try:
                self._q.put_nowait(mono)
            except queue.Full:
                pass   # 队列满了说明处理不过来，丢掉最旧的（这里简单丢弃最新的）

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.block_samples,
            device=self.device,
            callback=callback,
        )
        self._stream.start()

    def stop(self) -> None:
        super().stop()
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass

    def blocks(self) -> Iterator[np.ndarray]:
        if self._stream is None:
            self.open()
        while not self._stop.is_set():
            try:
                data = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            yield self._apply_gain(data)


class ScriptedMic(BaseMic):
    """
    脚本化音源：按一张"时间表"产出语音段和静音段。

    时间表 sim_schedule 里每项是 (开始秒, 结束秒)：
        在这些区间里输出"有声"的正弦波，区间外输出近乎静音。
    这样就能精确地在"机器人正在说话时"插入一段用户语音，稳定复现打断场景。
    """

    # 默认时间表（秒）。第 2 段特意安排在机器人第一轮回答的中间 → 触发打断。
    DEFAULT_SCHEDULE: List[Tuple[float, float]] = [
        (0.6, 2.4),      # 第 1 轮：用户完整说一句
        (4.2, 5.6),      # 第 2 轮：机器人正在说的时候插话 → 打断
        (10.5, 11.9),    # 第 3 轮：正常对话
        (15.0, 16.4),    # 第 4 轮
        (20.0, 21.2),    # 第 5 轮
    ]

    def __init__(self, *args, schedule: Optional[List[Tuple[float, float]]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.schedule = schedule or self.DEFAULT_SCHEDULE
        self._t0: Optional[float] = None

    @property
    def total_duration(self) -> float:
        return max(e for _s, e in self.schedule) + 2.0

    def _is_speech(self, t: float) -> bool:
        for s, e in self.schedule:
            if s <= t < e:
                return True
        return False

    def blocks(self) -> Iterator[np.ndarray]:
        self._t0 = time.perf_counter()
        n = self.block_samples
        i = 0
        while not self._stop.is_set():
            # 按真实时间推进：这一块应该在 t 时刻被采集
            target = self._t0 + i * (n / self.sample_rate)
            delay = target - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            t = i * (n / self.sample_rate)
            if t > self.total_duration:
                break

            if self._is_speech(t):
                # 220 Hz 正弦波 + 轻微幅度抖动，模拟说话时的能量起伏
                tt = np.arange(n) / self.sample_rate
                amp = 0.25 * (1.0 + 0.15 * np.sin(2 * np.pi * 3.0 * t))
                block = (amp * np.sin(2 * np.pi * 220.0 * tt) * 32767).astype(np.int16)
            else:
                # 静音不是绝对 0：加一点噪声更接近真实麦克风
                block = (np.random.randn(n) * 3.0).astype(np.int16)

            i += 1
            yield self._apply_gain(block)


# =============================================================================
# 工具：列出可用音频设备（第一次跑 mic 模式前先看这个）
# =============================================================================
def list_devices() -> None:
    import sounddevice as sd
    print(sd.query_devices())
    print("\n默认输入设备：", sd.query_devices(kind="input")["name"])
    print("默认输出设备：", sd.query_devices(kind="output")["name"])
    print("\n把 config.yaml 里 audio.mic_device / speaker_device 填成上面的编号即可指定设备。")
