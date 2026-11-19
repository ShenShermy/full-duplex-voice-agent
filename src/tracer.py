# -*- coding: utf-8 -*-
"""
tracer.py —— 埋点与延迟日志

【为什么这个文件是整个项目最值钱的部分】
面试时你说"我做的语音 Agent 首字延迟 700 ms"，面试官会问"怎么测的？分项多少？"。
答不上来 = 这句话是编的。这个文件就是答案：每个环节打一个时间戳，落成 JSONL，
再用 tools/latency.py 出分项统计。

【事件字典】—— 全项目只用这几个名字，别乱改，latency.py 依赖它们
    user_speech_start    VAD 判定用户开始说话
    user_speech_end      endpointing 判定用户说完（延迟的起算点！）
    asr_final            ASR 返回最终转写
    llm_first_token      LLM 吐第一个 token
    tts_first_audio      第一帧 TTS 音频生成出来
    bot_speech_start     第一帧音频真正写进声卡
    bot_speech_end       机器人这一轮说完（播放队列空了）
    barge_in_detected    VAD 判定用户插话
    bot_audio_stopped    播放缓冲已经清空（真正停下来）

【时钟】
全项目必须用同一个时钟：time.perf_counter()（单调递增，不受系统改时间影响）。
绝对数值没有意义，有意义的是两个事件之差。
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, Optional


class Tracer:
    """线程安全的埋点记录器。"""

    def __init__(self, path: str):
        self.path = path
        # 每次启动清空旧日志：一次运行 = 一份数据，别把多次运行混在一起算 P90
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self._f = open(path, "w", encoding="utf-8")
        self._lock = threading.Lock()   # 采集线程和播放线程都会写，必须加锁
        self._t0 = time.perf_counter()

    # ------------------------------------------------------------------ 写
    def log(self, turn: int, event: str, t: Optional[float] = None, **extra: Any) -> None:
        """
        turn  : 第几轮对话（从 1 开始）。latency.py 按 turn 配对事件。
        event : 上面字典里的事件名。
        t     : 时间戳；不传就取当前时刻。允许外部传入，保证"事件发生的瞬间"和
                "记录的时刻"一致（比如 VAD 在音频回调里算出的时刻）。
        extra : 想额外记录的信息（比如识别的文本、打断时的播放进度），
                latency.py 会忽略，但排查问题时很有用。
        """
        ts = t if t is not None else time.perf_counter()
        rec = {"turn": int(turn), "event": event, "t": round(ts, 6)}
        if extra:
            rec.update(extra)
        with self._lock:
            self._f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._f.flush()     # 立刻落盘：就算程序崩了，已发生的事件也不丢

    # ------------------------------------------------------------------ 读
    def elapsed(self) -> float:
        """从启动到现在过了多少秒。"""
        return time.perf_counter() - self._t0

    def close(self) -> None:
        with self._lock:
            try:
                self._f.close()
            except Exception:
                pass

    def __enter__(self) -> "Tracer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def now() -> float:
    """
    统一取时刻的入口。

    为什么单独包一个函数？因为以后想换成更精确的时钟（比如声卡时间戳）时，
    只改这一个地方，不用全项目搜索。
    """
    return time.perf_counter()


def summarize(path: str) -> Dict[str, Any]:
    """
    轻量汇总（不依赖 tools/latency.py），用于程序退出时在终端打一行结论。
    真正的报告请看 tools/latency.py 输出的表格。
    """
    import statistics

    events: Dict[int, Dict[str, float]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            events.setdefault(r["turn"], {})[r["event"]] = r["t"]

    def delta(a: str, b: str, lo: int = 0):
        vals = []
        for turn, ev in events.items():
            if a in ev and b in ev and ev[b] >= ev[a]:
                vals.append((ev[b] - ev[a]) * 1000)
        return vals

    e2e = delta("user_speech_end", "tts_first_audio")
    barges = delta("barge_in_detected", "bot_audio_stopped")
    return {
        "rounds": len(events),
        "e2e_first_audio_ms": round(statistics.median(e2e), 1) if e2e else None,
        "barge_in_stop_ms": round(statistics.median(barges), 1) if barges else None,
        "barge_in_count": len(barges),
    }
