# -*- coding: utf-8 -*-
"""
agent.py —— 对话状态机：把 VAD / ASR / LLM / TTS / 播放串成一个"会打断的"语音 Agent

这是整个项目最核心的一个文件。面试如果只问一个问题，大概率是：
    "你的全双工/打断是怎么实现的？"
答案就在下面这段状态机和 _on_barge_in() 里。

【线程模型】（先看懂这个，代码就好读了）

    主线程  ── 一直读麦克风 → VAD 判定 → 攒用户语音 → 触发一轮回答
       │
       └─ 每轮到回答时，起一个 worker 线程（_respond）：
            ASR → LLM(流式) → 切句 → TTS(流式) → 塞进播放队列

    播放线程 ── BaseSpeaker 内部，20 ms 一小片地往声卡写，随时可被 flush 清空

为什么主线程不能自己生成回答？
    因为生成一次要 1~3 秒。主线程一旦阻塞，麦克风就不读了 —— 用户这期间插话
    你根本听不见，"实时"就成了假的。把生成放到 worker，主线程永远在听。

【状态机】

    LISTENING ──VAD 检测到说话──▶ USER_SPEAKING ──VAD 判停──▶ THINKING ──首帧音频──▶ BOT_SPEAKING
        ▲                                                                              │
        │                                                                              │
        └──────────────── 播放结束 ────────────────────────────────────────────────┘
                                                                                       │
        USER_SPEAKING ◀── 用户在 BOT_SPEAKING/THINKING 时开口（= 打断 barge-in）──────┘

【打断处理的三步（顺序不能错）】
    1. stop_event.set()        → 让 LLM 的 token 循环和 TTS 合成循环立刻停手
    2. speaker.flush()         → 清空播放队列，返回"已经播出去的比例"
    3. history 只写播出去的部分 → 模型下一轮才不会以为自己说过用户没听到的话
"""
from __future__ import annotations

import threading
import time
from enum import Enum
from typing import List, Optional

import numpy as np

from .audio_io import AudioChunk
from .dialog import History, SentenceSplitter, played_text
from .tts import synth_to_speaker_rate
from .tracer import now


class State(str, Enum):
    """对话状态。用 Enum 而不是字符串，拼错就会立刻报错，不会静默出错。"""
    LISTENING = "listening"          # 在听，没人在说话
    USER_SPEAKING = "user_speaking"  # 用户正在说
    THINKING = "thinking"            # 用户说完了，我们在算
    BOT_SPEAKING = "bot_speaking"    # 机器人在说


class VoiceAgent:
    """
    参数说明（都是已经构造好的对象，方便替换成 mock 做测试）：
        cfg       : config.Config
        tracer    : 埋点，写 JSONL
        vad       : 语音活动检测
        mic       : 音频来源（真麦克风 / 脚本化音源）
        speaker   : 播放器（真扬声器 / 模拟）
        asr/llm/tts : 三个模型
    """

    def __init__(self, cfg, tracer, vad, mic, speaker, asr, llm, tts,
                 history: Optional[History] = None):
        self.cfg = cfg
        self.tracer = tracer
        self.vad = vad
        self.mic = mic
        self.speaker = speaker
        self.asr = asr
        self.llm = llm
        self.tts = tts
        self.history = history or History(max_turns=cfg.run.history_turns,
                                          max_chars=cfg.run.history_chars)

        self.sample_rate = cfg.audio.sample_rate
        self.block_samples = max(1, int(self.sample_rate * cfg.audio.block_ms / 1000))

        self.state = State.LISTENING
        self.turn = 0                     # 当前是第几轮（一次用户发言 = 一轮）
        self._seg_id = 0                  # 第几句机器人语音（一句话一个 seg_id）

        # 每轮一个 stop 事件：打断时 set()，工厂和合成循环都盯着它
        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._hist_lock = threading.Lock()
        self._assistant_committed = True  # 本轮 assistant 消息是否已写进历史

        # 打断候选（VAD 刚报 start 时先记一笔，持续够久才真的打断，避免被噪声误触发）
        self._barge_at: Optional[float] = None

        # 把播放器的"开始/结束"回调接到埋点和状态机
        speaker.on_start = self._on_bot_start
        speaker.on_end = self._on_bot_end
        self._turn_for_callback = 0
        self._bot_turn = 0

    # =========================================================================
    # 对外入口
    # =========================================================================
    def run(self) -> None:
        """主循环：读麦克风 → 判 VAD → 组织一轮对话。直到音源结束或 Ctrl+C。"""
        print(f"[agent] 采样率 {self.sample_rate} Hz，块长 {self.cfg.audio.block_ms} ms "
              f"（{self.block_samples} 点）")
        print(f"[agent] 最大轮数 {self.cfg.run.max_turns}，模式 {self.cfg.run.mode}")
        self.speaker.start()

        buf: List[np.ndarray] = []          # 当前这句用户语音的音频块
        max_utt_samples = int(self.cfg.vad.max_utterance_s * self.sample_rate)
        buffered = 0

        try:
            for block in self.mic.blocks():
                at = now()
                ev = self.vad.process(block, at)

                # ---------------- 用户开口 ----------------
                if ev.kind == "start":
                    # 轮数用完：又来了新的一句就收（已处理 max_turns 轮，不再接新的一轮）
                    if self.turn >= self.cfg.run.max_turns:
                        print(f"[agent] 已达最大轮数 {self.cfg.run.max_turns}，结束")
                        break

                    if self.state in (State.THINKING, State.BOT_SPEAKING):
                        # 机器人在说话/在想的时候用户开口 = 打断候选
                        if self.cfg.barge_in.enabled:
                            self._barge_at = at
                            self._begin_utterance(block, barge_candidate=True)
                            buf, buffered = [block], len(block)
                            continue
                        # 没开打断：忽略（当作回声），继续播
                        continue

                    if self.state == State.LISTENING:
                        self._begin_utterance(block)
                        buf, buffered = [block], len(block)
                        continue

                # ---------------- 打断确认 ----------------
                if self._barge_at is not None and self.state == State.USER_SPEAKING:
                    if ev.kind == "end":
                        # 判定为噪声/回声，取消这次打断候选（录音继续，但不算打断）
                        self._barge_at = None
                    else:
                        extra_ms = max(0, self.cfg.barge_in.min_speech_ms - self.cfg.vad.min_speech_ms)
                        if (at - self._barge_at) * 1000.0 >= extra_ms:
                            self._on_barge_in(at)
                            self._barge_at = None

                # ---------------- 用户继续说 ----------------
                if self.state == State.USER_SPEAKING:
                    if ev.kind != "start":
                        buf.append(block)
                        buffered += len(block)
                    if buffered >= max_utt_samples:
                        # 说太久了，强制截断，避免内存涨爆
                        print(f"[agent] 单句超过 {self.cfg.vad.max_utterance_s}s，强制截断")
                        ev_end = True
                    else:
                        ev_end = (ev.kind == "end")

                    if ev_end:
                        self.tracer.log(self.turn, "user_speech_end")
                        pcm = np.concatenate(buf) if buf else np.zeros(0, np.int16)
                        buf, buffered = [], 0
                        self._launch_respond(pcm)

                # ---------------- 轮数用完了就停 ----------------
                # 只在"空闲等用户说话"的时候才退，否则会把最后一轮的回答掐掉
                if self.turn >= self.cfg.run.max_turns and self.state == State.LISTENING:
                    print(f"[agent] 已达最大轮数 {self.cfg.run.max_turns}，结束")
                    break

        except KeyboardInterrupt:
            print("\n[agent] 收到 Ctrl+C，收尾…")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """收尾：停 worker、停播放、停麦克风。"""
        self._stop.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=3.0)
        try:
            self.mic.stop()
        except Exception:
            pass
        try:
            self.speaker.stop()
        except Exception:
            pass

    # =========================================================================
    # 一轮的开始 / 结束
    # =========================================================================
    def _begin_utterance(self, first_block: np.ndarray, barge_candidate: bool = False) -> None:
        """用户开口的第一帧：开新一轮、打点、切状态。"""
        self.turn += 1
        self._turn_for_callback = self.turn
        self.state = State.USER_SPEAKING
        self._assistant_committed = False
        # 注意：这里**不能**新建 self._stop。
        # self._stop 属于"正在生成回答的那一轮"，打断时要 set 的正是它；
        # 如果开新一轮就换一个新的 Event，打断就会 set 到新事件上，
        # 旧 worker 完全收不到信号、继续往播放队列里塞音频
        # （这个 bug 实测时表现为：打断了，日志也说停了，声音却接着响）。
        self._seg_id = 0
        self.tracer.log(self.turn, "user_speech_start")
        if barge_candidate:
            # 先记一笔"可能是打断"，真正确认在 _on_barge_in
            self.tracer.log(self.turn, "barge_candidate")
        _ = first_block

    def _launch_respond(self, pcm: np.ndarray) -> None:
        """用户说完了，把回答放到 worker 线程，主线程继续听。"""
        self.state = State.THINKING
        # 新一轮回答用自己的 stop 事件，并登记为"当前活跃事件"，供打断时 set
        self._stop = threading.Event()
        stop = self._stop
        turn = self.turn
        if self._worker and self._worker.is_alive():
            # 理论上不该发生（打断时会 set stop），保险起见等一下旧线程
            self._worker.join(timeout=2.0)
        self._worker = threading.Thread(
            target=self._respond, args=(pcm, turn, stop), name=f"respond-{turn}", daemon=True)
        self._worker.start()

    # =========================================================================
    # 回答生成（worker 线程）
    # =========================================================================
    def _respond(self, pcm: np.ndarray, turn: int, stop: threading.Event) -> None:
        """
        一轮完整回答：ASR → 写历史 → LLM 流式 → 切句 → TTS 流式 → 入播放队列。
        每个可能长耗时的地方都检查 stop，保证打断能在 20~50 ms 内生效。
        """
        try:
            # ---------- 1. ASR ----------
            text = self.asr.transcribe(pcm, self.sample_rate)
            self.tracer.log(turn, "asr_final")
            if stop.is_set():
                return
            if not text:
                print(f"[turn {turn}] 没识别出内容，跳过")
                if not stop.is_set():
                    self.state = State.LISTENING
                return
            print(f"\n[turn {turn}] 用户：{text}")

            with self._hist_lock:
                self.history.add_user(text)
                msgs = self.history.messages()

            # ---------- 2. LLM 流式 + 切句 ----------
            splitter = SentenceSplitter()
            full: List[str] = []          # 模型生成的全文（用于正常播完时写历史）
            spoken: List[str] = []        # 真正送去合成的话（调试用）
            first_token = True
            got_any_audio = False

            def play(sentence: str) -> bool:
                """把一句话合成并塞进播放队列。返回 False 表示被打断。"""
                nonlocal got_any_audio
                nonlocal_stop = stop
                if not sentence.strip():
                    return True
                self._seg_id += 1
                sid = self._seg_id
                for chunk in synth_to_speaker_rate(self.tts, sentence, self.sample_rate):
                    if nonlocal_stop.is_set():
                        return False
                    if not got_any_audio:
                        got_any_audio = True
                        # 首帧音频时间：这是"端到端首字延迟"的终点
                        self.tracer.log(turn, "tts_first_audio")
                    self.speaker.enqueue(AudioChunk(pcm=chunk, seg_id=sid, text=sentence,
                                                    index=0, is_last=False))
                spoken.append(sentence)
                return True

            try:
                for token in self.llm.stream(msgs):
                    if stop.is_set():
                        break
                    full.append(token)
                    if first_token:
                        first_token = False
                        self.tracer.log(turn, "llm_first_token")
                    for sentence in splitter.push(token):
                        if not play(sentence):
                            return
                if not stop.is_set():
                    for sentence in splitter.flush():
                        if not play(sentence):
                            return
            except Exception as e:
                print(f"[turn {turn}] LLM/合成出错：{e}")

            if not got_any_audio:
                # 一个字都没播出去：历史里不要留空 assistant
                with self._hist_lock:
                    self.history.drop_last_assistant()
                if not stop.is_set():
                    self.state = State.LISTENING
                return

            # ---------- 3. 收尾标记：告诉播放器"这一轮的消息发完了" ----------
            self.speaker.enqueue(AudioChunk(pcm=np.zeros(0, np.int16), seg_id=self._seg_id,
                                            text="", index=-1, is_last=True))

            if stop.is_set():
                return   # 已经被打断，历史由 _on_barge_in 写

            # ---------- 4. 等播完，把完整回答写进历史 ----------
            self._wait_speaker(stop, timeout=60.0)
            if not stop.is_set():
                with self._hist_lock:
                    if not self._assistant_committed:
                        self._assistant_committed = True
                        self.history.add_assistant("".join(full))
                print(f"[turn {turn}] 机器人：{''.join(full)}")
                self.state = State.LISTENING
        except Exception as e:
            import traceback
            print(f"[turn {turn}] 回答流程异常：{e}")
            traceback.print_exc()
            if not stop.is_set():
                self.state = State.LISTENING

    # =========================================================================
    # 打断（主线程调用）
    # =========================================================================
    def _on_barge_in(self, at: float) -> None:
        """
        用户真的插话了。这三步的顺序不能换：
            1) 停生成  2) 停播放并拿到"播了多少"  3) 按播出的部分写历史
        """
        print(f"\n[turn {self.turn}] ⚡ 检测到打断，停止播报")
        self.tracer.log(self.turn, "barge_in_detected", at=at)

        # 1) 让 LLM/TTS 循环立刻退出（set 的是"当前这一轮回答"的事件）
        self._stop.set()

        # 2) 清空播放队列，拿到每句话的播放比例
        progress = []
        try:
            progress = self.speaker.flush()
        except Exception as e:
            print(f"[agent] flush 出错：{e}")

        # flush() 内部会等播放线程确认"真的停了"，last_flush_ms 就是真实停声耗时
        stop_ms = round(getattr(self.speaker, "last_flush_ms", 0.0), 1)
        self.tracer.log(self.turn, "bot_audio_stopped",
                        ms=stop_ms, since_detect_ms=round((now() - at) * 1000.0, 1))

        # 3) 只把"用户真的听到的"写进历史
        heard = played_text(progress)
        with self._hist_lock:
            if not self._assistant_committed:
                self._assistant_committed = True
                if heard:
                    self.history.add_assistant(heard)
                    print(f"[turn {self.turn}] 已播出部分计为：{heard}")
                else:
                    self.history.drop_last_assistant()

    # =========================================================================
    # 播放回调 & 等待
    # =========================================================================
    def _on_bot_start(self) -> None:
        # 用户正在说话时不要把状态抢回 BOT_SPEAKING（打断后旧音频的余波会走到这里）
        if self.state != State.USER_SPEAKING:
            self.state = State.BOT_SPEAKING
        if self.cfg.vad.echo_gate.enabled:
            self.vad.set_echo_mode(True)      # 播放期抬高 VAD 阈值，防止自己的声音触发打断
        # 记住"这段语音属于哪一轮"：on_end 可能晚一点才来（打断时由 flush 触发），
        # 那时候 self.turn 已经变成下一轮了，用它就会把 bot_speech_dur pairs 打乱
        self._bot_turn = self._turn_for_callback
        self.tracer.log(self._bot_turn, "bot_speech_start")

    def _on_bot_end(self) -> None:
        if self.cfg.vad.echo_gate.enabled:
            self.vad.set_echo_mode(False)
        self.tracer.log(self._bot_turn, "bot_speech_end")
        if self.state == State.BOT_SPEAKING:
            self.state = State.LISTENING

    def _wait_speaker(self, stop: threading.Event, timeout: float = 60.0) -> None:
        """等播放队列自然播完（或被 stop 打断）。"""
        t0 = now()
        while not stop.is_set() and self.speaker.is_playing:
            if now() - t0 > timeout:
                print("[agent] 等待播放超时")
                return
            time.sleep(0.02)
