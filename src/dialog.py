# -*- coding: utf-8 -*-
"""
dialog.py —— 对话历史管理（上下文）+ 流式切句

这个文件回答 JD 里"上下文保持"和"动态轮次"两个要求，也是打断功能能不能正确的
**核心**。请务必理解下面这段话，面试时能讲清楚它，你就赢了一半：

    用户打断时，模型已经"以为"自己说完了整段话，但用户只听到了一半。
    如果把这整段都写进历史，历史里就有一条"机器人说过的话"，而用户根本没听到。
    下一轮用户会问"你刚才说什么？"，模型指着一条用户没听过的内容回答 —— 对话就崩了。

    正确做法：只把**真正播放出去的部分**写进历史。
    这就是 audio_io.BaseSpeaker.flush() 为什么要返回 [(seg_id, text, fraction)] 的原因。

【另外两个细节】
  max_turns / max_chars：上下文不能无限长。语音场景下历史越长，首 token 越慢，
  所以按"最近 N 轮 + 总字数上限"双重截断，且优先保留最近一轮。
  SentenceSplitter：LLM 是逐 token 吐字的，但 TTS 需要"一句完整的话"才能合成出
  自然的语调。所以要在流式过程中边收边切句。
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

# 句子结束符：中文句号/问号/叹号/分号 + 英文对应符号，以及换行
_SENT_END = "。！？；!?;\n"
# 这些符号会让句子"继续"（比如"等等，……"里的逗号不算结尾）
_SOFT = "，,、：:"


class History:
    """
    对话历史。只存 user / assistant 两种角色，system prompt 由 llm 层单独注入。

    参数
      max_turns : 最多保留几轮（一轮 = 一问一答）
      max_chars : 历史总字数上限，超了就从最老的开始丢
    """

    def __init__(self, max_turns: int = 6, max_chars: int = 1200):
        self._msgs: List[Dict[str, str]] = []
        self._max_turns = max_turns
        self._max_chars = max_chars

    # ---------- 写 ----------
    def add_user(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        self._msgs.append({"role": "user", "content": text})
        self._trim()

    def add_assistant(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            # 完全没播出去（一开口就被打断）：不要留空消息，否则 API 会报错
            return
        self._msgs.append({"role": "assistant", "content": text})
        self._trim()

    def drop_last_assistant(self) -> None:
        """回滚最后一条 assistant（用于"生成失败、一个字都没播"的情况）。"""
        if self._msgs and self._msgs[-1]["role"] == "assistant":
            self._msgs.pop()

    # ---------- 读 ----------
    def messages(self) -> List[Dict[str, str]]:
        """返回给 LLM 的消息列表（副本，防止外部改坏内部状态）。"""
        return [dict(m) for m in self._msgs]

    def last_user(self) -> str:
        for m in reversed(self._msgs):
            if m["role"] == "user":
                return m["content"]
        return ""

    def __len__(self) -> int:
        return len(self._msgs)

    def __repr__(self) -> str:
        return f"<History {len(self._msgs)} msgs, {sum(len(m['content']) for m in self._msgs)} chars>"

    # ---------- 截断 ----------
    def _trim(self) -> None:
        # 1) 按轮数截断：保留最后 max_turns*2 条
        limit = self._max_turns * 2
        if len(self._msgs) > limit:
            self._msgs = self._msgs[-limit:]

        # 2) 按总字数截断：从最老的开始丢，但保证最后两条（最近一问一答）一定留着
        def total() -> int:
            return sum(len(m["content"]) for m in self._msgs)

        while total() > self._max_chars and len(self._msgs) > 2:
            self._msgs.pop(0)

        # 3) 丢掉开头可能残留的孤立 assistant（上一轮的回答），保证历史从 user 开始
        while self._msgs and self._msgs[0]["role"] != "user":
            self._msgs.pop(0)


class SentenceSplitter:
    """
    把流式 token 切成"可以送去合成"的句子。

    为什么不能简单地按 len 累加到 20 字就切：
      那样会切在词中间，TTS 读出来的语调是断的。
    为什么要设 min_len：
      模型经常先吐"好的，"，这种 3 个字的碎片单独合成，语调会很怪，
      所以攒够一定长度再切（默认 8 字）。

    用法：
        sp = SentenceSplitter()
        for token in llm.stream(msgs):
            for sentence in sp.push(token):
                play(sentence)
        for sentence in sp.flush():      # 别忘了最后剩下的半句
            play(sentence)
    """

    def __init__(self, min_len: int = 8, max_len: int = 60, soft_min_len: int = 12):
        self._buf = ""
        self._min = min_len
        self._max = max_len
        # 软断点长度：逗号/顿号/冒号这类停顿，攒够这么长也切
        self._soft = soft_min_len

    def push(self, token: str) -> List[str]:
        self._buf += token
        out: List[str] = []
        while True:
            idx = self._find_cut(self._buf)
            if idx < 0:
                break
            out.append(self._buf[:idx + 1].strip())
            self._buf = self._buf[idx + 1:]
        # 兜底：万一模型一直在吐逗号没有句号，攒到 max_len 也强制切，避免无限等
        while len(self._buf) >= self._max:
            out.append(self._buf[:self._max].strip())
            self._buf = self._buf[self._max:]
        return [s for s in out if s]

    def flush(self) -> List[str]:
        tail = self._buf.strip()
        self._buf = ""
        return [tail] if tail else []

    def _find_cut(self, buf: str) -> int:
        """返回应该在这个位置（含）截断的下标，-1 表示还不够长、先等着。"""
        for i, ch in enumerate(buf):
            if ch in _SENT_END:
                # 太短的句子先攒着（比如"好。"），但遇到换行就切
                if len(buf[:i + 1].strip()) >= self._min or ch == "\n":
                    return i
            elif ch in _SOFT and len(buf[:i + 1].strip()) >= self._soft:
                # 软断点：逗号/顿号/冒号。攒够 soft_min_len 就切，
                # 好处是首帧音频能早到 —— 等整句说完再合成，首字延迟会白白多几百毫秒。
                return i
        return -1


def played_text(progress: List[Tuple[int, str, float]]) -> str:
    """
    把播放器 flush() 返回的进度列表，还原成"用户实际听到的文字"。

    progress 里每一项是 (seg_id, text, fraction)：
      fraction = 该片段已经播出的比例（0~1）。1.0 表示整句播完了。
    被打断的那一句只算播出去的那部分字数 —— 这是"近似"，但足够用：
    语音播报速度是均匀的，所以"播出比例≈字数比例"。

    参数是 (seg_id, text, fraction) 三元的列表，按 seg_id 升序。
    """
    out: List[str] = []
    for _seg_id, text, frac in sorted(progress, key=lambda x: x[0]):
        if frac >= 0.999:
            out.append(text)
        elif frac > 0:
            n = max(1, int(round(len(text) * frac)))
            out.append(text[:n])
    joined = "".join(out).strip()
    # 收尾：如果最后一个字是逗号之类，去掉它，避免历史里出现半句逗号
    return joined.rstrip(_SOFT + "… ")
