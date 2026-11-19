#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""latency.py — 语音交互链路延迟测量与报告（框架无关，纯标准库）

用途：给一个 Full-duplex / 级联语音 Agent 的 demo 量出可写进简历的数字。
你只需要在 pipeline 的回调里埋点，把事件写成 JSONL，然后跑这个脚本。

事件（每行一个 JSON，t 为 time.time() 或 perf_counter 的秒数）：

    {"turn": 1, "event": "user_speech_start",  "t": 12.301}
    {"turn": 1, "event": "user_speech_end",    "t": 13.874}   # VAD 判停
    {"turn": 1, "event": "asr_final",          "t": 14.021}   # 最终转写返回
    {"turn": 1, "event": "llm_first_token",    "t": 14.402}
    {"turn": 1, "event": "tts_first_audio",    "t": 14.980}   # 第一帧音频送出
    {"turn": 1, "event": "bot_speech_start",   "t": 14.985}
    {"turn": 1, "event": "bot_speech_end",     "t": 18.220}
    {"turn": 2, "event": "user_speech_start",  "t": 16.500}   # 机器人在说时用户插话
    {"turn": 2, "event": "barge_in_detected",  "t": 16.640}   # VAD 触发打断
    {"turn": 2, "event": "bot_audio_stopped",  "t": 16.702}   # 播放侧真正停下

用法：
    python latency.py events.jsonl            # markdown 表
    python latency.py events.jsonl --json     # 机器可读

判读标准（真实产品级参考）：
    e2e_first_audio   首字延迟    < 800 ms 可用，< 500 ms 自然，> 1.5 s 明显卡顿
    barge_in_stop     打断停止    < 250 ms 才像人，> 500 ms 会被用户重复打断
    asr_latency       判停到转写  < 300 ms（流式 ASR）；整段 Whisper 通常 500 ms+
"""
import json
import sys
from collections import defaultdict


def load(path):
    turns = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            turns[ev["turn"]].setdefault(ev["event"], []).append(float(ev["t"]))
    return turns


def first(turns, turn, name):
    ts = turns.get(turn, {}).get(name)
    return min(ts) if ts else None


def deltas(turns, a, b):
    """b - a，同一 turn 内，只取两边都存在且 b > a 的样本。"""
    out = []
    for turn in turns:
        ta, tb = first(turns, turn, a), first(turns, turn, b)
        if ta is not None and tb is not None and tb >= ta:
            out.append((tb - ta) * 1000.0)
    return out


def barge_ins(turns):
    out = []
    for turn in turns:
        td = first(turns, turn, "barge_in_detected")
        ts = first(turns, turn, "bot_audio_stopped")
        if td is not None and ts is not None and ts >= td:
            out.append((ts - td) * 1000.0)
    return out


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


METRICS = [
    ("asr_latency", "判停→最终转写", ("user_speech_end", "asr_final"), 300),
    ("llm_ttft", "转写→首个 token", ("asr_final", "llm_first_token"), 400),
    ("tts_latency", "首 token→首帧音频", ("llm_first_token", "tts_first_audio"), 300),
    ("e2e_first_audio", "判停→首帧音频（端到端首字延迟）", ("user_speech_end", "tts_first_audio"), 800),
    ("bot_speech_dur", "机器人单轮语音时长", ("bot_speech_start", "bot_speech_end"), None),
]


def report(turns):
    rows = []
    for key, label, (a, b), budget in METRICS:
        xs = deltas(turns, a, b)
        rows.append((key, label, budget, xs))
    barges = barge_ins(turns)
    rows.append(("barge_in_stop", "打断检测→音频真正停止", 250, barges))

    lines = [
        "| 指标 | 说明 | n | 均值 | P50 | P90 | 最大 | 预算 | 结论 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key, label, budget, xs in rows:
        if not xs:
            lines.append(f"| `{key}` | {label} | 0 | - | - | - | - | {budget or '-'} ms | 无数据 |")
            continue
        mean = sum(xs) / len(xs)
        verdict = "-"
        if budget:
            verdict = "达标" if pct(xs, 90) <= budget else "超预算"
        lines.append(
            "| `{}` | {} | {} | {:.0f} ms | {:.0f} ms | {:.0f} ms | {:.0f} ms | {} | {} |".format(
                key, label, len(xs), mean, pct(xs, 50), pct(xs, 90), max(xs),
                f"{budget} ms" if budget else "-", verdict
            )
        )

    rounds = len(turns)
    barge_rate = len(barges) / rounds * 100 if rounds else 0
    lines.append("")
    lines.append(f"- 轮次数：**{rounds}**；成功测到打断的轮次：**{len(barges)}**（{barge_rate:.0f}%）")
    over = [k for k, _l, b, xs in rows if b and xs and pct(xs, 90) > b]
    lines.append(f"- 超预算指标：{('、'.join('`' + k + '`' for k in over)) if over else '无'}")
    return "\n".join(lines), rows


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = sys.argv[1]
    as_json = "--json" in sys.argv
    turns = load(path)
    md, rows = report(turns)
    if as_json:
        payload = {
            "rounds": len(turns),
            "metrics": {
                key: {
                    "n": len(xs),
                    "mean_ms": round(sum(xs) / len(xs), 1) if xs else None,
                    "p50_ms": round(pct(xs, 50), 1) if xs else None,
                    "p90_ms": round(pct(xs, 90), 1) if xs else None,
                    "max_ms": round(max(xs), 1) if xs else None,
                }
                for key, _l, _b, xs in rows
            },
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(md)


if __name__ == "__main__":
    main()
