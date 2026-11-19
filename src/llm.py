# -*- coding: utf-8 -*-
"""
llm.py —— 对话大脑（大语言模型）

【为什么主推 openai_compat 而不是本地加载】
任何"OpenAI 兼容"的服务都走同一个协议 /v1/chat/completions + SSE 流式，
所以同一份代码可以对接：
    远程 DeepSeek / 阿里百炼 / OpenAI / 智谱
    本地 vLLM、llama.cpp server、Ollama、LM Studio
换后端只改 config.yaml 里的 base_url 和 model，代码一行不动。

对 8 GB 显存的笔记本来说，这是最实际的方案：
  - 想省显存：LLM 走 API，本地只留 ASR + TTS（两者加起来约 3~4 GB）
  - 想全本地：llama.cpp 加载 Qwen2.5-7B 的 Q4_K_M（约 4.4 GB 显存），见 docs/04

【流式输出为什么要"逐 token"给上层】
因为首 token 时间（TTFT）决定了打断体验：用户说完到听到第一个字之间的沉默，
超过 800 ms 就开始让人怀疑"它是不是没听见"。逐 token 返回，才能在第一个字
一出来就开始合成语音，而不是等整段生成完。
"""
from __future__ import annotations

import json
import time
from typing import Dict, Iterable, Iterator, List, Optional


# =============================================================================
# 统一接口
# =============================================================================
class BaseLLM:
    """
    LLM 统一接口。

    stream() 必须：
      - 逐 token（或小段）yield 文本
      - 支持被"半路抛弃"：上层打断时直接 stop iterating 即可，不需要清理
    """

    name = "base"

    def stream(self, messages: List[Dict[str, str]]) -> Iterator[str]:
        raise NotImplementedError

    def close(self) -> None:
        pass


# =============================================================================
# 0. 兜底：假模型（sim 模式）
# =============================================================================
class MockLLM(BaseLLM):
    """
    按顺序返回预设回复，并模拟 token 间隔。

    回复长度刻意控制在 20~45 字：
      - 语音回复越短，用户等得越少、打断机会越少
      - 这也是真实机器人产品的做法（"一到两句话"是行业惯例，不是偷懒）
    """

    name = "mock"

    REPLIES: List[str] = [
        "我是你的语音助手，可以陪你聊天，也能查时间和天气。",
        "延迟是这样测的：每个环节打时间戳，判停到第一帧声音出来大概七百毫秒。",
        "你插话我立刻就停，已经播出去的那部分才算我说过的话。",
        "不会乱，被打断的回复我只把播出去的写回历史，没播的就丢掉。",
        "好，那先到这里，有需要随时叫我。",
        "我还在学全双工这一块，慢慢补。",
    ]

    def __init__(self, token_delay_ms: int = 16):
        self._i = 0
        # 每个"token"（这里按 2 个字算）之间的间隔。
        # 16 ms ≈ 每秒 62 个 token、约 125 字/秒，接近真实在线模型的吐字速度；
        # 设得太大（比如 28 ms）会让"首字延迟"这个指标看起来很糟，
        # 但那其实是假模型慢，不是管线慢 —— 测指标时要把这两件事分开。
        self._delay = token_delay_ms / 1000.0

    def stream(self, messages: List[Dict[str, str]]) -> Iterator[str]:
        text = self.REPLIES[self._i % len(self.REPLIES)]
        self._i += 1
        # 按 2 个字一组往外吐，模拟真实 tokenizer 的粒度
        for k in range(0, len(text), 2):
            time.sleep(self._delay)
            yield text[k:k + 2]


# =============================================================================
# 1. 主力：任何 OpenAI 兼容服务
# =============================================================================
class OpenAICompatLLM(BaseLLM):
    """
    走 openai 官方 Python SDK（它也能连任何兼容服务）。

    这样做的好处：SSE 解析、重试、超时、错误处理都由 SDK 负责，
    你只需要关心 chat.completions.create(stream=True) 返回的 delta 文本。
    """

    name = "openai_compat"

    def __init__(self, base_url: str, model: str, api_key: str,
                 temperature: float = 0.7, max_tokens: int = 220, timeout_s: float = 30.0,
                 system_prompt: str = ""):
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("没装 openai。运行： pip install openai") from e

        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._system = system_prompt

    def stream(self, messages: List[Dict[str, str]]) -> Iterator[str]:
        msgs = list(messages)
        if self._system:
            msgs = [{"role": "system", "content": self._system}] + msgs

        resp = self._client.chat.completions.create(
            model=self._model,
            messages=msgs,          # type: ignore[arg-type]
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            stream=True,
        )
        for chunk in resp:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            piece = getattr(delta, "content", None)
            if piece:
                yield piece


# =============================================================================
# 2. 备选：本地 transformers + 4bit 量化
# =============================================================================
class LocalHFLLM(BaseLLM):
    """
    本地跑 Qwen2.5-7B-Instruct 的 4bit 版本，约 5 GB 显存。

    装：pip install transformers accelerate bitsandbytes
    模型：modelscope download --model Qwen/Qwen2.5-7B-Instruct-AWQ --local_dir models/Qwen2.5-7B-Instruct-AWQ

    为什么用 TextIteratorStreamer：
      它把 HuggingFace 的生成过程变成一个 Python 队列，主线程可以边生成边取，
      这就是"流式输出"在本地模型上的实现方式。
    """

    name = "hf_local"

    def __init__(self, model_path: str, device: str = "cuda", load_in_4bit: bool = True,
                 max_new_tokens: int = 200, system_prompt: str = ""):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer  # type: ignore
            import torch  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("没装 transformers / torch。见 requirements-gpu.txt") from e

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        kwargs = dict(device_map=device, trust_remote_code=True)
        if load_in_4bit:
            kwargs.update(load_in_4bit=True, torch_dtype=torch.float16)
        self._model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        self._max_new_tokens = max_new_tokens
        self._system = system_prompt
        self._streamer_cls = TextIteratorStreamer

    def stream(self, messages: List[Dict[str, str]]) -> Iterator[str]:
        import threading
        msgs = list(messages)
        if self._system:
            msgs = [{"role": "system", "content": self._system}] + msgs
        text = self._tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
        streamer = self._streamer_cls(self._tokenizer, skip_prompt=True, skip_special_tokens=True)

        def _run():
            with self._torch.no_grad():
                self._model.generate(**inputs, streamer=streamer,
                                     max_new_tokens=self._max_new_tokens,
                                     do_sample=True, temperature=0.7)

        threading.Thread(target=_run, daemon=True).start()
        for piece in streamer:
            if piece:
                yield piece


# =============================================================================
# 工厂
# =============================================================================
def build_llm(cfg) -> BaseLLM:
    backend = cfg.llm.backend.lower()
    try:
        if backend == "mock":
            return MockLLM()
        if backend == "openai_compat":
            o = cfg.llm.openai_compat
            print(f"[llm] {o.model} @ {o.base_url}")
            return OpenAICompatLLM(base_url=o.base_url, model=o.model, api_key=o.api_key(),
                                   temperature=o.temperature, max_tokens=o.max_tokens,
                                   timeout_s=o.timeout_s, system_prompt=cfg.llm.system_prompt)
        if backend == "hf_local":
            h = cfg.llm.hf_local
            print(f"[llm] 本地加载 {h.model_path}（4bit={h.load_in_4bit}）…")
            return LocalHFLLM(model_path=h.model_path, device=h.device,
                              load_in_4bit=h.load_in_4bit,
                              max_new_tokens=h.max_new_tokens,
                              system_prompt=cfg.llm.system_prompt)
    except Exception as e:
        print(f"[llm] 初始化 {backend} 失败，降级到 mock。原因：{e}")
        return MockLLM()
    raise ValueError(f"未知的 llm.backend：{cfg.llm.backend}（可选 mock / openai_compat / hf_local）")
