#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run.py —— 最外层入口（让你可以 `python run.py` 直接跑，不用记 `-m src.main`）

它做两件事：
  1. 保证项目根目录在 sys.path 里，然后把控制权交给 src/main.py；
  2. 把标准输出尽量切成 UTF-8 —— Windows 的控制台默认是 GBK（cp936），
     而本项目日志里有 ⚡ ✅ 这类字符，不处理的话会抛
     `UnicodeEncodeError: 'gbk' codec can't encode character`。
"""
import os
import sys

# ---- 1. 修 Windows 控制台编码（必须在任何 print 之前执行）----
# reconfigure 是 Python 3.7+ 才有的；用 errors="replace" 兜底：
# 万一终端确实显示不了某个字符，宁可显示成 "?" 也不要让整个程序崩掉。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# ---- 2. 把项目根目录加到 import 搜索路径的最前面 ----
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.main import main  # noqa: E402  (必须在 sys.path 处理好之后再 import)

if __name__ == "__main__":
    sys.exit(main())
