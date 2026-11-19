# -*- coding: utf-8 -*-
"""
miniyaml.py —— 一个"够用就好"的 YAML 子集解析器

【为什么需要它】
PyYAML 是唯一一个"非 numpy"的依赖。如果目标机器装不上包（内网、离线、pip 源不通），
整个项目就跑不起来 —— 而"一键跑通"是这个项目的第一目标。
所以这里手写一个只支持 config.yaml 用到的语法子集的小解析器：
有 PyYAML 就用 PyYAML（更完整），没有就自动退回这个。

支持的语法（正好覆盖 config.yaml）：
    key: value              标量
    key:                    嵌套映射
      sub: 1
    列表:                   块状列表
      - a
      - b
    行内列表: [0, 10, 5]
    多行文本: |              （保留换行，用于 system_prompt）
    # 注释、空行

不支持（也不需要）：锚点 &/*、多文档 ---、复杂键、tag。

这 130 行代码在面试里是很好的谈资：
    "为了让部署零依赖，我写了一个 YAML 子集解析器，有 PyYAML 时优先用它。"
"""
from __future__ import annotations

import re
from typing import Any, List, Tuple

_BOOL_TRUE = {"true", "yes", "on"}
_BOOL_FALSE = {"false", "no", "off"}
_NULL = {"null", "~", ""}


def safe_load(text: str) -> Any:
    """把 YAML 文本解析成 dict / list / 标量。"""
    rows = _preprocess(text)
    if not rows:
        return {}
    value, _ = _parse_block(rows, 0, rows[0][0])
    return value


def safe_load_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return safe_load(f.read())


# ---------------------------------------------------------------- 预处理
def _preprocess(text: str) -> List[Tuple[int, str]]:
    """返回 [(缩进, 内容)]，丢掉空行和纯注释行，且把块状多行文本合并成一行值。"""
    out: List[Tuple[int, str]] = []
    raw_lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    i = 0
    n = len(raw_lines)
    while i < n:
        raw = raw_lines[i]
        stripped = _strip_comment(raw).rstrip()
        if not stripped.strip():
            i += 1
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        content = stripped.strip()

        # 多行文本块： key: |  或  key: >
        m = re.match(r"^(.*?):\s*([|>])([-+]?)\s*$", content)
        if m:
            key_part, style = m.group(1), m.group(2)
            i += 1
            block: List[str] = []
            block_indent = None
            while i < n:
                nxt = raw_lines[i]
                if not nxt.strip():
                    block.append("")
                    i += 1
                    continue
                cur_indent = len(nxt) - len(nxt.lstrip(" "))
                if cur_indent <= indent:
                    break
                if block_indent is None:
                    block_indent = cur_indent
                block.append(nxt[block_indent:] if len(nxt) >= block_indent else nxt.strip())
                i += 1
            # 去掉尾部空行
            while block and not block[-1].strip():
                block.pop()
            joiner = "\n" if style == "|" else " "
            value = joiner.join(block)
            if joiner == "\n":
                value += "\n"
            out.append((indent, f"{key_part}: {_quote(value)}"))
            continue

        out.append((indent, content))
        i += 1
    return out


def _strip_comment(line: str) -> str:
    """去掉行尾注释：# 前面必须是空格或行首，且不在引号里。"""
    in_s = in_d = False
    for idx, ch in enumerate(line):
        if ch == "'" and not in_d:
            in_s = not in_s
        elif ch == '"' and not in_s:
            in_d = not in_d
        elif ch == "#" and not in_s and not in_d:
            if idx == 0 or line[idx - 1] in " \t":
                return line[:idx]
    return line


def _quote(s: str) -> str:
    """把可能含特殊字符的字符串包起来，解析时再拆掉。"""
    if "\n" in s:
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


# ---------------------------------------------------------------- 解析
def _parse_block(rows: List[Tuple[int, str]], i: int, indent: int) -> Tuple[Any, int]:
    if i >= len(rows):
        return None, i
    if rows[i][1].startswith("- "):
        return _parse_list(rows, i, indent)
    return _parse_map(rows, i, indent)


def _parse_map(rows: List[Tuple[int, str]], i: int, indent: int) -> Tuple[dict, int]:
    result = {}
    while i < len(rows):
        ind, content = rows[i]
        if ind < indent or content.startswith("- "):
            break
        if ind > indent:
            # 缩进异常（上一行已经处理过子块），跳过
            i += 1
            continue
        key, value_str = _split_key(content)
        if value_str is None:
            # 嵌套块
            next_indent, next_content = (rows[i + 1] if i + 1 < len(rows) else (indent, ""))
            if next_content.startswith("- "):
                value, i = _parse_list(rows, i + 1, next_indent)
            else:
                value, i = _parse_map(rows, i + 1, next_indent)
            result[key] = value
        else:
            result[key] = _parse_scalar(value_str)
            i += 1
    return result, i


def _parse_list(rows: List[Tuple[int, str]], i: int, indent: int) -> Tuple[list, int]:
    result = []
    while i < len(rows):
        ind, content = rows[i]
        if ind < indent or not content.startswith("- "):
            break
        item = content[2:].strip()
        result.append(_parse_scalar(item))
        i += 1
    return result, i


def _split_key(content: str) -> Tuple[str, Any]:
    """把 'key: value' 拆成 (key, 'value')；'key:' 返回 (key, None)。"""
    in_s = in_d = False
    for idx, ch in enumerate(content):
        if ch == "'" and not in_d:
            in_s = not in_s
        elif ch == '"' and not in_s:
            in_d = not in_d
        elif ch == ":" and not in_s and not in_d:
            key = content[:idx].strip()
            rest = content[idx + 1:].strip()
            return key, (rest if rest else None)
    return content.strip(), None


def _parse_scalar(s: Any) -> Any:
    if not isinstance(s, str):
        return s
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] == '"':
        return s[1:-1].replace('\\n', "\n").replace('\\"', '"').replace("\\\\", "\\")
    if len(s) >= 2 and s[0] == s[-1] == "'":
        return s[1:-1].replace("''", "'")

    low = s.lower()
    if low in _NULL:
        return None
    if low in _BOOL_TRUE:
        return True
    if low in _BOOL_FALSE:
        return False

    # 行内列表 [a, b, c]
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(p) for p in _split_inline(inner)]

    # 数字
    try:
        if re.fullmatch(r"[-+]?\d+", s):
            return int(s)
        if re.fullmatch(r"[-+]?\d*\.\d+([eE][-+]?\d+)?", s) or re.fullmatch(r"[-+]?\d+[eE][-+]?\d+", s):
            return float(s)
    except ValueError:
        pass

    return s


def _split_inline(s: str) -> List[str]:
    """按逗号切行内列表，忽略引号内的逗号。"""
    parts, buf, in_s, in_d = [], [], False, False
    for ch in s:
        if ch == "'" and not in_d:
            in_s = not in_s
        elif ch == '"' and not in_s:
            in_d = not in_d
        if ch == "," and not in_s and not in_d:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    return parts
