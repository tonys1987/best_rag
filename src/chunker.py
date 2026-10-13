"""结构感知分块器：按 markdown 标题层级分节，表格块原子化，带章节路径前缀。

替代旧的 30 行盲切。产出与现有 retrieval 层兼容的块结构：
    {"page": 起始行号(1-based), "lines": [start, end], "text": ...}

要点：
- 每个块前缀「【文档标题 › 章节路径】」，孤立检索时也带上下文；
- markdown 表格视为原子块：整体不超过 token 上限就独立成块，
  超限则按行分组、每组重复表头行；
- 章节超上限时在段落边界（空行）优先断开，单行超限整行成块不劈行。
"""
from typing import Dict, List, Optional

# token 计数用本地近似：tiktoken 首次运行需连 openaipublic 下载编码表，
# 国内网络易卡死，而分块只需求量级准确，近似即可。
# 近似规则：CJK 字符 1 字 ≈ 1 token（o200k 实测 0.7~1.1），其他字符 3.5 字符 ≈ 1 token。
_CJK_RANGES = (
    (0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x3040, 0x30FF),
    (0xAC00, 0xD7AF), (0xF900, 0xFAFF), (0x3000, 0x303F), (0xFF00, 0xFFEF),
)


def _tokens(text: str) -> int:
    cjk = 0
    other = 0
    for ch in text:
        cp = ord(ch)
        if any(lo <= cp <= hi for lo, hi in _CJK_RANGES):
            cjk += 1
        else:
            other += 1
    return cjk + int(other / 3.5) + 1


def _is_table_line(line: str) -> bool:
    s = line.strip()
    return s.startswith("|") and s.endswith("|") and s.count("|") >= 2


def _heading_level(line: str) -> Optional[int]:
    s = line.rstrip()
    if not s.startswith("#"):
        return None
    n = len(s) - len(s.lstrip("#"))
    if 1 <= n <= 6 and (len(s) == n or s[n] == " "):
        return n
    return None


def _split_sections(lines: List[str]) -> List[Dict]:
    """按标题把行切成节。返回 [{"headings": [...], "start": 行号1based, "lines": [...]}]"""
    sections: List[Dict] = []
    headings: List[str] = []  # headings[i] 为第 i+1 级标题文本
    cur: Optional[Dict] = None

    def flush():
        nonlocal cur
        if cur is not None:
            sections.append(cur)
        cur = None

    for i, ln in enumerate(lines, start=1):
        level = _heading_level(ln)
        if level is not None:
            flush()
            while len(headings) < level:
                headings.append("")
            headings[level - 1] = ln.lstrip("#").strip()
            del headings[level:]
            cur = {"headings": list(headings[:level]), "start": i, "lines": [ln]}
        else:
            if cur is None:
                cur = {"headings": [], "start": i, "lines": []}
            cur["lines"].append(ln)
    flush()
    return sections


def _make_prefix(headings: List[str], doc_title: str = "") -> str:
    parts = []
    for h in ([doc_title] if doc_title else []) + list(headings):
        if h and h not in parts:  # 文档标题与文内 h1 重复时只保留一个
            parts.append(h)
    return "【" + " › ".join(parts) + "】" if parts else ""


def _segments_of_section(sec: Dict) -> List[Dict]:
    """节内行 -> 连续段（表格段 / 文本段），带绝对行号。"""
    segs: List[Dict] = []
    i, n = 0, len(sec["lines"])
    while i < n:
        is_table = _is_table_line(sec["lines"][i])
        j = i
        while j < n and _is_table_line(sec["lines"][j]) == is_table:
            j += 1
        segs.append({"is_table": is_table, "start": sec["start"] + i, "lines": sec["lines"][i:j]})
        i = j
    return segs


def _split_big_table(seg: Dict, prefix: str, chunk_size: int) -> List[Dict]:
    """超长表格按行分组，每组重复表头行。"""
    header, body = seg["lines"][0], seg["lines"][1:]
    groups: List[List[str]] = []
    cur = [header]
    cur_tok = _tokens(header) + _tokens(prefix)
    for ln in body:
        t = _tokens(ln)
        if cur_tok + t > chunk_size and len(cur) > 1:
            groups.append(cur)
            cur, cur_tok = [header], _tokens(header) + _tokens(prefix)
        cur.append(ln)
        cur_tok += t
    if len(cur) > 1:
        groups.append(cur)

    out = []
    line_no = seg["start"]
    for g in groups:
        end = line_no + len(g) - 1
        out.append({
            "page": line_no,
            "lines": [line_no, end],
            "text": (prefix + "\n" if prefix else "") + "\n".join(g).strip(),
        })
        line_no = end + 1
    return out


def chunk_markdown(md_text: str, chunk_size: int = 500, doc_title: str = "") -> List[Dict]:
    """markdown 文本 -> 兼容现有 retrieval 层的块列表。chunk_size 单位：近似 token。"""
    lines = md_text.splitlines()
    chunks: List[Dict] = []

    for sec in _split_sections(lines):
        prefix = _make_prefix(sec["headings"], doc_title)
        for seg in _segments_of_section(sec):
            if seg["is_table"]:
                table_text = "\n".join(seg["lines"])
                if _tokens(table_text) + _tokens(prefix) <= chunk_size:
                    end = seg["start"] + len(seg["lines"]) - 1
                    chunks.append({
                        "page": seg["start"],
                        "lines": [seg["start"], end],
                        "text": (prefix + "\n" if prefix else "") + table_text.strip(),
                    })
                else:
                    chunks.extend(_split_big_table(seg, prefix, chunk_size))
                continue

            buf: List[str] = []
            buf_tok = 0
            buf_start: Optional[int] = None

            def emit(end_line: int):
                nonlocal buf, buf_tok, buf_start
                if buf and any(x.strip() for x in buf):
                    text = (prefix + "\n" if prefix else "") + "\n".join(buf).strip()
                    chunks.append({"page": buf_start, "lines": [buf_start, end_line], "text": text})
                buf, buf_tok, buf_start = [], 0, None

            offset = 0
            for ln in seg["lines"]:
                line_no = seg["start"] + offset
                offset += 1
                t = _tokens(ln)
                is_blank = not ln.strip()
                # 空行且缓冲已有一定体量 -> 段落边界，优先在此断开
                if is_blank and buf and buf_tok >= chunk_size * 0.5:
                    emit(line_no)
                    continue
                # 加上这行会超限 -> 先落盘当前缓冲
                if buf and buf_tok + t > chunk_size:
                    emit(line_no - 1)
                if buf_start is None:
                    buf_start = line_no
                buf.append(ln)
                buf_tok += t
            emit(seg["start"] + offset - 1)

    return [c for c in chunks if c["text"].strip()]


def chunk_parse_result(parse_result, chunk_size: int = 500) -> Dict:
    """ParseResult -> 与 chunked_reports 兼容的文档 JSON（metainfo + content.chunks）。"""
    from pathlib import Path
    title = parse_result.meta.get("title") or Path(parse_result.source_path).stem
    chunks = chunk_markdown(parse_result.markdown, chunk_size=chunk_size, doc_title=title)
    return {"metainfo": dict(parse_result.meta), "content": {"chunks": chunks}}
