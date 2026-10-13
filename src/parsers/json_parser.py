"""json 解析：递归展平为 key路径: 值 文本，保留层级语义。"""
from pathlib import Path
from typing import Any, List

from src.parsers.base import ParseResult, read_text_with_fallback

MAX_DEPTH = 16
MAX_VALUE_LEN = 500
MAX_ITEMS_SHOWN = 200  # 超长数组只展平前 N 项，避免单文件爆量


def _flatten(obj: Any, prefix: str, lines: List[str], depth: int = 0) -> None:
    if depth > MAX_DEPTH:
        lines.append(f"{prefix}: （嵌套过深，已截断）")
        return
    if isinstance(obj, dict):
        if not obj:
            lines.append(f"{prefix}: {{}}")
            return
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            _flatten(v, key, lines, depth + 1)
    elif isinstance(obj, (list, tuple)):
        if not obj:
            lines.append(f"{prefix}: []")
            return
        shown = obj[:MAX_ITEMS_SHOWN]
        for i, v in enumerate(shown):
            _flatten(v, f"{prefix}[{i}]", lines, depth + 1)
        if len(obj) > MAX_ITEMS_SHOWN:
            lines.append(f"{prefix}: （共 {len(obj)} 项，仅展平前 {MAX_ITEMS_SHOWN} 项）")
    else:
        s = str(obj).strip()
        if len(s) > MAX_VALUE_LEN:
            s = s[:MAX_VALUE_LEN] + "…"
        lines.append(f"{prefix}: {s}" if s else f"{prefix}: （空值）")


def parse_json(path: Path) -> ParseResult:
    text = read_text_with_fallback(path)
    import json
    data = json.loads(text)
    lines: List[str] = []
    _flatten(data, "", lines)
    md = f"# {path.stem}\n\n" + "\n".join(lines) + "\n"
    return ParseResult(
        markdown=md,
        meta={"title": path.stem, "format": "json", "flat_lines": len(lines)},
        source_path=str(path),
    )
