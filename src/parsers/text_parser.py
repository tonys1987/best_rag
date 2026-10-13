"""txt / md 解析：直读。md 保留标题层级供结构感知分块使用。"""
from pathlib import Path

from src.parsers.base import ParseResult, read_text_with_fallback


def parse_txt(path: Path) -> ParseResult:
    text = read_text_with_fallback(path)
    md = f"# {path.stem}\n\n{text.strip()}\n"
    return ParseResult(markdown=md, meta={"title": path.stem, "format": "txt"}, source_path=str(path))


def parse_md(path: Path) -> ParseResult:
    text = read_text_with_fallback(path)
    # 提取首个一级标题作为文档名；没有就补一个，保证分块层总有文档头
    title = path.stem
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# ") and not line.startswith("## "):
            title = line[2:].strip()
            break
    meta = {"title": title, "format": "md"}
    return ParseResult(markdown=text, meta=meta, source_path=str(path))
