"""doc / docx 解析：docx 用 python-docx 转保序 markdown；老 .doc 先用 Word COM 转 docx。"""
import os
import tempfile
from pathlib import Path
from typing import List, Optional

from src.parsers.base import ParseResult

# Word COM 另存格式常量
WD_FORMAT_XML_DOCUMENT = 16  # .docx


def _cell_text(cell) -> str:
    return " ".join(cell.text.split()).strip()


def _table_to_markdown(table) -> str:
    """表格转 markdown 表。空表返回空串。"""
    rows: List[List[str]] = []
    for row in table.rows:
        rows.append([_cell_text(c) for c in row.cells])
    if not rows:
        return ""
    # 合并重复表头行（python-docx 对合并单元格会重复返回）
    lines = []
    header = rows[0]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for r in rows[1:]:
        # 补齐列数，防错位
        r = (r + [""] * len(header))[: len(header)]
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def _para_to_markdown(para) -> str:
    style = (para.style.name or "").lower()
    text = para.text.strip()
    if not text:
        return ""
    if style.startswith("heading"):
        try:
            level = int(style.replace("heading", "").strip() or "1")
        except ValueError:
            level = 2
        level = max(1, min(level, 6))
        return "#" * level + " " + text
    if "title" in style:
        return "# " + text
    if "list" in style or "toc" in style:
        return "- " + text
    return text


def parse_docx(path: Path) -> ParseResult:
    from docx import Document as DocxDocument
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = DocxDocument(str(path))
    blocks: List[str] = [f"# {path.stem}"]

    # iter_inner_content 保证段落与表格按文档原始顺序输出（python-docx >= 1.1）
    try:
        items = list(doc.iter_inner_content())
    except AttributeError:
        items = list(doc.paragraphs) + list(doc.tables)

    for item in items:
        if isinstance(item, Paragraph):
            md = _para_to_markdown(item)
            if md:
                blocks.append(md)
        elif isinstance(item, Table):
            md = _table_to_markdown(item)
            if md:
                blocks.append(md)

    return ParseResult(
        markdown="\n\n".join(blocks) + "\n",
        meta={"title": path.stem, "format": "docx"},
        source_path=str(path),
    )


def doc_to_docx(doc_path: Path, out_dir: Path) -> Path:
    """Word COM 把老 .doc 另存为 .docx。需要机器装有 Microsoft Word。"""
    import win32com.client  # noqa:延迟导入，未装 pywin32 时给出清晰报错

    doc_path = doc_path.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (out_dir / (doc_path.stem + ".docx")).resolve()

    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    try:
        document = word.Documents.Open(str(doc_path), ReadOnly=True)
        document.SaveAs2(str(out_path), FileFormat=WD_FORMAT_XML_DOCUMENT)
        document.Close(False)
    finally:
        try:
            word.Quit()
        except Exception:
            # Word 偶发在 Quit 时报 RPC 错误（进程已自行退出），不影响转换产物
            pass
    if not out_path.exists():
        raise RuntimeError(f"Word COM 转换失败：{doc_path}")
    return out_path


def parse_doc(path: Path, work_dir: Optional[str] = None) -> ParseResult:
    out_dir = Path(work_dir) if work_dir else Path(tempfile.gettempdir()) / "rag_docx_cache"
    docx_path = doc_to_docx(path, out_dir)
    res = parse_docx(docx_path)
    res.source_path = str(path)
    res.meta["format"] = "doc"
    res.meta["converted_from"] = str(docx_path)
    return res
