"""多格式文档解析器统一入口。

用法：
    from src.parsers import parse_file
    res = parse_file("docs/报告.docx")   # -> ParseResult(markdown=..., meta=...)

支持：pdf / doc / docx / txt / md / markdown / json / xlsx / xls / csv
"""
from pathlib import Path

from src.parsers.base import ParseResult

SUPPORTED_EXTS = {".pdf", ".doc", ".docx", ".txt", ".md", ".markdown", ".json", ".xlsx", ".xls", ".csv"}


def parse_file(path, **kwargs) -> ParseResult:
    """按扩展名分发到对应解析器。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        from src.parsers.pdf_parser import parse_pdf
        return parse_pdf(path, output_dir=kwargs.get("output_dir", Path("data/parsed_md")))

    if ext == ".doc":
        from src.parsers.office_parser import parse_doc
        return parse_doc(path, work_dir=kwargs.get("work_dir"))

    if ext == ".docx":
        from src.parsers.office_parser import parse_docx
        return parse_docx(path)

    if ext in (".txt",):
        from src.parsers.text_parser import parse_txt
        return parse_txt(path)

    if ext in (".md", ".markdown"):
        from src.parsers.text_parser import parse_md
        return parse_md(path)

    if ext == ".json":
        from src.parsers.json_parser import parse_json
        return parse_json(path)

    if ext in (".xlsx", ".xls"):
        from src.parsers.table_parser import parse_excel
        return parse_excel(path)

    if ext == ".csv":
        from src.parsers.table_parser import parse_csv
        return parse_csv(path)

    raise ValueError(f"不支持的文档格式: {ext}（支持：{sorted(SUPPORTED_EXTS)}）")
