"""xlsx / xls / csv 表格解析：摘要块 + 行组块两路，全部渲染为 markdown。

设计约定：
- 每个工作表先产出一个「概览」小节（列名/类型/非空数/前3行样例），
  再按 ROWS_PER_GROUP 行一组渲染数据块。
- 列数 <= MAX_TABLE_COLS 时用 markdown 表格（适合对比类问题）；
  宽表退化为逐行 key:value 行（适合单格精确问题）。
- 每个数据块都带「工作表名 + 行范围」标题，分块层按标题切分后天然携带定位信息。
"""
import math
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.parsers.base import ParseResult, read_text_with_fallback

ROWS_PER_GROUP = 20
MAX_TABLE_COLS = 8
PREVIEW_ROWS = 3


def _fmt(v) -> str:
    """单元格值转显示文本：NaN/空 -> 空，长文本截断。"""
    if pd.isna(v):
        return ""
    s = str(v).strip()
    if len(s) > 200:
        s = s[:200] + "…"
    return s


def _sheet_overview(name: str, df: pd.DataFrame) -> str:
    lines = [f"#### 概览：工作表「{name}」（共 {len(df)} 行 × {len(df.columns)} 列）"]
    for col in df.columns:
        non_null = int(df[col].notna().sum())
        lines.append(f"- 列「{col}」：{df[col].dtype}，非空 {non_null} 条")
    if len(df) > 0:
        lines.append(f"前 {min(PREVIEW_ROWS, len(df))} 行样例：")
        for _, row in df.head(PREVIEW_ROWS).iterrows():
            kv = "；".join(f"{c}={_fmt(row[c])}" for c in df.columns if _fmt(row[c]))
            lines.append(f"- {kv}")
    return "\n".join(lines)


def _group_as_table(name: str, df: pd.DataFrame, start: int, end: int) -> str:
    part = df.iloc[start:end]
    title = f"#### 数据：工作表「{name}」第 {start + 1}-{min(end, len(df))} 行（共 {len(df)} 行）"
    cols = list(df.columns)
    lines = [title, "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in part.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _group_as_kv(name: str, df: pd.DataFrame, start: int, end: int) -> str:
    part = df.iloc[start:end]
    title = f"#### 数据：工作表「{name}」第 {start + 1}-{min(end, len(df))} 行（共 {len(df)} 行，宽表逐行键值）"
    lines = [title]
    for idx, row in part.iterrows():
        kv = "；".join(f"{c}={_fmt(row[c])}" for c in df.columns if _fmt(row[c]))
        lines.append(f"- 第 {idx + 1} 行：{kv}")
    return "\n".join(lines)


def _render_sheet(name: str, df: pd.DataFrame) -> List[str]:
    blocks = [f"### 工作表「{name}」", _sheet_overview(name, df)]
    wide = len(df.columns) > MAX_TABLE_COLS
    n = len(df)
    if n == 0:
        blocks.append("（此工作表无数据行）")
        return blocks
    groups = math.ceil(n / ROWS_PER_GROUP)
    for g in range(groups):
        start, end = g * ROWS_PER_GROUP, (g + 1) * ROWS_PER_GROUP
        if wide:
            blocks.append(_group_as_kv(name, df, start, end))
        else:
            blocks.append(_group_as_table(name, df, start, end))
    return blocks


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols: List[str] = []
    for i, c in enumerate(df.columns):
        c = str(c).strip() if c is not None else ""
        cols.append(c if c else f"列{i + 1}")
    df.columns = cols
    return df


def parse_excel(path: Path) -> ParseResult:
    """xlsx/xls：读全部工作表。"""
    sheets: Dict[str, pd.DataFrame] = pd.read_excel(path, sheet_name=None)
    blocks = [f"# {path.stem}"]
    for name, df in sheets.items():
        df = _clean_columns(df)
        blocks.extend(_render_sheet(name, df))
    meta = {"title": path.stem, "format": path.suffix.lstrip("."), "sheets": list(sheets.keys())}
    return ParseResult(markdown="\n\n".join(blocks) + "\n", meta=meta, source_path=str(path))


def parse_csv(path: Path) -> ParseResult:
    """csv：编码回退（utf-8-sig → gbk），可用分隔符自动嗅探。"""
    text = read_text_with_fallback(path)
    df = None
    for sep in (None, ",", "\t", ";"):
        try:
            df = pd.read_csv(pd.io.common.StringIO(text), sep=sep, engine="python")
            if df.shape[1] >= 1:
                break
        except Exception:
            continue
    if df is None:
        raise ValueError(f"CSV 解析失败（编码/分隔符无法识别）：{path}")
    df = _clean_columns(df)
    blocks = [f"# {path.stem}"]
    blocks.extend(_render_sheet("CSV", df))
    meta = {"title": path.stem, "format": "csv"}
    return ParseResult(markdown="\n\n".join(blocks) + "\n", meta=meta, source_path=str(path))
