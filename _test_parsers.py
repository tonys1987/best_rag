# -*- coding: utf-8 -*-
"""解析器验收测试：8 种格式真实文件 -> markdown，逐项断言。"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, str(Path(__file__).resolve().parent))

TEST_DIR = Path("data/_parse_test")
TEST_DIR.mkdir(parents=True, exist_ok=True)


def make_samples():
    # 样本已齐则跳过生成（Word COM 较慢且偶发 Quit RPC 报错，不影响产物）
    needed = ["sample.md", "sample.txt", "sample.json", "sample.csv",
              "sample.xlsx", "sample.docx", "sample.doc"]
    if all((TEST_DIR / n).exists() for n in needed):
        return
    # md
    (TEST_DIR / "sample.md").write_text(
        "# 项目说明\n\n## 背景\n本系统用于企业文档问答。\n\n## 目标\n支持多格式解析与检索。\n",
        encoding="utf-8")
    # txt
    (TEST_DIR / "sample.txt").write_text(
        "这是一份纯文本说明文档。\n第二行内容：公司成立于2015年。\n", encoding="utf-8")
    # json（嵌套 + 数组）
    (TEST_DIR / "sample.json").write_text(json.dumps({
        "company": "示例科技", "founded": 2015,
        "departments": [{"name": "研发", "headcount": 120}, {"name": "销售", "headcount": 60}],
        "address": {"city": "武汉", "district": "光谷"},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    # csv（25行触发两个行组）
    lines = ["产品,季度,销量,单价"] + [f"产品{i % 3 + 1},Q{i // 6 + 1},{100 + i},{9.9 + i}" for i in range(25)]
    (TEST_DIR / "sample.csv").write_text("\n".join(lines), encoding="utf-8")
    # xlsx（Sheet1 常规 5 列 / Sheet2 宽表 10 列）
    import pandas as pd
    with pd.ExcelWriter(TEST_DIR / "sample.xlsx", engine="openpyxl") as w:
        pd.DataFrame({"姓名": ["张三", "李四"], "部门": ["研发", "销售"],
                      "入职年份": [2020, 2021], "工资": [15000, 12000], "城市": ["武汉", "上海"]}
                     ).to_excel(w, sheet_name="员工表", index=False)
        wide = pd.DataFrame({f"指标{i}": range(3) for i in range(10)})
        wide.insert(0, "年份", [2022, 2023, 2024])
        wide.to_excel(w, sheet_name="宽表", index=False)
    # docx（标题 + 段落 + 表格）
    from docx import Document
    doc = Document()
    doc.add_heading("产品手册", level=1)
    doc.add_heading("产品简介", level=2)
    doc.add_paragraph("本产品支持多格式文档解析。")
    t = doc.add_table(rows=2, cols=3)
    for j, h in enumerate(["型号", "价格", "库存"]):
        t.rows[0].cells[j].text = h
    for j, v in enumerate(["A1", "999元", "42"]):
        t.rows[1].cells[j].text = v
    doc.save(TEST_DIR / "sample.docx")
    # doc（Word COM 从 docx 另存为 97-2003 格式，得到真 .doc）
    import win32com.client
    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    try:
        d = word.Documents.Open(str((TEST_DIR / "sample.docx").resolve()), ReadOnly=True)
        d.SaveAs2(str((TEST_DIR / "sample.doc").resolve()), FileFormat=0)  # wdFormatDocument97
        d.Close(False)
    finally:
        try:
            word.Quit()
        except Exception:
            pass  # Word 有时已自行退出，Quit 报 RPC 错误不影响产物


def run():
    from src.parsers import parse_file, SUPPORTED_EXTS
    make_samples()

    cases = ["sample.md", "sample.txt", "sample.json", "sample.csv",
             "sample.xlsx", "sample.docx", "sample.doc"]
    for name in cases:
        p = TEST_DIR / name
        res = parse_file(p)
        md = res.markdown
        assert len(md) > 30, f"{name} 产物过短"
        assert "#" in md or "：" in md, f"{name} 疑似空结构"
        print(f"[OK] {name:14s} -> {len(md):6d} 字符  meta={res.meta}")
        print("     预览: " + md.strip().splitlines()[0][:60])

    # 内容断言
    xlsx_md = parse_file(TEST_DIR / "sample.xlsx").markdown
    assert "员工表" in xlsx_md and "宽表" in xlsx_md and "逐行键值" in xlsx_md
    assert "第 1-2 行" in xlsx_md or "共 2 行" in xlsx_md
    csv_md = parse_file(TEST_DIR / "sample.csv").markdown
    assert "第 21-25 行" in csv_md, "csv 应有两个行组"
    doc_md = parse_file(TEST_DIR / "sample.doc").markdown
    assert "产品手册" in doc_md and "999元" in doc_md, "doc 表格/标题丢失"
    json_md = parse_file(TEST_DIR / "sample.json").markdown
    assert "address.city: 武汉" in json_md and "departments[0].name: 研发" in json_md
    print("\nALL PARSER TESTS PASSED (pdf 另测)")


if __name__ == "__main__":
    run()
