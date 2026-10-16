"""QA 准确性评估：3 道考题，ground truth 预先从库内分块人工核定。

Q1 长文档事实题（PDF 年报文本）: 厦门实业2019年营业收入 -> 20.02亿元
Q2 表格数值题（xlsx）: sample.xlsx 张三的工资 -> 15000
Q3 拒答能力（库内无此信息）: 厦门实业2020年营业收入 -> 应答"资料中未提及"类

评分：每题 0/1 正误；置信度 = (正确题数/3) 的系统准确率 + 检索分数支撑说明。
"""
import sys, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.enterprise_qa import EnterpriseQA

qa = EnterpriseQA()
CASES = [
    {
        "q": "厦门实业2019年的营业收入是多少？",
        "gt_check": lambda a: bool(re.search(r"20\.02", a)) and bool(re.search(r"(亿|亿元)", a)),
        "gt_desc": "人民币 20.02 亿元（源：厦门实业年报 行156-161）",
        "expect": "正确答出",
    },
    {
        "q": "sample.xlsx 里张三的工资是多少？",
        "gt_check": lambda a: "15000" in a or "15,000" in a,
        "gt_desc": "15000（源：sample.xlsx 表格块）",
        "expect": "正确答出",
    },
    {
        "q": "厦门实业2020年的营业收入是多少？",
        "gt_check": lambda a: ("未提及" in a or "未提供" in a or "无法" in a)
                              and not re.search(r"2020[^。；]*?(为|是)\s*[\d,\.]+\s*亿", a),
        "gt_desc": "库内仅有2019年报，正确行为是拒答/说明无该资料（可如实引用2019数据作对照）",
        "expect": "拒答",
    },
]

correct = 0
rows = []
for c in CASES:
    ans = qa.ask(c["q"], verbose=False)
    final = ans.get("final_answer", "")
    ok = c["gt_check"](final)
    correct += ok
    top = ans["hits"][0] if ans.get("hits") else {}
    rows.append((c["q"], ok, final, ans.get("sources", []),
                 top.get("file_name", ""), top.get("combined_score", top.get("vector_score", 0))))

print("=" * 70)
for q, ok, final, srcs, tf, ts in rows:
    print(f"{'✅' if ok else '❌'} [{q}]")
    print(f"   答案: {final[:150]}")
    print(f"   来源: {[s['file_name'] + ':' + str(s.get('lines')) for s in srcs] or '无'}")
    print(f"   Top1: {tf} (score={ts})")
print("=" * 70)
print(f"准确率: {correct}/3 = {correct/3:.0%}")
