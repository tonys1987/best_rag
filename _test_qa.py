# -*- coding: utf-8 -*-
"""enterprise_qa 验收：全库检索 / 类型过滤 / 重排 / 端到端问答溯源。"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.enterprise_qa import EnterpriseQA


def run():
    qa = EnterpriseQA()
    print(f"已加载 {qa.doc_count} 个文档库")
    assert qa.doc_count >= 6, "知识库应至少 6 个文档（上轮 doc_store 测试遗留）"

    # 1) 全库检索：公司成立时间应命中 sample.txt
    hits = qa.retrieve("公司成立于哪一年", top_k=5)
    assert hits, "检索无结果"
    print("[1] 全库检索 Top5:")
    for h in hits:
        print(f"    v={h['vector_score']:.3f} r={h.get('rerank_score', 0):.3f} "
              f"{h['file_name']:<16} {h['text'][:28].replace(chr(10), ' ')}")
    top1 = hits[0]
    assert top1["file_name"] == "sample.txt" and "2016" in top1["text"], \
        f"Top1 应为含 2016 的 sample.txt，实际 {top1['file_name']}"

    # 2) 类型过滤：只搜 xlsx
    hits_x = qa.retrieve("张三的工资", top_k=3, file_type="xlsx")
    assert hits_x and all(h["file_type"] == "xlsx" for h in hits_x)
    assert any("15000" in h["text"] or "张三" in h["text"] for h in hits_x)
    print(f"[2] file_type 过滤 OK（{len(hits_x)} 个 xlsx 块）")

    # 3) 端到端问答：事实型
    ans = qa.ask("公司成立于哪一年？")
    assert "2016" in ans["final_answer"], f"答案应含 2016: {ans['final_answer']}"
    assert ans["sources"], "应有来源"
    assert ans["sources"][0]["file_name"] == "sample.txt"
    print(f"[3] 事实问答 OK: {ans['final_answer']}")
    print(f"    来源: {qa.format_sources(ans)}")

    # 4) 端到端问答：表格数值
    ans2 = qa.ask("张三的工资是多少？", file_type="xlsx")
    assert "15000" in ans2["final_answer"], f"答案应含 15000: {ans2['final_answer']}"
    print(f"[4] 表格数值问答 OK: {ans2['final_answer']}")

    # 5) 端到端问答：PDF 内容
    ans3 = qa.ask("What company is this annual report about?", file_type="pdf")
    assert "holley" in ans3["final_answer"].lower(), f"应回答 Holley: {ans3['final_answer']}"
    print(f"[5] PDF 英文问答 OK: {ans3['final_answer'][:60]}")

    # 6) 拒答能力：问知识库里没有的东西
    ans4 = qa.ask("公司2024年在火星的分支机构有哪些？")
    print(f"[6] 无资料问题回答: {ans4['final_answer'][:60]}")

    print("\nALL ENTERPRISE_QA TESTS PASSED")


if __name__ == "__main__":
    run()
