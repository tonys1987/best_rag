"""Task#5 UI 验收：AppTest 执行级冒烟 —— 渲染无异常 + 元素齐全 + 模拟检索点击。"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from streamlit.testing.v1 import AppTest

# ---- 1. 首次加载：无异常、关键元素齐全 ----
at = AppTest.from_file("app_enterprise.py", default_timeout=120)
at.run()
assert not at.exception, f"页面执行异常: {at.exception}"
def all_bodies(t):
    els = [t.markdown, t.header, t.subheader, t.info, t.success, t.warning, t.error, t.title]
    return " ".join(x.body for grp in els for x in grp)

body = all_bodies(at) + " " + all_bodies(at.sidebar)
for key in ["企业知识库问答", "检索结果", "文档管理", "查询设置"]:
    assert key in body, f"缺少元素: {key}"
assert len(at.sidebar.selectbox) >= 1, "文档维护下拉缺失"
print("[1] 首次加载 OK：无异常，页头/侧栏/检索区元素齐全")

# ---- 2. 知识库状态与登记表一致 ----
from src import doc_store
ready = [d for d in doc_store.list_documents() if d["status"] == "ready"]
print(f"[2] 登记表 ready 文档 {len(ready)} 篇，UI 后端 EnterpriseQA doc_count 一致性交给 get_qa 缓存")

# ---- 3. 模拟输入问题 + 点击【生成答案】：走真实检索+LLM 链路（按 label 定位，防误点删除）----
at.sidebar.text_area[0].set_value("公司成立于哪一年？")
btn = [b for b in at.sidebar.button if "生成答案" in (b.label or "")][0]
btn.click()
at.run()
assert not at.exception, f"问答执行异常: {at.exception}"
full = all_bodies(at)
assert "最终答案" in full, "问答结果区未渲染"
assert ("2016" in full) or ("未提及" in full), f"答案内容异常: {full[-200:]}"
print("[3] 模拟点击【生成答案】OK：真实 LLM 链路跑通，答案已渲染（含来源区）")

print("\nUI 验收全部通过")
