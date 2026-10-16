"""企业知识库 Streamlit UI：文档管理 + 全库检索问答。

启动：
    streamlit run app_enterprise.py --server.port 8502

布局（对照 UI界面参考-完成.png 的版式）：
    左侧：文档管理（上传/入库/重建/删除/清单） + 查询设置（问题/过滤/Top-K/重排）
    右侧：检索结果（逐块卡片） + 分步推理 / 推理摘要 / 来源 / 最终答案

后端全部走 src/doc_store.py（登记表）与 src/enterprise_qa.py（检索问答），
旧比赛 pipeline（app_streamlit.py）不参与。
"""
import time
from pathlib import Path

import streamlit as st

from src import doc_store
from src.enterprise_qa import EnterpriseQA

st.set_page_config(page_title="BEST_RAG", page_icon="📚", layout="wide")

# 收紧侧栏留白，避免出现上下滚动条
st.markdown("""<style>
[data-testid="stSidebar"] .block-container {padding-top: 0.8rem; padding-bottom: 0.8rem;}
[data-testid="stSidebar"] [data-testid="stExpander"] {border: none;}
</style>""", unsafe_allow_html=True)

# ---------------- 缓存后端 ----------------

@st.cache_resource
def get_qa() -> EnterpriseQA:
    return EnterpriseQA()


def refresh_qa():
    get_qa().refresh()
    get_qa.clear()


# ---------------- 页头 ----------------

st.markdown("""
<div style='background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%); padding: 20px 0; border-radius: 12px; text-align: center;'>
    <h2 style='color: white; margin: 0;'>📚 BEST_RAG · 企业知识库问答</h2>
    <div style='color: #dfe6f3; font-size: 15px;'>
        多格式文档入库 | 结构感知分块 | 向量检索 + gte-rerank 重排 + qwen 结构化问答 | 来源可溯源
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- 左侧栏 ----------------

with st.sidebar:
    docs = doc_store.list_documents()

    # ---- 文档管理（收进折叠面板，默认收起，避免侧栏溢出滚动条）----
    with st.expander("🗂 文档管理（上传 / 清单 / 维护）"):
        uploads = st.file_uploader(
            "上传文档（pdf/doc/docx/txt/md/json/xlsx/xls/csv）",
            accept_multiple_files=True,
            type=["pdf", "doc", "docx", "txt", "md", "markdown", "json", "xlsx", "xls", "csv"],
        )
        if uploads and st.button("📥 入库所选文件", use_container_width=True):
            save_dir = Path("data/uploads")
            save_dir.mkdir(parents=True, exist_ok=True)
            ok, fail = 0, 0
            prog = st.progress(0.0)
            for i, up in enumerate(uploads):
                dest = save_dir / up.name
                dest.write_bytes(up.getvalue())
                try:
                    doc_store.ingest_file(dest)
                    ok += 1
                except Exception as e:
                    st.warning(f"入库失败 {up.name}: {e}")
                    fail += 1
                prog.progress((i + 1) / len(uploads))
            prog.empty()
            st.success(f"入库完成：成功 {ok}，失败 {fail}")
            refresh_qa()

        with st.expander(f"📚 文档清单（{len(docs)} 条，ready {sum(1 for d in docs if d['status']=='ready')}）"):
            for d in docs:
                icon = {"ready": "✅", "failed": "❌", "pending": "⏳"}.get(d["status"], "•")
                st.markdown(
                    f"{icon} **{d['file_name']}**（{d['file_type']}，{d['chunk_count']} 块）"
                    + (f"  \n⚠️ {d['error'][:60]}" if d["error"] else "")
                )

        if docs:
            target = st.selectbox(
                "选择文档进行维护",
                docs,
                format_func=lambda d: f"{d['file_name']} [{d['status']}]",
            )
            c1, c2 = st.columns(2)
            if c1.button("🔄 重建", use_container_width=True):
                try:
                    doc_store.rebuild_file(target["doc_id"])
                    st.success("重建完成")
                    refresh_qa()
                except Exception as e:
                    st.error(f"重建失败: {e}")
            if c2.button("🗑 删除", use_container_width=True):
                doc_store.delete_file(target["doc_id"])
                st.success("已删除")
                refresh_qa()

    # ---- 查询设置 ----
    question = st.text_area("输入问题", height=70, placeholder="例如：公司成立于哪一年？")

    # 固定展示全部支持格式，避免"没看到的类型以为不支持"；选无文档的类型时检索结果为空
    ready_types = sorted({d["file_type"] for d in docs if d["status"] == "ready"})
    all_types = sorted({ext.lstrip(".").lower() for ext in doc_store.SUPPORTED_EXTS})
    file_type = st.selectbox("限定文档类型（可选）", ["全部"] + all_types)
    if ready_types:
        st.caption(f"当前库内已有类型：{ '、'.join(ready_types) }")
    top_k = st.slider("返回相关块数量 Top-K", 1, 10, 5)
    use_rerank = st.checkbox("启用 LLM 重排（gte-rerank-v2）", value=True)

    search_btn = st.button("🔎 搜索文档", use_container_width=True)
    answer_btn = st.button("🚀 生成答案", use_container_width=True, type="primary")

# ---------------- 主区渲染 ----------------

file_type_arg = None if (not file_type or file_type == "全部") else file_type
qa = get_qa()

st.markdown("<h3 style='margin-top: 20px;'>📄 检索结果</h3>", unsafe_allow_html=True)
st.caption(f"知识库共 {qa.doc_count} 篇 ready 文档参与检索")


def render_hits(hits: list):
    for i, h in enumerate(hits, start=1):
        loc = f"行{h['lines'][0]}-{h['lines'][1]}" if h.get("lines") else "全文"
        score_txt = f"向量 {h['vector_score']}"
        if "combined_score" in h:
            score_txt += f" ｜ 重排 {h.get('rerank_score')} ｜ 综合 {h['combined_score']}"
        with st.expander(f"结果 {i}：{h['file_name']}（{loc}） — {score_txt}"):
            st.markdown(h["text"])


def render_answer(ans: dict):
    st.markdown("**分步推理：**")
    st.info(ans.get("step_by_step_analysis") or "（无）")
    st.markdown("**推理摘要：**")
    st.success(ans.get("reasoning_summary") or "（无）")
    if ans.get("sources"):
        st.markdown("**引用来源：**")
        st.markdown(qa.format_sources(ans))
    st.markdown("**最终答案：**")
    st.markdown(
        f"<div style='background:#f6f8fa;padding:16px;border-radius:8px;font-size:17px;'>"
        f"{ans.get('final_answer') or '（无）'}</div>",
        unsafe_allow_html=True,
    )
    if ans.get("elapsed"):
        st.caption(f"端到端耗时 {ans['elapsed']}s")


if search_btn and question.strip():
    t0 = time.time()
    with st.spinner("检索中..."):
        try:
            hits = qa.retrieve(question, top_k=top_k, file_type=file_type_arg,
                               use_rerank=use_rerank)
        except ValueError as e:
            st.warning(str(e))
            hits = []
    if not hits:
        st.info("没有检索到相关内容。")
    else:
        st.caption(f"检索 {len(hits)} 块，耗时 {time.time() - t0:.2f}s" + ("（含 LLM 重排）" if use_rerank else "（未启用重排，按向量相似度排序）"))
        render_hits(hits)
elif search_btn:
    st.warning("请先输入问题")

if answer_btn and question.strip():
    with st.spinner("检索 + LLM 问答中，请稍候..." if not use_rerank else "检索 + 重排 + LLM 问答中，请稍候..."):
        try:
            ans = qa.ask(question, top_k=top_k, file_type=file_type_arg,
                         use_rerank=use_rerank)
        except ValueError as e:
            st.warning(str(e))
            ans = None
    if ans is not None:
        if ans.get("hits"):
            st.markdown("**检索到的资料片段：**")
            render_hits(ans["hits"])
        st.markdown("<h3 style='margin-top: 16px;'>💡 问答结果</h3>", unsafe_allow_html=True)
        render_answer(ans)
elif answer_btn:
    st.warning("请先输入问题")

if not (search_btn or answer_btn):
    st.info("请在左侧输入问题，选择【🔎 搜索文档】查看检索结果，或【🚀 生成答案】获取带来源的完整回答。")
