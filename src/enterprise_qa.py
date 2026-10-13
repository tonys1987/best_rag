"""企业知识库问答层：全库统一检索 + 重排 + 来源溯源问答。

与旧 pipeline（company_name 路由）解耦：
    检索范围 = 登记表中所有 status='ready' 的文档，跨库合并 Top-K，
    支持按 file_type / doc_id 过滤；重排走 LLMClient.rerank_texts(gte-rerank-v2)；
    问答走 LLMClient.chat_structured(qwen3.7-max)，答案带来源编号可溯源。

典型用法：
    qa = EnterpriseQA()
    hits = qa.retrieve("公司成立时间", top_k=5)
    ans = qa.ask("公司成立于哪一年？")
    print(ans["final_answer"], ans["sources"])
"""
import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
from pydantic import BaseModel

from src import doc_store
from src.chunker import _tokens  # 复用本地近似 token 计数

# ---------------- 问答 Schema ----------------

class QAAnswer(BaseModel):
    """企业问答结构化输出。used_sources 为上下文中来源编号（1-based）。"""
    step_by_step_analysis: str
    reasoning_summary: str
    final_answer: str
    used_sources: List[int]


SYSTEM_PROMPT = (
    "你是企业内部知识库问答助手。请严格根据提供的【检索资料】回答问题，"
    "不要编造资料之外的数字和事实；资料不足以回答时，明确说\"资料中未提及\"。\n"
    "回答要求：\n"
    "1. step_by_step_analysis 中逐步分析资料；\n"
    "2. final_answer 用简洁的中文直接回答问题本体；\n"
    "3. used_sources 列出你实际依据的资料来源编号（1-based 整数列表）。"
)


class EnterpriseQA:
    """全库检索问答器。懒加载所有 ready 文档的向量库与分块，支持 refresh()。"""

    def __init__(self, rerank: bool = True, rerank_weight: float = 0.7,
                 candidate_k: int = 20, top_k: int = 5):
        self.rerank = rerank
        self.rerank_weight = rerank_weight
        self.candidate_k = candidate_k
        self.top_k = top_k
        self._lock = threading.Lock()
        self._dbs: Dict[str, dict] = {}  # doc_id -> {"index": faiss, "chunks": [...], "metainfo": {...}}
        self.refresh()

    # ---------- 库加载 ----------

    def refresh(self) -> int:
        """重新扫描登记表，加载所有 ready 文档。返回文档数。"""
        with self._lock:
            self._dbs.clear()
            for d in doc_store.list_documents():
                if d["status"] != "ready":
                    continue
                chunk_path = doc_store.CHUNKED_DIR / f"{d['doc_id']}.json"
                faiss_path = doc_store.VECTOR_DIR / f"{d['doc_id']}.faiss"
                if not chunk_path.exists() or not faiss_path.exists():
                    continue
                try:
                    with open(chunk_path, "r", encoding="utf-8") as f:
                        doc = json.load(f)
                    # Windows 中文路径兜底：从临时目录读
                    tmp = os.path.join(tempfile.gettempdir(), f"{d['doc_id']}.faiss")
                    if not os.path.exists(tmp):
                        shutil.copyfile(str(faiss_path), tmp)
                    index = faiss.read_index(tmp)
                    self._dbs[d["doc_id"]] = {
                        "index": index,
                        "chunks": doc["content"]["chunks"],
                        "metainfo": doc["metainfo"],
                    }
                except Exception as e:
                    print(f"[EnterpriseQA] 加载失败 {d['file_name']}: {e}")
            return len(self._dbs)

    @property
    def doc_count(self) -> int:
        return len(self._dbs)

    # ---------- 检索 ----------

    def retrieve(self, query: str, top_k: Optional[int] = None,
                 file_type: Optional[str] = None, doc_id: Optional[str] = None) -> List[dict]:
        """全库检索：逐库 Top-N -> 合并 -> (可选)重排融合 -> Top-K。

        返回块列表，附 doc/file/type/lines/vector_score/rerank_score。
        """
        from LLMClient import embed_query, rerank_texts

        if not query.strip():
            raise ValueError("检索词不能为空")
        top_k = top_k or self.top_k
        want_type = file_type.lower() if file_type else None

        candidates = []
        with self._lock:
            dbs = dict(self._dbs)
        if not dbs:
            raise ValueError("知识库为空：请先通过 doc_store.ingest_file 入库文档")

        q_vec = np.array([embed_query(query)], dtype=np.float32)
        per_db_k = max(self.candidate_k, top_k)
        for did, db in dbs.items():
            meta = db["metainfo"]
            if want_type and meta.get("file_type", "").lower() != want_type:
                continue
            if doc_id and did != doc_id:
                continue
            k = min(per_db_k, db["index"].ntotal)
            if k == 0:
                continue
            scores, ids = db["index"].search(q_vec, k)
            for score, idx in zip(scores[0], ids[0]):
                if idx < 0:
                    continue
                chunk = db["chunks"][idx]
                candidates.append({
                    "doc_id": did,
                    "file_name": meta.get("file_name", ""),
                    "file_type": meta.get("file_type", ""),
                    "title": meta.get("title", ""),
                    "lines": chunk.get("lines"),
                    "text": chunk["text"],
                    "vector_score": round(float(score), 4),
                })

        if not candidates:
            return []

        # 重排：gte-rerank-v2 真实相关性分数，与向量余弦加权融合
        if self.rerank and len(candidates) > 1:
            texts = [c["text"] for c in candidates]
            ranked = rerank_texts(query, texts, top_n=len(texts))
            rerank_score_by_idx = {r["index"]: r["relevance_score"] for r in ranked}
            for i, c in enumerate(candidates):
                c["rerank_score"] = round(rerank_score_by_idx.get(i, 0.0), 4)
                c["combined_score"] = round(
                    self.rerank_weight * c["rerank_score"] + (1 - self.rerank_weight) * c["vector_score"],
                    4,
                )
            candidates.sort(key=lambda x: x["combined_score"], reverse=True)
        else:
            candidates.sort(key=lambda x: x["vector_score"], reverse=True)

        return candidates[:top_k]

    # ---------- 问答 ----------

    @staticmethod
    def _format_context(hits: List[dict]) -> str:
        parts = []
        for i, h in enumerate(hits, start=1):
            loc = f"行{h['lines'][0]}-{h['lines'][1]}" if h.get("lines") else "全文"
            parts.append(
                f"[来源{i}] {h['file_name']}（{h['file_type']}，{loc}）\n"
                f"\"\"\"\n{h['text']}\n\"\"\""
            )
        return "\n\n---\n\n".join(parts)

    def ask(self, question: str, top_k: Optional[int] = None,
            file_type: Optional[str] = None, verbose: bool = True) -> dict:
        """端到端问答：检索 -> 拼上下文 -> LLM 结构化输出 -> 来源回填。"""
        from LLMClient import chat_structured, last_usage

        t0 = time.time()
        hits = self.retrieve(question, top_k=top_k, file_type=file_type)
        if not hits:
            return {
                "question": question, "final_answer": "资料中未提及。",
                "step_by_step_analysis": "", "reasoning_summary": "",
                "sources": [], "hits": [],
            }
        context = self._format_context(hits)
        user_prompt = f"【检索资料】\n{context}\n\n【问题】\n{question}"
        answer = chat_structured(SYSTEM_PROMPT, user_prompt, QAAnswer)

        used = answer.get("used_sources") or []
        sources = []
        for s in used:
            if isinstance(s, int) and 1 <= s <= len(hits):
                h = hits[s - 1]
                sources.append({
                    "file_name": h["file_name"], "file_type": h["file_type"],
                    "title": h["title"], "lines": h["lines"],
                })

        result = {
            "question": question,
            "step_by_step_analysis": answer.get("step_by_step_analysis", ""),
            "reasoning_summary": answer.get("reasoning_summary", ""),
            "final_answer": answer.get("final_answer", ""),
            "sources": sources,
            "hits": hits,
            "elapsed": round(time.time() - t0, 2),
        }
        if verbose:
            print(f"[QA] {question[:30]} -> {result['final_answer'][:50]}  "
                  f"(检索{len(hits)}块, {result['elapsed']}s, tokens={last_usage})")
        return result

    def format_sources(self, result: dict) -> str:
        """把答案的 sources 渲染成可读引用串（Streamlit 直接用）。"""
        lines = []
        for s in result.get("sources", []):
            loc = f"行{s['lines'][0]}-{s['lines'][1]}" if s.get("lines") else "全文"
            lines.append(f"- {s['file_name']}（{s['file_type']}，{loc}）")
        return "\n".join(lines)
