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
                 candidate_k: int = 20, top_k: int = 5, min_score: float = 0.30,
                 bm25_weight: float = 0.30):
        self.rerank = rerank
        self.rerank_weight = rerank_weight
        self.candidate_k = candidate_k
        self.top_k = top_k
        # 相关度阈值：开重排按 combined_score 过滤，否则按 hybrid_score 过滤；
        # 低于阈值的块不进入问答上下文（实测无关问题 Top1 约 0.13-0.17，相关普通问题 0.41+）
        self.min_score = min_score
        # 混合检索：hybrid = (1-bm25_weight)*向量分 + bm25_weight*BM25归一分。
        # 0 = 纯向量（旧行为），1 = 纯 BM25；BM25 侧 jieba 分词、按查询内全局最大值归一到 0-1
        self.bm25_weight = max(0.0, min(1.0, bm25_weight))
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
                    entry = {
                        "index": index,
                        "chunks": doc["content"]["chunks"],
                        "metainfo": doc["metainfo"],
                        "bm25": None,
                    }
                    self._build_bm25(entry)
                    self._dbs[d["doc_id"]] = entry
                except Exception as e:
                    print(f"[EnterpriseQA] 加载失败 {d['file_name']}: {e}")
            return len(self._dbs)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """BM25 分词：jieba 切词 + 小写，仅保留含字母/数字/汉字的词元。"""
        import jieba
        return [t for t in jieba.lcut(text.lower()) if any(ch.isalnum() for ch in t)]

    def _build_bm25(self, entry: dict) -> None:
        """为单个文档构建 BM25 索引；依赖缺失时降级为无 BM25（混合权重自动失效）。"""
        try:
            from rank_bm25 import BM25Okapi
            corpus = [self._tokenize(c["text"]) for c in entry["chunks"]]
            entry["bm25"] = BM25Okapi(corpus) if corpus and any(corpus) else None
        except ImportError as e:
            if not getattr(self, "_bm25_warned", False):
                print(f"[EnterpriseQA] BM25 依赖缺失，混合检索降级为纯向量: {e}")
                self._bm25_warned = True
            entry["bm25"] = None

    @property
    def doc_count(self) -> int:
        return len(self._dbs)

    # ---------- 检索 ----------

    def retrieve(self, query: str, top_k: Optional[int] = None,
                 file_type: Optional[str] = None, doc_id: Optional[str] = None,
                 use_rerank: Optional[bool] = None,
                 min_score: Optional[float] = None,
                 bm25_weight: Optional[float] = None) -> List[dict]:
        """混合检索：向量(faiss) + BM25 双路召回 -> 加权融合 -> (可选)重排 -> 阈值过滤 -> Top-K。

        use_rerank / min_score / bm25_weight 为 None 时跟随构造配置，否则以调用方为准
        （min_score<=0 关闭阈值；bm25_weight=0 纯向量、1 纯 BM25）。
        返回块列表，附 vector_score / bm25_score / hybrid_score / rerank_score / combined_score。
        """
        from LLMClient import embed_query, rerank_texts

        if not query.strip():
            raise ValueError("检索词不能为空")
        top_k = top_k or self.top_k
        do_rerank = self.rerank if use_rerank is None else use_rerank
        w_b = self.bm25_weight if bm25_weight is None else max(0.0, min(1.0, bm25_weight))
        w_v = 1.0 - w_b
        want_type = file_type.lower() if file_type else None

        with self._lock:
            dbs = dict(self._dbs)
        if not dbs:
            raise ValueError("知识库为空：请先通过 doc_store.ingest_file 入库文档")

        # ---- 双路召回：按 (doc_id, 块下标) 合并到一张表，缺席的路记 0 分 ----
        merged: Dict[tuple, dict] = {}

        if w_v > 0:
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
                    merged[(did, int(idx))] = self._candidate(db, did, int(idx), float(score))

        if w_b > 0:
            tokens = self._tokenize(query)
            if tokens:
                raw_scores = {}  # (did, idx) -> 原始 BM25 分
                for did, db in dbs.items():
                    if db["bm25"] is None or (want_type and db["metainfo"].get("file_type", "").lower() != want_type):
                        continue
                    if doc_id and did != doc_id:
                        continue
                    raw = db["bm25"].get_scores(tokens)
                    for idx in np.argsort(raw)[::-1][:max(self.candidate_k, top_k)]:
                        if raw[idx] > 0:
                            raw_scores[(did, int(idx))] = float(raw[idx])
                gmax = max(raw_scores.values(), default=0.0)
                if gmax > 0:
                    for (did, idx), raw in raw_scores.items():
                        norm = raw / gmax  # 按本次查询全局最大值归一到 0-1
                        if (did, idx) in merged:
                            merged[(did, idx)]["bm25_score"] = round(norm, 4)
                        else:
                            merged[(did, idx)] = self._candidate(dbs[did], did, idx, 0.0, bm25=norm)

        if not merged:
            return []

        candidates = list(merged.values())
        for c in candidates:
            c["hybrid_score"] = round(w_v * c["vector_score"] + w_b * c["bm25_score"], 4)
        candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)

        # ---- 重排：对融合后的候选池精排，与 hybrid 分按 rerank_weight 加权 ----
        if do_rerank and len(candidates) > 1:
            texts = [c["text"] for c in candidates]
            ranked = rerank_texts(query, texts, top_n=len(texts))
            rerank_score_by_idx = {r["index"]: r["relevance_score"] for r in ranked}
            for i, c in enumerate(candidates):
                c["rerank_score"] = round(rerank_score_by_idx.get(i, 0.0), 4)
                c["combined_score"] = round(
                    self.rerank_weight * c["rerank_score"] + (1 - self.rerank_weight) * c["hybrid_score"],
                    4,
                )
            candidates.sort(key=lambda x: x["combined_score"], reverse=True)

        # ---- 相关度阈值：低于阈值的块不返回（问答回馈给拒答分支，避免拿不相关块硬答）----
        thr = self.min_score if min_score is None else min_score
        if thr and thr > 0:
            key = "combined_score" if "combined_score" in candidates[0] else "hybrid_score"
            candidates = [c for c in candidates if c[key] >= thr]

        return candidates[:top_k]

    @staticmethod
    def _candidate(db: dict, did: str, idx: int, vector_score: float,
                   bm25: Optional[float] = None) -> dict:
        """从库条目组装候选块记录（BM25 侧新命中时 vector_score 记 0）。"""
        chunk = db["chunks"][idx]
        meta = db["metainfo"]
        return {
            "doc_id": did,
            "file_name": meta.get("file_name", ""),
            "file_type": meta.get("file_type", ""),
            "title": meta.get("title", ""),
            "lines": chunk.get("lines"),
            "text": chunk["text"],
            "vector_score": round(vector_score, 4),
            "bm25_score": round(bm25, 4) if bm25 is not None else 0.0,
        }

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
            file_type: Optional[str] = None, verbose: bool = True,
            use_rerank: Optional[bool] = None,
            min_score: Optional[float] = None,
            bm25_weight: Optional[float] = None) -> dict:
        """端到端问答：混合检索(含相关度阈值) -> 无命中则拒答 -> 拼上下文 -> LLM 结构化输出 -> 来源回填。"""
        from LLMClient import chat_structured, last_usage

        t0 = time.time()
        hits = self.retrieve(question, top_k=top_k, file_type=file_type,
                             use_rerank=use_rerank, min_score=min_score,
                             bm25_weight=bm25_weight)
        if not hits:
            thr = self.min_score if min_score is None else (min_score or 0)
            # 拒答分支：相关度不足时不调 LLM，直接返回（省 token 且行为确定）
            return {
                "question": question,
                "final_answer": "资料中未提及。当前知识库中没有与问题相关的内容，请确认相关资料是否已入库。",
                "step_by_step_analysis": f"检索未命中任何相关度不低于 {thr:.2f} 的资料块，直接拒答。",
                "reasoning_summary": "问题与知识库内容相关度过低。",
                "sources": [], "hits": [],
                "elapsed": round(time.time() - t0, 2),
                "refused_by_threshold": True,
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
