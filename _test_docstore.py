# -*- coding: utf-8 -*-
"""doc_store 验收：入库/幂等/更新换代/删除/向量正确性。"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import faiss

from src import doc_store
from src.parsers import parse_file
from src.chunker import chunk_markdown

TEST_DIR = Path("data/_parse_test")


def faiss_ready(doc_id):
    p = doc_store.VECTOR_DIR / f"{doc_id}.faiss"
    idx = faiss.read_index(str(p))
    return idx


def run():
    # 1) 批量入库：7 个样例 + 1 个真 PDF
    files = [TEST_DIR / n for n in ["sample.md", "sample.txt", "sample.json",
                                    "sample.csv", "sample.xlsx", "sample.docx",
                                    "sample.doc"]] + [Path("src/dummy_report.pdf")]
    for f in files:
        doc_store.ingest_file(f)

    docs = doc_store.list_documents()
    assert len(docs) == 8, f"应有 8 条登记，实际 {len(docs)}"
    assert all(d["status"] == "ready" for d in docs), "应全部 ready"
    print("[1] 批量入库 8 文件 OK")
    doc_store.print_registry()

    # 2) 产物校验：faiss 维度/数量、chunk json 结构
    for d in docs:
        idx = faiss_ready(d["doc_id"])
        assert idx.ntotal == d["chunk_count"] > 0, f"{d['file_name']} faiss 块数不符"
        assert idx.d == 1536, f"向量维度应为 1536（text-embedding-v1），实际 {idx.d}"
        cj = json.loads((doc_store.CHUNKED_DIR / f"{d['doc_id']}.json").read_text(encoding="utf-8"))
        assert cj["metainfo"]["doc_id"] == d["doc_id"]
        assert len(cj["content"]["chunks"]) == d["chunk_count"]
        assert all("text" in c and "lines" in c for c in cj["content"]["chunks"])
    print("[2] faiss/分块 JSON 结构校验 OK")

    # 3) 幂等：同内容重复入库应跳过
    r = doc_store.ingest_file(TEST_DIR / "sample.txt")
    assert r["status"] == "ready" and len(doc_store.list_documents()) == 8
    print("[3] 幂等跳过 OK")

    # 4) 文件更新：改内容 -> 重新入库 -> 新 doc_id，旧产物清除
    txt = TEST_DIR / "sample.txt"
    old_hash_before = {d["doc_id"] for d in docs if d["file_name"] == "sample.txt"}
    txt.write_text("这是更新后的内容：公司实际成立于2016年3月，注册资本五千万。\n", encoding="utf-8")
    r2 = doc_store.ingest_file(txt)
    docs_now = doc_store.list_documents()
    assert len(docs_now) == 8, "更新后总数应仍为 8（旧版本被清除）"
    assert r2["doc_id"] != list(old_hash_before)[0], "内容变化 doc_id 应变化"
    assert not (doc_store.VECTOR_DIR / f"{list(old_hash_before)[0]}.faiss").exists(), "旧 faiss 应删除"
    print("[4] 文件更新换代 OK ->", r2["doc_id"][:8], f"{r2['chunk_count']} 块")

    # 5) 向量正确性冒烟：查询向量与入库向量可对齐（余弦最高应命中相关块）
    from LLMClient import embed_query
    q = np.array([embed_query("公司成立时间")], dtype=np.float32)
    idx = faiss.read_index(str(doc_store.VECTOR_DIR / f"{r2['doc_id']}.faiss"))
    scores, ids = idx.search(q, 1)
    chunks = json.loads((doc_store.CHUNKED_DIR / f"{r2['doc_id']}.json").read_text(encoding="utf-8"))
    top_text = chunks["content"]["chunks"][ids[0][0]]["text"]
    assert "2016" in top_text, f"检索命中应含 2016，实际: {top_text[:50]}"
    print(f"[5] 向量检索冒烟 OK (score={scores[0][0]:.3f}, 命中: {top_text[:36]}…)")

    # 6) 删除
    assert doc_store.delete_file("sample.md")
    docs_after = doc_store.list_documents()
    assert len(docs_after) == 7
    md_doc = [d for d in docs if d["file_name"] == "sample.md"][0]
    assert not (doc_store.VECTOR_DIR / f"{md_doc['doc_id']}.faiss").exists()
    assert not (doc_store.CHUNKED_DIR / f"{md_doc['doc_id']}.json").exists()
    print("[6] 删除 OK")

    print("\nALL DOC_STORE TESTS PASSED")


if __name__ == "__main__":
    run()
