"""文档登记表与全库管理：单文件入库 / 重建 / 删除 / 清单。

布局：
    data/enterprise_db/registry.db            SQLite 登记表
    data/enterprise_db/chunked_reports/{doc_id}.json   分块结果
    data/enterprise_db/vector_dbs/{doc_id}.faiss       向量索引

约定：
    doc_id = 文件内容 sha1 —— 同内容幂等跳过；文件更新后 hash 变化，
    重新入库会自动清掉旧版本产物（单文件维度重建，不再全量重跑）。
    入库链路：parse_file -> chunk_markdown -> embed_texts -> faiss。
"""
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
DB_DIR = PROJECT_DIR / "data" / "enterprise_db"
CHUNKED_DIR = DB_DIR / "chunked_reports"
VECTOR_DIR = DB_DIR / "vector_dbs"
DB_PATH = DB_DIR / "registry.db"

SUPPORTED_EXTS = {".pdf", ".doc", ".docx", ".txt", ".md", ".markdown", ".json", ".xlsx", ".xls", ".csv"}


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _connect() -> sqlite3.Connection:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE IF NOT EXISTS documents (
            doc_id      TEXT PRIMARY KEY,
            file_path   TEXT NOT NULL,
            file_name   TEXT NOT NULL,
            file_type   TEXT NOT NULL,
            title       TEXT,
            file_hash   TEXT NOT NULL,
            status      TEXT NOT NULL DEFAULT 'pending',
            chunk_count INTEGER DEFAULT 0,
            error       TEXT,
            created_at  TEXT,
            updated_at  TEXT
        )"""
    )
    return conn


def file_hash(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _remove_artifacts(doc_id: str) -> None:
    for p in (CHUNKED_DIR / f"{doc_id}.json", VECTOR_DIR / f"{doc_id}.faiss"):
        try:
            p.unlink()
        except FileNotFoundError:
            pass


def _find_by_identifier(conn: sqlite3.Connection, identifier: str) -> Optional[sqlite3.Row]:
    """按 doc_id / file_name / 文件路径 定位登记记录。"""
    row = conn.execute("SELECT * FROM documents WHERE doc_id=?", (identifier,)).fetchone()
    if row is None:
        row = conn.execute(
            "SELECT * FROM documents WHERE file_name=? ORDER BY updated_at DESC", (identifier,)
        ).fetchone()
    if row is None:
        row = conn.execute(
            "SELECT * FROM documents WHERE file_path=? ORDER BY updated_at DESC", (identifier,)
        ).fetchone()
    return row


def ingest_file(path, chunk_size: int = 500, verbose: bool = True) -> dict:
    """解析 -> 分块 -> 向量化 -> faiss -> 登记。同内容幂等；文件更新自动清旧版本。

    失败时登记 status='failed' 并抛出异常。
    """
    from src.parsers import parse_file
    from src.chunker import chunk_markdown
    from LLMClient import embed_texts

    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() not in SUPPORTED_EXTS:
        raise ValueError(f"不支持的格式: {path.suffix}（支持 {sorted(SUPPORTED_EXTS)}）")

    h = file_hash(path)
    conn = _connect()
    try:
        row = conn.execute("SELECT * FROM documents WHERE doc_id=?", (h,)).fetchone()
        if row and row["status"] == "ready":
            if verbose:
                print(f"跳过（内容未变，已入库）: {path.name}")
            return dict(row)

        # 同路径旧版本：清掉旧 doc_id 的产物与记录
        old = conn.execute(
            "SELECT doc_id FROM documents WHERE file_path=? AND doc_id<>?", (str(path), h)
        ).fetchone()
        if old:
            _remove_artifacts(old["doc_id"])
            conn.execute("DELETE FROM documents WHERE doc_id=?", (old["doc_id"],))
            if verbose:
                print(f"检测到文件更新，清除旧版本: {old['doc_id'][:8]}…")

        doc_id = h
        error = None
        status = "failed"
        chunk_count = 0
        title = path.stem
        file_type = path.suffix.lstrip(".").lower()
        try:
            res = parse_file(path)
            title = res.meta.get("title") or path.stem
            file_type = res.meta.get("format") or file_type
            chunks = chunk_markdown(res.markdown, chunk_size=chunk_size, doc_title=title)
            if not chunks:
                raise RuntimeError("分块结果为空")
            chunk_count = len(chunks)

            # 写分块 JSON（与 retrieval 层兼容的结构）
            CHUNKED_DIR.mkdir(parents=True, exist_ok=True)
            metainfo = {
                "doc_id": doc_id,
                "title": title,
                "file_name": path.name,
                "file_type": file_type,
                "file_path": str(path),
                "file_hash": h,
                "chunk_count": chunk_count,
            }
            with open(CHUNKED_DIR / f"{doc_id}.json", "w", encoding="utf-8") as f:
                json.dump(
                    {"metainfo": metainfo, "content": {"chunks": chunks}},
                    f, ensure_ascii=False, indent=2,
                )

            # 向量化 + faiss（Windows 中文路径：先写临时目录再搬）
            texts = [c["text"] for c in chunks]
            vectors = embed_texts(texts)
            arr = np.array(vectors, dtype=np.float32)
            index = faiss.IndexFlatIP(arr.shape[1])
            index.add(arr)
            VECTOR_DIR.mkdir(parents=True, exist_ok=True)
            tmp = os.path.join(tempfile.gettempdir(), f"{doc_id}.faiss")
            if os.path.exists(tmp):
                os.remove(tmp)
            faiss.write_index(index, tmp)
            shutil.move(tmp, str(VECTOR_DIR / f"{doc_id}.faiss"))
            status = "ready"
            if verbose:
                print(f"入库完成: {path.name} -> {chunk_count} 块")
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            if verbose:
                print(f"入库失败 {path.name}: {error}")
            raise
        finally:
            conn.execute(
                """INSERT INTO documents
                   (doc_id, file_path, file_name, file_type, title, file_hash,
                    status, chunk_count, error, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(doc_id) DO UPDATE SET
                     status=excluded.status, chunk_count=excluded.chunk_count,
                     error=excluded.error, updated_at=excluded.updated_at""",
                (doc_id, str(path), path.name, file_type, title, h,
                 status, chunk_count, error, _now(), _now()),
            )
            conn.commit()

        row = conn.execute("SELECT * FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
        return dict(row)
    finally:
        conn.close()


def ingest_dir(dir_path, verbose: bool = True) -> List[dict]:
    """批量入库目录下所有支持的文件。单个失败不中断，最后汇总。"""
    dir_path = Path(dir_path)
    files = sorted(p for p in dir_path.iterdir()
                   if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS)
    if verbose:
        print(f"待入库 {len(files)} 个文件: {dir_path}")
    results = []
    failed = []
    for p in files:
        try:
            results.append(ingest_file(p, verbose=verbose))
        except Exception as e:
            failed.append((p.name, str(e)))
    if verbose:
        print(f"\n汇总: 成功 {len(results)}，失败 {len(failed)}")
        for name, err in failed:
            print(f"  [失败] {name}: {err[:100]}")
    return results


def rebuild_file(identifier, verbose: bool = True) -> dict:
    """按 doc_id / file_name / 路径 定位后删除产物并重新入库。"""
    conn = _connect()
    try:
        row = _find_by_identifier(conn, identifier)
        if row is None:
            raise ValueError(f"登记表中找不到: {identifier}")
        file_path = row["file_path"]
    finally:
        conn.close()
    _remove_artifacts(row["doc_id"])
    conn = _connect()
    try:
        conn.execute("DELETE FROM documents WHERE doc_id=?", (row["doc_id"],))
        conn.commit()
    finally:
        conn.close()
    if verbose:
        print(f"重建: {row['file_name']}")
    return ingest_file(file_path, verbose=verbose)


def delete_file(identifier, verbose: bool = True) -> bool:
    """按 doc_id / file_name / 路径 删除登记记录与产物。返回是否删除。"""
    conn = _connect()
    try:
        row = _find_by_identifier(conn, identifier)
        if row is None:
            if verbose:
                print(f"未找到，无需删除: {identifier}")
            return False
        _remove_artifacts(row["doc_id"])
        conn.execute("DELETE FROM documents WHERE doc_id=?", (row["doc_id"],))
        conn.commit()
        if verbose:
            print(f"已删除: {row['file_name']} ({row['doc_id'][:8]}…)")
        return True
    finally:
        conn.close()


def list_documents() -> List[dict]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM documents ORDER BY updated_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def print_registry() -> None:
    docs = list_documents()
    print(f"{'状态':<6} {'类型':<5} {'块数':>4}  {'文件名':<30} 标题 / 错误")
    print("-" * 90)
    for d in docs:
        print(f"{d['status']:<6} {d['file_type']:<5} {d['chunk_count']:>4}  {d['file_name']:<30} "
              f"{(d['title'] or d['error'] or '')[:40]}")
    print(f"\n共 {len(docs)} 条登记")
