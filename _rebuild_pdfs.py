"""重建所有 PDF 文档，使图片 OCR 回填生效。"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import doc_store

docs = [d for d in doc_store.list_documents()
        if d["file_type"] == "pdf" and d["status"] == "ready"]
print(f"待重建 {len(docs)} 份 PDF")
for d in docs:
    print("→", d["file_name"])
    doc_store.rebuild_file(d["doc_id"])

print("\n重建完成，登记表现状：")
doc_store.print_registry()
