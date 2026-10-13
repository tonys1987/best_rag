"""pdf 解析：本地 MinerU（magic_pdf + ModelScope 模型），GPU 自动检测。

复用 batch_run.py 已验证的调用链：PymuDocDataset + doc_analyze(doclayout_yolo)
+ pipe_ocr/txt_mode + dump_md。设备检测有 CUDA 用 GPU，否则 CPU。
"""
import json
import os
from pathlib import Path

from src.parsers.base import ParseResult

PROJECT_DIR = Path(__file__).resolve().parents[2]
LOCAL_MODEL_DIR = PROJECT_DIR / "modelscope_models"
CONFIG_PATH = Path.home() / "magic-pdf.json"


def detect_device() -> str:
    """torch.cuda.is_available() -> cuda，否则 cpu。

    注意 magic-pdf.json 的 device-mode 合法值是 cuda/cpu（torch 设备名），
    写 "gpu" 会让 magic-pdf 把它原样传给 model.to() 而报错。
    """
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def ensure_magic_pdf_config() -> Path:
    """确保 magic-pdf.json 指向本项目模型目录，并按当前硬件写入 device-mode。

    模型缺失时提示先跑 batch_run.py 下载（modelscope_models/ 已有则零下载）。
    """
    if not (LOCAL_MODEL_DIR / "models").exists() or not (LOCAL_MODEL_DIR / "layoutreader").exists():
        raise RuntimeError(
            f"MinerU 模型缺失（期望目录：{LOCAL_MODEL_DIR}/models 与 /layoutreader）。"
            "请先运行: venv\\Scripts\\python batch_run.py  完成模型准备。"
        )
    layoutreader_dir = str(LOCAL_MODEL_DIR / "layoutreader")
    device = detect_device()

    data = {}
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            data = {}

    if data.get("models-dir") != str(LOCAL_MODEL_DIR / "models") \
            or data.get("layoutreader-model-dir") != layoutreader_dir \
            or data.get("device-mode") != device:
        data["models-dir"] = str(LOCAL_MODEL_DIR / "models")
        data["layoutreader-model-dir"] = layoutreader_dir
        data["device-mode"] = device
        data.setdefault("table-config", {"model": "rapid_table", "enable": False})
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    return CONFIG_PATH


def parse_pdf(path: Path, output_dir: Path, method: str = "auto") -> ParseResult:
    """本地 MinerU 解析单个 PDF，返回 markdown。method: auto/txt/ocr。"""
    config_path = ensure_magic_pdf_config()
    os.environ["MINERU_TOOLS_CONFIG_JSON"] = str(config_path)

    # 延迟导入：magic_pdf 依赖较重，且必须在设置配置环境变量之后导入
    from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
    from magic_pdf.data.dataset import PymuDocDataset
    from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
    from magic_pdf.config.enums import SupportedPdfParseMethod

    path = Path(path)
    output_dir = Path(output_dir)
    local_image_dir = output_dir / "images"
    local_image_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_writer = FileBasedDataWriter(str(local_image_dir))
    md_writer = FileBasedDataWriter(str(output_dir))

    reader = FileBasedDataReader("")
    pdf_bytes = reader.read(str(path))

    ds = PymuDocDataset(pdf_bytes)
    doc_type = ds.classify()

    if method == "ocr":
        use_ocr = True
    elif method == "txt":
        use_ocr = False
    else:  # auto：按文档类型
        use_ocr = doc_type == SupportedPdfParseMethod.OCR

    layout_model = "doclayout_yolo"
    infer_result = ds.apply(doc_analyze, layout_model=layout_model, ocr=use_ocr)
    pipe_result = infer_result.pipe_ocr_mode(image_writer) if use_ocr else infer_result.pipe_txt_mode(image_writer)

    md_name = f"{path.stem}.md"
    pipe_result.dump_md(md_writer, md_name, os.path.basename(str(local_image_dir)))
    md_path = output_dir / md_name

    md_text = md_path.read_text(encoding="utf-8")
    return ParseResult(
        markdown=md_text,
        meta={
            "title": path.stem,
            "format": "pdf",
            "device": detect_device(),
            "parse_method": "ocr" if use_ocr else "txt",
            "md_path": str(md_path),
        },
        source_path=str(path),
    )
