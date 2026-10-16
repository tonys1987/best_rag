"""图片内容 OCR 回填：让图片里的表格/文字可被检索。

MinerU 对部分表格只输出 ![](images/x.jpg) 图片引用（table-config enable=False 时），
纯文本检索永远命不中。本模块用 RapidOCR 把引用图片识别为文本，
追加在原图引用之后（【图片内容OCR】块），再入库即可命中。

带磁盘缓存（<images_dir>/../ocr_cache/<图片名>.txt）：
MinerU 的图片名即内容 hash，同图重复入库零开销。
"""
import re
from pathlib import Path

# 匹配 MinerU markdown 中的图片引用：![](images/xxx.jpg)
IMG_RE = re.compile(r"!\[[^\]]*\]\((images/[^)\s]+)\)")

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        from rapidocr_onnxruntime import RapidOCR
        _engine = RapidOCR()
    return _engine


def ocr_image_file(img_path: Path, cache_dir: Path) -> str:
    """OCR 单张图片，返回按阅读顺序拼接的文本。带缓存。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / (img_path.stem + ".txt")
    if cache.exists():
        return cache.read_text(encoding="utf-8")

    eng = _get_engine()
    res, _ = eng(str(img_path))

    lines = []
    if res:
        # 按 y 坐标聚成行（y 差 <=12 视为同一行），行内按 x 排序
        items = sorted(res, key=lambda r: (min(p[1] for p in r[0]), min(p[0] for p in r[0])))
        rows, last_y = [], None
        for box, text, _score in items:
            y = min(p[1] for p in box)
            if last_y is not None and abs(y - last_y) <= 12:
                rows[-1].append(text)
            else:
                rows.append([text])
            last_y = y
        lines = [" | ".join(r) if len(r) > 1 else r[0] for r in rows]

    text = "\n".join(lines)
    cache.write_text(text, encoding="utf-8")
    return text


def augment_markdown(md_text: str, images_dir: Path, cache_dir: Path = None):
    """对 markdown 中每处图片引用追加 OCR 文本块。

    返回 (新markdown, ocr成功张数)。图片缺失或识别为空则保持原样。
    """
    images_dir = Path(images_dir)
    if cache_dir is None:
        cache_dir = images_dir.parent / "ocr_cache"
    count = 0

    def repl(m: re.Match) -> str:
        nonlocal count
        tag = m.group(0)
        img = images_dir / Path(m.group(1)).name
        if not img.exists():
            return tag
        try:
            text = ocr_image_file(img, cache_dir)
        except Exception as e:  # OCR 引擎异常不阻断入库
            print(f"[ocr_images] 识别失败 {img.name}: {e}")
            return tag
        if not text.strip():
            return tag
        count += 1
        return tag + "\n\n【图片内容OCR】\n" + text

    return IMG_RE.sub(repl, md_text), count
