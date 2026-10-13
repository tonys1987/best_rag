"""解析器公共基础：统一返回结构、编码回退读取。"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass
class ParseResult:
    """所有解析器的统一产出：markdown 文本 + 元信息。

    chunking 层只看 markdown；meta 供登记表和引用定位使用。
    """
    markdown: str
    meta: Dict = field(default_factory=dict)
    source_path: str = ""

    @property
    def title(self) -> str:
        return self.meta.get("title", Path(self.source_path).stem)


def read_text_with_fallback(path: Path) -> str:
    """读文本：utf-8-sig 优先，失败回退 gbk，再失败按 latin-1 兜底不报错。"""
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def result_to_json(res: ParseResult) -> str:
    return json.dumps(res.meta, ensure_ascii=False, indent=2)
