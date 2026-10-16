"""统一模型网关：项目内所有 LLM / Embedding / 重排调用都从这里走。

配置全部来自项目根目录 .env：
  api_base_url      OpenAI 兼容接口地址（chat / embedding 走这里，当前指向 agicto）
  api_key           主网关密钥
  model             问答用大模型名（如 deepseek-v4.1-flash）
  dashscope_api_key 重排专用（gte-rerank-v2 为 DashScope 原生模型，兼容网关代理不了；
                    未配置时回退 api_key）

embedding 固定 text-embedding-v1（1536 维，与既有向量库兼容；换 v3/更高版需全量重建向量库），
重排固定 gte-rerank-v2（DashScope 原生）。
主网关可随时通过 .env 的 api_base_url 整体切换，代码零改动。
"""
import os
import sys
import io
import time
from typing import Dict, List, Optional, Type

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
from pydantic import BaseModel
from json_repair import repair_json
from tenacity import retry, wait_fixed, stop_after_attempt

try:  # Windows 控制台 GBK 兜底，失败不影响主流程
    if hasattr(sys.stdout, "buffer") and (sys.stdout.encoding or "").lower() not in ("utf-8", "utf8"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

# override=True：项目 .env 优先于系统环境变量（机器上残留过旧的用户级 api_base_url，
# 会把请求打到错误网关，这里强制以 .env 为准）
load_dotenv(override=True)
model = os.getenv("model")
key = os.getenv("api_key")
url = os.getenv("api_base_url")

serpapi_key = os.getenv("serpapi_key")

if not key or not url:
    raise RuntimeError("缺少模型网关配置：请在项目根目录 .env 中配置 api_key 与 api_base_url")

EMBED_MODEL = "text-embedding-v1"  # 1536 维固定，不支持 dimensions 参数；与旧向量库兼容
# EMBED_MODEL = "qwen3.7-text-embedding"
RERANK_MODEL = "gte-rerank-v2"
EMBED_BATCH = 25  # DashScope text-embedding-v1 单次批量上限

llm = ChatOpenAI(
    api_key=key,
    base_url=url,
    model=model,
    temperature=0.1,
    timeout=None,
    max_retries=2,
)

embeddings = OpenAIEmbeddings(
    api_key=key,
    base_url=url,
    model=EMBED_MODEL,
    check_embedding_ctx_length=False,
    chunk_size=EMBED_BATCH,
)

# 最近一次调用的 token 用量（结构化调用会丢失 usage，这里兜底记录）
last_usage: Dict = {}


def _normalize(vec: List[float]) -> List[float]:
    """L2 归一化：FAISS IndexFlatIP 内积即余弦相似度，分数越高越相关。"""
    import math
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        raise RuntimeError("embedding 向量范数为 0，疑似接口返回异常")
    return [x / norm for x in vec]


@retry(wait=wait_fixed(5), stop=stop_after_attempt(3))
def embed_query(text: str) -> List[float]:
    """单条查询向量（已 L2 归一化）。"""
    if not text or not text.strip():
        raise ValueError("embed_query 输入不能为空")
    vec = embeddings.embed_query(text)
    return _normalize(vec)


@retry(wait=wait_fixed(5), stop=stop_after_attempt(3))
def embed_texts(texts: List[str]) -> List[List[float]]:
    """批量文档向量：过滤空串 + 归一化。批量切分由 OpenAIEmbeddings(chunk_size) 处理。

    返回顺序与传入的过滤后文本一一对应。
    """
    cleaned = [t for t in texts if isinstance(t, str) and t.strip()]
    if not cleaned:
        raise ValueError("embed_texts 输入全为空字符串")
    vectors = embeddings.embed_documents(cleaned)
    return [_normalize(v) for v in vectors]


def chat(system_content: str, human_content: str, temperature: float = 0.1) -> str:
    """普通对话，返回文本。"""
    global last_usage
    global llm
    resp = llm.bind(temperature=temperature).invoke(
        [
            ("system", system_content),
            ("human", human_content),
        ]
    )
    usage = resp.response_metadata.get("token_usage") or {}
    last_usage = {
        "model": resp.response_metadata.get("model_name", model),
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
    }
    return resp.content


def chat_structured(
    system_content: str,
    human_content: str,
    schema: Type[BaseModel],
    temperature: float = 0.1,
) -> Dict:
    """结构化输出：优先 function calling，失败回退 json_mode + json_repair。

    返回 schema 校验后的 dict。
    """
    global last_usage
    # 第一次尝试：原生结构化（DashScope 兼容模式对 qwen 系支持 tool calling）
    try:
        structured_llm = llm.bind(temperature=temperature).with_structured_output(schema)
        result = structured_llm.invoke(
            [
                ("system", system_content),
                ("human", human_content),
            ]
        )
        if isinstance(result, BaseModel):
            last_usage = {"model": model, "input_tokens": None, "output_tokens": None}
            return result.model_dump()
        raise RuntimeError(f"with_structured_output 返回类型异常: {type(result)}")
    except Exception as e:
        print(f"[LLMClient] 结构化输出首次尝试失败，回退 json_mode: {type(e).__name__}: {e}")

    # 回退：json_mode + 提示词内嵌 schema
    schema_json = schema.model_json_schema()
    fallback_system = (
        f"{system_content}\n\n"
        "你必须只输出一个合法 JSON 对象，不要输出任何其他文字或代码块标记。\n"
        f"JSON 必须符合如下 JSON Schema：\n{schema_json}"
    )
    raw = chat(fallback_system, human_content, temperature=temperature)
    repaired = repair_json(raw)
    parsed = schema.model_validate_json(repaired)
    return parsed.model_dump()


def get_ali_rerank(top_n: int = 3):
    """阿里 gte-rerank-v2 重排器实例（LangChain DocumentCompressor 接口）。"""
    from langchain_community.document_compressors.dashscope_rerank import DashScopeRerank
    return DashScopeRerank(
        model=RERANK_MODEL,
        dashscope_api_key=key,
        top_n=top_n,
    )


@retry(wait=wait_fixed(5), stop=stop_after_attempt(3))
def rerank_texts(query: str, texts: List[str], top_n: Optional[int] = None) -> List[Dict]:
    """对文本列表做 gte-rerank-v2 重排。

    直接裸调 dashscope.TextReRank（langchain_community 的 DashScopeRerank 封装
    与新版 SDK 不兼容，吞错后报 output.results 为 None，故不使用）。
    返回 [{'index': 原始下标, 'text': 文本, 'relevance_score': 分数}]，按分数降序。
    """
    if not texts:
        return []
    import dashscope
    # gte-rerank-v2 是 DashScope 原生模型：主网关切到兼容代理（如 agicto）时，
    # 重排仍需 DashScope key，由 .env 的 dashscope_api_key 单独提供
    dashscope.api_key = os.getenv("dashscope_api_key") or key
    from dashscope import TextReRank
    resp = TextReRank.call(
        model=RERANK_MODEL,
        query=query,
        documents=texts,
        top_n=top_n or len(texts),
        return_documents=False,
    )
    if not resp.output or not resp.output.results:
        raise RuntimeError(f"重排接口返回异常: status={resp.status_code} code={resp.code} message={resp.message}")
    results = []
    for r in resp.output.results:
        results.append({
            "index": r.index,
            "text": texts[r.index],
            "relevance_score": float(r.relevance_score),
        })
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    return results
