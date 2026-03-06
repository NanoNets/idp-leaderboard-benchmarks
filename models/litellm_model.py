"""
LiteLLM-based model adapter for multi-provider model calls.

Supports any provider LiteLLM handles (Anthropic, OpenAI, Gemini, etc.)
via a unified interface matching the signatures used by the benchmark runners.

API keys are read from environment variables automatically by LiteLLM:
  ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, etc.
"""

import base64
import io
import logging
import os
import time
from pathlib import Path

import httpx

# Auto-load .env from repo root if present
_REPO_ROOT = Path(__file__).resolve().parents[1]
_env_file = _REPO_ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip("'\""))

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 8192
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF = 10.0
DEFAULT_RETRY_MAX = 120.0


def _ensure_litellm():
    try:
        import litellm
        return litellm
    except ImportError:
        raise ImportError("litellm is required. Install with: pip install litellm")


def complete(
    messages: list[dict],
    model_id: str,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout: int = 300,
) -> str:
    """Call a model via LiteLLM and return the response text.

    Args:
        messages: OpenAI-format message list (supports vision content parts).
        model_id: LiteLLM model identifier, e.g. "anthropic/claude-sonnet-4-6".
        max_tokens: Maximum tokens in the response.
        temperature: Sampling temperature.
        max_retries: Number of retry attempts for transient errors.
        timeout: Request timeout in seconds.
    """
    litellm = _ensure_litellm()

    call_kwargs = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "drop_params": True,
        "timeout": timeout,
    }

    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            response = litellm.completion(**call_kwargs)
            return response.choices[0].message.content or ""
        except Exception as exc:
            last_exc = exc
            status = getattr(exc, "status_code", None)
            msg = str(exc).lower()
            retryable = (
                status in {408, 429, 500, 502, 503, 504}
                or any(m in msg for m in ["timeout", "rate limit", "too many requests",
                                          "bad gateway", "service unavailable"])
            )
            if not retryable or attempt >= max_retries:
                raise
            sleep_for = min(DEFAULT_RETRY_BACKOFF * (2 ** (attempt - 1)), DEFAULT_RETRY_MAX)
            logger.warning(
                "LiteLLM call failed (attempt %d/%d) for %s; retrying in %.0fs: %s",
                attempt, max_retries, model_id, sleep_for, type(exc).__name__,
            )
            time.sleep(sleep_for)

    raise last_exc  # unreachable but satisfies type checker


def _file_to_data_uri(
    file_path: str | Path | None = None,
    file_url: str = "",
    page_num: int = 0,
    dpi: int = 200,
) -> str:
    """Convert a local file or remote URL to a base64 data URI for vision models."""
    if file_url:
        resp = httpx.get(file_url, timeout=60, follow_redirects=True)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "image/png")
        if "pdf" in content_type:
            return _pdf_bytes_to_data_uri(resp.content, page_num, dpi)
        mime = "image/jpeg" if "jpeg" in content_type or "jpg" in content_type else "image/png"
        b64 = base64.b64encode(resp.content).decode()
        return f"data:{mime};base64,{b64}"

    if file_path:
        file_path = Path(file_path)
        if file_path.suffix.lower() == ".pdf":
            return _pdf_bytes_to_data_uri(file_path.read_bytes(), page_num, dpi)
        mime_map = {
            ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".gif": "image/gif", ".webp": "image/webp",
        }
        mime = mime_map.get(file_path.suffix.lower(), "image/png")
        b64 = base64.b64encode(file_path.read_bytes()).decode()
        return f"data:{mime};base64,{b64}"

    raise ValueError("Provide either file_path or file_url")


def _pdf_bytes_to_data_uri(pdf_bytes: bytes, page_num: int = 0, dpi: int = 200) -> str:
    """Render a PDF page to a PNG data URI."""
    import fitz
    from PIL import Image

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(page_num)
    pix = page.get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def extract_text(
    file_path: str | Path | None = None,
    *,
    file_url: str = "",
    model_id: str,
    custom_instructions: str = "",
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    **kwargs,
) -> str:
    """Convert a document to markdown text using a vision LLM via LiteLLM.

    Drop-in replacement for models.nanonets.extract_text when using
    third-party models (Claude, GPT, Gemini, etc.).
    """
    data_uri = _file_to_data_uri(file_path=file_path, file_url=file_url)
    prompt = custom_instructions or (
        "Convert this document page to markdown. "
        "Preserve all text, tables, and equations exactly as shown."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return complete(messages, model_id, max_tokens=max_tokens,
                    temperature=temperature, max_retries=max_retries)


def chat(
    image_url: str = "",
    prompt: str = "",
    *,
    file_path: str | Path | None = None,
    model_id: str,
    page_num: int = 0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    dpi: int = 200,
    **kwargs,
) -> str:
    """Chat with a vision model about a document image via LiteLLM.

    Drop-in replacement for models.nanonets.chat when using
    third-party models (Claude, GPT, Gemini, etc.).
    """
    data_uri = _file_to_data_uri(
        file_path=file_path, file_url=image_url, page_num=page_num, dpi=dpi,
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return complete(messages, model_id, max_tokens=max_tokens,
                    temperature=temperature, max_retries=max_retries)
