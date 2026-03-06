"""
Thin wrapper around the Nanonets OCR2+ API.

Supports both Extract (sync) and Chat (OpenAI-compatible) endpoints.
All functions accept either a local file path or a remote URL via file_url.
"""

import base64
import io
import json
import mimetypes
import os
from pathlib import Path

import httpx

DEFAULT_BASE_URL = "https://extraction-api.nanonets.com"
DEFAULT_CHAT_MODEL = "nanonets/Nanonets-OCR-s"
DEFAULT_TIMEOUT = 300


def _get_api_key() -> str:
    key = os.environ.get("NANONETS_API_KEY", "")
    if not key:
        raise RuntimeError("NANONETS_API_KEY environment variable not set")
    return key


def _parse_response(body: dict) -> tuple[str, list]:
    """Parse the Extract API response into (markdown_text, bbox_elements).

    Handles two response formats:
      - With metadata:  body.result.markdown.content + body.result.markdown.metadata.bounding_boxes.elements
      - Without:        body.result (string) or body.result (list of page dicts)
    """
    if not body:
        return "", []
    result = body.get("result", "")

    if isinstance(result, dict):
        md = result.get("markdown") or {}
        if isinstance(md, dict):
            text = md.get("content", "")
            metadata = md.get("metadata") or {}
            bboxes = metadata.get("bounding_boxes") or {}
            elements = bboxes.get("elements", [])
            return text, elements

    if isinstance(result, list):
        text = "\n\n".join(
            p.get("markdown", p.get("text", ""))
            for p in result
            if isinstance(p, dict)
        )
        return text, []

    return str(result), []


def _guess_mime(file_path: str | Path) -> str:
    ext = Path(file_path).suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
        ".bmp": "image/bmp",
    }.get(ext, mimetypes.guess_type(str(file_path))[0] or "application/pdf")


def extract(
    file_path: str | Path | None = None,
    *,
    file_url: str = "",
    custom_instructions: str = "",
    prompt_mode: str = "",
    include_metadata: str = "",
    output_format: str = "markdown",
    json_options: str = "",
    model_type: str = "",
    base_url: str = DEFAULT_BASE_URL,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict:
    """Call the Nanonets Extract (sync) endpoint. Returns the raw JSON response.

    Provide exactly one of file_path (local file) or file_url (remote URL).
    When file_url is given, the API fetches the file server-side — no download needed.
    Use model_type to select a specific model variant (e.g. "nanonets-ocr-3").
    """
    api_key = _get_api_key()
    data: dict[str, str] = {"output_format": output_format}
    if custom_instructions:
        data["custom_instructions"] = custom_instructions
    if prompt_mode in ("append", "replace"):
        data["prompt_mode"] = prompt_mode
    if include_metadata:
        data["include_metadata"] = include_metadata
    if json_options:
        data["json_options"] = json_options
    if model_type:
        data["model_type"] = model_type

    if file_url:
        data["file_url"] = file_url
        resp = httpx.post(
            f"{base_url}/api/v1/extract/sync",
            headers={"Authorization": f"Bearer {api_key}"},
            data=data,
            timeout=timeout,
        )
    elif file_path:
        mime = _guess_mime(file_path)
        with open(file_path, "rb") as f:
            files = {"file": (Path(file_path).name, f, mime)}
            resp = httpx.post(
                f"{base_url}/api/v1/extract/sync",
                headers={"Authorization": f"Bearer {api_key}"},
                files=files,
                data=data,
                timeout=timeout,
            )
    else:
        raise ValueError("Provide either file_path or file_url")

    resp.raise_for_status()
    return resp.json()


def extract_text(
    file_path: str | Path | None = None, *, file_url: str = "", **kwargs
) -> str:
    """Convenience: call extract() and return just the markdown text."""
    body = extract(file_path, file_url=file_url, **kwargs)
    text, _ = _parse_response(body)
    return text


def extract_with_bbox(
    file_path: str | Path | None = None, *, file_url: str = "", **kwargs
) -> tuple[str, list]:
    """Call extract() with bounding boxes and return (text, bbox_elements).

    Each element in bbox_elements is a dict with keys like:
      {"type": "header", "content": "...", "bounding_box": {...}}

    Element types include: header, footer, number, text, formula,
    paragraph_title, doc_title, table, aside_text, image, footnote, etc.
    """
    kwargs.setdefault("include_metadata", "bounding_boxes")
    body = extract(file_path, file_url=file_url, **kwargs)
    return _parse_response(body)


def extract_json(
    file_path: str | Path | None = None,
    *,
    file_url: str = "",
    json_options: str = "",
    custom_instructions: str = "",
    prompt_mode: str = "",
    model_type: str = "",
    base_url: str = DEFAULT_BASE_URL,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict | list:
    """Call extract() with output_format=json and return the parsed JSON content.

    Use json_options to specify a field schema (for KIE) or row template (for TABLE).
    Examples:
      json_options='{"date": "", "name": ""}'          # KIE field schema
      json_options='[{"col1": "", "col2": ""}]'        # TABLE row template
    """
    body = extract(
        file_path, file_url=file_url, output_format="json",
        json_options=json_options, custom_instructions=custom_instructions,
        prompt_mode=prompt_mode, model_type=model_type,
        base_url=base_url, timeout=timeout,
    )
    result = body.get("result", {})
    if isinstance(result, dict):
        json_result = result.get("json") or {}
        if isinstance(json_result, dict):
            content = json_result.get("content", {})
            if isinstance(content, str):
                return json.loads(content)
            return content
    return result if result else {}


def chat(
    image_url: str = "",
    prompt: str = "",
    *,
    file_path: str | Path | None = None,
    page_num: int = 0,
    model: str = DEFAULT_CHAT_MODEL,
    model_type: str = "",
    base_url: str = DEFAULT_BASE_URL,
    timeout: int = DEFAULT_TIMEOUT,
    dpi: int = 200,
) -> str:
    """Call the Nanonets Chat endpoint (OpenAI-compatible).

    Provide either image_url (remote PNG/JPEG URL) or file_path (local PDF).
    When image_url is given, the image is fetched and base64-encoded.
    When file_path is given, the PDF page is rendered to PNG locally.
    """
    api_key = _get_api_key()

    if image_url:
        resp = httpx.get(image_url, timeout=60, follow_redirects=True)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "image/png")
        if "jpeg" in content_type or "jpg" in content_type:
            mime = "image/jpeg"
        else:
            mime = "image/png"
        b64 = base64.b64encode(resp.content).decode()
        data_uri = f"data:{mime};base64,{b64}"
    elif file_path:
        import fitz  # PyMuPDF -- only needed for local PDF rendering
        from PIL import Image

        doc = fitz.open(str(file_path))
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        doc.close()
        b64 = base64.b64encode(buf.getvalue()).decode()
        data_uri = f"data:image/png;base64,{b64}"
    else:
        raise ValueError("Provide either image_url or file_path")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    payload: dict = {"model": model, "messages": messages}
    if model_type:
        payload["model_type"] = model_type

    resp = httpx.post(
        f"{base_url}/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    body = resp.json()
    return body["choices"][0]["message"]["content"]


def chat_messages(
    messages: list[dict],
    *,
    model: str = DEFAULT_CHAT_MODEL,
    model_type: str = "",
    base_url: str = DEFAULT_BASE_URL,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """Send pre-built OpenAI-format messages directly to the Nanonets Chat endpoint.

    Useful when the caller already has messages with embedded base64 images
    (e.g. from docext message builders).
    """
    api_key = _get_api_key()
    payload: dict = {"model": model, "messages": messages}
    if model_type:
        payload["model_type"] = model_type

    resp = httpx.post(
        f"{base_url}/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    body = resp.json()
    return body["choices"][0]["message"]["content"]
