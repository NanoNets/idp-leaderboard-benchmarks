#!/usr/bin/env python3
"""
Populate the IDP prediction cache by calling the Nanonets API.

Routes tasks to the appropriate endpoint:
  - KIE          -> Extract API with output_format=json + json_options
  - OCR, VQA, TABLE -> Chat API (OpenAI-compatible)

TABLE goes through Chat because the Extract API's JSON mode is
KIE-oriented (flat single-object output) and cannot return multi-row
table data.

Uses docext for dataset loading and message construction.

Requires:
  pip install -e ../docext

Usage:
  python benchmarks/idp/run.py --workers 5
  python benchmarks/idp/run.py --model nanonets --tasks KIE OCR
"""

import argparse
import base64
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO, StringIO
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from models.nanonets import chat_messages, extract_json

_MODEL_TYPE = ""
_PROVIDER = "nanonets"
_LITELLM_MODEL_ID = ""
_REQUEST_DELAY = 0.0

try:
    from docext.benchmark.benchmark import NanonetsIDPBenchmark
    from docext.benchmark.tasks import (
        get_KIE_messages,
        get_OCR_messages,
        get_VQA_messages,
        get_TABLE_messages,
    )
except ImportError:
    print(
        'ERROR: docext not installed.\n'
        '  Run: pip install -e ../docext',
        file=sys.stderr,
    )
    sys.exit(1)


CHAT_TASKS = {"OCR", "VQA", "TABLE"}
EXTRACT_TASKS = {"KIE"}

TASK_MESSAGE_BUILDERS = {
    "KIE": get_KIE_messages,
    "OCR": get_OCR_messages,
    "VQA": get_VQA_messages,
    "TABLE": get_TABLE_messages,
}


def _extract_json_from_response(raw: str) -> str:
    """Strip thinking/reasoning tags and code fences to extract pure JSON."""
    text = raw.strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    if text.startswith("```"):
        text = re.sub(r'^```\w*\n?', '', text)
        text = re.sub(r'\n?```$', '', text).strip()
    return text


def _call_litellm_kie(data, max_retries: int) -> dict | list:
    """Call LiteLLM for a KIE item — builds a vision message with the image and a JSON prompt."""
    from models.litellm_model import complete as litellm_complete, _file_to_data_uri

    image_path = data.image_paths[0]
    field_names = [f.label for f in (data.fields or []) if f is not None]
    labels_text = ", ".join(field_names) if field_names else "provided field labels"

    data_uri = _file_to_data_uri(file_path=image_path)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": data_uri}},
            {"type": "text", "text": (
                "You are doing key information extraction from a document. "
                "Return ONLY a valid JSON object and nothing else. "
                f"Use exactly these keys: {labels_text}. "
                "Do not add extra keys. For missing values return empty string."
            )},
        ],
    }]

    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            raw = litellm_complete(messages, _LITELLM_MODEL_ID, max_retries=1)
            stripped = _extract_json_from_response(raw)
            return json.loads(stripped)
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries:
                backoff = min(30 * (2 ** (attempt - 1)), 120)
                time.sleep(backoff)
    raise RuntimeError(f"LiteLLM KIE failed after {max_retries} attempts: {last_error}")


def _call_litellm_chat(messages: list, task: str, max_retries: int) -> str:
    """Call LiteLLM with pre-built OpenAI-format messages from docext."""
    from models.litellm_model import complete as litellm_complete

    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            raw = litellm_complete(messages, _LITELLM_MODEL_ID, max_retries=1)
            if "</think>" in raw:
                raw = raw.split("</think>", 1)[1].strip()
            if task == "TABLE":
                raw = _postprocess_table_response(raw)
            return raw
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries:
                backoff = min(30 * (2 ** (attempt - 1)), 120)
                time.sleep(backoff)
    raise RuntimeError(f"LiteLLM {task} failed after {max_retries} attempts: {last_error}")


def _call_extract_kie(data, max_retries: int) -> dict | list:
    """Call extract_json for a KIE item, matching the Nanobench approach.

    Sends json_options as a field list and custom_instructions with a strict
    KIE prompt that forces JSON-only output with exact keys.
    """
    image_path = data.image_paths[0]
    field_names = [f.label for f in (data.fields or []) if f is not None]
    json_options = json.dumps(field_names, ensure_ascii=False)
    labels_text = ", ".join(field_names) if field_names else "provided field labels"
    custom_instructions = (
        "You are doing key information extraction from a document. "
        "Return ONLY a valid JSON object and nothing else. "
        f"Use exactly these keys: {labels_text}. "
        "Do not add extra keys. For missing values return empty string."
    )

    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            result = extract_json(
                file_path=image_path,
                json_options=json_options,
                custom_instructions=custom_instructions,
                prompt_mode="replace",
                model_type=_MODEL_TYPE,
            )
            return result
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"KIE extract failed after {max_retries} attempts: {last_error}")


def _postprocess_table_response(raw: str) -> str:
    """Convert HTML/markdown table responses from Chat to JSON row arrays.

    The Chat endpoint often returns HTML tables. We convert them to the
    JSON array-of-row-objects format expected by the benchmark scorer.
    """
    stripped = raw.strip()
    if "</think>" in stripped:
        stripped = stripped.split("</think>", 1)[1].strip()

    if stripped.startswith("```"):
        stripped = re.sub(r'^```\w*\n?', '', stripped)
        stripped = re.sub(r'\n?```$', '', stripped).strip()

    if stripped.startswith("[") or stripped.startswith("{"):
        return stripped

    if "<table" in stripped.lower():
        try:
            dfs = pd.read_html(StringIO(stripped))
            if dfs:
                df = dfs[0].fillna("")
                rows = df.to_dict(orient="records")
                return json.dumps(rows, ensure_ascii=False)
        except Exception:
            pass

    if "|" in stripped:
        try:
            lines = [
                l for l in stripped.split("\n")
                if l.strip() and not all(c in "-| :" for c in l.strip())
            ]
            if len(lines) >= 2 and "|" in lines[0]:
                headers = [h.strip() for h in lines[0].split("|") if h.strip()]
                rows = []
                for line in lines[1:]:
                    vals = [v.strip() for v in line.split("|") if v.strip()]
                    row = {h: (vals[i] if i < len(vals) else "") for i, h in enumerate(headers)}
                    rows.append(row)
                if rows:
                    return json.dumps(rows, ensure_ascii=False)
        except Exception:
            pass

    return stripped


MAX_IMAGES_PER_REQUEST = 20


def _compress_messages(messages: list) -> list:
    """Compress base64 images in messages to keep the payload within API limits.

    Adapts resolution and quality based on the number of images:
    - 1 image:  up to 1568px, JPEG quality 85
    - 2-5:      up to 1024px, JPEG quality 75
    - 6-15:     up to 768px,  JPEG quality 60
    - 16+:      up to 512px,  JPEG quality 50
    Caps total images at MAX_IMAGES_PER_REQUEST.
    """
    total_imgs = sum(
        1 for msg in messages if isinstance(msg.get("content"), list)
        for b in msg["content"] if b.get("type") == "image_url"
    )
    if total_imgs <= 1:
        max_dim, quality = 1568, 85
    elif total_imgs <= 5:
        max_dim, quality = 1024, 75
    elif total_imgs <= 15:
        max_dim, quality = 768, 60
    else:
        max_dim, quality = 512, 50

    compressed = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            compressed.append(msg)
            continue

        new_content = []
        img_count = 0
        for block in content:
            if block.get("type") == "image_url":
                img_count += 1
                if img_count > MAX_IMAGES_PER_REQUEST:
                    continue
                url = block["image_url"]["url"]
                if url.startswith("data:image/"):
                    try:
                        _, b64_data = url.split(",", 1)
                        img = Image.open(BytesIO(base64.b64decode(b64_data)))
                        w, h = img.size
                        if max(w, h) > max_dim:
                            scale = max_dim / max(w, h)
                            img = img.resize(
                                (int(w * scale), int(h * scale)), Image.LANCZOS
                            )
                        if img.mode in ("RGBA", "P"):
                            img = img.convert("RGB")
                        buf = BytesIO()
                        img.save(buf, format="JPEG", quality=quality)
                        new_b64 = base64.b64encode(buf.getvalue()).decode()
                        new_content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{new_b64}"},
                        })
                    except Exception:
                        new_content.append(block)
                else:
                    new_content.append(block)
            else:
                new_content.append(block)
        compressed.append({**msg, "content": new_content})
    return compressed


def _extract_vqa_question(text: str) -> str:
    """Extract the raw question from docext's VQA template format.

    Handles: "Answer the following question based on the images shared: {q}. Do not ..."
    """
    marker = "Answer the following question based on the images shared:"
    if marker in text:
        after = text.split(marker, 1)[1].strip()
        suffix = ". Do not give any explanation"
        idx = after.find(suffix)
        if idx > 0:
            return after[:idx].strip()
        return after.split(".")[0].strip() if "." in after else after.strip()
    for m in ["Question:", "question:"]:
        if m in text:
            return text.split(m, 1)[1].strip()
    lines = [l.strip() for l in text.split("\n") if l.strip() and "?" in l]
    return lines[-1] if lines else text.strip()


def _rewrite_messages_for_vqa(messages: list) -> list:
    """Rewrite docext VQA messages into a clean single-turn prompt.

    Matches the Nanobench approach: extract all images and the question,
    then build a concise prompt asking for a direct answer.
    """
    images = []
    texts = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            if content:
                texts.append(content)
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "image_url":
                    images.append(part)
                elif part.get("type") == "text" and part.get("text"):
                    texts.append(part["text"])

    combined_text = "\n".join(texts)
    question = _extract_vqa_question(combined_text)

    content_parts = list(images)
    content_parts.append({
        "type": "text",
        "text": (
            "Answer the following question based on the document image. "
            "Return ONLY the answer as a single concise value "
            "(number, word, or short phrase). No explanation, no JSON, no markdown.\n\n"
            f"Question: {question}"
        ),
    })
    return [{"role": "user", "content": content_parts}]


def _rewrite_messages_for_table(messages: list) -> list:
    """Rewrite docext messages into a single user turn with table-specific prompt.

    Extracts all base64 images and constructs a clean single-turn message
    that asks for a JSON array of row objects (matching Nanobench approach).
    """
    images = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    images.append(part)

    content_parts = list(images)
    content_parts.append({
        "type": "text",
        "text": (
            "Extract ALL tables from this document image. "
            "Return the result as a JSON array of row objects, where each "
            "object represents one row with column headers as keys. Example:\n"
            '[{"Col1":"val1","Col2":"val2"},{"Col1":"val3","Col2":"val4"}]\n'
            "Return ONLY valid JSON. No HTML, no markdown, no explanation."
        ),
    })
    return [{"role": "user", "content": content_parts}]


def _call_chat(data, template: dict, task: str, max_retries: int) -> str:
    """Call chat_messages for OCR/VQA/TABLE using docext message builders."""
    builder = TASK_MESSAGE_BUILDERS[task]
    messages = builder(data, template)

    if task == "TABLE":
        messages = _rewrite_messages_for_table(messages)
    elif task == "VQA":
        messages = _rewrite_messages_for_vqa(messages)

    messages = _compress_messages(messages)

    if _PROVIDER == "litellm":
        return _call_litellm_chat(messages, task, max_retries)

    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            raw = chat_messages(messages, model_type=_MODEL_TYPE)
            if task == "TABLE":
                raw = _postprocess_table_response(raw)
            return raw
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"{task} chat failed after {max_retries} attempts: {last_error}")


def _process_item(
    data, template: dict, task: str, cache_file: Path,
    dataset_name: str, index: int, max_retries: int,
) -> dict:
    """Process a single item: check cache, call API, save result."""
    if cache_file.exists():
        return {"status": "cached", "task": task, "dataset": dataset_name, "index": index}

    try:
        if task == "KIE":
            if _PROVIDER == "litellm":
                endpoint = "litellm_kie"
                response = _call_litellm_kie(data, max_retries)
            else:
                endpoint = "extract_json"
                response = _call_extract_kie(data, max_retries)
        else:
            endpoint = "chat"
            response = _call_chat(data, template, task, max_retries)

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_data = {
            "task": task,
            "dataset": dataset_name,
            "index": index,
            "endpoint": endpoint,
            "response": response,
            "timestamp": time.time(),
        }
        cache_file.write_text(
            json.dumps(cache_data, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        if _REQUEST_DELAY > 0:
            time.sleep(_REQUEST_DELAY)
        return {"status": "ok", "task": task, "dataset": dataset_name, "index": index}
    except Exception as e:
        return {
            "status": "error", "task": task, "dataset": dataset_name,
            "index": index, "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Populate IDP prediction cache via Nanonets API")
    parser.add_argument("--model", type=str, default="nanonets")
    parser.add_argument("--config", type=str, default=None, help="Override IDP config YAML")
    parser.add_argument("--tasks", type=str, nargs="*", default=None, help="Only run specific tasks")
    parser.add_argument("--datasets", type=str, nargs="*", default=None, help="Only run specific datasets")
    parser.add_argument("--skip-datasets", type=str, nargs="*", default=None, help="Skip specific datasets")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--model-type", type=str, default="", help="API model_type field (e.g. nanonets-ocr-3)")
    parser.add_argument("--provider", type=str, choices=["nanonets", "litellm"], default="nanonets",
                        help="API provider: nanonets (default) or litellm")
    parser.add_argument("--model-id", type=str, default="",
                        help="LiteLLM model identifier (e.g. openai/gpt-5.4-2026-03-05)")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Seconds to sleep between successful API calls (rate limit protection)")
    args = parser.parse_args()

    global _MODEL_TYPE, _PROVIDER, _LITELLM_MODEL_ID, _REQUEST_DELAY
    _MODEL_TYPE = args.model_type
    _PROVIDER = args.provider
    _LITELLM_MODEL_ID = args.model_id
    _REQUEST_DELAY = args.delay

    if _PROVIDER == "litellm" and not _LITELLM_MODEL_ID:
        print("ERROR: --model-id is required when using --provider litellm", file=sys.stderr)
        sys.exit(1)

    config_path = Path(args.config) if args.config else Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        bench_cfg = yaml.safe_load(f)

    cache_root = REPO_ROOT / "caches" / args.model / "idp"
    cache_root.mkdir(parents=True, exist_ok=True)

    all_tasks = bench_cfg.get("tasks", [])
    tasks_to_run = set(args.tasks) if args.tasks else set(all_tasks)
    templates = {
        "KIE": bench_cfg.get("KIE_default_template", {}),
        "OCR": bench_cfg.get("OCR_default_template", {}),
        "VQA": bench_cfg.get("VQA_default_template", {}),
        "TABLE": bench_cfg.get("TABLE_default_template", {}),
    }

    print(f"=== IDP Runner (model={args.model}) ===\n")
    print(f"  Tasks: {sorted(tasks_to_run)}")
    print(f"  Config: {config_path}")
    print(f"  Cache: {cache_root}")

    idp_benchmark = NanonetsIDPBenchmark(str(config_path))
    datasets = idp_benchmark.datasets
    print(f"  Datasets loaded: {len(datasets)}\n")

    total_ok = total_cached = total_errors = 0

    for dataset in datasets:
        ds_name = getattr(dataset, "name", dataset.__class__.__name__)
        task = dataset.task
        if task not in tasks_to_run:
            continue
        if args.datasets and ds_name not in args.datasets:
            continue
        if args.skip_datasets and ds_name in args.skip_datasets:
            print(f"  SKIPPING {ds_name} (--skip-datasets)")
            continue

        data_items = list(getattr(dataset, "data", []))
        max_samples = bench_cfg.get("max_samples_per_dataset", 1000)
        if len(data_items) > max_samples:
            data_items = data_items[:max_samples]

        ds_cache = cache_root / ds_name
        ds_cache.mkdir(parents=True, exist_ok=True)

        endpoint_map = {"KIE": "Extract JSON", "OCR": "Chat", "VQA": "Chat", "TABLE": "Chat"}
        endpoint_label = endpoint_map.get(task, "Chat")
        print(f"  {ds_name} ({task} via {endpoint_label}): {len(data_items)} items")

        template = templates.get(task, {})
        jobs = []
        for idx, data in enumerate(data_items):
            cache_file = ds_cache / f"{idx}.json"
            jobs.append((data, template, task, cache_file, ds_name, idx, args.max_retries))

        ok = cached = errors = 0
        t0 = time.time()

        if args.workers <= 1:
            for job in jobs:
                result = _process_item(*job)
                if result["status"] == "cached":
                    cached += 1
                elif result["status"] == "ok":
                    ok += 1
                else:
                    errors += 1
                    print(f"    ERROR [{result['index']}]: {result.get('error', '')[:100]}")
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(_process_item, *job): job[5] for job in jobs}
                for future in as_completed(futures):
                    result = future.result()
                    if result["status"] == "cached":
                        cached += 1
                    elif result["status"] == "ok":
                        ok += 1
                    else:
                        errors += 1
                        print(f"    ERROR [{result['index']}]: {result.get('error', '')[:100]}", flush=True)
                    done = ok + cached + errors
                    if ok > 0 and done % 50 == 0:
                        elapsed = time.time() - t0
                        rate = ok / elapsed if elapsed > 0 else 0
                        print(f"    [{done}/{len(jobs)}] {rate:.1f} new/s  errors={errors}", flush=True)

        dt = time.time() - t0
        print(f"    Done in {dt:.0f}s: {ok} new, {cached} cached, {errors} errors")
        total_ok += ok
        total_cached += cached
        total_errors += errors

    print(f"\n=== Summary ===")
    print(f"  New: {total_ok}, Cached: {total_cached}, Errors: {total_errors}")
    print(f"  Cache: {cache_root}")


if __name__ == "__main__":
    main()
