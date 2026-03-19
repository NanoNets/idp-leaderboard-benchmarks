#!/usr/bin/env python3
"""
Populate the olmOCR prediction cache by calling the Nanonets API.

Downloads ground truth from HuggingFace, discovers unique PDFs per category,
calls the appropriate Nanonets endpoint (chat / extract / extract+bbox),
and writes raw output to caches/{model}/olmocr/raw/{category}/{stem}.md.

Usage:
  python benchmarks/olmocr/run.py --workers 10
  python benchmarks/olmocr/run.py --model nanonets --categories arxiv_math tables
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from models.nanonets import extract_text, extract_with_bbox, chat

# Global config set from CLI args, passed to all API calls
_MODEL_TYPE = ""
_PROVIDER = "nanonets"
_LITELLM_MODEL_ID = ""
_REQUEST_DELAY = 0.0

# ---------------------------------------------------------------------------
# Prompts (inlined to keep this file self-contained)
# ---------------------------------------------------------------------------

OLMOCR_CHAT = (
    "Below is the image of one page of a PDF document.\n"
    "Return the plain text representation of this document "
    "as if you were reading it naturally.\n"
    "\n"
    "Rules:\n"
    "- Turn ALL equations and math symbols into LaTeX notation. "
    "Use \\( and \\) for inline math and \\[ and \\] for display/block math. "
    "Do NOT use unicode math symbols -- always use LaTeX commands instead "
    "(e.g., \\in not ∈, \\rightarrow not →).\n"
    "- Convert tables into markdown table format with proper column alignment.\n"
    "- Preserve all visible text with high fidelity, including headers, "
    "footers, references, and footnotes.\n"
    "- Maintain the correct multi-column reading order.\n"
    "- Read any natural handwriting.\n"
    "- If there is no text at all, output null.\n"
    "- Do not hallucinate.\n"
    "- Do not output image descriptions, alt-text, or <img> tags."
)

EXTRACT_LONG_TINY_TEXT = (
    "Extract ALL visible text from this document with maximum fidelity.\n"
    "Pay special attention to small, tiny, or fine-print text -- every word matters.\n"
    "Preserve the exact wording, spelling, and punctuation as shown in the document.\n"
    "Include footnotes, captions, labels, and marginal text.\n"
    "Do not skip or summarize any content."
)

EXTRACT_OLD_SCANS = (
    "Extract ALL visible text from this scanned document with maximum fidelity.\n"
    "Maintain the correct reading order -- read the natural flow of the text as a human would.\n"
    "Preserve exact wording, spelling, and punctuation from the original scan.\n"
    "Do NOT include page numbers, running headers, running footers, or library stamps.\n"
    "Do not describe images or illustrations."
)

EXTRACT_MULTI_COLUMN = (
    "Extract ALL text from this document maintaining the correct reading order.\n"
    "For multi-column layouts, read each column completely from top to bottom "
    "before moving to the next column (left to right).\n"
    "Do not interleave text from different columns.\n"
    "Preserve the exact wording and structure.\n"
    "Include all footnotes and references at the end, in the order they appear."
)

EXTRACT_TABLES = (
    "Extract all content from this document.\n"
    "For tables, use markdown table format with pipe (|) delimiters "
    "and proper column alignment.\n"
    "Preserve all cell values exactly as shown -- numbers, text, and symbols must be accurate.\n"
    "Include column headers and row headers.\n"
    "Keep multi-row and multi-column spanning cells as close to the original structure as possible.\n"
    "Use --- separator row between header and data rows."
)

# ---------------------------------------------------------------------------
# URL templates
# ---------------------------------------------------------------------------

HF_JSONL_URL = (
    "https://huggingface.co/datasets/allenai/olmOCR-bench"
    "/resolve/main/bench_data/{jsonl_name}"
)
HF_PDF_URL = (
    "https://huggingface.co/datasets/allenai/olmOCR-bench"
    "/resolve/main/bench_data/pdfs/{pdf_path}"
)
HF_PNG_URL = (
    "https://huggingface.co/datasets/shhdwi/olmocr-pre-rendered"
    "/resolve/main/images/{category}/{stem}_pg1.png"
)

JSONL_TO_CATEGORY = {
    "arxiv_math.jsonl": "arxiv_math",
    "old_scans_math.jsonl": "old_scans_math",
    "headers_footers.jsonl": "headers_footers",
    "long_tiny_text.jsonl": "long_tiny_text",
    "old_scans.jsonl": "old_scans",
    "multi_column.jsonl": "multi_column",
    "table_tests.jsonl": "tables",
}

# ---------------------------------------------------------------------------
# Category routing config
# ---------------------------------------------------------------------------

CATEGORY_CONFIG = {
    "arxiv_math": {"mode": "chat", "prompt": OLMOCR_CHAT},
    "old_scans_math": {"mode": "chat", "prompt": OLMOCR_CHAT},
    "headers_footers": {"mode": "extract_bbox", "custom_instructions": "", "prompt_mode": ""},
    "long_tiny_text": {"mode": "extract", "custom_instructions": EXTRACT_LONG_TINY_TEXT, "prompt_mode": "replace"},
    "old_scans": {"mode": "extract", "custom_instructions": EXTRACT_OLD_SCANS, "prompt_mode": "replace"},
    "multi_column": {"mode": "extract", "custom_instructions": EXTRACT_MULTI_COLUMN, "prompt_mode": "replace"},
    "tables": {"mode": "extract", "custom_instructions": EXTRACT_TABLES, "prompt_mode": "replace"},
}

# ---------------------------------------------------------------------------
# Ground truth fetching
# ---------------------------------------------------------------------------

def fetch_ground_truth(jsonl_name: str, cache_dir: Path) -> list[dict]:
    cached = cache_dir / jsonl_name
    if cached.exists():
        lines = cached.read_text(encoding="utf-8").strip().splitlines()
        return [json.loads(l) for l in lines]
    url = HF_JSONL_URL.format(jsonl_name=jsonl_name)
    resp = httpx.get(url, follow_redirects=True, timeout=60)
    resp.raise_for_status()
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(resp.content)
    lines = resp.text.strip().splitlines()
    return [json.loads(l) for l in lines]


# ---------------------------------------------------------------------------
# API call logic
# ---------------------------------------------------------------------------

def call_api(pdf_path: str, category: str) -> tuple[str, list | None]:
    config = CATEGORY_CONFIG.get(category, {"mode": "extract", "custom_instructions": "", "prompt_mode": ""})
    pdf_stem = Path(pdf_path).stem
    pdf_url = HF_PDF_URL.format(pdf_path=pdf_path)

    if _PROVIDER == "litellm":
        return _call_api_litellm(pdf_path, category, config), None

    if config["mode"] == "chat":
        png_url = HF_PNG_URL.format(category=category, stem=pdf_stem)
        text = chat(image_url=png_url, prompt=config["prompt"], model_type=_MODEL_TYPE)
        return text, None

    if config["mode"] == "extract_bbox":
        text, elements = extract_with_bbox(
            file_url=pdf_url,
            custom_instructions=config.get("custom_instructions", ""),
            prompt_mode=config.get("prompt_mode", ""),
            model_type=_MODEL_TYPE,
        )
        return text, elements

    text = extract_text(
        file_url=pdf_url,
        custom_instructions=config.get("custom_instructions", ""),
        prompt_mode=config.get("prompt_mode", ""),
        model_type=_MODEL_TYPE,
    )
    return text, None


OLMOCR_FALLBACK_PROMPT = (
    "Extract all text from this document image. "
    "Preserve the structure, headings, paragraphs, and reading order. "
    "Convert equations to LaTeX. Convert tables to markdown tables. "
    "Do not guess content that is not visible."
)


def _call_api_litellm(pdf_path: str, category: str, config: dict) -> str:
    from models.litellm_model import (
        ContentFilterError,
        SAFETY_SETTINGS_BLOCK_NONE,
        chat as litellm_chat,
        extract_text as litellm_extract,
    )

    pdf_stem = Path(pdf_path).stem

    if config["mode"] == "chat":
        png_url = HF_PNG_URL.format(category=category, stem=pdf_stem)
        try:
            return litellm_chat(
                image_url=png_url, prompt=config["prompt"], model_id=_LITELLM_MODEL_ID,
                max_retries=1,
            )
        except ContentFilterError:
            return litellm_chat(
                image_url=png_url, prompt=OLMOCR_FALLBACK_PROMPT,
                model_id=_LITELLM_MODEL_ID, safety_settings=SAFETY_SETTINGS_BLOCK_NONE,
                max_retries=1,
            )

    prompt = config.get("custom_instructions", "") or (
        "Convert this document page to markdown. "
        "Preserve all text, tables, and equations exactly as shown."
    )
    pdf_url = HF_PDF_URL.format(pdf_path=pdf_path)
    try:
        return litellm_extract(
            file_url=pdf_url, model_id=_LITELLM_MODEL_ID, custom_instructions=prompt,
            max_retries=1,
        )
    except ContentFilterError:
        return litellm_extract(
            file_url=pdf_url, model_id=_LITELLM_MODEL_ID,
            custom_instructions=OLMOCR_FALLBACK_PROMPT,
            safety_settings=SAFETY_SETTINGS_BLOCK_NONE,
            max_retries=1,
        )


def _api_worker(pdf_path: str, category: str, raw_md: Path, raw_bbox: Path | None, max_retries: int) -> dict:
    if raw_md.exists():
        return {"status": "cached", "pdf": pdf_path, "category": category}

    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            text, elements = call_api(pdf_path, category)
            if "</think>" in text:
                text = text.split("</think>", 1)[1].strip()
            if not text.strip():
                raise RuntimeError("API returned empty content")
            raw_md.parent.mkdir(parents=True, exist_ok=True)
            raw_md.write_text(text, encoding="utf-8")
            if elements is not None and raw_bbox is not None:
                raw_bbox.write_text(json.dumps(elements, ensure_ascii=False), encoding="utf-8")
            if _REQUEST_DELAY > 0:
                time.sleep(_REQUEST_DELAY)
            return {"status": "ok", "pdf": pdf_path, "category": category}
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries:
                backoff = min(30 * (2 ** (attempt - 1)), 120)
                time.sleep(backoff)

    # Fallback: try chat mode via pre-rendered PNG
    config = CATEGORY_CONFIG.get(category, {})
    if config.get("mode") in ("extract", "extract_bbox"):
        try:
            pdf_stem = Path(pdf_path).stem
            png_url = HF_PNG_URL.format(category=category, stem=pdf_stem)
            if _PROVIDER == "litellm":
                from models.litellm_model import (
                    SAFETY_SETTINGS_BLOCK_NONE,
                    chat as litellm_chat,
                )
                text = litellm_chat(
                    image_url=png_url, prompt=OLMOCR_FALLBACK_PROMPT,
                    model_id=_LITELLM_MODEL_ID,
                    safety_settings=SAFETY_SETTINGS_BLOCK_NONE,
                    max_retries=1,
                )
            else:
                text = chat(image_url=png_url, prompt=OLMOCR_CHAT, model_type=_MODEL_TYPE)
            if "</think>" in text:
                text = text.split("</think>", 1)[1].strip()
            if text.strip():
                raw_md.parent.mkdir(parents=True, exist_ok=True)
                raw_md.write_text(text, encoding="utf-8")
                return {"status": "ok_fallback", "pdf": pdf_path, "category": category}
        except Exception as fb_err:
            last_error = f"extract: {last_error}; chat fallback: {fb_err}"

    return {"status": "error", "pdf": pdf_path, "category": category, "error": last_error}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Populate olmOCR prediction cache")
    parser.add_argument("--model", type=str, default="nanonets", help="Model name (determines cache folder)")
    parser.add_argument("--categories", type=str, nargs="*", default=None, help="Only run specific categories")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (recommended: 5-15)")
    parser.add_argument("--model-type", type=str, default="", help="API model_type field (e.g. nanonets-ocr-3)")
    parser.add_argument("--provider", type=str, choices=["nanonets", "litellm"], default="nanonets",
                        help="API provider: nanonets (default) or litellm")
    parser.add_argument("--model-id", type=str, default="",
                        help="LiteLLM model identifier (e.g. anthropic/claude-sonnet-4-6)")
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

    raw_dir = REPO_ROOT / "caches" / args.model / "olmocr" / "raw"
    gt_cache = REPO_ROOT / "ground_truth"
    gt_cache.mkdir(parents=True, exist_ok=True)
    categories_to_run = args.categories or list(CATEGORY_CONFIG.keys())

    print(f"=== olmOCR Runner (model={args.model}) ===\n")

    jobs: list[tuple[str, str, Path, Path | None]] = []
    seen_pdfs: dict[str, set] = {}

    for jsonl_name, category in JSONL_TO_CATEGORY.items():
        if category not in categories_to_run:
            continue
        tests = fetch_ground_truth(jsonl_name, gt_cache)
        print(f"  {category}: {len(tests)} tests from {jsonl_name}")
        if category not in seen_pdfs:
            seen_pdfs[category] = set()
        for test in tests:
            pdf_path = test["pdf"]
            pdf_stem = Path(pdf_path).stem
            if pdf_stem in seen_pdfs[category]:
                continue
            seen_pdfs[category].add(pdf_stem)
            raw_md = raw_dir / category / f"{pdf_stem}.md"
            raw_bbox = None
            if CATEGORY_CONFIG.get(category, {}).get("mode") == "extract_bbox":
                raw_bbox = raw_dir / category / f"{pdf_stem}_bbox.json"
            jobs.append((pdf_path, category, raw_md, raw_bbox))

    cached = sum(1 for _, _, md, _ in jobs if md.exists())
    remaining = len(jobs) - cached
    print(f"\nTotal: {len(jobs)} unique PDFs, Cached: {cached}, Remaining: {remaining}")
    print(f"Workers: {args.workers}")

    if remaining == 0:
        print("All raw predictions cached. Nothing to do.\n")
        return

    t0 = time.time()
    done = ok = errors = 0
    error_list: list[dict] = []

    if args.workers <= 1:
        for pdf_path, category, raw_md, raw_bbox in jobs:
            result = _api_worker(pdf_path, category, raw_md, raw_bbox, args.max_retries)
            if result["status"] == "cached":
                continue
            done += 1
            if result["status"] in ("ok", "ok_fallback"):
                ok += 1
                if result["status"] == "ok_fallback":
                    print(f"  FALLBACK: {result['category']}/{Path(result['pdf']).stem}", flush=True)
            else:
                errors += 1
                error_list.append(result)
                print(f"  ERROR: {result['pdf']}: {result.get('error', '')[:80]}")
            if done % 25 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (remaining - done) / rate / 60 if rate > 0 else 0
                print(f"  [{done}/{remaining}] {rate:.1f}/s  ETA={eta:.0f}min  errors={errors}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_api_worker, pdf_path, category, raw_md, raw_bbox, args.max_retries): (pdf_path, category)
                for pdf_path, category, raw_md, raw_bbox in jobs
            }
            for future in as_completed(futures):
                result = future.result()
                if result["status"] == "cached":
                    continue
                done += 1
                if result["status"] in ("ok", "ok_fallback"):
                    ok += 1
                    if result["status"] == "ok_fallback":
                        print(f"  FALLBACK: {result['category']}/{Path(result['pdf']).stem}", flush=True)
                else:
                    errors += 1
                    error_list.append(result)
                    print(f"  ERROR: {result['pdf']}: {result.get('error', '')[:80]}", flush=True)
                if done % 25 == 0 or done == remaining:
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (remaining - done) / rate / 60 if rate > 0 else 0
                    print(f"  [{done}/{remaining}] {rate:.1f}/s  ETA={eta:.0f}min  errors={errors}", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. {ok} new + {cached} cached, {errors} errors.")
    if error_list:
        print(f"\nFailed ({len(error_list)}):")
        for e in error_list[:20]:
            print(f"  {e['category']}/{Path(e['pdf']).stem}: {e.get('error', '')[:100]}")
    print(f"\nRaw cache: {raw_dir}")


if __name__ == "__main__":
    main()
