#!/usr/bin/env python3
"""
Populate the OmniDocBench prediction cache by calling the Nanonets API.

Reads the ground truth JSON (OmniDocBench.json) to discover all page images,
converts each to markdown via the Nanonets Extract API, and caches the output
in caches/{model}/omnidocbench/{image_stem}.md.

Usage:
  python benchmarks/omnidocbench/run.py --workers 10
  python benchmarks/omnidocbench/run.py --omnidoc-root /path/to/OmniDocBench
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from models.nanonets import extract_text

_MODEL_TYPE = ""
_PROVIDER = "nanonets"
_LITELLM_MODEL_ID = ""
_REQUEST_DELAY = 0.0

OMNIDOC_EXTRACT_PROMPT = (
    "Extract the text from the above document as if you were reading it naturally.\n"
    "Return the tables in html format using <table>, <tr>, <th>, and <td> tags.\n"
    "Return all equations in LaTeX representation. "
    "Use \\( and \\) for inline math and \\[ and \\] for display/block math. "
    "Do NOT use unicode math symbols — always use LaTeX commands.\n"
    "If there is an image in the document and image caption is not present, "
    "add a small description of the image inside the <img></img> tag; "
    "otherwise, add the image caption inside <img></img>.\n"
    "Watermarks should be wrapped in brackets. "
    "Ex: <watermark>OFFICIAL COPY</watermark>.\n"
    "Page numbers should be wrapped in brackets. "
    "Ex: <page_number>14</page_number>.\n"
    "Maintain the original document structure, headings, paragraphs, "
    "and reading order.\n"
    "Do not guess or infer content not visible in the document.\n"
    "Prefer using ☐ and ☑ for check boxes."
)

HF_OMNIDOC_IMAGE_URL = (
    "https://huggingface.co/datasets/opendatalab/OmniDocBench"
    "/resolve/main/images/{image_path}"
)


def discover_pages(gt_json_path: Path) -> list[str]:
    """Read OmniDocBench.json and return list of image paths."""
    with open(gt_json_path) as f:
        data = json.load(f)
    pages = []
    for page in data:
        img_path = (page.get("page_info") or {}).get("image_path", "")
        if img_path:
            pages.append(img_path)
    return sorted(set(pages))


def _convert_worker(
    image_path: str,
    pred_dir: Path,
    omnidoc_root: Path | None,
    max_retries: int,
) -> dict:
    stem = Path(image_path).stem
    out_file = pred_dir / f"{stem}.md"
    if out_file.exists():
        return {"status": "cached", "image": image_path}

    # Try local file first, fall back to HuggingFace URL
    local_img = None
    if omnidoc_root:
        local_candidate = omnidoc_root / "images" / image_path
        if local_candidate.exists():
            local_img = local_candidate

    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            if _PROVIDER == "litellm":
                text = _call_litellm(image_path, local_img)
            elif local_img:
                text = extract_text(
                    file_path=str(local_img),
                    custom_instructions=OMNIDOC_EXTRACT_PROMPT,
                    prompt_mode="replace",
                    model_type=_MODEL_TYPE,
                )
            else:
                file_url = HF_OMNIDOC_IMAGE_URL.format(image_path=image_path)
                text = extract_text(
                    file_url=file_url,
                    custom_instructions=OMNIDOC_EXTRACT_PROMPT,
                    prompt_mode="replace",
                    model_type=_MODEL_TYPE,
                )
            if "</think>" in text:
                text = text.split("</think>", 1)[1].strip()
            if not text.strip():
                raise RuntimeError("API returned empty content")
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(text, encoding="utf-8")
            if _REQUEST_DELAY > 0:
                time.sleep(_REQUEST_DELAY)
            return {"status": "ok", "image": image_path}
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries:
                backoff = min(30 * (2 ** (attempt - 1)), 120)
                time.sleep(backoff)

    return {"status": "error", "image": image_path, "error": last_error}


OMNIDOC_FALLBACK_PROMPT = (
    "Extract all text from this document image. "
    "Preserve the structure, headings, paragraphs, and reading order. "
    "Return tables as markdown tables. "
    "Return equations in LaTeX using \\( \\) for inline and \\[ \\] for block math. "
    "Do not guess content that is not visible."
)


def _call_litellm(image_path: str, local_img: Path | None) -> str:
    from models.litellm_model import (
        ContentFilterError,
        SAFETY_SETTINGS_BLOCK_NONE,
        extract_text as litellm_extract,
    )

    kwargs = dict(model_id=_LITELLM_MODEL_ID, custom_instructions=OMNIDOC_EXTRACT_PROMPT)
    if local_img:
        kwargs["file_path"] = str(local_img)
    else:
        kwargs["file_url"] = HF_OMNIDOC_IMAGE_URL.format(image_path=image_path)

    kwargs["max_retries"] = 1
    try:
        return litellm_extract(**kwargs)
    except ContentFilterError:
        kwargs["custom_instructions"] = OMNIDOC_FALLBACK_PROMPT
        kwargs["safety_settings"] = SAFETY_SETTINGS_BLOCK_NONE
        return litellm_extract(**kwargs)


def main():
    parser = argparse.ArgumentParser(description="Populate OmniDocBench prediction cache")
    parser.add_argument("--model", type=str, default="nanonets")
    parser.add_argument("--omnidoc-root", type=str, default=None,
                        help="Path to OmniDocBench repo root (has OmniDocBench.json + images/)")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
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

    # Find OmniDocBench root
    omnidoc_root = None
    if args.omnidoc_root:
        omnidoc_root = Path(args.omnidoc_root).resolve()
    else:
        # Auto-detect: look for OmniDocBench/ alongside the repo
        for candidate in [
            REPO_ROOT.parent / "OmniDocBench",
            REPO_ROOT / "OmniDocBench",
        ]:
            if (candidate / "OmniDocBench.json").exists():
                omnidoc_root = candidate
                break

    if omnidoc_root is None or not (omnidoc_root / "OmniDocBench.json").exists():
        print("ERROR: Cannot find OmniDocBench.json.", file=sys.stderr)
        print("  Provide --omnidoc-root /path/to/OmniDocBench", file=sys.stderr)
        sys.exit(1)

    gt_json = omnidoc_root / "OmniDocBench.json"
    pred_dir = REPO_ROOT / "caches" / args.model / "omnidocbench"
    pred_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== OmniDocBench Runner (model={args.model}) ===\n")
    print(f"  GT: {gt_json}")
    print(f"  Cache: {pred_dir}")

    pages = discover_pages(gt_json)
    print(f"  Pages: {len(pages)}")

    cached = sum(1 for p in pages if (pred_dir / f"{Path(p).stem}.md").exists())
    remaining = len(pages) - cached
    print(f"  Cached: {cached}, Remaining: {remaining}")
    print(f"  Workers: {args.workers}")

    if remaining == 0:
        print("\nAll pages cached. Nothing to do.")
        return

    t0 = time.time()
    done = ok = errors = 0

    if args.workers <= 1:
        for img_path in pages:
            result = _convert_worker(img_path, pred_dir, omnidoc_root, args.max_retries)
            if result["status"] == "cached":
                continue
            done += 1
            if result["status"] == "ok":
                ok += 1
            else:
                errors += 1
                print(f"  ERROR: {result['image']}: {result.get('error', '')[:80]}")
            if done % 25 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                print(f"  [{done}/{remaining}] {rate:.1f}/s  errors={errors}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_convert_worker, img_path, pred_dir, omnidoc_root, args.max_retries): img_path
                for img_path in pages
            }
            for future in as_completed(futures):
                result = future.result()
                if result["status"] == "cached":
                    continue
                done += 1
                if result["status"] == "ok":
                    ok += 1
                else:
                    errors += 1
                    print(f"  ERROR: {result['image']}: {result.get('error', '')[:80]}", flush=True)
                if done % 25 == 0 or done == remaining:
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    print(f"  [{done}/{remaining}] {rate:.1f}/s  errors={errors}", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. {ok} new + {cached} cached, {errors} errors.")
    print(f"Cache: {pred_dir}")


if __name__ == "__main__":
    main()
