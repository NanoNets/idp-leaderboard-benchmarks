#!/usr/bin/env python3
"""
Targeted re-run for empty/content-filtered prediction caches via LiteLLM.

Scans existing caches for a model, identifies empty (zero-byte) .md files,
and re-runs only those through LiteLLM to fill gaps from content-filter failures.

Usage:
  python scripts/rerun_empty.py --model claude-sonnet-4-6 \
    --model-id anthropic/claude-sonnet-4-6 \
    --benchmarks olmocr omnidocbench \
    --workers 10
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Auto-load .env if present (no extra dependency needed)
_env_file = REPO_ROOT / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))

# ---------------------------------------------------------------------------
# olmOCR constants (duplicated from benchmarks/olmocr/run.py to keep standalone)
# ---------------------------------------------------------------------------

HF_PDF_URL = (
    "https://huggingface.co/datasets/allenai/olmOCR-bench"
    "/resolve/main/bench_data/pdfs/{pdf_path}"
)
HF_PNG_URL = (
    "https://huggingface.co/datasets/shhdwi/olmocr-pre-rendered"
    "/resolve/main/images/{category}/{stem}_pg1.png"
)
HF_JSONL_URL = (
    "https://huggingface.co/datasets/allenai/olmOCR-bench"
    "/resolve/main/bench_data/{jsonl_name}"
)
HF_OMNIDOC_IMAGE_URL = (
    "https://huggingface.co/datasets/opendatalab/OmniDocBench"
    "/resolve/main/images/{image_path}"
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

OLMOCR_CHAT = (
    "Below is the image of one page of a PDF document.\n"
    "Return the plain text representation of this document "
    "as if you were reading it naturally.\n\n"
    "Rules:\n"
    "- Turn ALL equations and math symbols into LaTeX notation. "
    "Use \\( and \\) for inline math and \\[ and \\] for display/block math. "
    "Do NOT use unicode math symbols -- always use LaTeX commands instead "
    "(e.g., \\in not \u2208, \\rightarrow not \u2192).\n"
    "- Convert tables into markdown table format with proper column alignment.\n"
    "- Preserve all visible text with high fidelity, including headers, "
    "footers, references, and footnotes.\n"
    "- Maintain the correct multi-column reading order.\n"
    "- Read any natural handwriting.\n"
    "- If there is no text at all, output null.\n"
    "- Do not hallucinate.\n"
    "- Do not output image descriptions, alt-text, or <img> tags."
)

EXTRACT_PROMPTS = {
    "long_tiny_text": (
        "Extract ALL visible text from this document with maximum fidelity.\n"
        "Pay special attention to small, tiny, or fine-print text -- every word matters.\n"
        "Preserve the exact wording, spelling, and punctuation as shown in the document.\n"
        "Include footnotes, captions, labels, and marginal text.\n"
        "Do not skip or summarize any content."
    ),
    "old_scans": (
        "Extract ALL visible text from this scanned document with maximum fidelity.\n"
        "Maintain the correct reading order -- read the natural flow of the text as a human would.\n"
        "Preserve exact wording, spelling, and punctuation from the original scan.\n"
        "Do NOT include page numbers, running headers, running footers, or library stamps.\n"
        "Do not describe images or illustrations."
    ),
    "multi_column": (
        "Extract ALL text from this document maintaining the correct reading order.\n"
        "For multi-column layouts, read each column completely from top to bottom "
        "before moving to the next column (left to right).\n"
        "Do not interleave text from different columns.\n"
        "Preserve the exact wording and structure.\n"
        "Include all footnotes and references at the end, in the order they appear."
    ),
    "tables": (
        "Extract all content from this document.\n"
        "For tables, use markdown table format with pipe (|) delimiters "
        "and proper column alignment.\n"
        "Preserve all cell values exactly as shown -- numbers, text, and symbols must be accurate.\n"
        "Include column headers and row headers.\n"
        "Keep multi-row and multi-column spanning cells as close to the original structure as possible.\n"
        "Use --- separator row between header and data rows."
    ),
    "headers_footers": (
        "Convert this document page to markdown. "
        "Preserve all text, tables, and equations exactly as shown."
    ),
}

CHAT_CATEGORIES = {"arxiv_math", "old_scans_math"}

OMNIDOC_EXTRACT_PROMPT = (
    "Extract the text from the above document as if you were reading it naturally.\n"
    "Return the tables in html format using <table>, <tr>, <th>, and <td> tags.\n"
    "Return all equations in LaTeX representation. "
    "Use \\( and \\) for inline math and \\[ and \\] for display/block math. "
    "Do NOT use unicode math symbols \u2014 always use LaTeX commands.\n"
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
    "Prefer using \u2610 and \u2611 for check boxes."
)


# ---------------------------------------------------------------------------
# Ground truth helpers
# ---------------------------------------------------------------------------

def _fetch_gt_jsonl(jsonl_name: str, gt_cache: Path) -> list[dict]:
    import httpx as _httpx
    cached = gt_cache / jsonl_name
    if cached.exists():
        lines = cached.read_text(encoding="utf-8").strip().splitlines()
        return [json.loads(l) for l in lines]
    url = HF_JSONL_URL.format(jsonl_name=jsonl_name)
    resp = _httpx.get(url, follow_redirects=True, timeout=60)
    resp.raise_for_status()
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(resp.content)
    lines = resp.text.strip().splitlines()
    return [json.loads(l) for l in lines]


def _build_olmocr_stem_to_pdf(gt_cache: Path) -> dict[str, dict]:
    """Map pdf_stem -> {pdf_path, category} from the ground truth JSONLs."""
    mapping: dict[str, dict] = {}
    for jsonl_name, category in JSONL_TO_CATEGORY.items():
        tests = _fetch_gt_jsonl(jsonl_name, gt_cache)
        for test in tests:
            pdf_path = test["pdf"]
            stem = Path(pdf_path).stem
            if stem not in mapping:
                mapping[stem] = {"pdf_path": pdf_path, "category": category}
    return mapping


# ---------------------------------------------------------------------------
# Scan for empty caches
# ---------------------------------------------------------------------------

def find_empty_olmocr(model_cache: Path) -> list[dict]:
    """Find empty .md files in olmocr/raw/{category}/."""
    raw_dir = model_cache / "olmocr" / "raw"
    if not raw_dir.exists():
        return []
    empties = []
    for cat_dir in sorted(raw_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        category = cat_dir.name
        for md_file in sorted(cat_dir.glob("*.md")):
            if md_file.stat().st_size == 0:
                empties.append({
                    "path": md_file,
                    "stem": md_file.stem,
                    "category": category,
                    "benchmark": "olmocr",
                })
    return empties


def find_empty_omnidocbench(model_cache: Path) -> list[dict]:
    """Find empty .md files in omnidocbench/."""
    pred_dir = model_cache / "omnidocbench"
    if not pred_dir.exists():
        return []
    # Check quick_match/ (Nanobench layout) first, then top-level
    qm_dir = pred_dir / "quick_match"
    search_dir = qm_dir if qm_dir.is_dir() else pred_dir
    empties = []
    for md_file in sorted(search_dir.glob("*.md")):
        if md_file.stat().st_size == 0:
            empties.append({
                "path": md_file,
                "stem": md_file.stem,
                "benchmark": "omnidocbench",
            })
    return empties


# ---------------------------------------------------------------------------
# Re-run workers
# ---------------------------------------------------------------------------

def _rerun_olmocr_item(item: dict, model_id: str, stem_map: dict) -> dict:
    from models.litellm_model import chat as litellm_chat, extract_text as litellm_extract

    stem = item["stem"]
    category = item["category"]
    md_path: Path = item["path"]

    info = stem_map.get(stem)
    if not info:
        return {"status": "skip", "reason": f"no GT mapping for {stem}", **item}

    pdf_path = info["pdf_path"]

    try:
        if category in CHAT_CATEGORIES:
            png_url = HF_PNG_URL.format(category=category, stem=stem)
            text = litellm_chat(image_url=png_url, prompt=OLMOCR_CHAT, model_id=model_id)
        else:
            prompt = EXTRACT_PROMPTS.get(category, EXTRACT_PROMPTS["headers_footers"])
            pdf_url = HF_PDF_URL.format(pdf_path=pdf_path)
            text = litellm_extract(file_url=pdf_url, model_id=model_id, custom_instructions=prompt)

        md_path.write_text(text, encoding="utf-8")
        filled = bool(text.strip())
        return {"status": "ok" if filled else "still_empty", "stem": stem, "category": category,
                "chars": len(text)}
    except Exception as exc:
        return {"status": "error", "stem": stem, "category": category, "error": str(exc)[:200]}


def _rerun_omnidoc_item(item: dict, model_id: str) -> dict:
    from models.litellm_model import extract_text as litellm_extract

    stem = item["stem"]
    md_path: Path = item["path"]

    image_path = f"{stem}.jpg"
    file_url = HF_OMNIDOC_IMAGE_URL.format(image_path=image_path)

    try:
        text = litellm_extract(file_url=file_url, model_id=model_id,
                               custom_instructions=OMNIDOC_EXTRACT_PROMPT)
        md_path.write_text(text, encoding="utf-8")
        filled = bool(text.strip())
        return {"status": "ok" if filled else "still_empty", "stem": stem, "chars": len(text)}
    except Exception as exc:
        return {"status": "error", "stem": stem, "error": str(exc)[:200]}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Re-run empty prediction caches via LiteLLM",
    )
    parser.add_argument("--model", required=True, help="Model cache folder name (e.g. claude-sonnet-4-6)")
    parser.add_argument("--model-id", required=True,
                        help="LiteLLM model identifier (e.g. anthropic/claude-sonnet-4-6)")
    parser.add_argument("--benchmarks", nargs="+", choices=["olmocr", "omnidocbench"],
                        default=["olmocr", "omnidocbench"])
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true", help="List empties without re-running")
    args = parser.parse_args()

    model_cache = REPO_ROOT / "caches" / args.model
    if not model_cache.exists():
        print(f"ERROR: Cache directory not found: {model_cache}", file=sys.stderr)
        sys.exit(1)

    gt_cache = REPO_ROOT / "ground_truth"
    gt_cache.mkdir(parents=True, exist_ok=True)

    # Collect empty files
    empties: list[dict] = []
    if "olmocr" in args.benchmarks:
        olmocr_empties = find_empty_olmocr(model_cache)
        empties.extend(olmocr_empties)
        print(f"olmOCR: {len(olmocr_empties)} empty files")
        for cat in sorted({e["category"] for e in olmocr_empties}):
            n = sum(1 for e in olmocr_empties if e["category"] == cat)
            print(f"  {cat}: {n}")
    if "omnidocbench" in args.benchmarks:
        omnidoc_empties = find_empty_omnidocbench(model_cache)
        empties.extend(omnidoc_empties)
        print(f"OmniDocBench: {len(omnidoc_empties)} empty files")

    if not empties:
        print("\nNo empty files found. Nothing to re-run.")
        return

    print(f"\nTotal: {len(empties)} empty predictions to re-run")

    if args.dry_run:
        for e in empties:
            cat = e.get("category", "")
            print(f"  [{e['benchmark']}] {cat}/{e['stem']}")
        return

    # Build stem-to-PDF mapping for olmOCR
    stem_map = {}
    if any(e["benchmark"] == "olmocr" for e in empties):
        print("\nFetching olmOCR ground truth for PDF path mapping...")
        stem_map = _build_olmocr_stem_to_pdf(gt_cache)
        print(f"  Mapped {len(stem_map)} PDF stems")

    print(f"\nRe-running with model_id={args.model_id}, workers={args.workers}\n")
    t0 = time.time()
    ok = still_empty = errors = skipped = 0

    def _dispatch(item):
        if item["benchmark"] == "olmocr":
            return _rerun_olmocr_item(item, args.model_id, stem_map)
        return _rerun_omnidoc_item(item, args.model_id)

    if args.workers <= 1:
        for i, item in enumerate(empties, 1):
            result = _dispatch(item)
            status = result["status"]
            if status == "ok":
                ok += 1
            elif status == "still_empty":
                still_empty += 1
                print(f"  STILL EMPTY: {result.get('category', '')}/{result['stem']}")
            elif status == "error":
                errors += 1
                print(f"  ERROR: {result.get('category', '')}/{result['stem']}: "
                      f"{result.get('error', '')[:100]}")
            elif status == "skip":
                skipped += 1
            if i % 10 == 0 or i == len(empties):
                elapsed = time.time() - t0
                print(f"  [{i}/{len(empties)}] {elapsed:.0f}s  ok={ok} still_empty={still_empty} errors={errors}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_dispatch, item): item for item in empties}
            done = 0
            for future in as_completed(futures):
                result = future.result()
                done += 1
                status = result["status"]
                if status == "ok":
                    ok += 1
                elif status == "still_empty":
                    still_empty += 1
                    print(f"  STILL EMPTY: {result.get('category', '')}/{result['stem']}")
                elif status == "error":
                    errors += 1
                    print(f"  ERROR: {result.get('category', '')}/{result['stem']}: "
                          f"{result.get('error', '')[:100]}")
                elif status == "skip":
                    skipped += 1
                if done % 10 == 0 or done == len(empties):
                    elapsed = time.time() - t0
                    print(f"  [{done}/{len(empties)}] {elapsed:.0f}s  "
                          f"ok={ok} still_empty={still_empty} errors={errors} skipped={skipped}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Filled:      {ok}")
    print(f"  Still empty: {still_empty}")
    print(f"  Errors:      {errors}")
    print(f"  Skipped:     {skipped}")


if __name__ == "__main__":
    main()
