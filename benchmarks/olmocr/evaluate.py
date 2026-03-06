#!/usr/bin/env python3
"""
Evaluate olmOCR predictions using the OFFICIAL olmocr benchmark.

Two modes:
  python benchmarks/olmocr/evaluate.py                   # evaluate raw cache directly
  python benchmarks/olmocr/evaluate.py --postprocess      # postprocess first, then evaluate

Setup (one-time):
  pip install "olmocr[bench]"
  python -m playwright install chromium
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

try:
    from olmocr.bench.tests import BaselineTest, load_tests
    from olmocr.bench.benchmark import evaluate_candidate
    from olmocr.bench.utils import calculate_bootstrap_ci
except ImportError:
    print(
        'ERROR: olmocr[bench] not installed.\n'
        '  Run: pip install "olmocr[bench]"\n'
        '       python -m playwright install chromium',
        file=sys.stderr,
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

HF_JSONL_URL = (
    "https://huggingface.co/datasets/allenai/olmOCR-bench"
    "/resolve/main/bench_data/{jsonl_name}"
)

JSONL_FILES = [
    "arxiv_math.jsonl",
    "old_scans_math.jsonl",
    "headers_footers.jsonl",
    "long_tiny_text.jsonl",
    "old_scans.jsonl",
    "multi_column.jsonl",
    "table_tests.jsonl",
]

JSONL_TO_CATEGORY = {
    "arxiv_math.jsonl": "arxiv_math",
    "old_scans_math.jsonl": "old_scans_math",
    "headers_footers.jsonl": "headers_footers",
    "long_tiny_text.jsonl": "long_tiny_text",
    "old_scans.jsonl": "old_scans",
    "multi_column.jsonl": "multi_column",
    "table_tests.jsonl": "tables",
}

CATEGORY_CONFIG = {
    "arxiv_math": {"mode": "chat"},
    "old_scans_math": {"mode": "chat"},
    "headers_footers": {"mode": "extract_bbox"},
    "long_tiny_text": {"mode": "extract"},
    "old_scans": {"mode": "extract"},
    "multi_column": {"mode": "extract"},
    "tables": {"mode": "extract"},
}


def prepare_raw_for_eval(raw_dir: Path) -> Path:
    """Create a dir with symlinks named {stem}_pg1_repeat0.md for each raw {stem}.md.

    The olmocr evaluator expects files matching {category}/{stem}_pg1_repeat*.md
    but the raw cache stores them as {category}/{stem}.md.

    Auto-detects Nanobench format where files already have the _pg*_repeat* suffix
    and uses them directly without re-adding the suffix.
    """
    import re
    pg_repeat_re = re.compile(r'_pg\d+_repeat\d+\.md$')

    eval_dir = raw_dir.parent / "raw_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    for cat_dir in sorted(raw_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        eval_cat = eval_dir / cat_dir.name
        eval_cat.mkdir(parents=True, exist_ok=True)
        for md_file in cat_dir.glob("*.md"):
            if pg_repeat_re.search(md_file.name):
                target = eval_cat / md_file.name
            else:
                target = eval_cat / f"{md_file.stem}_pg1_repeat0.md"
            if not target.exists():
                os.symlink(md_file.resolve(), target)

    return eval_dir


def download_jsonl(jsonl_name: str, gt_dir: Path) -> Path:
    cached = gt_dir / jsonl_name
    if cached.exists():
        return cached
    url = HF_JSONL_URL.format(jsonl_name=jsonl_name)
    print(f"  Downloading {jsonl_name}...")
    resp = httpx.get(url, follow_redirects=True, timeout=60)
    resp.raise_for_status()
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(resp.content)
    return cached


# ---------------------------------------------------------------------------
# Post-processing bridge
# ---------------------------------------------------------------------------

def run_postprocess(raw_dir: Path, pred_dir: Path) -> dict[str, int]:
    """Read raw cache, apply per-category post-processing, write to predictions dir."""
    from benchmarks.olmocr.postprocess import postprocess

    stats: dict[str, int] = {"processed": 0, "skipped_empty": 0}

    for category, config in CATEGORY_CONFIG.items():
        raw_cat = raw_dir / category
        if not raw_cat.exists():
            continue
        pred_cat = pred_dir / category
        pred_cat.mkdir(parents=True, exist_ok=True)

        for raw_md in sorted(raw_cat.glob("*.md")):
            stem = raw_md.stem
            out_file = pred_cat / f"{stem}_pg1_repeat0.md"
            raw_text = raw_md.read_text(encoding="utf-8")
            if not raw_text.strip():
                out_file.write_text("", encoding="utf-8")
                stats["skipped_empty"] += 1
                continue

            bbox_elements = None
            if config["mode"] == "extract_bbox":
                bbox_file = raw_cat / f"{stem}_bbox.json"
                if bbox_file.exists():
                    try:
                        bbox_elements = json.loads(bbox_file.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        pass

            processed = postprocess(raw_text, category, bbox_elements)
            out_file.write_text(processed, encoding="utf-8")
            stats["processed"] += 1

    return stats


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(pred_dir: Path, candidate_name: str, gt_dir: Path, skip_baseline: bool = False):
    print("Loading ground truth...")
    all_tests = []
    test_to_jsonl = {}

    for jsonl_name in JSONL_FILES:
        jsonl_path = download_jsonl(jsonl_name, gt_dir)
        tests = load_tests(str(jsonl_path))
        for test in tests:
            test_to_jsonl[test.id] = jsonl_name
        all_tests.extend(tests)
        print(f"  {JSONL_TO_CATEGORY[jsonl_name]:20s}: {len(tests):5d} tests")

    print(f"\n  Total: {len(all_tests)} tests")

    pdf_basenames = sorted(set(test.pdf for test in all_tests))
    print(f"  Unique PDFs referenced: {len(pdf_basenames)}")

    if not skip_baseline:
        for pdf in pdf_basenames:
            if not any(t.type == "baseline" for t in all_tests if t.pdf == pdf):
                bt = BaselineTest(id=f"{pdf}_baseline", pdf=pdf, page=1, type="baseline")
                all_tests.append(bt)
                test_to_jsonl[bt.id] = "baseline"

    print(f"\nEvaluating candidate: {candidate_name}")
    print(f"  Predictions dir: {pred_dir}\n")

    (
        overall_score,
        total_tests,
        candidate_errors,
        test_failures,
        test_type_breakdown,
        all_test_scores,
        test_results,
    ) = evaluate_candidate(str(pred_dir), all_tests, pdf_basenames, force=True)

    # Per-JSONL scoring (official leaderboard method)
    jsonl_results = {}
    jsonl_scores = []
    jsonl_file_sizes = []

    for test in all_tests:
        jsonl_file = test_to_jsonl.get(test.id, "unknown")
        if jsonl_file not in jsonl_results:
            jsonl_results[jsonl_file] = {"total": 0, "passed": 0, "scores": []}
        jsonl_results[jsonl_file]["total"] += 1

        if not candidate_errors and hasattr(test, "pdf") and hasattr(test, "page"):
            pdf_name = test.pdf
            page = test.page
            if pdf_name in test_results and page in test_results.get(pdf_name, {}):
                for t, passed, _ in test_results[pdf_name][page]:
                    if t.id == test.id:
                        score = 1.0 if passed else 0.0
                        jsonl_results[jsonl_file]["scores"].append(score)
                        if passed:
                            jsonl_results[jsonl_file]["passed"] += 1
                        break

    for jsonl_file, results in jsonl_results.items():
        if results["scores"]:
            jsonl_file_sizes.append(len(results["scores"]))
            jsonl_scores.extend(results["scores"])

    ci = calculate_bootstrap_ci(jsonl_scores, n_bootstrap=1000, ci_level=0.95, splits=jsonl_file_sizes) if jsonl_scores else (0.0, 0.0)

    # Report
    print("\n" + "=" * 70)
    print(f"  Candidate: {candidate_name}")
    print("=" * 70)

    if candidate_errors:
        print("\n  ERRORS:")
        for err in candidate_errors[:20]:
            print(f"    {err}")

    jsonl_pass_rates = []
    print("\n  Results by category (JSONL file):")
    for jsonl_file in JSONL_FILES + ["baseline"]:
        results = jsonl_results.get(jsonl_file)
        if not results or results["total"] == 0:
            continue
        if results["scores"]:
            pass_rate = results["passed"] / results["total"]
            jsonl_pass_rates.append(pass_rate)
            category = JSONL_TO_CATEGORY.get(jsonl_file, jsonl_file)
            print(f"    {category:20s}: {pass_rate*100:5.1f}%  ({results['passed']}/{results['total']})")
        else:
            category = JSONL_TO_CATEGORY.get(jsonl_file, jsonl_file)
            print(f"    {category:20s}: no predictions found")

    print("\n  Results by test type:")
    for ttype in sorted(test_type_breakdown.keys()):
        scores = test_type_breakdown[ttype]
        avg = sum(scores) / len(scores) * 100 if scores else 0.0
        print(f"    {ttype:12s}: {avg:5.1f}%  ({len(scores)} tests)")

    per_category_score = sum(jsonl_pass_rates) / len(jsonl_pass_rates) if jsonl_pass_rates else 0.0
    half_width = ((ci[1] - ci[0]) / 2) * 100

    print(f"\n  {'─' * 50}")
    print(f"  OVERALL SCORE: {per_category_score * 100:.1f}% ± {half_width:.1f}%")
    print(f"  (average of per-JSONL-file pass rates, 95% CI: [{ci[0]*100:.1f}%, {ci[1]*100:.1f}%])")
    print(f"  Total tests evaluated: {total_tests}")

    return per_category_score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate olmOCR predictions (official benchmark)")
    parser.add_argument("--model", type=str, default="nanonets", help="Model name (selects cache folder)")
    parser.add_argument("--postprocess", action="store_true", help="Run post-processing before evaluation")
    parser.add_argument("--predictions", type=str, default=None, help="Override: evaluate this directory directly")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--output-failed", type=str, default=None)
    args = parser.parse_args()

    gt_dir = REPO_ROOT / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)

    if args.predictions:
        pred_dir = Path(args.predictions).resolve()
        candidate_name = pred_dir.name
    elif args.postprocess:
        raw_dir = REPO_ROOT / "caches" / args.model / "olmocr" / "raw"
        pred_dir = REPO_ROOT / "caches" / args.model / "olmocr" / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        print(f"=== Post-processing raw cache → predictions ===\n")
        stats = run_postprocess(raw_dir, pred_dir)
        print(f"  Processed: {stats['processed']}, Empty: {stats['skipped_empty']}\n")
        candidate_name = f"{args.model}-postprocessed"
    else:
        raw_dir = REPO_ROOT / "caches" / args.model / "olmocr" / "raw"
        print(f"=== Preparing raw cache for evaluation ===\n")
        pred_dir = prepare_raw_for_eval(raw_dir)
        candidate_name = f"{args.model}-raw"

    if not pred_dir.exists():
        print(f"ERROR: Predictions directory does not exist: {pred_dir}", file=sys.stderr)
        print(f"  Run: python benchmarks/olmocr/run.py --model {args.model}", file=sys.stderr)
        print(f"  Or migrate Nanobench caches: python scripts/migrate_caches.py --source ../all_prediction_caches", file=sys.stderr)
        sys.exit(1)

    evaluate(pred_dir, candidate_name, gt_dir, skip_baseline=args.skip_baseline)


if __name__ == "__main__":
    main()
