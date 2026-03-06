"""
Consolidate all benchmark results into nanonets-benchmarks/results/.

Scans all_prediction_caches/results/{model}/runs/*/ and latest/ to find
the highest-scoring result per benchmark for each leaderboard model, then
copies them into a clean flat structure:

  nanonets-benchmarks/results/{model}/{benchmark}.json
  nanonets-benchmarks/results/{model}/best_runs.json

Also updates the upload script's source to point here.

Usage:
  python scripts/consolidate_results.py
"""

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent
SCATTERED_DIR = ROOT / "all_prediction_caches" / "results"
CONSOLIDATED_DIR = ROOT / "nanonets-benchmarks" / "results"

ALLOWED_MODELS = {
    "nanonets": "Nanonets OCR2+",
    "gpt-5.2-2025-12-11": "GPT-5.2",
    "gemini-3-flash": "Gemini-3-Flash",
    "gpt-5-mini-2025-08-07": "GPT-5-Mini",
    "gemini-3-pro": "Gemini-3-Pro",
    "llama-3.2-vision-11b": "Llama-3.2-Vision-11B",
    "pixtral-12b": "Pixtral-12B",
    "gpt-5-nano-2025-08-07": "GPT-5-Nano",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "claude-opus-4-6": "Claude Opus 4.6",
    "claude-haiku-4-5": "Claude Haiku 4.5",
    "datalab-marker": "Datalab Marker",
    "glm-ocr": "GLM-OCR",
    "gpt-4.1": "GPT-4.1",
    "ministral-8b": "Ministral-8B",
}

NO_IDP_MODELS = {"datalab-marker"}
VALID_BENCHMARKS = {"idp", "olmocr", "omnidocbench"}


MIN_SAMPLES = {"idp": 1000, "olmocr": 1000, "omnidocbench": 500}


def find_best_result(model_dir: Path, bench: str) -> tuple[Path | None, float, int, str]:
    """Return (path, score, num_samples, run_name) for the best publish run.

    Prefers runs with sufficient samples (filtering out canary/test runs).
    Falls back to any run if no publish run qualifies.
    """
    search_dirs = []
    latest = model_dir / "latest"
    if latest.is_dir():
        search_dirs.append(("latest", latest))
    runs_dir = model_dir / "runs"
    if runs_dir.is_dir():
        for d in sorted(runs_dir.iterdir()):
            if d.is_dir():
                search_dirs.append((d.name, d))

    candidates = []
    for run_name, d in search_dirs:
        f = d / f"{bench}.json"
        if not f.is_file() or f.stat().st_size < 1000:
            continue
        try:
            with open(f) as fh:
                data = json.load(fh)
            score = data.get("overall_score", 0) or 0
            samples = data.get("num_samples", 0) or 0
        except Exception:
            continue
        is_publish = "publish" in run_name or run_name == "latest"
        candidates.append((f, score, samples, run_name, is_publish))

    if not candidates:
        return None, 0.0, 0, ""

    min_s = MIN_SAMPLES.get(bench, 500)
    full_runs = [c for c in candidates if c[2] >= min_s and c[4]]
    if not full_runs:
        full_runs = [c for c in candidates if c[2] >= min_s]
    if not full_runs:
        full_runs = candidates

    best = max(full_runs, key=lambda c: c[1])
    return best[0], best[1], best[2], best[3]


def main():
    CONSOLIDATED_DIR.mkdir(parents=True, exist_ok=True)
    
    total_copied = 0
    summary = []

    for raw_name in sorted(ALLOWED_MODELS.keys()):
        display = ALLOWED_MODELS[raw_name]
        model_dir = SCATTERED_DIR / raw_name

        if not model_dir.is_dir():
            print(f"  {display:30s} DIRECTORY NOT FOUND: {model_dir}")
            continue

        out_dir = CONSOLIDATED_DIR / raw_name
        out_dir.mkdir(parents=True, exist_ok=True)

        valid = sorted(VALID_BENCHMARKS - ({"idp"} if raw_name in NO_IDP_MODELS else set()))
        best_runs = {}

        for bench in valid:
            best_path, score, samples, run_name = find_best_result(model_dir, bench)

            if best_path is None:
                print(f"  {display:30s} {bench:15s} MISSING!")
                continue

            dest = out_dir / f"{bench}.json"
            shutil.copy2(best_path, dest)
            total_copied += 1

            best_runs[bench] = {
                "source_run": run_name,
                "source_path": str(best_path),
                "overall_score": round(score, 6),
                "num_samples": samples,
            }

            print(f"  {display:30s} {bench:15s} score={score:.4f}  samples={samples:5d}  from={run_name}")

        with open(out_dir / "best_runs.json", "w") as f:
            json.dump(best_runs, f, indent=2)

        benchmarks_found = list(best_runs.keys())
        summary.append((display, raw_name, benchmarks_found))

    print(f"\n{'='*70}")
    print(f"Consolidated {total_copied} result files into {CONSOLIDATED_DIR}")
    print(f"\nSummary ({len(summary)} models):")
    for display, raw_name, benches in summary:
        marks = "  ".join(f"{b}" for b in ["idp", "olmocr", "omnidocbench"] if b in benches)
        print(f"  {display:30s} {marks}")


if __name__ == "__main__":
    main()
