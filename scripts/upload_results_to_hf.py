"""
Upload benchmark results to HuggingFace.

Creates/updates the dataset repo: shhdwi/idp-leaderboard-results

Reads pre-consolidated results from nanonets-benchmarks/results/{model}/
(produced by consolidate_results.py).

Structure on HF:
  index.json                        # manifest of models + benchmarks
  results/{model}/{benchmark}.json  # per-sample results (trimmed)

Usage:
  python scripts/consolidate_results.py   # run first to pick best results
  python scripts/upload_results_to_hf.py
"""

import json
import os
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "nanonets-benchmarks" / "results"
REPO_ID = "shhdwi/idp-leaderboard-results"

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

VALID_BENCHMARKS = {"idp", "olmocr", "omnidocbench"}

EXCLUDE_DATASETS = {
    "ocr_handwriting_rotated",
    "nanonets_longdocbench",
}


def build_staging_dir(staging: Path):
    """Read consolidated results and build trimmed JSONs + index for HF upload."""
    results_out = staging / "results"
    results_out.mkdir(parents=True, exist_ok=True)

    index = {"models": []}

    for raw_name, model_name in sorted(ALLOWED_MODELS.items(), key=lambda x: x[1]):
        model_dir = RESULTS_DIR / raw_name
        if not model_dir.is_dir():
            print(f"  {model_name}: DIRECTORY NOT FOUND ({model_dir})")
            continue

        benchmarks = {}
        model_out = results_out / model_name
        model_out.mkdir(parents=True, exist_ok=True)

        for bench in sorted(VALID_BENCHMARKS):
            src = model_dir / f"{bench}.json"
            if not src.is_file():
                continue

            with open(src) as f:
                data = json.load(f)

            trimmed = trim_results(data)
            num = trimmed["num_samples"]

            benchmarks[bench] = {
                "num_samples": num,
                "overall_score": data.get("overall_score", 0),
            }

            out_path = model_out / f"{bench}.json"
            with open(out_path, "w") as f:
                json.dump(trimmed, f, separators=(",", ":"))

            print(f"  {model_name}/{bench}: {num} samples, score={data.get('overall_score', 0):.4f}")

        if benchmarks:
            index["models"].append({
                "model_name": model_name,
                "benchmarks": benchmarks,
            })

    index_path = staging / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nIndex: {len(index['models'])} models")
    return index


def load_olmocr_ground_truth() -> dict[str, str]:
    """Load real ground truth text from OLMoCR JSONL files, keyed by sample ID."""
    gt_dir = ROOT / "nanonets-benchmarks" / "ground_truth"
    gt_map: dict[str, str] = {}

    for jsonl_file in gt_dir.glob("*.jsonl"):
        with open(jsonl_file) as f:
            for line in f:
                entry = json.loads(line)
                sid = entry.get("id", "")
                t = entry.get("type", "")

                if t == "math":
                    gt_map[sid] = entry.get("math", "")
                elif t in ("present", "absent"):
                    gt_map[sid] = f"[{t}] {entry.get('text', '')}"
                elif t == "order":
                    gt_map[sid] = f"before: {entry.get('before', '')}\nafter: {entry.get('after', '')}"
                elif t == "table":
                    parts = [f"cell: {entry.get('cell', '')}"]
                    for k in ("left_heading", "top_heading", "up", "down", "left", "right"):
                        v = entry.get(k)
                        if v and v != "None":
                            parts.append(f"{k}: {v}")
                    gt_map[sid] = "\n".join(parts)
                elif t == "baseline":
                    gt_map[sid] = "(baseline check)"

    return gt_map


_olmocr_gt_cache: dict[str, str] | None = None


def get_olmocr_gt() -> dict[str, str]:
    global _olmocr_gt_cache
    if _olmocr_gt_cache is None:
        _olmocr_gt_cache = load_olmocr_ground_truth()
        print(f"  Loaded {len(_olmocr_gt_cache)} OLMoCR ground truth entries")
    return _olmocr_gt_cache


def trim_results(data: dict) -> dict:
    """Keep only the fields needed for the explorer."""
    benchmark = data.get("benchmark", "")
    olmocr_gt = get_olmocr_gt() if benchmark == "olmocr" else {}

    trimmed_samples = []
    for s in data.get("samples", []):
        if s.get("dataset", "") in EXCLUDE_DATASETS:
            continue

        pred = s.get("prediction", "")
        gt = s.get("ground_truth", "")

        if benchmark == "olmocr":
            sid = s.get("sample_id", "")
            gt = olmocr_gt.get(sid, gt)

        trimmed_samples.append({
            "id": s.get("sample_id", ""),
            "task": s.get("task", ""),
            "dataset": s.get("dataset", ""),
            "score": s.get("score"),
            "source_file": s.get("source_file", ""),
            "prediction": pred,
            "ground_truth": gt,
        })

    return {
        "benchmark": data.get("benchmark"),
        "model_name": data.get("model_name"),
        "overall_score": data.get("overall_score"),
        "num_samples": len(trimmed_samples),
        "breakdown": data.get("breakdown"),
        "samples": trimmed_samples,
    }


def upload_to_hf(staging: Path):
    """Push staging directory to HuggingFace."""
    token = os.environ.get("HF_WRITE_TOKEN")
    if not token:
        print("\nHF_WRITE_TOKEN not set. Staging directory ready at:")
        print(f"  {staging}")
        print("Set HF_WRITE_TOKEN and re-run, or upload manually:")
        print(f"  huggingface-cli upload {REPO_ID} {staging} --repo-type dataset")
        return

    api = HfApi(token=token)
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True, private=False)
    api.upload_folder(
        folder_path=str(staging),
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message="Update leaderboard results and IDP images",
    )
    print(f"\nUploaded to https://huggingface.co/datasets/{REPO_ID}")


def main():
    staging = Path(tempfile.mkdtemp(prefix="idp-leaderboard-results-"))
    print(f"Staging directory: {staging}\n")

    print("=== Building results JSONs ===")
    build_staging_dir(staging)

    print("\n=== Uploading to HuggingFace ===")
    upload_to_hf(staging)


if __name__ == "__main__":
    main()
