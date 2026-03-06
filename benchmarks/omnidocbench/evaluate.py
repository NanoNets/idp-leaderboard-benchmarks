#!/usr/bin/env python3
"""
Evaluate OmniDocBench predictions using the official upstream pipeline.

Runs OmniDocBench/pdf_validation.py inside Docker for exact metric parity
(CDM requires xelatex which ships in the official Docker image).

Usage:
  python benchmarks/omnidocbench/evaluate.py --docker         # Docker (EC2)
  python benchmarks/omnidocbench/evaluate.py --host            # direct on host (no CDM)
  python benchmarks/omnidocbench/evaluate.py --omnidoc-root /path/to/OmniDocBench
"""

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

DOCKER_IMAGE = "sunyuefeng/omnidocbench-env:v1.5"


def _build_eval_config(gt_data_path: Path, pred_dir: Path) -> dict:
    return {
        "end2end_eval": {
            "metrics": {
                "text_block": {"metric": ["Edit_dist"]},
                "display_formula": {"metric": ["Edit_dist", "CDM"]},
                "table": {"metric": ["TEDS", "Edit_dist"]},
                "reading_order": {"metric": ["Edit_dist"]},
            },
            "dataset": {
                "dataset_name": "end2end_dataset",
                "ground_truth": {"data_path": str(gt_data_path)},
                "prediction": {"data_path": str(pred_dir)},
                "match_method": "quick_match",
            },
        }
    }


def _run_eval_docker(
    omnidoc_root: Path,
    config_path: Path,
    work_dir: Path,
    pred_dir: Path | None = None,
    docker_image: str = DOCKER_IMAGE,
    pull: bool = False,
    cpus: float | None = None,
    memory: str | None = None,
):
    if pull:
        print(f"Pulling Docker image: {docker_image}")
        subprocess.run(["docker", "pull", docker_image], check=True)

    python_cmd = (
        f"python {shlex.quote(str(omnidoc_root / 'pdf_validation.py'))} "
        f"--config {shlex.quote(str(config_path))}"
    )
    docker_limits: list[str] = []
    if cpus is not None:
        docker_limits.extend(["--cpus", str(cpus)])
    if memory is not None:
        docker_limits.extend(["--memory", memory])

    volume_mounts = [
        "-v", f"{omnidoc_root}:{omnidoc_root}",
        "-v", f"{work_dir}:{work_dir}",
    ]
    if pred_dir and pred_dir.resolve() != omnidoc_root.resolve() and not str(pred_dir.resolve()).startswith(str(omnidoc_root.resolve())):
        volume_mounts.extend(["-v", f"{pred_dir.resolve()}:{pred_dir.resolve()}"])

    cmd = [
        "docker", "run", "--rm",
        *docker_limits,
        *volume_mounts,
        "-w", str(work_dir),
        "-e", f"PYTHONPATH={omnidoc_root}",
        docker_image,
        "/bin/bash", "-lc",
        f"env PYTHONUNBUFFERED=1 {python_cmd}",
    ]

    print(f"Running: {' '.join(cmd[:8])} ...")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"OmniDocBench Docker eval failed (exit {result.returncode})")


def _run_eval_host(omnidoc_root: Path, config_path: Path, work_dir: Path):
    cmd = [
        sys.executable,
        str(omnidoc_root / "pdf_validation.py"),
        "--config", str(config_path),
    ]
    env = {"PYTHONPATH": str(omnidoc_root), "PYTHONUNBUFFERED": "1"}
    import os
    full_env = {**os.environ, **env}

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=full_env, cwd=str(work_dir), capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"OmniDocBench host eval failed (exit {result.returncode})")


def _parse_results(result_dir: Path) -> dict:
    """Parse upstream result JSONs and compute v1.5 overall score."""
    scores = {}
    for f in result_dir.glob("*.json"):
        data = json.loads(f.read_text(encoding="utf-8"))
        scores[f.stem] = data

    # Extract headline metrics
    text_edit = None
    formula_cdm = None
    table_teds = None
    read_order_edit = None

    for name, data in scores.items():
        if "text_block" in name and "Edit_dist" in name:
            text_edit = data.get("ALL_page_avg")
        if "display_formula" in name and "CDM" in name:
            formula_cdm = data.get("all")
        if "table" in name and "TEDS" in name:
            table_teds = data.get("ALL_page_avg")
        if "reading_order" in name and "Edit_dist" in name:
            read_order_edit = data.get("ALL_page_avg")

    print("\n" + "=" * 60)
    print("  OmniDocBench Results")
    print("=" * 60)
    print(f"  Text Edit Distance:    {text_edit}")
    print(f"  Formula CDM:           {formula_cdm}")
    print(f"  Table TEDS:            {table_teds}")
    print(f"  Reading Order Edit:    {read_order_edit}")

    overall = None
    if text_edit is not None and table_teds is not None and formula_cdm is not None:
        overall = ((1 - text_edit) * 100 + table_teds + formula_cdm) / 3
        print(f"\n  OVERALL (v1.5): {overall:.2f}%")
        print(f"  Formula: ((1 - {text_edit:.4f})*100 + {table_teds:.2f} + {formula_cdm:.2f}) / 3")
    else:
        print("\n  WARNING: Missing metrics for v1.5 overall score.")
        if formula_cdm is None:
            print("  CDM metric requires Docker (xelatex). Use --docker flag.")

    return {"text_edit": text_edit, "formula_cdm": formula_cdm, "table_teds": table_teds,
            "read_order_edit": read_order_edit, "overall": overall}


def main():
    parser = argparse.ArgumentParser(description="Evaluate OmniDocBench predictions (official upstream eval)")
    parser.add_argument("--model", type=str, default="nanonets")
    parser.add_argument("--omnidoc-root", type=str, default=None)
    parser.add_argument("--docker", action="store_true", help="Run eval in Docker (required for CDM/v1.5 parity)")
    parser.add_argument("--host", action="store_true", help="Run eval on host (no CDM)")
    parser.add_argument("--pull", action="store_true", help="Pull Docker image before running")
    parser.add_argument("--cpus", type=float, default=None)
    parser.add_argument("--memory", type=str, default=None, help="Docker memory limit (e.g. 24g)")
    parser.add_argument("--pred-dir", type=str, default=None,
                        help="Override prediction cache directory (e.g. for post-processed predictions)")
    args = parser.parse_args()

    if not args.docker and not args.host:
        print("ERROR: Specify --docker or --host", file=sys.stderr)
        sys.exit(1)

    # Find OmniDocBench root
    omnidoc_root = None
    if args.omnidoc_root:
        omnidoc_root = Path(args.omnidoc_root).resolve()
    else:
        for candidate in [REPO_ROOT.parent / "OmniDocBench", REPO_ROOT / "OmniDocBench"]:
            if (candidate / "OmniDocBench.json").exists():
                omnidoc_root = candidate
                break

    if omnidoc_root is None or not (omnidoc_root / "OmniDocBench.json").exists():
        print("ERROR: Cannot find OmniDocBench.json.", file=sys.stderr)
        sys.exit(1)

    if args.pred_dir:
        pred_dir = Path(args.pred_dir).resolve()
    else:
        pred_dir = REPO_ROOT / "caches" / args.model / "omnidocbench"
    if not pred_dir.exists():
        print(f"ERROR: Prediction cache not found: {pred_dir}", file=sys.stderr)
        print(f"  Run: python benchmarks/omnidocbench/run.py --model {args.model}", file=sys.stderr)
        print(f"  Or migrate Nanobench caches: python scripts/migrate_caches.py --source ../all_prediction_caches", file=sys.stderr)
        sys.exit(1)

    # Auto-detect Nanobench layout: {model}/quick_match/*.md
    qm_dir = pred_dir / "quick_match"
    if qm_dir.exists() and any(qm_dir.glob("*.md")) and not any(pred_dir.glob("*.md")):
        print(f"  (detected Nanobench cache layout: using quick_match/ subdirectory)")
        pred_dir = qm_dir

    gt_data_path = omnidoc_root / "OmniDocBench.json"

    # Set up work dir
    work_dir = Path(tempfile.mkdtemp(prefix="omnidoc_eval_", dir=str(omnidoc_root)))
    eval_config = _build_eval_config(gt_data_path, pred_dir)
    config_path = work_dir / "eval_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(eval_config, f, default_flow_style=False)

    result_dir = work_dir / "result"
    result_dir.mkdir(exist_ok=True)

    print(f"=== OmniDocBench Eval (model={args.model}) ===\n")
    print(f"  Predictions: {pred_dir}")
    print(f"  GT: {gt_data_path}")
    print(f"  Work dir: {work_dir}")

    if args.docker:
        _run_eval_docker(omnidoc_root, config_path, work_dir, pred_dir=pred_dir,
                         pull=args.pull, cpus=args.cpus, memory=args.memory)
    else:
        _run_eval_host(omnidoc_root, config_path, work_dir)

    _parse_results(result_dir)


if __name__ == "__main__":
    main()
