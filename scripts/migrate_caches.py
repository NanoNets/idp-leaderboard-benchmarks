#!/usr/bin/env python3
"""
Migrate prediction caches from the Nanobench all_prediction_caches directory
into the nanonets-benchmarks per-model layout.

Source (Nanobench / all_prediction_caches):
  data/idp/prediction_cache/{model}_{hash}.json          (flat, all models mixed)
  data/omnidocbench/prediction_cache/{model}/quick_match/ (*.md)
  data/olmocr-bench/prediction_cache/{model}/{category}/  (*.md with _pg*_repeat* suffix)

Target (nanonets-benchmarks):
  caches/{model}/idp/prediction_cache/{hash}.json
  caches/{model}/omnidocbench/{stem}.md
  caches/{model}/olmocr/raw/{category}/{stem}.md

Usage:
  python scripts/migrate_caches.py --source ../all_prediction_caches
  python scripts/migrate_caches.py --source ../all_prediction_caches --benchmarks idp
  python scripts/migrate_caches.py --source ../all_prediction_caches --models chandra nanonets
  python scripts/migrate_caches.py --source ../all_prediction_caches --dry-run
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def migrate_idp(source_dir: Path, target_root: Path, models: set[str] | None, dry_run: bool):
    """Split flat {model}_{hash}.json files into per-model directories."""
    idp_cache = source_dir / "data" / "idp" / "prediction_cache"
    if not idp_cache.exists():
        print(f"  SKIP: IDP cache not found at {idp_cache}")
        return

    pattern = re.compile(r'^(.+?)_([a-f0-9]{32})\.json$')
    files = sorted(idp_cache.glob("*.json"))
    print(f"  IDP: {len(files)} cache files found")

    # Skip models that already have native (index-based) IDP caches
    native_models: set[str] = set()
    for d in target_root.iterdir():
        idp_dir = d / "idp"
        if idp_dir.is_dir() and any(
            c.is_dir() and c.name != "prediction_cache" for c in idp_dir.iterdir()
        ):
            native_models.add(d.name)
    if native_models:
        print(f"  IDP: skipping {len(native_models)} models with native caches: {sorted(native_models)}")

    copied = 0
    skipped = 0
    for f in files:
        m = pattern.match(f.name)
        if not m:
            continue
        model_name, file_hash = m.group(1), m.group(2)
        if models and model_name not in models:
            skipped += 1
            continue
        if model_name in native_models:
            skipped += 1
            continue

        dest_dir = target_root / model_name / "idp" / "prediction_cache"
        dest_file = dest_dir / f"{file_hash}.json"

        if dest_file.exists():
            skipped += 1
            continue

        if dry_run:
            print(f"    [DRY] {f.name} → {dest_file.relative_to(target_root)}")
        else:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest_file)
        copied += 1

    print(f"  IDP: {copied} copied, {skipped} skipped")


def migrate_omnidocbench(source_dir: Path, target_root: Path, models: set[str] | None, dry_run: bool):
    """Copy {model}/quick_match/*.md → caches/{model}/omnidocbench/*.md"""
    omni_cache = source_dir / "data" / "omnidocbench" / "prediction_cache"
    if not omni_cache.exists():
        print(f"  SKIP: OmniDocBench cache not found at {omni_cache}")
        return

    model_dirs = sorted(d for d in omni_cache.iterdir() if d.is_dir())
    print(f"  OmniDocBench: {len(model_dirs)} model directories found")

    total_copied = 0
    for model_dir in model_dirs:
        model_name = model_dir.name
        if models and model_name not in models:
            continue

        qm_dir = model_dir / "quick_match"
        src_dir = qm_dir if qm_dir.exists() else model_dir
        md_files = sorted(src_dir.glob("*.md"))
        if not md_files:
            continue

        dest_dir = target_root / model_name / "omnidocbench"
        copied = 0
        for md in md_files:
            dest_file = dest_dir / md.name
            if dest_file.exists():
                continue
            if dry_run:
                if copied == 0:
                    print(f"    [DRY] {model_name}: {len(md_files)} files → omnidocbench/")
            else:
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(md, dest_file)
            copied += 1

        total_copied += copied
        if not dry_run and copied:
            print(f"    {model_name}: {copied} files copied")

    print(f"  OmniDocBench: {total_copied} total files copied")


def migrate_olmocr(source_dir: Path, target_root: Path, models: set[str] | None, dry_run: bool):
    """Copy {model}/{category}/*.md → caches/{model}/olmocr/raw/{category}/*.md

    Strips _pg*_repeat* suffix if present (nanonets-benchmarks uses plain stems).
    """
    olmocr_cache = source_dir / "data" / "olmocr-bench" / "prediction_cache"
    if not olmocr_cache.exists():
        print(f"  SKIP: olmOCR cache not found at {olmocr_cache}")
        return

    pg_suffix = re.compile(r'(_pg\d+)?(_repeat\d+)?\.md$')
    model_dirs = sorted(d for d in olmocr_cache.iterdir() if d.is_dir())
    print(f"  olmOCR: {len(model_dirs)} model directories found")

    total_copied = 0
    for model_dir in model_dirs:
        model_name = model_dir.name
        if models and model_name not in models:
            continue

        cat_dirs = sorted(d for d in model_dir.iterdir() if d.is_dir())
        copied = 0
        for cat_dir in cat_dirs:
            category = cat_dir.name
            md_files = sorted(cat_dir.glob("*.md"))
            if not md_files:
                continue

            dest_dir = target_root / model_name / "olmocr" / "raw" / category
            for md in md_files:
                clean_name = pg_suffix.sub('.md', md.name)
                dest_file = dest_dir / clean_name
                if dest_file.exists():
                    continue
                if dry_run:
                    if copied == 0:
                        print(f"    [DRY] {model_name}/{category}: {len(md_files)} files")
                else:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(md, dest_file)
                copied += 1

        total_copied += copied
        if not dry_run and copied:
            print(f"    {model_name}: {copied} files copied")

    print(f"  olmOCR: {total_copied} total files copied")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate Nanobench prediction caches into nanonets-benchmarks per-model layout"
    )
    parser.add_argument(
        "--source", type=str, required=True,
        help="Path to all_prediction_caches directory (or Nanobench data root)",
    )
    parser.add_argument(
        "--target", type=str, default=None,
        help="Target caches directory (default: nanonets-benchmarks/caches)",
    )
    parser.add_argument("--models", type=str, nargs="*", default=None, help="Only migrate specific models")
    parser.add_argument(
        "--benchmarks", type=str, nargs="*", default=None,
        choices=["idp", "omnidocbench", "olmocr"],
        help="Only migrate specific benchmarks",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be copied without copying")
    args = parser.parse_args()

    source_dir = Path(args.source).resolve()
    if not source_dir.exists():
        print(f"ERROR: Source directory not found: {source_dir}", file=sys.stderr)
        sys.exit(1)

    target_root = Path(args.target).resolve() if args.target else REPO_ROOT / "caches"
    models = set(args.models) if args.models else None
    benchmarks = set(args.benchmarks) if args.benchmarks else {"idp", "omnidocbench", "olmocr"}

    print(f"=== Cache Migration ===")
    print(f"  Source: {source_dir}")
    print(f"  Target: {target_root}")
    if models:
        print(f"  Models: {sorted(models)}")
    if args.dry_run:
        print(f"  Mode:   DRY RUN\n")
    else:
        print()

    if "idp" in benchmarks:
        migrate_idp(source_dir, target_root, models, args.dry_run)
    if "omnidocbench" in benchmarks:
        migrate_omnidocbench(source_dir, target_root, models, args.dry_run)
    if "olmocr" in benchmarks:
        migrate_olmocr(source_dir, target_root, models, args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
