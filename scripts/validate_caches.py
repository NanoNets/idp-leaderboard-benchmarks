#!/usr/bin/env python3
"""
Validate prediction caches for completeness, format, and quality.

Checks all 3 benchmarks (IDP, OmniDocBench, olmOCR) for a given model:
  - Structural: files exist, non-zero-byte, readable
  - Format: JSON parseable (IDP), valid UTF-8 (md files)
  - Quality: non-empty content, no degenerate n-gram repetition

Inspired by olmOCR's BaselineTest quality gate pattern.

Usage:
  python scripts/validate_caches.py --model chandra
  python scripts/validate_caches.py --model nanonets --benchmarks idp olmocr
  python scripts/validate_caches.py --model chandra --verbose
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

OLMOCR_CATEGORIES = [
    "arxiv_math", "old_scans_math", "headers_footers",
    "long_tiny_text", "old_scans", "multi_column", "tables",
]

OMNIDOCBENCH_EXPECTED = 1352


# ---------------------------------------------------------------------------
# Baseline quality checks (adapted from olmOCR BaselineTest)
# ---------------------------------------------------------------------------

def check_non_empty(content: str) -> tuple[bool, str]:
    alphanum = sum(1 for c in content if c.isalnum())
    if alphanum == 0:
        return False, "no alphanumeric characters"
    return True, ""


def check_repetition(content: str, max_repeats: int = 30) -> tuple[bool, str]:
    """Lightweight n-gram repetition detector (n=1..5) on the last 2000 chars."""
    tail = content[-2000:] if len(content) > 2000 else content
    if not tail.strip():
        return True, ""

    for n in range(1, 6):
        if len(tail) < n * 2:
            continue
        count = 1
        i = len(tail) - n
        gram = tail[i:]
        while i >= n:
            prev = tail[i - n:i]
            if prev == gram:
                count += 1
                i -= n
            else:
                break
        if count > max_repeats:
            return False, f"{count} repeating {n}-grams at end"

    return True, ""


def run_quality_checks(content: str) -> list[str]:
    """Return list of failure reasons (empty = all passed)."""
    failures = []
    ok, msg = check_non_empty(content)
    if not ok:
        failures.append(msg)
        return failures
    ok, msg = check_repetition(content)
    if not ok:
        failures.append(msg)
    return failures


# ---------------------------------------------------------------------------
# Per-benchmark validators
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkReport:
    benchmark: str
    total_files: int = 0
    expected_files: int | None = None
    empty_files: int = 0
    unreadable: int = 0
    malformed: int = 0
    degenerate: int = 0
    bad_files: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return (
            self.total_files > 0
            and self.empty_files == 0
            and self.unreadable == 0
            and self.malformed == 0
            and self.degenerate == 0
            and (self.expected_files is None or self.total_files >= self.expected_files)
        )

    @property
    def status(self) -> str:
        if self.total_files == 0:
            return "NO CACHE"
        return "PASS" if self.passed else "FAIL"

    def summary_lines(self, verbose: bool = False) -> list[str]:
        lines = []
        count_str = str(self.total_files)
        if self.expected_files is not None:
            count_str += f" / {self.expected_files} expected"
        lines.append(f"  Files:       {count_str}")
        if self.empty_files:
            lines.append(f"  Empty:       {self.empty_files}")
        if self.unreadable:
            lines.append(f"  Unreadable:  {self.unreadable}")
        if self.malformed:
            lines.append(f"  Malformed:   {self.malformed}")
        if self.degenerate:
            lines.append(f"  Degenerate:  {self.degenerate}")
        lines.append(f"  STATUS:      {self.status}")
        if verbose and self.bad_files:
            lines.append(f"  Bad files ({len(self.bad_files)}):")
            for bf in self.bad_files[:20]:
                lines.append(f"    - {bf}")
            if len(self.bad_files) > 20:
                lines.append(f"    ... and {len(self.bad_files) - 20} more")
        return lines


def _detect_idp_format(idp_root: Path) -> str:
    """Detect cache format. Prefer native if dataset subdirs exist."""
    has_native = any(
        d.is_dir() and d.name != "prediction_cache"
        for d in idp_root.iterdir()
    )
    if has_native:
        return "native"
    pc = idp_root / "prediction_cache"
    if pc.exists() and any(pc.glob("*.json")):
        return "nanobench"
    return "native"


def validate_idp(model: str, caches_root: Path, verbose: bool) -> BenchmarkReport:
    report = BenchmarkReport(benchmark="IDP")
    idp_root = caches_root / model / "idp"
    if not idp_root.exists():
        return report

    fmt = _detect_idp_format(idp_root)

    if fmt == "nanobench":
        pc_dir = idp_root / "prediction_cache"
        files = sorted(pc_dir.glob("*.json"))
        report.total_files = len(files)
        for f in files:
            if f.stat().st_size == 0:
                report.empty_files += 1
                report.bad_files.append(f"{f.name}: zero bytes")
                continue
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except Exception as e:
                report.malformed += 1
                report.bad_files.append(f"{f.name}: {e}")
                continue
            text = data.get("text", "")
            if not text or not text.strip():
                report.empty_files += 1
                report.bad_files.append(f"{f.name}: empty text")
                continue
            problems = run_quality_checks(text)
            if problems:
                report.degenerate += 1
                report.bad_files.append(f"{f.name}: {'; '.join(problems)}")
    else:
        ds_dirs = sorted(d for d in idp_root.iterdir() if d.is_dir() and d.name != "prediction_cache")
        for ds_dir in ds_dirs:
            files = sorted(ds_dir.glob("*.json"))
            for f in files:
                report.total_files += 1
                if f.stat().st_size == 0:
                    report.empty_files += 1
                    report.bad_files.append(f"{ds_dir.name}/{f.name}: zero bytes")
                    continue
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                except Exception as e:
                    report.malformed += 1
                    report.bad_files.append(f"{ds_dir.name}/{f.name}: {e}")
                    continue
                response = data.get("response", "")
                content = response if isinstance(response, str) else json.dumps(response)
                if not content or not content.strip():
                    report.empty_files += 1
                    report.bad_files.append(f"{ds_dir.name}/{f.name}: empty response")
                    continue
                problems = run_quality_checks(content)
                if problems:
                    report.degenerate += 1
                    report.bad_files.append(f"{ds_dir.name}/{f.name}: {'; '.join(problems)}")

    return report


def validate_omnidocbench(model: str, caches_root: Path, verbose: bool) -> BenchmarkReport:
    report = BenchmarkReport(benchmark="OmniDocBench", expected_files=OMNIDOCBENCH_EXPECTED)
    pred_dir = caches_root / model / "omnidocbench"
    if not pred_dir.exists():
        return report

    # Handle Nanobench quick_match/ subdirectory
    qm = pred_dir / "quick_match"
    if qm.exists() and any(qm.glob("*.md")) and not any(pred_dir.glob("*.md")):
        pred_dir = qm

    files = sorted(pred_dir.glob("*.md"))
    report.total_files = len(files)

    for f in files:
        if f.stat().st_size == 0:
            report.empty_files += 1
            report.bad_files.append(f"{f.name}: zero bytes")
            continue
        try:
            content = f.read_text(encoding="utf-8")
        except Exception as e:
            report.unreadable += 1
            report.bad_files.append(f"{f.name}: {e}")
            continue
        problems = run_quality_checks(content)
        if problems:
            report.degenerate += 1
            report.bad_files.append(f"{f.name}: {'; '.join(problems)}")

    return report


def validate_olmocr(model: str, caches_root: Path, verbose: bool) -> BenchmarkReport:
    report = BenchmarkReport(benchmark="olmOCR")
    raw_dir = caches_root / model / "olmocr" / "raw"
    if not raw_dir.exists():
        return report

    pg_re = re.compile(r'_pg\d+_repeat\d+\.md$')
    cat_counts: dict[str, int] = {}

    for cat_dir in sorted(raw_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        cat = cat_dir.name
        md_files = sorted(cat_dir.glob("*.md"))
        cat_counts[cat] = len(md_files)

        for f in md_files:
            report.total_files += 1
            if f.stat().st_size == 0:
                report.empty_files += 1
                report.bad_files.append(f"{cat}/{f.name}: zero bytes")
                continue
            try:
                content = f.read_text(encoding="utf-8")
            except Exception as e:
                report.unreadable += 1
                report.bad_files.append(f"{cat}/{f.name}: {e}")
                continue
            problems = run_quality_checks(content)
            if problems:
                report.degenerate += 1
                report.bad_files.append(f"{cat}/{f.name}: {'; '.join(problems)}")

        # Also validate any _bbox.json sidecar files
        for bj in cat_dir.glob("*_bbox.json"):
            try:
                json.loads(bj.read_text(encoding="utf-8"))
            except Exception as e:
                report.malformed += 1
                report.bad_files.append(f"{cat}/{bj.name}: {e}")

    report._cat_counts = cat_counts  # stash for reporting
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Validate prediction caches for completeness and quality")
    parser.add_argument("--model", type=str, required=True, help="Model name to validate")
    parser.add_argument(
        "--benchmarks", type=str, nargs="*", default=None,
        choices=["idp", "omnidocbench", "olmocr"],
        help="Only validate specific benchmarks (default: all)",
    )
    parser.add_argument("--caches-dir", type=str, default=None, help="Override caches root directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="List individual bad files")
    args = parser.parse_args()

    caches_root = Path(args.caches_dir).resolve() if args.caches_dir else REPO_ROOT / "caches"
    benchmarks = set(args.benchmarks) if args.benchmarks else {"idp", "omnidocbench", "olmocr"}

    print(f"=== Cache Validation: {args.model} ===")
    print(f"  Root: {caches_root / args.model}\n")

    reports: list[BenchmarkReport] = []

    if "idp" in benchmarks:
        r = validate_idp(args.model, caches_root, args.verbose)
        fmt = _detect_idp_format(caches_root / args.model / "idp") if (caches_root / args.model / "idp").exists() else "n/a"
        print(f"IDP ({fmt} format):")
        for line in r.summary_lines(args.verbose):
            print(line)
        print()
        reports.append(r)

    if "omnidocbench" in benchmarks:
        r = validate_omnidocbench(args.model, caches_root, args.verbose)
        print("OmniDocBench:")
        for line in r.summary_lines(args.verbose):
            print(line)
        print()
        reports.append(r)

    if "olmocr" in benchmarks:
        r = validate_olmocr(args.model, caches_root, args.verbose)
        print("olmOCR:")
        cat_counts = getattr(r, "_cat_counts", {})
        if cat_counts:
            for cat in OLMOCR_CATEGORIES:
                n = cat_counts.get(cat, 0)
                mark = "OK" if n > 0 else "MISSING"
                print(f"  {cat:20s}: {n:5d} files  {mark}")
            print()
        for line in r.summary_lines(args.verbose):
            print(line)
        print()
        reports.append(r)

    # Overall
    present = [r for r in reports if r.total_files > 0]
    passed = [r for r in present if r.passed]
    missing = [r for r in reports if r.total_files == 0]

    print("=" * 50)
    if missing:
        print(f"  Missing: {', '.join(r.benchmark for r in missing)}")
    print(f"  Result:  {len(passed)}/{len(present)} benchmarks PASS")

    if len(passed) < len(present):
        sys.exit(1)


if __name__ == "__main__":
    main()
