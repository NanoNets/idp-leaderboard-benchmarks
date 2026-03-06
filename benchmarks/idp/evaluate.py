#!/usr/bin/env python3
"""
Evaluate IDP predictions using the official docext metrics.

Supports two cache formats:
  native    - caches/{model}/idp/{dataset_name}/{index}.json   (nanonets-benchmarks)
  nanobench - caches/{model}/idp/prediction_cache/{hash}.json  (Nanobench flat cache)

Loads datasets via docext, reads cached API responses, parses them into
Prediction objects, and scores with the appropriate metric function.

Requires:
  pip install -e ../docext

Usage:
  python benchmarks/idp/evaluate.py
  python benchmarks/idp/evaluate.py --model nanonets
  python benchmarks/idp/evaluate.py --model chandra --cache-format nanobench
  python benchmarks/idp/evaluate.py --tasks KIE OCR
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

try:
    from docext.benchmark.benchmark import NanonetsIDPBenchmark
    from docext.benchmark.metrics.kie import get_kie_metrics
    from docext.benchmark.metrics.ocr import get_ocr_metrics
    from docext.benchmark.metrics.tables import get_table_metrics
    from docext.benchmark.metrics.vqa import (
        get_vqa__metric_for_multiple_possible_answers,
        get_vqa_metrics,
    )
    from docext.benchmark.vlm_datasets.ds import (
        BenchmarkData,
        PredField,
        Prediction,
        Table,
        VQA,
    )
except ImportError:
    print(
        'ERROR: docext not installed.\n'
        '  Run: pip install -e ../docext',
        file=sys.stderr,
    )
    sys.exit(1)

try:
    from docext.benchmark.tasks import (
        get_KIE_messages,
        get_OCR_messages,
        get_VQA_messages,
        get_TABLE_messages,
    )
    _HAS_TASK_BUILDERS = True
except ImportError:
    _HAS_TASK_BUILDERS = False

try:
    import json_repair
except ImportError:
    json_repair = None


def _parse_kie_response(response) -> dict:
    """Parse a KIE response (dict from extract_json) into {label: value}."""
    if isinstance(response, dict):
        return response
    if isinstance(response, str):
        if json_repair:
            parsed = json_repair.repair_json(response, ensure_ascii=False, return_objects=True)
        else:
            parsed = json.loads(response)
        if isinstance(parsed, list):
            merged = {}
            for item in parsed:
                if isinstance(item, dict):
                    merged.update(item)
            return merged
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _reshape_flat_table(flat: dict) -> pd.DataFrame:
    """Reshape Nanonets' flat {Col_X_rowY: val} dict into a DataFrame.

    The API returns keys like 'Col_1_row1', 'Col_2_row1', 'Col_1_row2', etc.
    We reconstruct columns (Col_1, Col_2, ...) and rows (row1, row2, ...).
    """
    import re
    pattern = re.compile(r'^(.+?)_row(\d+)$')
    cells: dict[tuple[str, int], str] = {}
    cols: set[str] = set()
    rows: set[int] = set()

    for key, val in flat.items():
        m = pattern.match(key)
        if m:
            col_name, row_idx = m.group(1), int(m.group(2))
            cells[(col_name, row_idx)] = str(val) if val is not None else ""
            cols.add(col_name)
            rows.add(row_idx)

    if not cols or not rows:
        return pd.DataFrame([flat]) if flat else pd.DataFrame()

    sorted_cols = sorted(cols)
    sorted_rows = sorted(rows)
    data = []
    for r in sorted_rows:
        row = {c: cells.get((c, r), "") for c in sorted_cols}
        data.append(row)
    return pd.DataFrame(data)


def _parse_table_response(response) -> list[pd.DataFrame]:
    """Parse a TABLE response into a list of DataFrames.

    Handles multiple formats:
      - JSON string (from Chat post-processing): '[{"col":"val"}, ...]'
      - HTML string (fallback): '<table>...</table>'
      - List of dicts: [{"col": "val"}, ...] -> DataFrame
      - Nanonets flat dict (legacy): {"Col_1_row1": "val", ...} -> reshape
      - List of lists: [[row], ...] -> DataFrame
    """
    if isinstance(response, pd.DataFrame):
        return [response]

    raw = response
    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped.startswith("```"):
            import re
            stripped = re.sub(r'^```\w*\n?', '', stripped)
            stripped = re.sub(r'\n?```$', '', stripped).strip()

        if "<table" in stripped.lower():
            try:
                from io import StringIO
                dfs = pd.read_html(StringIO(stripped))
                if dfs:
                    return [dfs[0].fillna("")]
            except Exception:
                pass

        try:
            if json_repair:
                raw = json_repair.repair_json(stripped, ensure_ascii=False, return_objects=True)
            else:
                raw = json.loads(stripped)
        except Exception:
            return [pd.DataFrame()]

    try:
        if isinstance(raw, dict):
            if any("_row" in k for k in raw.keys()):
                return [_reshape_flat_table(raw)]
            return [pd.DataFrame([raw])]
        if isinstance(raw, list):
            if len(raw) > 0 and isinstance(raw[0], list):
                return [pd.DataFrame(item) for item in raw]
            if len(raw) > 0 and isinstance(raw[0], dict):
                return [pd.DataFrame(raw)]
            return [pd.DataFrame(raw)]
    except Exception:
        pass
    return [pd.DataFrame()]


def _build_prediction(data: BenchmarkData, response, task: str) -> Prediction:
    """Build a docext Prediction from ground truth data + cached API response."""
    if task == "KIE":
        parsed = _parse_kie_response(response)
        pred_fields = [
            PredField(
                label=label,
                value=value if isinstance(value, str) else ("" if value is None else json.dumps(value)),
                confidence=-1.0,
            )
            for label, value in parsed.items()
        ]
        return Prediction(
            gt=data,
            pred=BenchmarkData(
                image_paths=data.image_paths,
                extraction_type=data.extraction_type,
                fields=pred_fields,
            ),
        )

    if task == "OCR":
        text = response.strip() if isinstance(response, str) else str(response)
        return Prediction(
            gt=data,
            pred=BenchmarkData(
                image_paths=data.image_paths,
                extraction_type=data.extraction_type,
                ocr_text=text,
            ),
        )

    if task == "VQA":
        answer = response.strip() if isinstance(response, str) else str(response)
        return Prediction(
            gt=data,
            pred=BenchmarkData(
                image_paths=data.image_paths,
                extraction_type=data.extraction_type,
                vqa=VQA(
                    question=data.vqa.question if data.vqa else "",
                    answer=answer,
                ),
            ),
        )

    if task == "TABLE":
        dfs = _parse_table_response(response)
        tables = [
            Table(table=df, columns=df.columns.tolist())
            for df in dfs
        ]
        return Prediction(
            gt=data,
            pred=BenchmarkData(
                image_paths=data.image_paths,
                extraction_type=data.extraction_type,
                tables=tables,
            ),
        )

    raise ValueError(f"Unknown task: {task}")


def _score_predictions(predictions: list[Prediction], task: str, dataset_name: str) -> float:
    """Run the appropriate docext metric on a list of Predictions."""
    if task == "KIE":
        return get_kie_metrics(predictions)
    if task == "OCR":
        return get_ocr_metrics(predictions)
    if task == "VQA":
        if dataset_name == "docvqa":
            return get_vqa__metric_for_multiple_possible_answers(predictions)
        return get_vqa_metrics(predictions)
    if task == "TABLE":
        return get_table_metrics(predictions)
    raise ValueError(f"Unknown task: {task}")


# ---------------------------------------------------------------------------
# Nanobench cache compatibility
# ---------------------------------------------------------------------------

_TASK_MSG_BUILDERS = {
    "KIE": "get_KIE_messages",
    "OCR": "get_OCR_messages",
    "VQA": "get_VQA_messages",
    "TABLE": "get_TABLE_messages",
}


def _nanobench_output_format(task: str) -> str:
    if task in {"KIE", "TABLE", "VQA"}:
        return "json"
    return "markdown"


def _nanobench_json_options(data, task: str) -> str | None:
    if task == "KIE":
        fields = getattr(data, "fields", None) or []
        names = [f.label for f in fields if getattr(f, "label", None)]
        return json.dumps(names, ensure_ascii=False) if names else None
    if task == "TABLE":
        tables = getattr(data, "tables", None) or []
        if tables and getattr(tables[0], "columns", None):
            return json.dumps(list(tables[0].columns), ensure_ascii=False)
        return None
    if task == "VQA":
        return json.dumps(["answer"], ensure_ascii=False)
    return None


def _nanobench_prompt_mode(task: str) -> str | None:
    if task in {"KIE", "OCR", "VQA", "CLASSIFICATION", "TABLE"}:
        return "replace"
    return None


def _nanobench_task_instruction(data, task: str) -> str | None:
    if task == "KIE":
        fields = getattr(data, "fields", None) or []
        labels = [f.label for f in fields if getattr(f, "label", None)]
        labels_text = ", ".join(labels) if labels else "provided field labels"
        return (
            "You are doing key information extraction from a document. "
            "Return ONLY a valid JSON object and nothing else. "
            f"Use exactly these keys: {labels_text}. "
            "Do not add extra keys. For missing values return empty string."
        )
    if task == "TABLE":
        return (
            "Extract the table and return ONLY valid JSON. "
            "Do not include markdown, prose, code fences, or explanations."
        )
    if task == "VQA":
        return (
            'Answer the question from the document and return ONLY valid JSON '
            'in this exact shape: {"answer": "<final answer>"}. '
            "The answer must be a single concise value (number/word/short phrase), "
            "not an array/object/table. "
            "If the answer is numeric, return only the number token (no units/sentence). "
            "If the answer is binary, return exactly Yes or No. "
            "No explanation, no markdown, no extra fields."
        )
    if task == "OCR":
        return (
            "Return only the exact OCR text content as plain text. "
            "No markdown, no tags, no LaTeX, no explanations."
        )
    return None


def _nanobench_cache_hash(messages: list, data, task: str, adapter: str = "litellm") -> str:
    """Compute the MD5 hash matching Nanobench's idp_current.py cache key logic."""
    use_chat = task in ("VQA", "TABLE") and adapter == "nanonets"
    cache_payload = {
        "messages": messages,
        "output_format": _nanobench_output_format(task),
        "json_options": _nanobench_json_options(data, task),
        "prompt_mode": _nanobench_prompt_mode(task),
        "endpoint": "chat_completions" if use_chat else "extraction",
    }
    return hashlib.md5(json.dumps(cache_payload, sort_keys=True).encode()).hexdigest()


def _build_nanobench_messages(data, template: dict, task: str) -> list[dict]:
    """Build messages the same way Nanobench does (docext builders + task instruction)."""
    if not _HAS_TASK_BUILDERS:
        raise ImportError(
            "docext.benchmark.tasks not available. "
            "Install docext with: pip install -e ../docext"
        )
    builder_name = _TASK_MSG_BUILDERS.get(task)
    if not builder_name:
        raise ValueError(f"No message builder for task: {task}")

    builder_fn = {
        "get_KIE_messages": get_KIE_messages,
        "get_OCR_messages": get_OCR_messages,
        "get_VQA_messages": get_VQA_messages,
        "get_TABLE_messages": get_TABLE_messages,
    }[builder_name]

    messages = builder_fn(data, template)
    instruction = _nanobench_task_instruction(data, task)
    if instruction:
        messages.append({"role": "user", "content": instruction})
    return messages


def _parse_nanobench_response(text: str, task: str):
    """Parse Nanobench's {"text": ...} cache value into a response suitable for _build_prediction."""
    if task == "KIE":
        if json_repair:
            parsed = json_repair.repair_json(text, ensure_ascii=False, return_objects=True)
        else:
            parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            merged = {}
            for item in parsed:
                if isinstance(item, dict):
                    merged.update(item)
            return merged
        return {}
    return text


def _detect_cache_format(cache_root: Path, model_name: str) -> str:
    """Auto-detect cache format based on directory structure."""
    native_dir = cache_root
    nanobench_dir = cache_root / "prediction_cache"
    if nanobench_dir.exists() and any(nanobench_dir.glob("*.json")):
        return "nanobench"
    if any(native_dir.iterdir()):
        for child in native_dir.iterdir():
            if child.is_dir() and child.name != "prediction_cache":
                return "native"
    return "native"


def _eval_native(
    dataset, ds_name: str, task: str, cache_root: Path, max_samples: int,
) -> tuple[list[Prediction], int]:
    """Read predictions from native format: {dataset}/{index}.json"""
    ds_cache = cache_root / ds_name
    if not ds_cache.exists():
        return [], -1

    data_items = list(getattr(dataset, "data", []))
    if len(data_items) > max_samples:
        data_items = data_items[:max_samples]

    predictions = []
    missing = 0
    for idx, data in enumerate(data_items):
        cache_file = ds_cache / f"{idx}.json"
        if not cache_file.exists():
            missing += 1
            continue
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            response = cached.get("response", "")
            pred = _build_prediction(data, response, task)
            predictions.append(pred)
        except Exception as e:
            print(f"    WARNING: Failed to parse {cache_file.name}: {e}")
            missing += 1

    return predictions, missing


def _eval_nanobench(
    dataset, ds_name: str, task: str, cache_root: Path,
    model_name: str, template: dict, adapter: str, max_samples: int,
) -> tuple[list[Prediction], int]:
    """Read predictions from Nanobench flat hash cache: prediction_cache/{hash}.json"""
    pred_cache_dir = cache_root / "prediction_cache"
    if not pred_cache_dir.exists():
        return [], -1

    data_items = list(getattr(dataset, "data", []))
    if len(data_items) > max_samples:
        data_items = data_items[:max_samples]

    predictions = []
    missing = 0
    for idx, data in enumerate(data_items):
        try:
            messages = _build_nanobench_messages(data, template, task)
            file_hash = _nanobench_cache_hash(messages, data, task, adapter)
            cache_file = pred_cache_dir / f"{file_hash}.json"

            if not cache_file.exists():
                cache_file_with_model = pred_cache_dir / f"{model_name}_{file_hash}.json"
                if cache_file_with_model.exists():
                    cache_file = cache_file_with_model
                else:
                    missing += 1
                    continue

            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            response_text = cached.get("text", "")
            response = _parse_nanobench_response(response_text, task)
            pred = _build_prediction(data, response, task)
            predictions.append(pred)
        except Exception as e:
            print(f"    WARNING: Failed to process item {idx}: {e}")
            missing += 1

    return predictions, missing


def main():
    parser = argparse.ArgumentParser(description="Evaluate IDP predictions using docext metrics")
    parser.add_argument("--model", type=str, default="nanonets")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--tasks", type=str, nargs="*", default=None, help="Only eval specific tasks")
    parser.add_argument("--datasets", type=str, nargs="*", default=None, help="Only eval specific datasets")
    parser.add_argument(
        "--cache-format", type=str, choices=["native", "nanobench", "auto"], default="auto",
        help="Cache format: native (index-based), nanobench (flat hash-based), or auto-detect",
    )
    parser.add_argument(
        "--nanobench-config", type=str, default=None,
        help="Path to Nanobench benchmark YAML (for hash computation in nanobench mode)",
    )
    parser.add_argument(
        "--adapter", type=str, default="litellm",
        help="Adapter name for Nanobench hash computation (litellm, nanonets, openai)",
    )
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        bench_cfg = yaml.safe_load(f)

    nanobench_cfg = bench_cfg
    if args.nanobench_config:
        with open(args.nanobench_config) as f:
            nanobench_cfg = yaml.safe_load(f)

    cache_root = REPO_ROOT / "caches" / args.model / "idp"
    if not cache_root.exists():
        print(f"ERROR: Cache not found: {cache_root}", file=sys.stderr)
        print(f"  Run: python benchmarks/idp/run.py --model {args.model}", file=sys.stderr)
        print(f"  Or migrate Nanobench caches: python scripts/migrate_caches.py --source ../all_prediction_caches", file=sys.stderr)
        sys.exit(1)

    cache_format = args.cache_format
    if cache_format == "auto":
        cache_format = _detect_cache_format(cache_root, args.model)

    all_tasks = bench_cfg.get("tasks", [])
    tasks_to_eval = set(args.tasks) if args.tasks else set(all_tasks)

    print(f"=== IDP Evaluation (model={args.model}) ===\n")
    print(f"  Cache: {cache_root}")
    print(f"  Format: {cache_format}")
    print(f"  Tasks: {sorted(tasks_to_eval)}")

    idp_benchmark = NanonetsIDPBenchmark(str(config_path))
    all_datasets = idp_benchmark.datasets
    print(f"  Datasets loaded: {len(all_datasets)}\n")

    nanobench_templates = {}
    if cache_format == "nanobench":
        for task_key in ["KIE", "OCR", "VQA", "TABLE"]:
            nanobench_templates[task_key] = nanobench_cfg.get(f"{task_key}_default_template", {})

    results: dict[str, dict] = {}
    max_samples = bench_cfg.get("max_samples_per_dataset", 1000)

    for dataset in all_datasets:
        ds_name = getattr(dataset, "name", dataset.__class__.__name__)
        task = dataset.task
        if task not in tasks_to_eval:
            continue
        if args.datasets and ds_name not in args.datasets:
            continue

        if cache_format == "nanobench":
            template = nanobench_templates.get(task, {})
            predictions, missing = _eval_nanobench(
                dataset, ds_name, task, cache_root,
                args.model, template, args.adapter, max_samples,
            )
        else:
            predictions, missing = _eval_native(
                dataset, ds_name, task, cache_root, max_samples,
            )

        if missing == -1:
            print(f"  {ds_name} ({task}): SKIPPED (no cache)")
            continue

        if not predictions:
            print(f"  {ds_name} ({task}): 0 predictions (need to run inference first)")
            continue

        score = _score_predictions(predictions, task, ds_name)
        results[ds_name] = {"task": task, "score": score, "n": len(predictions), "missing": missing}
        status = f"  {ds_name} ({task}): {score*100:.1f}%  ({len(predictions)} scored"
        if missing:
            status += f", {missing} missing"
        status += ")"
        print(status)

    # Summary
    print("\n" + "=" * 60)
    print(f"  IDP Results (model={args.model})")
    print("=" * 60)

    task_scores: dict[str, list[float]] = {}
    for ds_name, info in sorted(results.items()):
        task = info["task"]
        score = info["score"]
        print(f"  {ds_name:45s} ({task:5s}): {score*100:6.1f}%")
        task_scores.setdefault(task, []).append(score)

    if task_scores:
        print()
        print("  --- Per-Task Averages ---")
        all_dataset_scores = []
        for task in sorted(task_scores):
            scores = task_scores[task]
            avg = sum(scores) / len(scores)
            all_dataset_scores.extend(scores)
            print(f"  {task:15s}: {avg*100:6.1f}%  ({len(scores)} datasets)")

        overall = sum(all_dataset_scores) / len(all_dataset_scores)
        print(f"\n  OVERALL (unweighted mean): {overall*100:.1f}%  ({len(all_dataset_scores)} datasets)")


if __name__ == "__main__":
    main()
