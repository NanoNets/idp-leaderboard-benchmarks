# idp-leaderboard-benchmarks

Prediction cache generation and evaluation pipeline for [idp-leaderboard.org](https://idp-leaderboard.org).

Three benchmarks, each testing a different axis of document understanding:

| Benchmark | What it tests | Scale |
|-----------|--------------|-------|
| **OlmOCR Bench** | Math, tables, reading order, text presence/absence | 1,403 pages · 7,010 tests |
| **OmniDocBench** | Text extraction, formula recognition, table structure, reading order | 1,355 pages · 18K+ samples |
| **IDP Core** | Key info extraction, OCR, table parsing, visual QA | 5,376 samples across 4 tasks |

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

For IDP benchmark, also install [docext](https://github.com/NanoNets/docext):
```bash
pip install -e ../docext
```

For OlmOCR math rendering (KaTeX):
```bash
python -m playwright install chromium
```

## Running Prediction Caches

Every benchmark follows the same pattern: **run** populates the prediction cache, **evaluate** scores it.

### OlmOCR Bench

```bash
# Generate predictions (cached to caches/{model}/olmocr/)
python benchmarks/olmocr/run.py --model nanonets --workers 10

# Evaluate raw predictions
python benchmarks/olmocr/evaluate.py --model nanonets

# Evaluate with post-processing (category-aware cleanup)
python benchmarks/olmocr/evaluate.py --model nanonets --postprocess
```

### OmniDocBench

Requires the OmniDocBench dataset. On EC2, use Docker for full eval (CDM metric needs xelatex).

```bash
# Generate predictions
python benchmarks/omnidocbench/run.py --model nanonets --workers 10

# Full eval via Docker (EC2 recommended)
python benchmarks/omnidocbench/evaluate.py --model nanonets --docker

# Partial eval on host (no CDM metric)
python benchmarks/omnidocbench/evaluate.py --model nanonets --host
```

### IDP Core

Requires `docext` installed locally.

```bash
# Generate predictions
python benchmarks/idp/run.py --model nanonets --workers 5

# Evaluate
python benchmarks/idp/evaluate.py --model nanonets
```

## Adding a New Model

1. Create `models/{name}.py` with a predict function (see `models/nanonets.py` or `models/litellm_model.py` for examples)
2. Run all three benchmarks with `--model {name}`
3. Results land in `results/{name}/`

For LiteLLM-compatible APIs (OpenAI, Anthropic, Google, etc.), use the built-in adapter:
```bash
python benchmarks/olmocr/run.py --model gpt-5.2 --adapter litellm
```

## Publishing Results

After generating caches and evaluations:

```bash
# Consolidate best results per model into results/
python scripts/consolidate_results.py

# Upload to HuggingFace (powers the Results Explorer on the leaderboard)
python scripts/upload_results_to_hf.py
```

The upload script reads from `results/{model}/{benchmark}.json` and pushes trimmed per-sample data to `shhdwi/idp-leaderboard-results` on HuggingFace.

## Directory Layout

```
models/                     API clients (one per model)
benchmarks/
  olmocr/                   OlmOCR Bench eval pipeline
  omnidocbench/             OmniDocBench eval pipeline
  idp/                      IDP Core eval pipeline
scripts/
  consolidate_results.py    Merge best runs into canonical results/
  upload_results_to_hf.py   Push results to HuggingFace dataset
  validate_caches.py        Sanity-check prediction caches
  migrate_caches.py         Restructure legacy cache layouts
caches/{model}/             Prediction caches (gitignored, auto-created)
ground_truth/               Auto-downloaded ground truth (gitignored)
results/{model}/            Evaluation output (gitignored)
```

## Environment Variables

| Variable | Required for | Description |
|----------|-------------|-------------|
| `NANONETS_API_KEY` | Nanonets model | API key for extraction-api.nanonets.com |
| `OPENAI_API_KEY` | GPT models | OpenAI API key |
| `ANTHROPIC_API_KEY` | Claude models | Anthropic API key |
| `GOOGLE_API_KEY` | Gemini models | Google AI API key |
| `HF_WRITE_TOKEN` | Publishing | HuggingFace write token for uploading results |
