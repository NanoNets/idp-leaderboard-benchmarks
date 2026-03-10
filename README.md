# idp-leaderboard-benchmarks

Prediction cache generation and evaluation pipeline for [idp-leaderboard.org](https://idp-leaderboard.org).

Three benchmarks, each testing a different axis of document understanding:

| Benchmark | What it tests | Scale |
|-----------|--------------|-------|
| **OlmOCR Bench** | Math, tables, reading order, text presence/absence | 1,403 pages · 7,010 tests |
| **OmniDocBench** | Text extraction, formula recognition, table structure, reading order | 1,355 pages · 18K+ samples |
| **IDP Core** | Key info extraction, OCR, table parsing, visual QA | 6,406 samples across 4 tasks |

## Data Sources

All benchmark data is fetched automatically — no manual downloads needed for most workflows.

| Benchmark | Ground truth | Images / PDFs |
|-----------|-------------|---------------|
| **OlmOCR Bench** | JSONL from [`allenai/olmOCR-bench`](https://huggingface.co/datasets/allenai/olmOCR-bench) (auto-cached to `ground_truth/`) | PDFs + pre-rendered PNGs from HuggingFace |
| **OmniDocBench** | `OmniDocBench.json` (local, from [`opendatalab/OmniDocBench`](https://huggingface.co/datasets/opendatalab/OmniDocBench)) | Local if available, falls back to HuggingFace URLs |
| **IDP Core** | Loaded via `docext` from HuggingFace datasets | Embedded in dataset (base64) |

OmniDocBench is the only benchmark that needs a local file (`OmniDocBench.json`). Clone the dataset repo or point `--omnidoc-root` at a directory containing it.

## Quick Start

Requires **Python 3.10+**.

**Automated setup** (recommended for first-time users):

```bash
./setup.sh
```

This creates a virtualenv, installs all dependencies (including Playwright for olmOCR evaluation), sets up a `.env` template, and validates the install.

**Manual setup:**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium   # needed for olmOCR evaluation
```

For IDP benchmark, also install [docext](https://github.com/NanoNets/docext):
```bash
git clone https://github.com/NanoNets/docext.git ../docext
pip install -e ../docext
```

For OmniDocBench, clone the dataset:
```bash
git clone https://huggingface.co/datasets/opendatalab/OmniDocBench ../OmniDocBench
```

Set your API keys (create a `.env` file or export directly):
```bash
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="AIza..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Running Benchmarks via LiteLLM (GPT, Gemini, Claude, etc.)

All three benchmarks support `--provider litellm --model-id <litellm_model_id>` for any model that LiteLLM can route to. This is the primary way to benchmark third-party models.

LiteLLM model IDs follow the `provider/model-name` format, e.g.:
- `openai/gpt-5.4-2026-03-05`
- `gemini/gemini-3.1-pro-preview`
- `anthropic/claude-sonnet-4-6`

### OlmOCR Bench

```bash
python benchmarks/olmocr/run.py \
  --model gpt-5.4 \
  --provider litellm \
  --model-id openai/gpt-5.4-2026-03-05 \
  --workers 25
```

### OmniDocBench

```bash
python benchmarks/omnidocbench/run.py \
  --model gpt-5.4 \
  --provider litellm \
  --model-id openai/gpt-5.4-2026-03-05 \
  --omnidoc-root ~/Nanobench/data/omnidocbench \
  --workers 25
```

### IDP Core

```bash
python benchmarks/idp/run.py \
  --model gpt-5.4 \
  --provider litellm \
  --model-id openai/gpt-5.4-2026-03-05 \
  --workers 25
```

## Running Benchmarks via Nanonets API

For the Nanonets model, use the default provider (no `--provider` flag needed):

```bash
python benchmarks/olmocr/run.py --model nanonets --workers 10
python benchmarks/omnidocbench/run.py --model nanonets --workers 10
python benchmarks/idp/run.py --model nanonets --workers 5
```

## Evaluation

After prediction caches are populated, run evaluation:

```bash
# OlmOCR
python benchmarks/olmocr/evaluate.py --model gpt-5.4
python benchmarks/olmocr/evaluate.py --model gpt-5.4 --postprocess

# OmniDocBench (Docker recommended for full CDM metric)
python benchmarks/omnidocbench/evaluate.py --model gpt-5.4 --docker
python benchmarks/omnidocbench/evaluate.py --model gpt-5.4 --host   # partial, no CDM

# IDP
python benchmarks/idp/evaluate.py --model gpt-5.4
```

## Publishing Results

After generating caches and evaluations:

```bash
python scripts/consolidate_results.py
```

## Directory Layout

```
setup.sh                      Automated first-time setup
models/
  litellm_model.py            LiteLLM adapter (OpenAI, Anthropic, Google, etc.)
  nanonets.py                 Nanonets OCR2+ API client
benchmarks/
  olmocr/                     OlmOCR Bench — run, evaluate, postprocess
  omnidocbench/               OmniDocBench — run, evaluate, postprocess
  idp/                        IDP Core — run, evaluate
scripts/
  consolidate_results.py      Merge best runs into canonical results/
  validate_caches.py          Sanity-check prediction caches
  migrate_caches.py           Restructure legacy cache layouts
  rerun_empty.py              Re-run empty/failed cache entries
caches/{model}/               Prediction caches (gitignored, auto-created)
ground_truth/                 Auto-downloaded ground truth (gitignored)
results/{model}/              Evaluation output JSON files
```

## Environment Variables

| Variable | Required for | Description |
|----------|-------------|-------------|
| `OPENAI_API_KEY` | GPT models | OpenAI API key (auto-read by LiteLLM) |
| `GOOGLE_API_KEY` | Gemini models | Google AI API key (auto-read by LiteLLM) |
| `ANTHROPIC_API_KEY` | Claude models | Anthropic API key (auto-read by LiteLLM) |
| `NANONETS_API_KEY` | Nanonets model | API key for extraction-api.nanonets.com |
