#!/usr/bin/env python3
"""
Post-processing for OmniDocBench predictions.

Cleans model output to improve CDM (formula), text edit distance, and
table TEDS metrics.  Run standalone or import ``postprocess(text)`` in
other code.

Applies the following transforms:
  1. HTML entity decoding  (``&lt;`` → ``<``)
  2. Structural tag removal  (<header>, <page_number>, <footer>, <img>, <watermark>)
  3. Repetition truncation and paragraph deduplication
  4. Unicode math → LaTeX conversion  (``²`` → ``^{2}``, ``α`` → ``\\alpha``)
  5. Display-formula delimiter normalization  (``\\[…\\]`` → ``$$…$$``, promote solo ``$…$``)
  6. Placeholder removal  (``□`` lines, empty checkbox sequences)

Usage:
  # Apply to a whole cache directory (copy into a new dir)
  python benchmarks/omnidocbench/postprocess.py \\
      --src caches/nanonets-ocr-3/omnidocbench \\
      --dst caches/nanonets-ocr-3/omnidocbench_pp

  # In-place overwrite
  python benchmarks/omnidocbench/postprocess.py \\
      --src caches/nanonets-ocr-3/omnidocbench --inplace
"""

import argparse
import html
import re
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Unicode → LaTeX maps  (shared with olmocr/postprocess.py)
# ---------------------------------------------------------------------------

UNICODE_SUBSCRIPT = {
    "\u2080": "0", "\u2081": "1", "\u2082": "2", "\u2083": "3",
    "\u2084": "4", "\u2085": "5", "\u2086": "6", "\u2087": "7",
    "\u2088": "8", "\u2089": "9",
    "\u2090": "a", "\u2091": "e", "\u2092": "o", "\u2093": "x",
    "\u2095": "h", "\u2096": "k", "\u2097": "l", "\u2098": "m",
    "\u2099": "n", "\u209a": "p", "\u209b": "s", "\u209c": "t",
    "\u1d62": "i", "\u1d63": "r", "\u1d64": "u", "\u1d65": "v",
    "\u2c7c": "j",
}

UNICODE_SUPERSCRIPT = {
    "\u00b2": "2", "\u00b3": "3", "\u00b9": "1",
    "\u2070": "0", "\u2074": "4", "\u2075": "5", "\u2076": "6",
    "\u2077": "7", "\u2078": "8", "\u2079": "9",
    "\u207a": "+", "\u207b": "-", "\u207c": "=",
    "\u207d": "(", "\u207e": ")",
    "\u207f": "n", "\u2071": "i",
}

UNICODE_MATH = {
    "\u00d7": r"\times", "\u00f7": r"\div",
    "\u2264": r"\leq", "\u2265": r"\geq", "\u2260": r"\neq",
    "\u00b1": r"\pm", "\u221e": r"\infty",
    "\u2211": r"\sum", "\u220f": r"\prod", "\u222b": r"\int",
    "\u221a": r"\sqrt", "\u2202": r"\partial", "\u2207": r"\nabla",
    "\u2208": r"\in", "\u2209": r"\notin",
    "\u2282": r"\subset", "\u2283": r"\supset",
    "\u222a": r"\cup", "\u2229": r"\cap",
    "\u2205": r"\emptyset",
    "\u2203": r"\exists", "\u2200": r"\forall",
    "\u21d2": r"\Rightarrow", "\u21d4": r"\Leftrightarrow",
    "\u2192": r"\to", "\u2190": r"\leftarrow",
    "\u2248": r"\approx", "\u2261": r"\equiv",
    "\u03b1": r"\alpha", "\u03b2": r"\beta", "\u03b3": r"\gamma",
    "\u03b4": r"\delta", "\u03b5": r"\epsilon", "\u03b6": r"\zeta",
    "\u03b7": r"\eta", "\u03b8": r"\theta", "\u03b9": r"\iota",
    "\u03ba": r"\kappa", "\u03bb": r"\lambda", "\u03bc": r"\mu",
    "\u03bd": r"\nu", "\u03be": r"\xi", "\u03c0": r"\pi",
    "\u03c1": r"\rho", "\u03c3": r"\sigma", "\u03c4": r"\tau",
    "\u03c5": r"\upsilon", "\u03c6": r"\phi", "\u03c7": r"\chi",
    "\u03c8": r"\psi", "\u03c9": r"\omega",
    "\u0393": r"\Gamma", "\u0394": r"\Delta", "\u0398": r"\Theta",
    "\u039b": r"\Lambda", "\u039e": r"\Xi", "\u03a0": r"\Pi",
    "\u03a3": r"\Sigma", "\u03a6": r"\Phi", "\u03a8": r"\Psi",
    "\u03a9": r"\Omega",
}

_SUB_RE = re.compile(
    "([A-Za-z0-9}\\]])([" + "".join(re.escape(c) for c in UNICODE_SUBSCRIPT) + "]+)"
)
_SUP_RE = re.compile(
    "([A-Za-z0-9}\\]])([" + "".join(re.escape(c) for c in UNICODE_SUPERSCRIPT) + "]+)"
)


def _replace_sub(m):
    base = m.group(1)
    chars = "".join(UNICODE_SUBSCRIPT.get(c, c) for c in m.group(2))
    return base + "_{" + chars + "}"


def _replace_sup(m):
    base = m.group(1)
    chars = "".join(UNICODE_SUPERSCRIPT.get(c, c) for c in m.group(2))
    return base + "^{" + chars + "}"


def _convert_unicode_math(text: str) -> str:
    text = _SUB_RE.sub(_replace_sub, text)
    text = _SUP_RE.sub(_replace_sup, text)
    for uni, latex in UNICODE_MATH.items():
        if uni in text:
            text = text.replace(uni, " " + latex + " ")
    return text


# ---------------------------------------------------------------------------
# Structural tag removal
# ---------------------------------------------------------------------------

def _strip_img_tags(content: str) -> str:
    content = re.sub(r"<img[^>]*>.*?</img>", "", content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r"<img[^>]*/?>", "", content, flags=re.IGNORECASE)
    return content


def _strip_structural_tags(content: str) -> str:
    """Remove structural wrapper tags that the OmniDocBench evaluation ignores."""
    content = html.unescape(content)
    content = _strip_img_tags(content)
    for tag in ["page_number", "header", "watermark", "signature", "footer"]:
        content = re.sub(
            rf"<{tag}[^>]*>.*?</{tag}>", "", content,
            flags=re.DOTALL | re.IGNORECASE,
        )
        content = re.sub(rf"<{tag}[^>]*/?>", "", content, flags=re.IGNORECASE)
    return content


# ---------------------------------------------------------------------------
# Repetition / dedup
# ---------------------------------------------------------------------------

def _truncate_repetitions(content: str, max_repeats: int = 3) -> str:
    lines = content.split("\n")
    if len(lines) < 20:
        return content
    result = []
    repeat_count = 0
    prev_line = None
    in_table = False
    for line in lines:
        lower = line.strip().lower()
        if "<table" in lower:
            in_table = True
        if "</table>" in lower:
            in_table = False
        stripped = line.strip()
        if in_table:
            result.append(line)
            continue
        if stripped and stripped == prev_line:
            repeat_count += 1
            if repeat_count <= max_repeats:
                result.append(line)
        else:
            repeat_count = 0
            result.append(line)
            prev_line = stripped if stripped else prev_line
    return "\n".join(result)


def _deduplicate_paragraphs(content: str) -> str:
    blocks = content.split("\n\n")
    seen: set[str] = set()
    deduped = []
    for block in blocks:
        key = block.strip()
        if not key:
            continue
        if len(key) > 30 and key in seen and "<table" not in key.lower():
            continue
        seen.add(key)
        deduped.append(block)
    return "\n\n".join(deduped)


# ---------------------------------------------------------------------------
# Formula delimiter normalization
# ---------------------------------------------------------------------------

_DISPLAY_FORMULA_HINTS = re.compile(
    r"\\(?:frac|sum|prod|int|lim|begin|end|left|right|sqrt|"
    r"alpha|beta|gamma|delta|epsilon|theta|lambda|mu|sigma|omega|"
    r"Gamma|Delta|Sigma|Omega|Pi|"
    r"partial|nabla|infty|cdot|times|div|leq|geq|neq|approx|equiv|"
    r"text|mathrm|mathbf|mathcal|operatorname|displaystyle|"
    r"overline|underline|hat|bar|vec|dot|ddot|tilde|"
    r"binom|choose|atop)",
    re.IGNORECASE,
)

_STANDALONE_DOLLAR_RE = re.compile(
    r"^\s*\$(?!\$)(.*?)(?<!\$)\$\s*$", re.DOTALL
)


_INLINE_PAREN_RE = re.compile(
    r"\\\((.*?)\\\)", re.DOTALL
)

_DISPLAY_ONLY_HINTS = re.compile(
    r"\\(?:frac|sum|prod|int|iint|iiint|lim|begin|end|sqrt|"
    r"binom|displaystyle|left|right|operatorname)",
    re.IGNORECASE,
)


def _normalize_formula_delimiters(content: str) -> str:
    r"""Normalize formula delimiters for CDM compatibility.

    The upstream ``md_tex_filter`` recognises ``$$…$$`` and ``\[…\]``.
    Standardize ``\[…\]`` → ``$$…$$`` to avoid edge cases in
    tokenize_latex.
    """
    content = re.sub(
        r"\\\[(.*?)\\\]",
        lambda m: "$$" + m.group(1) + "$$",
        content,
        flags=re.DOTALL,
    )
    return content


# ---------------------------------------------------------------------------
# Placeholder removal
# ---------------------------------------------------------------------------

def _remove_placeholders(content: str) -> str:
    lines = content.split("\n")
    kept = []
    for line in lines:
        n = line.count("\u25a1")
        if n > 10 and n > len(line.strip()) * 0.5:
            continue
        kept.append(line)
    return "\n".join(kept)


def _convert_md_tables_to_html(content: str) -> str:
    """Convert markdown tables to HTML <table> for OmniDocBench TEDS eval.

    Only converts if the prediction has NO HTML tables already.
    """
    if re.search(r"<table", content, re.IGNORECASE):
        return content

    md_table_re = re.compile(
        r"(^\s*\|.*\|[ \t]*\n"
        r"(?:\s*\|[\s\-:|]+\|[ \t]*\n)"
        r"(?:\s*\|.*\|[ \t]*\n?)*)",
        re.MULTILINE,
    )

    def _md_to_html(m):
        block = m.group(0).strip()
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        if len(lines) < 2:
            return m.group(0)

        rows = []
        is_header = True
        for line in lines:
            if re.match(r"\|[\s\-:|]+\|$", line):
                is_header = False
                continue
            cells = [c.strip() for c in line.split("|")]
            cells = [c for c in cells if c != "" or cells.index(c) not in (0, len(cells)-1)]
            if not cells:
                cells = [c.strip() for c in line.strip("|").split("|")]
            tag = "th" if is_header else "td"
            row_html = "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>"
            rows.append(row_html)
            if is_header:
                is_header = False

        return "\n<table>" + "".join(rows) + "</table>\n"

    return md_table_re.sub(_md_to_html, content)


def _collapse_blank_lines(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def postprocess(text: str) -> str:
    """Apply OmniDocBench-specific post-processing to raw model output.

    Designed for the Nanonets OCR-3 (and OCR-2+) extract API output with
    ``prompt_mode='replace'``.
    """
    text = _truncate_repetitions(text)
    text = _deduplicate_paragraphs(text)
    text = _strip_structural_tags(text)
    text = _remove_placeholders(text)
    text = _convert_unicode_math(text)
    text = _normalize_formula_delimiters(text)
    text = _collapse_blank_lines(text)
    return text.strip() + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Post-process OmniDocBench prediction cache")
    parser.add_argument("--src", type=str, required=True, help="Source prediction cache directory")
    parser.add_argument("--dst", type=str, default=None, help="Destination directory (copy mode)")
    parser.add_argument("--inplace", action="store_true", help="Overwrite source files in-place")
    args = parser.parse_args()

    src = Path(args.src)
    if not src.is_dir():
        print(f"ERROR: {src} is not a directory")
        return

    if args.inplace:
        dst = src
    elif args.dst:
        dst = Path(args.dst)
        dst.mkdir(parents=True, exist_ok=True)
    else:
        print("ERROR: Specify --dst or --inplace")
        return

    md_files = sorted(src.glob("*.md"))
    print(f"Processing {len(md_files)} files: {src} → {dst}")

    changed = 0
    for f in md_files:
        try:
            original = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            original = f.read_text(encoding="utf-8", errors="replace")
        processed = postprocess(original)
        out_file = dst / f.name
        if processed != original:
            changed += 1
        out_file.write_text(processed, encoding="utf-8")

    print(f"Done. {changed}/{len(md_files)} files modified.")


if __name__ == "__main__":
    main()
