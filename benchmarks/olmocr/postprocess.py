"""
Post-processing for olmOCR benchmark predictions.

Single public entry point:
    postprocess(text, category, bbox_elements=None) -> str

Internally routes to the appropriate strategy based on category.
"""

import html
import re

# ---------------------------------------------------------------------------
# Bbox-aware filtering constants
# ---------------------------------------------------------------------------

BBOX_STRIP_TYPES = {"header", "footer", "number", "aside_text", "image"}

# ---------------------------------------------------------------------------
# Unicode → LaTeX maps
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
# Structural helpers
# ---------------------------------------------------------------------------

def _strip_img_tags(content: str) -> str:
    content = re.sub(r"<img[^>]*>.*?</img>", "", content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r"<img[^>]*/?>", "", content, flags=re.IGNORECASE)
    return content


def _strip_structural_tags_delete(content: str) -> str:
    content = html.unescape(content)
    content = _strip_img_tags(content)
    for tag in ["page_number", "header", "watermark", "signature", "footer"]:
        content = re.sub(rf"<{tag}[^>]*>.*?</{tag}>", "", content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(rf"<{tag}[^>]*/?>", "", content, flags=re.IGNORECASE)
    return content


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


def _remove_placeholders(content: str) -> str:
    lines = content.split("\n")
    kept = []
    for line in lines:
        n = line.count("\u25a1")
        if n > 10 and n > len(line.strip()) * 0.5:
            continue
        kept.append(line)
    return "\n".join(kept)


def _join_hyphenated_breaks(text: str) -> str:
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def _collapse_blank_lines(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text)


# ---------------------------------------------------------------------------
# Regex patterns for boundary cleanup
# ---------------------------------------------------------------------------

PAGE_HEADING = re.compile(r"^\s*#{1,6}\s*page\s+\d+\s*$", re.IGNORECASE)
PAGE_MARKER_LINE = re.compile(
    r"^\s*(?:#{1,6}\s*)?page\s+\d+(?:\s*(?:/|of)\s*\d+)?\s*$", re.IGNORECASE
)
PAGE_MARKER_STRICT = re.compile(
    r"^\s*(?:---\s*)?(?:page\s*\d+|p\.?\s*\d+|\d+\s*/\s*\d+)(?:\s*---\s*)?$",
    re.IGNORECASE,
)
BARE_NUMBER_LINE = re.compile(r"^\s*\*{0,2}\d{1,4}\*{0,2}\s*$")
BOLD_NUMBER_LINE = re.compile(r"^\s*\*{2}\d{1,4}\*{2}\s*$")
SEPARATOR_LINE = re.compile(r"^\s*-{3,}\s*$")
MD_IMG_LINE = re.compile(r"^\s*!\[[^\]]*\]\([^)]+\)\s*$")
BOUNDARY_BOILERPLATE_LINE = re.compile(
    r"^\s*(?:"
    r"please cite this article|all rights reserved|copyright|doi\s*:|"
    r"http[s]?://|www\.|cell press|available online|preprint|in press|"
    r"creative commons|arxiv:"
    r").*",
    re.IGNORECASE,
)
URL_LINE = re.compile(r"^\s*https?://\S+\s*$")
EMAIL_LINE = re.compile(r"^\s*[\w.+-]+@[\w.-]+\.\w+\s*$")
COPYRIGHT_LINE = re.compile(r"^\s*(?:©|\(c\)|copyright)\s", re.IGNORECASE)
DOI_ARXIV_LINE = re.compile(r"^\s*(?:doi\s*[:.]|arxiv\s*:)", re.IGNORECASE)
JOURNAL_META_LINE = re.compile(
    r"^\s*(?:please cite this article|all rights reserved|copyright|"
    r"doi\s*:|https?://|www\.|cell press|available online|"
    r"preprint|in press|creative commons|arxiv:).*",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Bbox-aware filtering
# ---------------------------------------------------------------------------

def _normalize_for_match(line: str) -> str:
    compact = re.sub(r"\s+", " ", line or "").strip().lower()
    compact = compact.strip("`*#_~[](){}<>:;.,-")
    return compact


def _strip_bbox_boilerplate(content: str, elements: list) -> str:
    if not content or not elements:
        return content
    candidates: set[str] = set()
    for el in elements:
        if not isinstance(el, dict):
            continue
        region_type = str(el.get("type", "")).lower()
        if region_type not in BBOX_STRIP_TYPES:
            continue
        text = str(el.get("content", "")).strip()
        if not text:
            bb = el.get("bounding_box", {})
            if isinstance(bb, dict):
                text = str(bb.get("text", "")).strip()
        norm = _normalize_for_match(text)
        if norm:
            candidates.add(norm)
            for subline in text.split("\n"):
                sub_norm = _normalize_for_match(subline)
                if sub_norm:
                    candidates.add(sub_norm)
    if not candidates:
        return content
    kept = []
    for line in content.splitlines():
        norm_line = _normalize_for_match(line)
        if norm_line and norm_line in candidates:
            continue
        kept.append(line)
    return "\n".join(kept)


def _strip_lines_matching(text: str, pattern: re.Pattern) -> str:
    return "\n".join(l for l in text.splitlines() if not pattern.match(l))


def _strip_boundary_boilerplate_windowed(text: str, window: int = 15) -> str:
    lines = text.splitlines()
    if not lines:
        return text
    n = len(lines)
    head = set(range(min(window, n)))
    tail = set(range(max(0, n - window), n))
    kept = []
    for i, line in enumerate(lines):
        if (i in head or i in tail) and BOUNDARY_BOILERPLATE_LINE.match(line):
            continue
        kept.append(line)
    return "\n".join(kept)


# ---------------------------------------------------------------------------
# Internal post-processing strategies
# ---------------------------------------------------------------------------

def _postprocess_tables(content: str) -> str:
    content = _strip_structural_tags_delete(content)
    lines = content.splitlines()
    kept = [l for l in lines if not PAGE_HEADING.match(l.strip())]
    content = "\n".join(kept)
    content = _collapse_blank_lines(content)
    return content.strip() + "\n"


def _postprocess_math(content: str) -> str:
    content = _strip_structural_tags_delete(content)
    content = _convert_unicode_math(content)
    lines = content.splitlines()
    kept = [l for l in lines if not PAGE_HEADING.match(l.strip())]
    content = "\n".join(kept)
    content = _collapse_blank_lines(content)
    return content.strip() + "\n"


def _postprocess_default(content: str) -> str:
    content = _truncate_repetitions(content)
    content = _deduplicate_paragraphs(content)
    content = _strip_structural_tags_delete(content)
    content = _remove_placeholders(content)
    content = _convert_unicode_math(content)

    lines = content.splitlines()
    n = len(lines)
    kept = []
    in_table_html = False
    in_table_md = False

    for idx, line in enumerate(lines):
        stripped = line.strip()
        lower = stripped.lower()
        if "<table" in lower:
            in_table_html = True
        if "</table>" in lower:
            in_table_html = False
        if "|" in stripped and not in_table_html:
            in_table_md = True
        elif in_table_md and not stripped:
            in_table_md = False

        in_any_table = in_table_html or in_table_md
        at_boundary = idx < 15 or idx >= n - 15

        if PAGE_HEADING.match(stripped):
            continue
        if PAGE_MARKER_LINE.match(stripped):
            continue
        if SEPARATOR_LINE.match(stripped) and not in_any_table:
            prev_is_table = idx > 0 and ("|" in lines[idx - 1] or "<t" in lines[idx - 1].lower())
            next_is_table = idx < n - 1 and ("|" in lines[idx + 1] or "<t" in lines[idx + 1].lower())
            if not prev_is_table and not next_is_table:
                continue
        if at_boundary and BARE_NUMBER_LINE.match(stripped):
            continue
        if BOLD_NUMBER_LINE.match(stripped):
            continue
        if MD_IMG_LINE.match(stripped):
            continue
        if at_boundary and BOUNDARY_BOILERPLATE_LINE.match(stripped):
            continue
        kept.append(line)

    content = "\n".join(kept)
    content = _join_hyphenated_breaks(content)
    content = _collapse_blank_lines(content)
    return content.strip() + "\n"


def _postprocess_headers_footers(content: str, elements: list) -> str:
    out = _strip_bbox_boilerplate(content, elements)
    out = _strip_img_tags(out)
    out = _join_hyphenated_breaks(out)
    out = _strip_lines_matching(out, PAGE_HEADING)
    out = _strip_lines_matching(out, PAGE_MARKER_STRICT)
    out = _strip_lines_matching(out, URL_LINE)
    out = _strip_lines_matching(out, EMAIL_LINE)
    out = _strip_lines_matching(out, COPYRIGHT_LINE)
    out = _strip_lines_matching(out, DOI_ARXIV_LINE)
    out = _strip_lines_matching(out, JOURNAL_META_LINE)
    out = _strip_lines_matching(out, BARE_NUMBER_LINE)
    out = _strip_boundary_boilerplate_windowed(out, window=15)
    out = _collapse_blank_lines(out)
    return out.strip() + "\n"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def postprocess(text: str, category: str, bbox_elements: list | None = None) -> str:
    """Apply category-appropriate post-processing to raw olmOCR output.

    Args:
        text: Raw markdown text from the API.
        category: olmOCR benchmark category (e.g. "arxiv_math", "tables").
        bbox_elements: Bounding box elements from extract_with_bbox (headers_footers only).

    Returns:
        Post-processed text ready for evaluation.
    """
    if category in ("arxiv_math", "old_scans_math"):
        return _postprocess_math(text)
    elif category == "headers_footers":
        return _postprocess_headers_footers(text, bbox_elements or [])
    elif category == "tables":
        return _postprocess_tables(text)
    else:
        return _postprocess_default(text)
