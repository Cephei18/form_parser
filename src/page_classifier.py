"""Detect non-fillable pages so the anchoring engine does not emit widgets on
instruction / checklist / reference / blank pages.

Multi-page forms routinely append pages that carry NO user-fillable fields:
office-use checklists, risk-disclosure ("Riskometer") pages, general
instructions, terms & conditions, and "intentionally left blank" pages.
Textract still reports KEY_VALUE_SET pairs, SELECTION_ELEMENTs (pre-printed
ticks) and table cells on them, so the engine would emit dozens of false
widgets. Evaluation already scores these as ``non_fillable_false_positives``;
this module lets the pipeline avoid producing them in the first place.

Detection is deliberately conservative — it keys off a SHORT top-of-page banner
heading (so a fillable page that merely *mentions* "instruction 7" or
"refer ... Riskometer ... last page" in a long sentence is never mis-flagged)
plus a blank-page check. ON by default; ``FORM_PARSER_PAGE_GATING_ENABLED=false``
restores the emit-everywhere behaviour.
"""
from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Any

# A page with fewer than this many words and no form fields is effectively blank.
BLANK_MAX_WORDS = 14

# A banner line is considered a page title only when it sits near the top and is
# short (a heading, not a sentence that happens to contain a keyword).
BANNER_MAX_TOP = 0.15
BANNER_MAX_WORDS = 7

# Substrings that mark a reference/non-fillable page when they appear in a short
# top banner line.
_BANNER_MARKERS = (
    "intentionally left blank",
    "checklist",
    "riskometer",
)

# Short title lines that are reference pages when the line *is* (or starts with)
# the marker — matched on the whole normalised line, never as a loose substring.
_TITLE_MARKERS = (
    "instructions",
    "general instructions",
    "instructions to investors",
    "how to fill",
    "terms and conditions",
    "frequently asked questions",
)


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def page_gating_enabled() -> bool:
    """True when non-fillable pages are suppressed (default ON)."""
    return _bool_env("FORM_PARSER_PAGE_GATING_ENABLED", True)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def non_fillable_pages(raw_response: dict[str, Any], parsed: dict[str, Any] | None = None) -> dict[int, str]:
    """Return ``{page_number: reason}`` for every page that carries no fillable
    fields (blank or a reference/instruction/checklist/riskometer page).

    ``parsed`` (the textract_parser output) is optional but recommended: a page
    bearing parsed field_items / checkboxes is never treated as blank, which
    keeps sparse real input pages safe even when the raw block stream is thin.
    """
    blocks = raw_response.get("Blocks", []) or []

    parsed_per_page: dict[int, int] = defaultdict(int)
    if parsed:
        for fi in parsed.get("field_items", []) or []:
            parsed_per_page[int(fi.get("page") or 1)] += 1
        for cb in parsed.get("checkboxes", []) or []:
            parsed_per_page[int(cb.get("page") or 1)] += 1

    words_per_page: dict[int, int] = defaultdict(int)
    keys_per_page: dict[int, int] = defaultdict(int)
    content_per_page: dict[int, int] = defaultdict(int)  # selection elements + cells
    lines_per_page: dict[int, list[tuple[float, str]]] = defaultdict(list)
    pages: set[int] = set()

    for block in blocks:
        btype = block.get("BlockType")
        page = int(block.get("Page") or 1)
        if btype == "PAGE":
            pages.add(page)
            continue
        if btype == "WORD":
            words_per_page[page] += 1
        elif btype == "LINE":
            top = float((block.get("Geometry") or {}).get("BoundingBox", {}).get("Top", 1.0))
            lines_per_page[page].append((top, _norm(block.get("Text"))))
        elif btype == "KEY_VALUE_SET" and "KEY" in (block.get("EntityTypes") or []):
            keys_per_page[page] += 1
        elif btype in {"SELECTION_ELEMENT", "CELL"}:
            content_per_page[page] += 1
        pages.add(page)

    result: dict[int, str] = {}
    for page in sorted(pages):
        words = words_per_page.get(page, 0)
        keys = keys_per_page.get(page, 0)
        content = content_per_page.get(page, 0)

        # Blank / near-empty page (e.g. "THIS PAGE IS INTENTIONALLY LEFT BLANK").
        # Requires no fields, no checkboxes and no table cells — otherwise a
        # sparse but real input page would be wrongly skipped.
        if words <= BLANK_MAX_WORDS and keys == 0 and content == 0 and parsed_per_page.get(page, 0) == 0:
            result[page] = "blank"
            continue

        banner_lines = [
            text for top, text in lines_per_page.get(page, []) if top <= BANNER_MAX_TOP and text
        ]
        marker = _match_banner(banner_lines)
        if marker:
            result[page] = marker
    return result


def _match_banner(banner_lines: list[str]) -> str | None:
    for text in banner_lines:
        word_count = len(text.split())
        if word_count == 0 or word_count > BANNER_MAX_WORDS:
            continue
        for m in _BANNER_MARKERS:
            if m in text:
                return f"banner:{m.split()[0]}"
        for m in _TITLE_MARKERS:
            if text == m or text.startswith(m + " ") or text.startswith(m + ":"):
                return f"title:{m.split()[0]}"
    return None
