"""
Document hierarchy / section detection (Phase 2.2).

Turns a flat ``Page -> Fields`` view into ``Document -> Section -> Field`` by
discovering section headings and assigning every field / checkbox / table to the
section whose reading-order band contains it.

Design constraints (deliberately narrow for this phase):
  * Pure + dependency-light (no cv2 / boto3) so it is trivially unit-testable.
  * Page-aware and multi-page-safe: ownership is resolved by ``(page, y)``
    reading position, so a section started on page 1 keeps owning items on
    page 2 until a new heading appears (sections carry across page breaks).
  * Additive: callers attach ``section`` + ``qualified_label`` to mappings; the
    raw ``label`` is never mutated, so existing contracts/tests are unaffected.

NOT in scope here: template systems, canonical schemas, review queues, advanced
table semantics. This module only detects sections and assigns ownership.
"""
from __future__ import annotations

import re
from typing import Any

# --- Canonical section vocabulary -------------------------------------------
# Ordered most-specific first; the first keyword whose phrase is a substring of
# the (lowercased) heading text wins. Person-section names ("applicant" etc.)
# and self-naming sections ("fatca", "declaration", "signature") are accepted on
# the keyword alone; the very generic ones that also appear inside ordinary
# field labels ("bank", "contact") require a structural cue (see below).
SECTION_KEYWORDS: list[tuple[str, set[str]]] = [
    ("signature", {"signature", "signatures", "authorised signatory", "authorized signatory"}),
    ("fatca", {"fatca", "crs", "tax residency", "tax residence", "foreign account"}),
    ("nominee", {"nominee", "nomination"}),
    ("guardian", {"guardian"}),
    ("joint_holder", {"second holder", "third holder", "joint holder", "second applicant",
                      "third applicant", "co-applicant", "co applicant", "other holder", "additional holder"}),
    ("applicant", {"applicant", "first holder", "sole holder", "primary holder", "first applicant",
                   "personal details", "investor details", "holder details", "details of applicant",
                   "details of first", "details of sole"}),
    ("holding_mode", {"holding mode", "mode of holding", "mode of operation", "operating instruction"}),
    ("contact", {"contact", "communication", "correspondence", "address details"}),
    ("bank", {"bank", "account details", "demat", "depository", "payment details"}),
    ("declaration", {"declaration", "declarations", "undertaking"}),
    ("kyc", {"kyc", "identity details", "proof of identity"}),
]

# Keyword types accepted on the keyword alone (they ARE section names).
_SELF_CUE_TYPES = {"signature", "fatca", "declaration", "nominee", "guardian",
                   "applicant", "joint_holder", "holding_mode"}
# Generic keyword types that also occur in plain field labels ("Bank Name",
# "Contact Number") and therefore require a structural cue to count as a heading.
_CUE_REQUIRED_TYPES = {"contact", "bank", "kyc"}

# Words that strongly signal a heading rather than a field label.
_CUE_WORDS = {"details", "detail", "section", "information", "particulars", "part"}

MAX_HEADING_WORDS = 8

ROOT_SECTION: dict[str, Any] = {
    "section_id": "document",
    "title": "Document",
    "type": "document",
    "page": 1,
    "y_start": 0.0,
    "y_end": 1.0,
    "confidence": 1.0,
    "reasons": ["document_root"],
    "heading_block_id": None,
}


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _slug(value: Any) -> str:
    text = _clean(value).lower()
    text = re.sub(r"^\s*\d+[\s.)\-:]*", "", text)
    return re.sub(r"[^a-z0-9]+", "", text)


def _contains_any(text: str, words: set[str]) -> bool:
    return any(word in text for word in words)


def _match_keyword(norm: str) -> str | None:
    for section_type, phrases in SECTION_KEYWORDS:
        if any(phrase in norm for phrase in phrases):
            return section_type
    return None


def _score_heading(
    line: dict[str, Any],
    field_key_slugs: set[str],
    metrics_by_page: dict[int, dict[str, float]],
) -> tuple[float, str, str, list[str]] | None:
    """Classify one text line as a section heading. Returns
    ``(confidence, section_type, title, reasons)`` or ``None`` if it is not a
    heading."""
    text = _clean(line.get("text"))
    if not text:
        return None
    words = text.split()
    if len(words) > MAX_HEADING_WORDS:
        return None  # headings are short; this is body text / a long label

    slug = _slug(text)
    if slug and slug in field_key_slugs:
        return None  # it is a field label, not a heading

    # "Key: value" shaped lines are fields, never headings.
    if ":" in text and text.split(":", 1)[1].strip():
        return None

    page = int(line.get("page") or 1)
    line_height = float((metrics_by_page.get(page) or {}).get("line_height") or 0.014)
    height = float((line.get("bbox") or {}).get("height") or 0.0)

    letters = [c for c in text if c.isalpha()]
    caps = bool(letters) and (sum(1 for c in letters if c.isupper()) / len(letters)) >= 0.7
    big = height >= line_height * 1.3
    numbered = bool(
        re.match(r"^\s*(section|part|step|annexure|schedule)\b", text.lower())
        or re.match(r"^\s*\(?[0-9ivxabc]+[.)]\s+\S", text.lower())
    )
    has_cue_word = _contains_any(text.lower(), _CUE_WORDS)
    structural_cue = big or caps or numbered or has_cue_word

    section_type = _match_keyword(text.lower())
    reasons: list[str] = []

    if section_type:
        if section_type in _CUE_REQUIRED_TYPES and not structural_cue:
            return None
        score = 0.6
        reasons.append(f"keyword:{section_type}")
    else:
        # Unknown / generic section: require strong typography to avoid turning
        # ordinary lines into spurious sections.
        if not (big and caps) and not (big and numbered):
            return None
        section_type = "unknown"
        score = 0.4

    if big:
        score += 0.15
        reasons.append("large_type")
    if caps:
        score += 0.12
        reasons.append("uppercase")
    if numbered:
        score += 0.08
        reasons.append("numbered")
    if has_cue_word:
        reasons.append("section_cue_word")
    if len(words) <= 5:
        score += 0.05

    return min(score, 1.0), section_type, text, reasons


def detect_sections(
    lines: list[dict[str, Any]],
    field_key_slugs: set[str],
    metrics_by_page: dict[int, dict[str, float]],
) -> list[dict[str, Any]]:
    """Detect section headings from text ``lines`` (each ``{text, bbox, page}``).

    Returns sections in reading order, each with a vertical ownership band
    (``y_start``/``y_end``). A heading owns everything from its position until
    the next heading on the same page (or the page bottom).
    """
    candidates: list[dict[str, Any]] = []
    for line in lines:
        scored = _score_heading(line, field_key_slugs, metrics_by_page)
        if scored is None:
            continue
        confidence, section_type, title, reasons = scored
        bbox = line.get("bbox") or {}
        candidates.append(
            {
                "title": title,
                "type": section_type,
                "page": int(line.get("page") or 1),
                "y_start": float(bbox.get("y") or 0.0),
                "bbox": bbox,
                "confidence": round(confidence, 4),
                "reasons": reasons,
                "heading_block_id": line.get("block_id"),
            }
        )

    candidates.sort(key=lambda s: (s["page"], s["y_start"]))

    sections: list[dict[str, Any]] = []
    for index, section in enumerate(candidates, start=1):
        # y_end: next heading on the same page, else the bottom of the page.
        y_end = 1.0
        for following in candidates[index:]:
            if following["page"] == section["page"]:
                y_end = following["y_start"]
            break
        section["section_id"] = f"section_{index}"
        section["y_end"] = y_end
        sections.append(section)
    return sections


class SectionIndex:
    """Resolves the owning section for an item at a given ``(page, y)`` reading
    position. Items above the first heading fall to the document root."""

    def __init__(self, sections: list[dict[str, Any]]):
        self.sections = sorted(sections, key=lambda s: (s["page"], s["y_start"]))
        self.root = dict(ROOT_SECTION)

    def owner(self, page: int, y: float) -> dict[str, Any]:
        position = (int(page or 1), float(y or 0.0))
        chosen = self.root
        for section in self.sections:
            if (section["page"], section["y_start"]) <= position:
                chosen = section
            else:
                break
        return chosen

    def band(self, section: dict[str, Any] | None) -> dict[str, Any] | None:
        """The (page, y_start, y_end) band for a non-root section, else None."""
        if not section or section.get("type") == "document":
            return None
        return {"page": int(section["page"]), "y_start": float(section["y_start"]), "y_end": float(section["y_end"])}


def section_summary(section: dict[str, Any] | None) -> dict[str, Any]:
    """Compact, mapping-embeddable view of a section owner."""
    section = section or ROOT_SECTION
    return {
        "section_id": section.get("section_id", "document"),
        "title": section.get("title", "Document"),
        "type": section.get("type", "document"),
        "page": int(section.get("page") or 1),
        "confidence": float(section.get("confidence") or 1.0),
    }


def qualify_label(section: dict[str, Any] | None, label: str) -> str:
    """``"Applicant Details › Name"`` for sectioned fields; raw label at root."""
    title = (section or {}).get("title")
    if not section or section.get("type") == "document" or not title:
        return label
    return f"{title} › {label}"


def build_hierarchy(sections: list[dict[str, Any]], mappings: list[dict[str, Any]]) -> dict[str, Any]:
    """Assemble the Document -> Section -> Field tree from finished mappings.

    Items are bucketed by their attached ``section.section_id`` (root items land
    under the synthetic ``document`` node). Purely for explainability/debugging.
    """
    by_id: dict[str, dict[str, Any]] = {ROOT_SECTION["section_id"]: {**section_summary(ROOT_SECTION), "children": []}}
    for section in sections:
        by_id[section["section_id"]] = {**section_summary(section), "pages": [], "children": []}

    for mapping in mappings or []:
        owner = (mapping.get("section") or {}).get("section_id", "document")
        node = by_id.get(owner) or by_id[ROOT_SECTION["section_id"]]
        node["children"].append(
            {
                "field_id": mapping.get("field_id"),
                "label": mapping.get("label"),
                "qualified_label": mapping.get("qualified_label", mapping.get("label")),
                "field_type": mapping.get("field_type"),
                "page": int(mapping.get("page") or 1),
            }
        )
        pages = node.setdefault("pages", [])
        page = int(mapping.get("page") or 1)
        if page not in pages:
            pages.append(page)

    ordered = [by_id[ROOT_SECTION["section_id"]]] + [by_id[s["section_id"]] for s in sections]
    return {
        "document": "Document",
        "section_count": len(sections),
        "sections": [node for node in ordered if node["children"] or node["section_id"] != "document"],
    }
