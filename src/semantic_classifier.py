"""Phase K — Semantic Classifier.

The system understands fields, regions, tables and assignment — but not what a
field *means*. "PAN", "Aadhaar", "Mobile Number", "Address", "Gender" are
treated as plain strings, which caps widget selection, confidence, assignment
quality and validation.

This module adds a rule-based (no ML, no embeddings, no LLM) semantic layer. It
classifies a field from its label / section / value into a ``SemanticField``
carrying the type, a preferred widget, preferred answer-region types, an
expected length and a validation pattern. The classifier's region preferences
are turned into **score priors** on candidate regions, so a field steers toward
geometrically-plausible answer regions for its meaning (Address -> multiline,
PAN -> comb, ...). Those adjusted scores flow into both legacy argmax and the
Phase H global assignment with no change to the assignment engine.

Gated behind ``FORM_PARSER_SEMANTICS_ENABLED`` (default OFF). When OFF the
classifier is never invoked and behaviour is byte-identical.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

from src.semantic_gazetteers import (
    EXPECTATIONS,
    REGION_PRIORS,
    SEMANTIC_KEYWORDS,
    UNKNOWN,
    VALIDATION_PATTERNS,
)


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def semantics_enabled() -> bool:
    """True when the Phase K semantic classifier is active."""
    return _bool_env("FORM_PARSER_SEMANTICS_ENABLED", False)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


# --- Core object ------------------------------------------------------------ #
@dataclass
class SemanticField:
    semantic_type: str
    confidence: float
    preferred_widget: str | None = None
    preferred_region_types: list[str] = field(default_factory=list)
    expected_length: int | None = None
    validation_pattern: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic_type": self.semantic_type,
            "confidence": round(float(self.confidence), 4),
            "preferred_widget": self.preferred_widget,
            "preferred_region_types": list(self.preferred_region_types),
            "expected_length": self.expected_length,
            "validation_pattern": self.validation_pattern,
            "metadata": self.metadata,
        }


# --- Normalisation + matching ----------------------------------------------- #
def _normalize(text: Any) -> str:
    s = re.sub(r"\s+", " ", str(text or "").strip().lower())
    # Drop a leading enumerator like "1.", "(a)", "iii)".
    s = re.sub(r"^\s*\(?[0-9ivxabc]+[.)\-:]\s*", "", s)
    return s


def _keyword_hit(normalized: str, keyword: str) -> bool:
    if " " in keyword or "/" in keyword or "#" in keyword:
        return keyword in normalized
    return re.search(rf"\b{re.escape(keyword)}\b", normalized) is not None


def _expectations_for(semantic_type: str) -> dict[str, Any]:
    return EXPECTATIONS.get(semantic_type, EXPECTATIONS[UNKNOWN])


def _build_field(semantic_type: str, confidence: float, meta: dict[str, Any]) -> SemanticField:
    exp = _expectations_for(semantic_type)
    validation_key = exp.get("validation")
    return SemanticField(
        semantic_type=semantic_type,
        confidence=round(_clamp01(confidence), 4),
        preferred_widget=exp.get("widget"),
        preferred_region_types=list(exp.get("regions") or []),
        expected_length=exp.get("length"),
        validation_pattern=VALIDATION_PATTERNS.get(validation_key) if validation_key else None,
        metadata=meta,
    )


def classify_field(
    label: str,
    section_title: str | None = None,
    value: str | None = None,
    nearby_text: str | None = None,
    table_headers: list[str] | None = None,
) -> SemanticField:
    """Classify a single field. Rule-based: keyword gazetteer over the label
    (and, as weaker signals, table headers / section / nearby text), plus a
    value-pattern confirmation boost. Returns UNKNOWN when nothing matches."""
    normalized = _normalize(label)
    search_spaces = [("label", normalized, 1.0)]
    if table_headers:
        search_spaces.append(("table_header", _normalize(" ".join(table_headers)), 0.85))
    if nearby_text:
        search_spaces.append(("nearby_text", _normalize(nearby_text), 0.6))
    if section_title:
        search_spaces.append(("section", _normalize(section_title), 0.5))

    best: tuple[float, str, str, str, str] | None = None  # (conf, type, keyword, source, normalized)
    for semantic_type, keywords in SEMANTIC_KEYWORDS:
        for source, space, source_weight in search_spaces:
            if not space:
                continue
            for keyword in keywords:
                if not _keyword_hit(space, keyword):
                    continue
                # Base confidence + specificity boosts, scaled by signal source.
                conf = 0.60
                if space == keyword:
                    conf += 0.25
                elif space.startswith(keyword) or space.endswith(keyword):
                    conf += 0.12
                if " " in keyword:
                    conf += 0.08
                conf *= source_weight
                if best is None or conf > best[0]:
                    best = (conf, semantic_type, keyword, source, normalized)
                break  # first keyword hit for this type+source is enough

    if best is None:
        return _build_field(UNKNOWN, 0.0, {"normalized_label": normalized, "matched_keyword": None, "source": None})

    conf, semantic_type, keyword, source, _ = best
    meta: dict[str, Any] = {"normalized_label": normalized, "matched_keyword": keyword, "source": source}

    # Value-pattern confirmation: if a value is present and matches the type's
    # validation pattern, lift confidence (and confirm against mis-fires).
    value_norm = str(value or "").strip()
    validation_key = _expectations_for(semantic_type).get("validation")
    if value_norm and validation_key and validation_key in VALIDATION_PATTERNS:
        if re.match(VALIDATION_PATTERNS[validation_key], value_norm):
            conf = max(conf, 0.9)
            meta["value_confirmed"] = True

    return _build_field(semantic_type, conf, meta)


# --- Candidate region family + prior application ---------------------------- #
_ANCHOR_TO_FAMILY = {
    "underline": "UNDERLINE",
    "dotted_underline": "DOTTED",
    "broken_underline": "BROKEN",
    "inline_dotted_field": "DOTTED",
    "empty_rectangle": "RECTANGLE",
    "adjacent_whitespace": "WHITESPACE",
    "value_block": "VALUE",
    "signature_region": "SIGNATURE",
    "photo_region": "PHOTO",
    "table_cell": "TABLE_CELL",
    "checkbox_region": "CHECKBOX",
    "comb_region": "COMB",
    "radio_region": "RADIO",
    "unresolved_label_region": "UNKNOWN",
}


def candidate_region_family(candidate: dict[str, Any]) -> str:
    """Map a candidate to a region family for prior lookup. Phase I answer
    regions carry their own region_type; otherwise the anchor_type is mapped."""
    art = candidate.get("answer_region_type")
    if art:
        return str(art)
    if candidate.get("table_input_cell"):
        return "TABLE_CELL"
    return _ANCHOR_TO_FAMILY.get(str(candidate.get("anchor_type")), "UNKNOWN")


def semantic_prior(semantic_field: SemanticField | None, region_family: str) -> float:
    """Unscaled prior for a (semantic type, region family) pair."""
    if not semantic_field:
        return 0.0
    return REGION_PRIORS.get(semantic_field.semantic_type, {}).get(region_family, 0.0)


def apply_semantic_priors(
    candidates: list[dict[str, Any]],
    semantic_field: SemanticField | None,
) -> list[dict[str, Any]]:
    """Return candidates with semantic priors folded into their scores.

    The delta is scaled by the classifier's confidence so a shaky label nudges
    less than a certain one. Records ``semantic_prior`` / ``semantic_type`` on
    each candidate for diagnostics. New dicts are returned (no mutation)."""
    if not semantic_field or semantic_field.semantic_type == UNKNOWN:
        return candidates
    priors = REGION_PRIORS.get(semantic_field.semantic_type, {})
    out: list[dict[str, Any]] = []
    for cand in candidates:
        family = candidate_region_family(cand)
        delta = priors.get(family, 0.0) * float(semantic_field.confidence)
        new = dict(cand)
        new["semantic_type"] = semantic_field.semantic_type
        if delta:
            new["score"] = round(_clamp01(float(cand.get("score", 0.0)) + delta), 4)
            new["semantic_prior"] = round(delta, 4)
            direction = "pref" if delta > 0 else "penalty"
            reasons = list(new.get("reasons", []))
            reasons.append(f"semantic_{semantic_field.semantic_type.lower()}_{direction}_{family.lower()}")
            new["reasons"] = reasons
        out.append(new)
    return out


# --- Diagnostics ------------------------------------------------------------ #
def build_semantic_diagnostics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """``records`` per resolved field: {field_id, label, semantic (to_dict),
    applied_prior, selected_family}."""
    type_counts: dict[str, int] = {}
    unknown_fields: list[dict[str, Any]] = []
    overrides: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    for rec in records:
        sem = rec.get("semantic") or {}
        stype = sem.get("semantic_type", UNKNOWN)
        type_counts[stype] = type_counts.get(stype, 0) + 1
        if stype == UNKNOWN:
            unknown_fields.append({"field_id": rec.get("field_id"), "label": rec.get("label")})
        applied = float(rec.get("applied_prior") or 0.0)
        if applied:
            overrides.append({
                "field_id": rec.get("field_id"),
                "label": rec.get("label"),
                "semantic_type": stype,
                "applied_prior": round(applied, 4),
                "selected_region_family": rec.get("selected_family"),
            })
        if sem.get("validation_pattern") or sem.get("expected_length") is not None:
            validations.append({
                "field_id": rec.get("field_id"),
                "semantic_type": stype,
                "expected_length": sem.get("expected_length"),
                "validation_pattern": sem.get("validation_pattern"),
            })
    return {
        "enabled": True,
        "field_count": len(records),
        "semantic_types": type_counts,
        "unknown_fields": unknown_fields,
        "semantic_overrides": overrides,
        "validation_expectations": validations,
    }
