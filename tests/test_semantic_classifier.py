"""Phase K — semantic classifier validation.

Classifier-level tests drive ``classify_field`` and ``apply_semantic_priors``
deterministically (no rasters, no AWS) for every required scenario: PAN,
Aadhaar, Address, Email, Phone, Gender, Date, Amount, table headers, unknown
labels, confidence integration and assignment integration. Integration tests
confirm the ``FORM_PARSER_SEMANTICS_ENABLED`` flag is fully reversible and emit
validation artifacts.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import logging

logging.disable(logging.CRITICAL)

from src import field_anchor_engine as fae
from src.semantic_classifier import (
    apply_semantic_priors,
    build_semantic_diagnostics,
    classify_field,
)
from src.semantic_gazetteers import (
    AADHAAR,
    ADDRESS,
    AMOUNT,
    DATE,
    EMAIL,
    GENDER,
    PAN,
    PHONE,
    UNKNOWN,
)

FLAG = "FORM_PARSER_SEMANTICS_ENABLED"


# --------------------------------------------------------------------------- #
# classify_field
# --------------------------------------------------------------------------- #
def test_pan():
    sf = classify_field("PAN")
    assert sf.semantic_type == PAN
    assert sf.preferred_widget == "comb"
    assert sf.expected_length == 10
    assert "COMB" in sf.preferred_region_types
    assert sf.validation_pattern  # PAN regex present


def test_pan_full_phrase():
    assert classify_field("Permanent Account Number").semantic_type == PAN


def test_aadhaar():
    sf = classify_field("Aadhaar Number")
    assert sf.semantic_type == AADHAAR
    assert sf.expected_length == 12


def test_address():
    sf = classify_field("Permanent Address")
    assert sf.semantic_type == ADDRESS
    assert sf.preferred_widget == "multiline"
    assert "MULTILINE" in sf.preferred_region_types


def test_email():
    sf = classify_field("Email ID")
    assert sf.semantic_type == EMAIL
    assert sf.validation_pattern


def test_phone():
    # "Mobile Number" must resolve to PHONE, not ACCOUNT_NUMBER / NUMBER.
    assert classify_field("Mobile Number").semantic_type == PHONE


def test_gender():
    sf = classify_field("Gender")
    assert sf.semantic_type == GENDER
    assert sf.preferred_widget == "radio"


def test_date():
    assert classify_field("Date of Birth").semantic_type == DATE
    assert classify_field("Date").semantic_type == DATE


def test_amount():
    assert classify_field("Amount").semantic_type == AMOUNT


def test_account_number_not_plain_number():
    from src.semantic_gazetteers import ACCOUNT_NUMBER
    assert classify_field("Account Number").semantic_type == ACCOUNT_NUMBER


def test_table_headers_context():
    # A blank/empty label classified via its column header.
    sf = classify_field("", table_headers=["Mobile"])
    assert sf.semantic_type == PHONE
    sf2 = classify_field("", table_headers=["Amount"])
    assert sf2.semantic_type == AMOUNT


def test_unknown_label():
    sf = classify_field("Xyzzy Plugh")
    assert sf.semantic_type == UNKNOWN
    assert sf.confidence == 0.0
    assert sf.preferred_region_types == []


def test_value_confirmation_boosts_confidence():
    weak = classify_field("Permanent Account Number")
    strong = classify_field("Permanent Account Number", value="ABCDE1234F")
    assert strong.confidence >= weak.confidence
    assert strong.metadata.get("value_confirmed") is True


# --------------------------------------------------------------------------- #
# apply_semantic_priors — assignment + confidence integration
# --------------------------------------------------------------------------- #
def _cand(anchor, score, answer_region_type=None):
    c = {"anchor_type": anchor, "score": score, "reasons": []}
    if answer_region_type:
        c["answer_region_type"] = answer_region_type
    return c


def test_assignment_address_prefers_multiline_over_comb():
    sf = classify_field("Permanent Address")
    comb = _cand("empty_rectangle", 0.85, answer_region_type="COMB")
    multiline = _cand("underline", 0.60, answer_region_type="MULTILINE")
    adjusted = apply_semantic_priors([comb, multiline], sf)
    winner = max(adjusted, key=lambda c: c["score"])
    assert winner["answer_region_type"] == "MULTILINE"


def test_assignment_date_prefers_comb_over_freeform():
    sf = classify_field("Date of Birth")
    comb = _cand("empty_rectangle", 0.60, answer_region_type="COMB")
    freeform = _cand("adjacent_whitespace", 0.80, answer_region_type="FREEFORM")
    adjusted = apply_semantic_priors([comb, freeform], sf)
    winner = max(adjusted, key=lambda c: c["score"])
    assert winner["answer_region_type"] == "COMB"


def test_confidence_address_in_comb_is_penalised():
    sf = classify_field("Permanent Address")
    comb = _cand("empty_rectangle", 0.80, answer_region_type="COMB")
    adjusted = apply_semantic_priors([comb], sf)
    assert adjusted[0]["score"] < 0.80
    assert adjusted[0]["semantic_prior"] < 0


def test_priors_skipped_for_unknown():
    sf = classify_field("Xyzzy")
    comb = _cand("empty_rectangle", 0.80, answer_region_type="COMB")
    adjusted = apply_semantic_priors([comb], sf)
    assert adjusted[0]["score"] == 0.80  # unchanged


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def test_diagnostics_shape():
    records = [
        {"field_id": "f1", "label": "PAN", "semantic": classify_field("PAN").to_dict(),
         "applied_prior": 0.36, "selected_family": "COMB"},
        {"field_id": "f2", "label": "Xyzzy", "semantic": classify_field("Xyzzy").to_dict(),
         "applied_prior": 0.0, "selected_family": "WHITESPACE"},
    ]
    diag = build_semantic_diagnostics(records)
    for key in ("enabled", "semantic_types", "unknown_fields", "semantic_overrides", "validation_expectations"):
        assert key in diag
    assert diag["semantic_types"].get(PAN) == 1
    assert len(diag["unknown_fields"]) == 1
    assert len(diag["semantic_overrides"]) == 1  # only the PAN field had an applied prior
    assert any(v["semantic_type"] == PAN for v in diag["validation_expectations"])


# --------------------------------------------------------------------------- #
# Integration through build_anchored_mappings
# --------------------------------------------------------------------------- #
def _blank_png(tmp_path) -> Path:
    img = np.full((1000, 800, 3), 255, dtype=np.uint8)
    path = tmp_path / "page1.png"
    cv2.imwrite(str(path), img)
    return path


def _word(bid, text, x, y, w=0.12, h=0.02, page=1):
    return {"Id": bid, "BlockType": "WORD", "Text": text, "Page": page,
            "Geometry": {"BoundingBox": {"Left": x, "Top": y, "Width": w, "Height": h}}}


def _raw(words):
    page = {"Id": "PAGE1", "BlockType": "PAGE", "Page": 1,
            "Geometry": {"BoundingBox": {"Left": 0, "Top": 0, "Width": 1, "Height": 1}},
            "Relationships": [{"Type": "CHILD", "Ids": [w["Id"] for w in words]}]}
    return {"Blocks": [page, *words]}


def _run(raw, parsed, image_path, enabled):
    if enabled:
        os.environ[FLAG] = "true"
    else:
        os.environ.pop(FLAG, None)
    try:
        return fae.build_anchored_mappings(raw, parsed, image_path)
    finally:
        os.environ.pop(FLAG, None)


def _signature(result):
    return sorted(
        (m.get("label"), m.get("anchoring", {}).get("anchor_type"), tuple(sorted(m["bbox"].items())))
        for m in result["mappings"]
    )


def _parsed():
    return {
        "field_items": [{"key": "PAN", "key_block_id": "K1", "value_block_ids": ["V1"], "value": ""}],
        "tables": [], "checkboxes": [],
    }


def test_integration_flag_off_is_unchanged(tmp_path):
    image_path = _blank_png(tmp_path)
    raw = _raw([_word("K1", "PAN", 0.10, 0.20), _word("V1", "", 0.55, 0.20, w=0.25)])
    parsed = _parsed()

    off = _run(raw, parsed, image_path, enabled=False)
    on = _run(raw, parsed, image_path, enabled=True)

    assert off["diagnostics"]["semantic_classifier"] == {"enabled": False}
    assert "semantic" not in off["mappings"][0]
    # ON classifies and enriches metadata.
    on_diag = on["diagnostics"]["semantic_classifier"]
    assert on_diag["enabled"] is True
    assert on_diag["semantic_types"].get(PAN) == 1
    assert on["mappings"][0]["semantic"]["semantic_type"] == PAN
    # The only candidate here is the value block (no comb/region competition),
    # so the rendered widget geometry is unchanged by semantics.
    assert _signature(off) == _signature(on)


def test_emit_validation_artifacts(tmp_path):
    image_path = _blank_png(tmp_path)
    raw = _raw([_word("K1", "PAN", 0.10, 0.20), _word("V1", "", 0.55, 0.20, w=0.25)])
    parsed = _parsed()

    off = _run(raw, parsed, image_path, enabled=False)
    on = _run(raw, parsed, image_path, enabled=True)
    diag = on["diagnostics"]["semantic_classifier"]

    out_dir = REPO_ROOT / "output" / "phase_k_semantics_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _dump(name, payload):
        (out_dir / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _dump("before_semantics.json", {"mappings": off["mappings"]})
    _dump("after_semantics.json", {"mappings": on["mappings"]})
    _dump("semantic_debug.json", diag)
    _dump("validation_metrics.json", {
        "field_count": diag["field_count"],
        "semantic_types": diag["semantic_types"],
        "unknown_field_count": len(diag["unknown_fields"]),
        "semantic_override_count": len(diag["semantic_overrides"]),
        "validation_expectation_count": len(diag["validation_expectations"]),
    })

    for name in ("before_semantics.json", "after_semantics.json", "semantic_debug.json", "validation_metrics.json"):
        assert (out_dir / name).exists()
    assert diag["semantic_types"].get(PAN) == 1
