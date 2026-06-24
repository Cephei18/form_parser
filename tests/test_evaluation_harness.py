"""Tests for the Phase 0 evaluation harness (src/evaluation).

Covers geometry, schema loading/validation, greedy matching, metric math, and an
end-to-end run over the bundled synthetic corpus form with exact expected
numbers. No AWS, no model — pure artifact scoring.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation.flags import EvalConfig
from src.evaluation.geometry import containment, iou
from src.evaluation.matching import match_widgets
from src.evaluation.metrics import compute_form_metrics
from src.evaluation.runner import evaluate_corpus
from src.evaluation.schema import SchemaError, load_ground_truth, load_predictions

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / "benchmarks" / "corpus"
EXAMPLE = CORPUS / "_example_synthetic"


# --- geometry -----------------------------------------------------------------
def test_iou_identical_is_one():
    b = {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2}
    assert iou(b, b) == pytest.approx(1.0)


def test_iou_disjoint_is_zero():
    a = {"x": 0.0, "y": 0.0, "width": 0.1, "height": 0.1}
    b = {"x": 0.5, "y": 0.5, "width": 0.1, "height": 0.1}
    assert iou(a, b) == 0.0


def test_containment_fully_inside():
    inner = {"x": 0.2, "y": 0.2, "width": 0.05, "height": 0.05}
    outer = {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}
    assert containment(inner, outer) == pytest.approx(1.0)


# --- schema -------------------------------------------------------------------
def test_load_example_ground_truth():
    form = load_ground_truth(EXAMPLE / "ground_truth.json")
    assert form.page_count == 2
    assert len(form.widgets) == 4
    assert form.fillable_pages() == {1}  # page 2 is "instructions"


def test_load_predictions_excludes_photo():
    preds = load_predictions(EXAMPLE / "predictions" / "mappings.json")
    # 7 records in the file, but the photo placeholder is excluded.
    assert len(preds) == 6
    assert all(p.wtype != "photo" for p in preds)


def test_invalid_type_raises(tmp_path):
    bad = tmp_path / "ground_truth.json"
    bad.write_text(
        json.dumps(
            {"form_id": "x", "page_count": 1, "widgets": [
                {"widget_id": "w1", "page": 1, "type": "not_a_type",
                 "bbox": {"x": 0, "y": 0, "width": 0.1, "height": 0.1}}
            ]}
        ),
        encoding="utf-8",
    )
    with pytest.raises(SchemaError):
        load_ground_truth(bad)


def test_out_of_range_page_raises(tmp_path):
    bad = tmp_path / "ground_truth.json"
    bad.write_text(
        json.dumps(
            {"form_id": "x", "page_count": 1, "widgets": [
                {"widget_id": "w1", "page": 3, "type": "text",
                 "bbox": {"x": 0, "y": 0, "width": 0.1, "height": 0.1}}
            ]}
        ),
        encoding="utf-8",
    )
    with pytest.raises(SchemaError):
        load_ground_truth(bad)


# --- matching + metrics on the synthetic example ------------------------------
def _example_metrics():
    form = load_ground_truth(EXAMPLE / "ground_truth.json")
    preds = load_predictions(EXAMPLE / "predictions" / "mappings.json")
    config = EvalConfig()
    result = match_widgets(list(form.widgets), preds, config)
    return form, preds, result, compute_form_metrics(form, preds, result, config)


def test_match_counts():
    _, _, result, _ = _example_metrics()
    assert (result.tp, result.fp, result.fn) == (2, 4, 2)


def test_overall_prf():
    *_, metrics = _example_metrics()
    o = metrics["overall"]
    assert o["precision"] == pytest.approx(0.3333, abs=1e-3)
    assert o["recall"] == pytest.approx(0.5, abs=1e-3)
    assert o["f1"] == pytest.approx(0.4, abs=1e-3)
    assert o["mean_iou"] == pytest.approx(0.847, abs=1e-2)


def test_duplicate_detection():
    *_, metrics = _example_metrics()
    # The two near-identical checkbox predictions form one duplicate pair.
    assert metrics["duplicates"]["duplicate_pairs"] == 1


def test_comb_fragmentation():
    *_, metrics = _example_metrics()
    frag = metrics["fragmentation"]
    assert frag["group_count"] == 1            # the SSN comb
    assert frag["fragmented_groups"] == 1      # split into 2 text preds
    assert frag["avg_fragments_per_group"] == pytest.approx(2.0)


def test_non_fillable_false_positive():
    *_, metrics = _example_metrics()
    nf = metrics["non_fillable_false_positives"]
    assert nf["predicted_widgets_on_non_fillable_pages"] == 1
    assert nf["non_fillable_pages"] == [2]


def test_per_type_breakdown():
    *_, metrics = _example_metrics()
    assert metrics["per_type"]["checkbox"]["recall"] == pytest.approx(1.0)
    assert metrics["per_type"]["comb"]["recall"] == pytest.approx(0.0)


# --- end to end ---------------------------------------------------------------
def test_evaluate_corpus_writes_artifacts(tmp_path):
    summary = evaluate_corpus(CORPUS, tmp_path)
    # benchmark_summary.json written at the run root.
    assert (tmp_path / "benchmark_summary.json").is_file()
    # The synthetic form was evaluated and diff images were produced.
    assert summary["corpus"]["evaluated"] >= 1
    diffs = list((tmp_path / "_example_synthetic").glob("diff_page_*.png"))
    assert len(diffs) == 2  # two pages
    assert (tmp_path / "_example_synthetic" / "evaluation.json").is_file()
    # Flag snapshot is recorded for attribution.
    assert "FORM_PARSER_GLOBAL_ASSIGNMENT_ENABLED" in summary["pipeline_flags"]


def test_fail_under_gate_via_cli(tmp_path):
    # Import the CLI main and exercise the regression-gate exit code.
    import scripts.evaluate as cli

    rc = cli.main(["--corpus", str(CORPUS), "--out", str(tmp_path), "--fail-under-f1", "0.99"])
    assert rc == 1  # synthetic F1 (0.4) is below 0.99
