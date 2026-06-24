"""Metric computation: the contract from ``docs/evaluation_harness.md``.

Headline detection metrics (precision / recall / F1 / mean-IoU) plus the
failure-mode-specific metrics that map 1:1 to the production problems:

* ``duplicate_rate``               -> "duplicate mappings"
* ``fragmentation``                -> "character boxes treated as independent fields"
                                       / "checkbox matrices as widgets"
* ``non_fillable_false_positives`` -> "non-interactive pages generating false positives"
* ``repeat_group_recall``          -> "repeated sections causing ambiguity"
* ``fillability_ratio``            -> does the output have ~the right widget count

Everything is computed from already-emitted artifacts; no model is run here.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.evaluation.flags import EvalConfig
from src.evaluation.geometry import containment, iou
from src.evaluation.matching import MatchResult
from src.evaluation.schema import CANONICAL_TYPES, GroundTruthForm, PredWidget


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def _group_widgets(form: GroundTruthForm) -> list:
    grouped = {"comb", "checkbox_group", "radio_group"}
    return [w for w in form.widgets if w.wtype in grouped or w.cell_count > 1]


def compute_form_metrics(
    form: GroundTruthForm,
    preds: list[PredWidget],
    result: MatchResult,
    config: EvalConfig,
) -> dict[str, Any]:
    tp, fp, fn = result.tp, result.fp, result.fn
    overall = _prf(tp, fp, fn)
    matched_ious = [m.iou for m in result.matches]
    overall["mean_iou"] = round(sum(matched_ious) / len(matched_ious), 4) if matched_ious else 0.0
    overall["type_match_rate"] = (
        round(sum(1 for m in result.matches if m.type_match) / len(result.matches), 4) if result.matches else 0.0
    )
    overall["family_match_rate"] = (
        round(sum(1 for m in result.matches if m.family_match) / len(result.matches), 4) if result.matches else 0.0
    )

    # --- Per-type precision/recall ------------------------------------------
    per_type: dict[str, dict[str, Any]] = {}
    matched_gt_ids = {m.gt.widget_id for m in result.matches}
    matched_pred_ids = {id(m.pred) for m in result.matches}
    for t in sorted(CANONICAL_TYPES):
        gt_t = [w for w in form.widgets if w.wtype == t]
        pred_t = [p for p in preds if p.wtype == t]
        t_tp = sum(1 for w in gt_t if w.widget_id in matched_gt_ids)
        t_fn = len(gt_t) - t_tp
        t_fp = sum(1 for p in pred_t if id(p) not in matched_pred_ids)
        if not (gt_t or pred_t):
            continue
        per_type[t] = {"gt": len(gt_t), "pred": len(pred_t), **_prf(t_tp, t_fp, t_fn)}

    # --- Duplicate predictions (overlapping each other) ---------------------
    by_page: dict[int, list[PredWidget]] = defaultdict(list)
    for p in preds:
        by_page[p.page].append(p)
    duplicate_pairs = 0
    for page_preds in by_page.values():
        for i in range(len(page_preds)):
            for j in range(i + 1, len(page_preds)):
                if iou(page_preds[i].bbox, page_preds[j].bbox) >= config.duplicate_iou:
                    duplicate_pairs += 1
    duplicates = {
        "duplicate_pairs": duplicate_pairs,
        "duplicate_rate": round(duplicate_pairs / len(preds), 4) if preds else 0.0,
        "threshold_iou": config.duplicate_iou,
    }

    # --- Fragmentation of grouped widgets -----------------------------------
    groups = _group_widgets(form)
    fragmented = 0
    fragment_counts: list[int] = []
    for g in groups:
        inside = [p for p in preds if p.page == g.page and containment(p.bbox, g.bbox) >= 0.5]
        fragment_counts.append(len(inside))
        if len(inside) > 1:
            fragmented += 1
    fragmentation = {
        "group_count": len(groups),
        "fragmented_groups": fragmented,
        "fragmentation_rate": round(fragmented / len(groups), 4) if groups else 0.0,
        "avg_fragments_per_group": round(sum(fragment_counts) / len(fragment_counts), 3) if fragment_counts else 0.0,
    }

    # --- Non-fillable-page false positives ----------------------------------
    fillable = form.fillable_pages()
    non_fillable_preds = [p for p in preds if p.page not in fillable]
    non_fillable_fp = {
        "predicted_widgets_on_non_fillable_pages": len(non_fillable_preds),
        "non_fillable_pages": sorted(set(range(1, form.page_count + 1)) - fillable),
    }

    # --- Repeated-section recall --------------------------------------------
    repeat_gts = [w for w in form.widgets if w.repeat_group_id]
    repeat_tp = sum(1 for w in repeat_gts if w.widget_id in matched_gt_ids)
    repeat_group = {
        "repeat_group_widgets": len(repeat_gts),
        "matched": repeat_tp,
        "repeat_group_recall": round(repeat_tp / len(repeat_gts), 4) if repeat_gts else None,
    }

    # --- Fillability (count fidelity) ---------------------------------------
    gt_n = len(form.widgets)
    fillability = {
        "gt_widget_count": gt_n,
        "pred_widget_count": len(preds),
        "fillability_ratio": round(len(preds) / gt_n, 4) if gt_n else None,
    }

    return {
        "form_id": form.form_id,
        "doc_class": form.doc_class,
        "failure_modes": list(form.failure_modes),
        "counts": {"tp": tp, "fp": fp, "fn": fn, "gt": gt_n, "pred": len(preds)},
        "overall": overall,
        "per_type": per_type,
        "duplicates": duplicates,
        "fragmentation": fragmentation,
        "non_fillable_false_positives": non_fillable_fp,
        "repeat_group": repeat_group,
        "fillability": fillability,
    }


def aggregate_metrics(form_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Corpus-wide rollup. Micro-averaged P/R/F1 (sum counts, then divide)."""
    if not form_metrics:
        return {"form_count": 0}

    tp = sum(m["counts"]["tp"] for m in form_metrics)
    fp = sum(m["counts"]["fp"] for m in form_metrics)
    fn = sum(m["counts"]["fn"] for m in form_metrics)
    micro = _prf(tp, fp, fn)

    ious = [m["overall"]["mean_iou"] for m in form_metrics if m["counts"]["tp"]]
    dup_pairs = sum(m["duplicates"]["duplicate_pairs"] for m in form_metrics)
    pred_total = sum(m["counts"]["pred"] for m in form_metrics)
    frag_groups = sum(m["fragmentation"]["group_count"] for m in form_metrics)
    frag_bad = sum(m["fragmentation"]["fragmented_groups"] for m in form_metrics)
    non_fillable = sum(m["non_fillable_false_positives"]["predicted_widgets_on_non_fillable_pages"] for m in form_metrics)

    # Macro-average F1 (each form weighted equally) complements the micro view.
    macro_f1 = round(sum(m["overall"]["f1"] for m in form_metrics) / len(form_metrics), 4)

    return {
        "form_count": len(form_metrics),
        "counts": {"tp": tp, "fp": fp, "fn": fn, "gt": tp + fn, "pred": pred_total},
        "micro": {**micro, "mean_iou": round(sum(ious) / len(ious), 4) if ious else 0.0},
        "macro_f1": macro_f1,
        "duplicate_rate": round(dup_pairs / pred_total, 4) if pred_total else 0.0,
        "fragmentation_rate": round(frag_bad / frag_groups, 4) if frag_groups else 0.0,
        "non_fillable_false_positives": non_fillable,
    }
