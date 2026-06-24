"""Greedy IoU matching of predictions to ground truth.

A deterministic, dependency-free matcher (no scipy needed): build every
candidate (pred, gt) pair on the same page whose IoU clears the threshold,
sort by IoU descending, and greedily lock in pairs, each prediction and each
GT widget used at most once. Greedy-by-IoU is the standard detection-eval
matcher and is stable for our box counts.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.evaluation.flags import EvalConfig
from src.evaluation.geometry import iou
from src.evaluation.schema import GTWidget, PredWidget, family_of


@dataclass(frozen=True)
class Match:
    gt: GTWidget
    pred: PredWidget
    iou: float
    type_match: bool
    family_match: bool


@dataclass(frozen=True)
class MatchResult:
    matches: list[Match]
    false_positives: list[PredWidget]  # predictions with no GT (over-detection)
    false_negatives: list[GTWidget]  # GT with no prediction (missed widgets)

    @property
    def tp(self) -> int:
        return len(self.matches)

    @property
    def fp(self) -> int:
        return len(self.false_positives)

    @property
    def fn(self) -> int:
        return len(self.false_negatives)


def match_widgets(
    gts: list[GTWidget],
    preds: list[PredWidget],
    config: EvalConfig,
) -> MatchResult:
    """Match a single form's predictions against its ground truth."""
    pairs: list[tuple[float, int, int]] = []  # (iou, gt_idx, pred_idx)
    for gi, gt in enumerate(gts):
        for pi, pred in enumerate(preds):
            if pred.page != gt.page:
                continue
            score = iou(gt.bbox, pred.bbox)
            if score < config.iou_threshold:
                continue
            if config.strict_type and pred.wtype != gt.wtype:
                continue
            pairs.append((score, gi, pi))

    # Greedy: highest IoU first. Ties broken by indices for determinism.
    pairs.sort(key=lambda t: (-t[0], t[1], t[2]))

    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[Match] = []
    for score, gi, pi in pairs:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        gt, pred = gts[gi], preds[pi]
        matches.append(
            Match(
                gt=gt,
                pred=pred,
                iou=score,
                type_match=(gt.wtype == pred.wtype),
                family_match=(family_of(gt.wtype) == family_of(pred.wtype)),
            )
        )

    fps = [p for i, p in enumerate(preds) if i not in used_pred]
    fns = [g for i, g in enumerate(gts) if i not in used_gt]
    return MatchResult(matches=matches, false_positives=fps, false_negatives=fns)
