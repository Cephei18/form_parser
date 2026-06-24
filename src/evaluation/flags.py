"""Feature flags + tunable config for the evaluation harness.

The harness is an offline developer/CI tool, so the flag below does **not** gate
the CLI (you always want to be able to run an evaluation locally). It exists to
gate any *inline* integration we may add later — e.g. emitting a
ground-truth comparison alongside a live pipeline run when a GT file happens to
be present. That hook does not exist yet; the flag defaults OFF so that when it
is added it stays dormant until explicitly enabled, exactly like every other
``FORM_PARSER_*`` phase flag in this codebase.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

_TRUE = {"1", "true", "yes", "on"}


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUE


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def eval_harness_enabled() -> bool:
    """Master switch for any *inline* (live-pipeline) evaluation hook.

    Default OFF. The offline CLI ignores this; it only governs future inline
    integration so the live path stays untouched until opted in.
    """
    return _bool_env("FORM_PARSER_EVAL_HARNESS_ENABLED", False)


# Canonical fraction-box convention shared with the pipeline mappings:
#   {"x", "y", "width", "height"} in page-local fractions, top-left origin.
BOX_KEYS = ("x", "y", "width", "height")


@dataclass(frozen=True)
class EvalConfig:
    """Tunable thresholds for a single evaluation run.

    All values are overridable from the CLI; the env fallbacks let CI pin them
    without flags. Defaults mirror the metrics contract in
    ``docs/evaluation_harness.md``.
    """

    # IoU at/above which a predicted widget may match a ground-truth widget.
    iou_threshold: float = field(default_factory=lambda: _float_env("FORM_PARSER_EVAL_IOU", 0.5))
    # IoU at/above which two predictions are considered a duplicate of each other.
    duplicate_iou: float = field(default_factory=lambda: _float_env("FORM_PARSER_EVAL_DUP_IOU", 0.7))
    # Minimum overlap for a prediction to count as "touching" a GT group (fragmentation).
    fragment_iou: float = field(default_factory=lambda: _float_env("FORM_PARSER_EVAL_FRAG_IOU", 0.1))
    # When True, a match also requires the canonical widget types to agree.
    strict_type: bool = field(default_factory=lambda: _bool_env("FORM_PARSER_EVAL_STRICT_TYPE", False))
    # Diff-image canvas size used only when a form ships no page raster.
    canvas_width: int = 1240
    canvas_height: int = 1754

    def to_dict(self) -> dict[str, Any]:
        return {
            "iou_threshold": self.iou_threshold,
            "duplicate_iou": self.duplicate_iou,
            "fragment_iou": self.fragment_iou,
            "strict_type": self.strict_type,
            "canvas_width": self.canvas_width,
            "canvas_height": self.canvas_height,
        }
