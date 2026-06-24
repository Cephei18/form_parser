"""Phase 0 — offline evaluation harness for the form parser.

This package is **fully isolated** from the live parsing/rendering pipeline:

* It imports nothing from ``src`` runtime modules (``textract_pipeline``,
  ``field_anchor_engine``, ``lambda_worker``, ...).
* No live-path code imports this package.
* It is never invoked inside the Lambda worker or the FastAPI request path.

Its only job is to compare a *prediction* (the ``mappings.json`` artifact the
pipeline already emits) against a hand-annotated *ground truth*, and produce
metrics + visual diffs. Because it only reads existing artifacts, running it can
never change production behaviour — it is additive and production-safe by
construction. An additional kill-switch (``FORM_PARSER_EVAL_HARNESS_ENABLED``)
gates any *future* inline integration; see :mod:`src.evaluation.flags`.

Public surface:

    from src.evaluation import evaluate_corpus, evaluate_form, EvalConfig
"""
from __future__ import annotations

from src.evaluation.flags import EvalConfig, eval_harness_enabled
from src.evaluation.runner import evaluate_corpus, evaluate_form

__all__ = [
    "EvalConfig",
    "eval_harness_enabled",
    "evaluate_corpus",
    "evaluate_form",
]

SCHEMA_VERSION = "1.0"
