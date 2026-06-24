"""Ground-truth & prediction schemas + loaders.

Two record kinds flow through the harness:

* :class:`GTWidget`   — a hand-annotated *logical* widget (the target).
* :class:`PredWidget` — a widget read from the pipeline's ``mappings.json``.

Both are normalised onto a shared canonical ``WidgetType`` so they can be
matched and scored. Annotators write ground truth as JSON (see
``benchmarks/corpus/README.md`` and the bundled example); the pipeline already
writes predictions. Nothing here imports live-pipeline code.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.evaluation.geometry import Box, normalize_box

SCHEMA_VERSION = "1.0"

# --- Canonical widget taxonomy -------------------------------------------------
# Logical widget types the harness scores. Ground truth is authored in these
# terms; predictions (pipeline ``field_type``) are normalised onto them.
TEXT_FAMILY = {"text", "multiline", "date", "comb"}
CHOICE_FAMILY = {"checkbox", "checkbox_group", "radio_group"}
OTHER_FAMILY = {"signature", "table_cell"}
CANONICAL_TYPES = TEXT_FAMILY | CHOICE_FAMILY | OTHER_FAMILY

# Pipeline ``field_type`` -> canonical type. Unknown strings fall back to "text"
# (recorded raw on the record so confusion analysis is still possible). "photo"
# is intentionally excluded upstream: it is a layout placeholder, not a widget.
PRED_TYPE_MAP = {
    "text": "text",
    "multiline": "multiline",
    "date": "date",
    "comb": "comb",
    "checkbox": "checkbox",
    "checkbox_group": "checkbox_group",
    "radio": "radio_group",
    "radio_group": "radio_group",
    "signature": "signature",
    "table_cell": "table_cell",
}
EXCLUDED_PRED_TYPES = {"photo"}


def family_of(canonical_type: str) -> str:
    if canonical_type in TEXT_FAMILY:
        return "text"
    if canonical_type in CHOICE_FAMILY:
        return "choice"
    return canonical_type


@dataclass(frozen=True)
class GTWidget:
    widget_id: str
    page: int
    wtype: str  # canonical
    bbox: Box
    label: str = ""
    section_path: tuple[str, ...] = ()
    cell_count: int = 1
    repeat_group_id: str | None = None
    fillable: bool = True


@dataclass(frozen=True)
class PredWidget:
    field_id: str
    page: int
    wtype: str  # canonical
    raw_type: str  # the original field_type string
    bbox: Box
    label: str = ""


@dataclass(frozen=True)
class GroundTruthForm:
    form_id: str
    doc_class: str
    page_count: int
    widgets: tuple[GTWidget, ...]
    page_types: dict[int, str]  # page -> "fillable" | "instructions" | "cover" | ...
    source: str = ""
    failure_modes: tuple[str, ...] = ()

    def fillable_pages(self) -> set[int]:
        # A page is fillable unless explicitly annotated as a non-fillable type.
        non_fillable = {"instructions", "cover", "terms", "signature_only", "blank"}
        return {p for p in range(1, self.page_count + 1) if self.page_types.get(p, "fillable") not in non_fillable}


class SchemaError(ValueError):
    """Raised when a ground-truth document is structurally invalid."""


# --- Loaders -------------------------------------------------------------------
def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise SchemaError(msg)


def load_ground_truth(path: str | Path) -> GroundTruthForm:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(data, dict), f"{path}: root must be an object")

    form_id = str(data.get("form_id") or path.parent.name)
    page_count = int(data.get("page_count") or 1)
    _require(page_count >= 1, f"{form_id}: page_count must be >= 1")

    page_types_raw = data.get("page_types") or {}
    page_types = {int(k): str(v) for k, v in page_types_raw.items()}

    widgets: list[GTWidget] = []
    seen_ids: set[str] = set()
    for i, w in enumerate(data.get("widgets") or []):
        _require(isinstance(w, dict), f"{form_id}: widget #{i} must be an object")
        box = normalize_box(w.get("bbox"))
        _require(box is not None, f"{form_id}: widget #{i} has an invalid bbox")
        wtype = str(w.get("type") or "text").strip().lower()
        _require(
            wtype in CANONICAL_TYPES,
            f"{form_id}: widget #{i} type {wtype!r} not in {sorted(CANONICAL_TYPES)}",
        )
        wid = str(w.get("widget_id") or f"gt{i + 1}")
        _require(wid not in seen_ids, f"{form_id}: duplicate widget_id {wid!r}")
        seen_ids.add(wid)
        page = int(w.get("page") or 1)
        _require(1 <= page <= page_count, f"{form_id}: widget {wid} page {page} out of range 1..{page_count}")
        _require(0.0 <= box["x"] <= 1.0 and 0.0 <= box["y"] <= 1.0, f"{form_id}: widget {wid} bbox not in 0..1 fractions")
        section = w.get("section_path") or []
        widgets.append(
            GTWidget(
                widget_id=wid,
                page=page,
                wtype=wtype,
                bbox=box,
                label=str(w.get("label") or ""),
                section_path=tuple(str(s) for s in section),
                cell_count=int(w.get("cell_count") or 1),
                repeat_group_id=(str(w["repeat_group_id"]) if w.get("repeat_group_id") else None),
                fillable=bool(w.get("fillable", True)),
            )
        )

    return GroundTruthForm(
        form_id=form_id,
        doc_class=str(data.get("doc_class") or "unknown"),
        page_count=page_count,
        widgets=tuple(widgets),
        page_types=page_types,
        source=str(data.get("source") or ""),
        failure_modes=tuple(str(m) for m in (data.get("failure_modes") or [])),
    )


def load_predictions(mappings_path: str | Path) -> list[PredWidget]:
    """Load predicted widgets from a pipeline ``mappings.json`` artifact.

    Excludes ``photo`` placeholders (not form widgets). Mappings with an
    unparseable bbox are skipped (recorded by the caller via the count delta).
    """
    mappings_path = Path(mappings_path)
    data = json.loads(mappings_path.read_text(encoding="utf-8"))
    _require(isinstance(data, list), f"{mappings_path}: mappings.json must be a JSON array")

    out: list[PredWidget] = []
    for i, m in enumerate(data):
        if not isinstance(m, dict):
            continue
        raw_type = str(m.get("field_type") or "text").strip().lower()
        if raw_type in EXCLUDED_PRED_TYPES:
            continue
        box = normalize_box(m.get("bbox"))
        if box is None:
            continue
        out.append(
            PredWidget(
                field_id=str(m.get("field_id") or f"pred{i + 1}"),
                page=int(m.get("page") or 1),
                wtype=PRED_TYPE_MAP.get(raw_type, "text"),
                raw_type=raw_type,
                bbox=box,
                label=str(m.get("label") or ""),
            )
        )
    return out
