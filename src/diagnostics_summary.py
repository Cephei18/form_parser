"""Task 2 — curated form-diagnostics summary for ``benchmark_summary.json``.

A single, flat block that explains *why* a form is hard: how many suspicious
checkboxes, how fragmented, how saturated the penalties are, how the matrix /
table classifier saw the page, etc. It is a pure, read-only projection of the
diagnostics the pipeline already produced — additive, no behaviour change, and
tolerant of every flag being OFF (missing sub-diagnostics collapse to defaults).
"""
from __future__ import annotations

from collections import Counter
from typing import Any


def _get(d: Any, *path: str, default: Any = None) -> Any:
    cur = d
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return cur if cur is not None else default


def build_form_diagnostics_summary(
    diagnostics: dict[str, Any],
    mappings: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Project the assembled pipeline diagnostics into a flat summary block."""
    mappings = mappings or []
    anchoring = diagnostics.get("anchoring") if isinstance(diagnostics.get("anchoring"), dict) else {}
    calibration = diagnostics.get("confidence_calibration") if isinstance(diagnostics.get("confidence_calibration"), dict) else {}

    total_widgets = len(mappings)

    # Per-page checkbox counts (the page-3 explosion signal).
    page_checkbox_counts = Counter(
        int(m.get("page") or 1) for m in mappings if isinstance(m, dict) and m.get("field_type") == "checkbox"
    )
    checkbox_total = sum(page_checkbox_counts.values())

    # Suspicious checkboxes (observe or enforce; whichever ran).
    cbv = anchoring.get("checkbox_validation") if isinstance(anchoring.get("checkbox_validation"), dict) else {}
    suspicious = int(cbv.get("would_reject_count") or cbv.get("rejected_count") or 0)

    duplicate_mappings = int(calibration.get("duplicate_mappings") or 0)
    ambiguity_count = int(calibration.get("ambiguous_mappings") or 0)

    comb = anchoring.get("comb_diagnostics") if isinstance(anchoring.get("comb_diagnostics"), dict) else {}
    table_intel = anchoring.get("table_intelligence") if isinstance(anchoring.get("table_intelligence"), dict) else {}
    dotted = anchoring.get("dotted_leader_source") if isinstance(anchoring.get("dotted_leader_source"), dict) else {}

    confidence_distribution = (
        _get(calibration, "penalty_saturation", "confidence_distribution")
        or _get(diagnostics, "confidence", "fields_by_confidence")
        or {}
    )

    return {
        "total_widgets": total_widgets,
        "checkbox_total": checkbox_total,
        "checkbox_share": round(checkbox_total / total_widgets, 4) if total_widgets else 0.0,
        "suspicious_checkboxes": suspicious,
        "duplicate_rate": round(duplicate_mappings / total_widgets, 4) if total_widgets else 0.0,
        "ambiguity_count": ambiguity_count,
        "comb_group_count": int(comb.get("likely_comb_group_count") or 0),
        "comb_cell_count": int(comb.get("total_cells_in_groups") or 0),
        "table_types": table_intel.get("table_types") or {},
        "page_checkbox_counts": {str(p): c for p, c in sorted(page_checkbox_counts.items())},
        "confidence_distribution": confidence_distribution,
        "penalty_saturation": calibration.get("penalty_saturation") or {"enabled": False},
        "dotted_leader_recall": dotted.get("recall") or {"enabled": dotted.get("enabled", False)},
        "table_intelligence_summary": {
            "enabled": bool(table_intel.get("enabled")),
            "table_count": int(table_intel.get("table_count") or 0),
            "table_types": table_intel.get("table_types") or {},
            "input_cell_count": (
                len(table_intel["input_cells"]) if isinstance(table_intel.get("input_cells"), list) else int(table_intel.get("input_cell_count") or 0)
            ),
        },
        "flags_active": {
            "checkbox_validation": cbv.get("mode") if cbv.get("enabled") else False,
            "comb_diagnostics": bool(comb.get("enabled")),
            "dotted_leader_diagnostics": bool(dotted.get("enabled")),
            "table_intelligence": bool(table_intel.get("enabled")),
            "confidence_calibration": bool(calibration.get("enabled")),
        },
    }
