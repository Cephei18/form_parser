from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable


SUPPORTED_WIDGET_TYPES = frozenset(
    {
        "text",
        "multiline",
        "checkbox",
        "radio",
        "comb",
        "signature",
    }
)


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def widget_registry_enabled() -> bool:
    """Default-off gate for the optional widget renderer registry."""
    return _bool_env("FORM_PARSER_WIDGET_REGISTRY_ENABLED", False)


def normalize_widget_type(value: Any) -> str | None:
    widget_type = str(value or "").strip().lower()
    if not widget_type:
        return None
    if widget_type not in SUPPORTED_WIDGET_TYPES:
        return None
    return widget_type


def declared_widget_type(mapping: dict[str, Any] | None) -> str | None:
    if not isinstance(mapping, dict):
        return None
    return normalize_widget_type(mapping.get("widget_type"))


def field_type_for_widget_type(widget_type: str) -> str:
    if widget_type in {"multiline"}:
        return "multiline"
    if widget_type in {"checkbox", "radio"}:
        return "checkbox"
    return "text"


@dataclass(frozen=True)
class WidgetRenderContext:
    field_name: str
    page_number: int
    mapping_index: int
    box_index: int
    pdf_x: float
    pdf_y: float
    width: float
    height: float
    page_width: float
    page_height: float


WidgetRenderer = Callable[[Any, dict[str, Any], dict[str, Any], WidgetRenderContext], None]


class WidgetRendererRegistry:
    def __init__(self) -> None:
        self._renderers: dict[str, WidgetRenderer] = {}

    def register(self, widget_type: str, renderer: WidgetRenderer) -> None:
        normalized = normalize_widget_type(widget_type)
        if normalized is None:
            raise ValueError(f"Unsupported widget_type: {widget_type!r}")
        self._renderers[normalized] = renderer

    def get(self, widget_type: str | None) -> WidgetRenderer | None:
        normalized = normalize_widget_type(widget_type)
        if normalized is None:
            return None
        return self._renderers.get(normalized)

    def render(
        self,
        canvas_obj: Any,
        widget_type: str,
        mapping: dict[str, Any],
        box: dict[str, Any],
        context: WidgetRenderContext,
    ) -> bool:
        renderer = self.get(widget_type)
        if renderer is None:
            return False
        renderer(canvas_obj, mapping, box, context)
        return True

    @property
    def widget_types(self) -> tuple[str, ...]:
        return tuple(sorted(self._renderers))


def build_widget_diagnostics(mappings: list[dict[str, Any]] | None) -> dict[str, Any]:
    safe_mappings = [m for m in (mappings or []) if isinstance(m, dict)]
    explicit_values = [
        str(mapping.get("widget_type") or "").strip().lower()
        for mapping in safe_mappings
        if str(mapping.get("widget_type") or "").strip()
    ]
    normalized_values = [normalize_widget_type(value) for value in explicit_values]
    unsupported = [
        value
        for value, normalized in zip(explicit_values, normalized_values)
        if normalized is None
    ]
    widget_counts = Counter(value for value in normalized_values if value is not None)
    field_type_counts = Counter(str(mapping.get("field_type") or "unknown") for mapping in safe_mappings)

    return {
        "feature_flag": "FORM_PARSER_WIDGET_REGISTRY_ENABLED",
        "enabled": widget_registry_enabled(),
        "supported_widget_types": sorted(SUPPORTED_WIDGET_TYPES),
        "explicit_widget_type_count": len(explicit_values),
        "legacy_mapping_count": len(safe_mappings) - len(explicit_values),
        "widget_type_counts": dict(sorted(widget_counts.items())),
        "unsupported_widget_type_count": len(unsupported),
        "unsupported_widget_types": sorted(set(unsupported)),
        "field_type_counts": dict(sorted(field_type_counts.items())),
        "reasoning": (
            "The registry is opt-in. When disabled, or when a mapping has no "
            "widget_type, rendering uses the legacy field_type path."
        ),
    }
