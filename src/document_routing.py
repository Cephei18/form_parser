from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SMALL = "SMALL"
MEDIUM = "MEDIUM"
LARGE = "LARGE"

ROUTE_LEGACY_SYNC = "LEGACY_SYNC"
ROUTE_SYNC = "SYNC"
ROUTE_PARALLEL_SYNC = "PARALLEL_SYNC"
ROUTE_ASYNC = "ASYNC_TEXTRACT"
ROUTE_JSON_REPLAY = "JSON_REPLAY"


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _file_size(path: Path | str | None) -> int:
    if path is None:
        return 0
    try:
        return Path(path).stat().st_size
    except OSError:
        return 0


def large_doc_routing_enabled() -> bool:
    return _bool_env("FORM_PARSER_LARGE_DOC_ROUTING_ENABLED", False)


def async_textract_enabled() -> bool:
    if os.getenv("FORM_PARSER_ASYNC_TEXTRACT_ENABLED") is not None:
        return _bool_env("FORM_PARSER_ASYNC_TEXTRACT_ENABLED", False)
    return _bool_env("FORM_PARSER_TEXTRACT_ASYNC", False)


def parallel_sync_enabled() -> bool:
    return _bool_env("FORM_PARSER_PARALLEL_SYNC_ENABLED", False)


@dataclass(frozen=True)
class DocumentRoutingConfig:
    enabled: bool = False
    async_textract_enabled: bool = False
    parallel_sync_enabled: bool = False
    small_max_pages: int = 5
    medium_max_pages: int = 20
    max_pages: int = 150
    max_textract_page_units: int = 150
    max_raster_bytes: int = 900 * 1024 * 1024
    max_source_bytes: int = 500 * 1024 * 1024
    parallel_max_workers: int = 4
    estimated_sync_seconds_per_page: float = 3.0
    timeout_budget_seconds: float = 90.0
    textract_cost_per_page_usd: float = 0.0

    @classmethod
    def from_env(cls) -> "DocumentRoutingConfig":
        small_max = max(1, _int_env("FORM_PARSER_SMALL_DOC_MAX_PAGES", 5))
        medium_max = max(small_max, _int_env("FORM_PARSER_MEDIUM_DOC_MAX_PAGES", 20))
        max_pages = max(medium_max, _int_env("FORM_PARSER_MAX_DOCUMENT_PAGES", 150))
        return cls(
            enabled=large_doc_routing_enabled(),
            async_textract_enabled=async_textract_enabled(),
            parallel_sync_enabled=parallel_sync_enabled(),
            small_max_pages=small_max,
            medium_max_pages=medium_max,
            max_pages=max_pages,
            max_textract_page_units=max(1, _int_env("FORM_PARSER_MAX_TEXTRACT_PAGE_UNITS", max_pages)),
            max_raster_bytes=max(1, _int_env("FORM_PARSER_MAX_RASTER_BYTES", 900 * 1024 * 1024)),
            max_source_bytes=max(1, _int_env("FORM_PARSER_MAX_SOURCE_BYTES", 500 * 1024 * 1024)),
            parallel_max_workers=max(1, _int_env("FORM_PARSER_PARALLEL_SYNC_MAX_WORKERS", 4)),
            estimated_sync_seconds_per_page=max(
                0.1,
                _float_env("FORM_PARSER_ESTIMATED_SYNC_SECONDS_PER_PAGE", 3.0),
            ),
            timeout_budget_seconds=max(1.0, _float_env("FORM_PARSER_LAMBDA_TIMEOUT_BUDGET_SECONDS", 90.0)),
            textract_cost_per_page_usd=max(
                0.0,
                _float_env("FORM_PARSER_TEXTRACT_COST_PER_PAGE_USD", 0.0),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DocumentSizeProfile:
    page_count: int
    raster_bytes: int
    source_bytes: int
    average_raster_bytes: int
    estimated_textract_page_units: int
    estimated_cost_usd: float | None
    estimated_serial_sync_seconds: float
    estimated_parallel_sync_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RoutingDecision:
    enabled: bool
    classification: str
    route: str
    reason: str
    document_size: DocumentSizeProfile
    config: DocumentRoutingConfig
    has_document_location: bool
    parallel_workers: int
    sync_pages: int
    async_pages: int
    guardrail_warnings: tuple[str, ...]
    degrade_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["document_size"] = self.document_size.to_dict()
        payload["config"] = self.config.to_dict()
        payload["guardrail_warnings"] = list(self.guardrail_warnings)
        return payload


def build_document_size_profile(
    page_images: list[tuple[int, Path]] | list[tuple[int, str]] | None,
    *,
    source_path: str | Path | None = None,
    source_bytes: int | None = None,
    config: DocumentRoutingConfig | None = None,
) -> DocumentSizeProfile:
    config = config or DocumentRoutingConfig.from_env()
    pages = [(int(page), Path(path)) for page, path in (page_images or [])]
    page_count = len(pages) or 1
    raster_bytes = sum(_file_size(path) for _, path in pages)
    resolved_source_bytes = int(source_bytes) if source_bytes is not None else _file_size(source_path)
    avg_raster = int(raster_bytes / page_count) if page_count else 0
    cost = round(page_count * config.textract_cost_per_page_usd, 6) if config.textract_cost_per_page_usd > 0 else None
    workers = max(1, min(config.parallel_max_workers, page_count))
    return DocumentSizeProfile(
        page_count=page_count,
        raster_bytes=raster_bytes,
        source_bytes=resolved_source_bytes,
        average_raster_bytes=avg_raster,
        estimated_textract_page_units=page_count,
        estimated_cost_usd=cost,
        estimated_serial_sync_seconds=round(page_count * config.estimated_sync_seconds_per_page, 4),
        estimated_parallel_sync_seconds=round(
            math.ceil(page_count / workers) * config.estimated_sync_seconds_per_page,
            4,
        ),
    )


def classify_document(profile: DocumentSizeProfile, config: DocumentRoutingConfig | None = None) -> str:
    config = config or DocumentRoutingConfig.from_env()
    if (
        profile.page_count > config.medium_max_pages
        or profile.estimated_textract_page_units > config.max_textract_page_units
        or profile.raster_bytes > config.max_raster_bytes
        or profile.source_bytes > config.max_source_bytes
        or profile.estimated_parallel_sync_seconds > config.timeout_budget_seconds
    ):
        return LARGE
    if profile.page_count > config.small_max_pages or profile.estimated_serial_sync_seconds > config.timeout_budget_seconds:
        return MEDIUM
    return SMALL


def _guardrails(profile: DocumentSizeProfile, config: DocumentRoutingConfig) -> list[str]:
    warnings: list[str] = []
    if profile.page_count > config.max_pages:
        warnings.append("page_count_exceeds_configured_limit")
    if profile.estimated_textract_page_units > config.max_textract_page_units:
        warnings.append("estimated_textract_page_units_exceeds_limit")
    if profile.raster_bytes > config.max_raster_bytes:
        warnings.append("estimated_raster_bytes_exceeds_limit")
    if profile.source_bytes > config.max_source_bytes:
        warnings.append("source_bytes_exceeds_limit")
    if profile.estimated_serial_sync_seconds > config.timeout_budget_seconds:
        warnings.append("serial_sync_timeout_risk")
    if profile.estimated_parallel_sync_seconds > config.timeout_budget_seconds:
        warnings.append("parallel_sync_timeout_risk")
    return warnings


def choose_routing(
    page_images: list[tuple[int, Path]] | list[tuple[int, str]] | None,
    *,
    source_path: str | Path | None = None,
    source_bytes: int | None = None,
    has_document_location: bool = False,
    config: DocumentRoutingConfig | None = None,
    extraction_mode: str | None = None,
) -> RoutingDecision:
    config = config or DocumentRoutingConfig.from_env()
    profile = build_document_size_profile(
        page_images,
        source_path=source_path,
        source_bytes=source_bytes,
        config=config,
    )
    classification = classify_document(profile, config)
    warnings = _guardrails(profile, config)
    page_count = profile.page_count

    if extraction_mode == ROUTE_JSON_REPLAY:
        return RoutingDecision(
            enabled=config.enabled,
            classification=classification,
            route=ROUTE_JSON_REPLAY,
            reason="json_replay_bypasses_textract_routing",
            document_size=profile,
            config=config,
            has_document_location=has_document_location,
            parallel_workers=0,
            sync_pages=0,
            async_pages=0,
            guardrail_warnings=tuple(warnings),
        )

    if not config.enabled:
        return RoutingDecision(
            enabled=False,
            classification=classification,
            route=ROUTE_LEGACY_SYNC,
            reason="large_document_routing_disabled",
            document_size=profile,
            config=config,
            has_document_location=has_document_location,
            parallel_workers=1,
            sync_pages=page_count,
            async_pages=0,
            guardrail_warnings=tuple(warnings),
        )

    workers = max(1, min(config.parallel_max_workers, page_count))
    if classification == LARGE:
        if config.async_textract_enabled and has_document_location:
            return RoutingDecision(
                enabled=True,
                classification=classification,
                route=ROUTE_ASYNC,
                reason="large_document_uses_start_document_analysis",
                document_size=profile,
                config=config,
                has_document_location=has_document_location,
                parallel_workers=0,
                sync_pages=0,
                async_pages=page_count,
                guardrail_warnings=tuple(warnings),
            )
        if config.parallel_sync_enabled:
            return RoutingDecision(
                enabled=True,
                classification=classification,
                route=ROUTE_PARALLEL_SYNC,
                reason="async_unavailable_degraded_to_bounded_parallel_sync",
                document_size=profile,
                config=config,
                has_document_location=has_document_location,
                parallel_workers=workers,
                sync_pages=page_count,
                async_pages=0,
                guardrail_warnings=tuple(warnings + ["async_textract_unavailable"]),
                degrade_reason="async_unavailable",
            )
        return RoutingDecision(
            enabled=True,
            classification=classification,
            route=ROUTE_SYNC,
            reason="async_and_parallel_unavailable_degraded_to_legacy_sync",
            document_size=profile,
            config=config,
            has_document_location=has_document_location,
            parallel_workers=1,
            sync_pages=page_count,
            async_pages=0,
            guardrail_warnings=tuple(warnings + ["async_textract_unavailable", "parallel_sync_unavailable"]),
            degrade_reason="async_and_parallel_unavailable",
        )

    if classification == MEDIUM and config.parallel_sync_enabled:
        return RoutingDecision(
            enabled=True,
            classification=classification,
            route=ROUTE_PARALLEL_SYNC,
            reason="medium_document_uses_bounded_parallel_sync",
            document_size=profile,
            config=config,
            has_document_location=has_document_location,
            parallel_workers=workers,
            sync_pages=page_count,
            async_pages=0,
            guardrail_warnings=tuple(warnings),
        )

    return RoutingDecision(
        enabled=True,
        classification=classification,
        route=ROUTE_SYNC,
        reason="small_document_or_parallel_sync_disabled",
        document_size=profile,
        config=config,
        has_document_location=has_document_location,
        parallel_workers=1,
        sync_pages=page_count,
        async_pages=0,
        guardrail_warnings=tuple(warnings),
    )
