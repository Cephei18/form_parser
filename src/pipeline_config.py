from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


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


def _path_env(name: str) -> Path | None:
    value = os.getenv(name)
    if not value:
        return None
    return Path(value)


def _pipeline_mode_env(default: str = "ocr") -> str:
    value = os.getenv("FORM_PARSER_PIPELINE_MODE", default)
    mode = str(value or default).strip().lower()
    if mode not in {"ocr", "textract", "hybrid"}:
        logger.warning("[config] invalid FORM_PARSER_PIPELINE_MODE=%r; falling back to ocr", value)
        return "ocr"
    return mode


@dataclass(frozen=True)
class PreprocessingConfig:
    enabled: bool = False
    denoise: bool = True
    contrast_enhancement: bool = True
    sharpen: bool = False
    adaptive_threshold: bool = False
    skew_correction: bool = False
    dpi_normalization: bool = False
    target_long_edge: int = 1650
    max_skew_degrees: float = 4.0

    @classmethod
    def from_env(cls) -> "PreprocessingConfig":
        return cls(
            enabled=_bool_env("FORM_PARSER_PREPROCESSING_ENABLED", False),
            denoise=_bool_env("FORM_PARSER_PREPROCESS_DENOISE", True),
            contrast_enhancement=_bool_env("FORM_PARSER_PREPROCESS_CONTRAST", True),
            sharpen=_bool_env("FORM_PARSER_PREPROCESS_SHARPEN", False),
            adaptive_threshold=_bool_env("FORM_PARSER_PREPROCESS_ADAPTIVE_THRESHOLD", False),
            skew_correction=_bool_env("FORM_PARSER_PREPROCESS_SKEW", False),
            dpi_normalization=_bool_env("FORM_PARSER_PREPROCESS_DPI_NORMALIZE", False),
            target_long_edge=max(300, _int_env("FORM_PARSER_PREPROCESS_TARGET_LONG_EDGE", 1650)),
            max_skew_degrees=max(0.0, _float_env("FORM_PARSER_PREPROCESS_MAX_SKEW_DEGREES", 4.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StructuralRefinementConfig:
    enabled: bool = False
    field_quality_enabled: bool = True
    section_grouping_enabled: bool = True
    ownership_propagation_enabled: bool = True
    table_aware_enabled: bool = True
    min_field_quality_score: float = 0.32

    @classmethod
    def from_env(cls) -> "StructuralRefinementConfig":
        return cls(
            enabled=_bool_env("FORM_PARSER_STRUCTURAL_REFINEMENT_ENABLED", False),
            field_quality_enabled=_bool_env("FORM_PARSER_FIELD_QUALITY_REFINEMENT_ENABLED", True),
            section_grouping_enabled=_bool_env("FORM_PARSER_SECTION_GROUPING_ENABLED", True),
            ownership_propagation_enabled=_bool_env("FORM_PARSER_OWNERSHIP_PROPAGATION_ENABLED", True),
            table_aware_enabled=_bool_env("FORM_PARSER_TABLE_AWARE_REFINEMENT_ENABLED", True),
            min_field_quality_score=max(
                0.0,
                min(1.0, _float_env("FORM_PARSER_MIN_FIELD_QUALITY_SCORE", 0.32)),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _str_env(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


@dataclass(frozen=True)
class StorageConfig:
    """Artifact storage backend for the Textract pipeline (serverless migration).

    Defaults to ``local`` so the existing on-disk workflow is unchanged. Set
    ``FORM_PARSER_ARTIFACT_BACKEND=s3`` plus a processed bucket to publish run
    artifacts to S3 (the first Lambda-compatible step). OCR is unaffected.
    """

    backend: str = "local"
    processed_bucket: str | None = None
    prefix: str = "textract"
    region: str | None = None

    @classmethod
    def from_env(cls) -> "StorageConfig":
        backend = _str_env("FORM_PARSER_ARTIFACT_BACKEND", "local").lower()
        if backend not in {"local", "s3"}:
            logger.warning("[config] invalid FORM_PARSER_ARTIFACT_BACKEND=%r; falling back to local", backend)
            backend = "local"
        region = os.getenv("FORM_PARSER_AWS_REGION") or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
        return cls(
            backend=backend,
            processed_bucket=os.getenv("FORM_PARSER_PROCESSED_BUCKET") or None,
            prefix=_str_env("FORM_PARSER_ARTIFACT_PREFIX", "textract"),
            region=region or None,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PipelineConfig:
    preprocessing: PreprocessingConfig
    structural_refinement: StructuralRefinementConfig
    storage: StorageConfig
    widget_registry_enabled: bool = False
    comb_detection_enabled: bool = False
    comb_min_confidence: float = 0.84
    radio_grouping_enabled: bool = False
    radio_min_confidence: float = 0.86
    confidence_pipeline_enabled: bool = False
    review_queue_enabled: bool = False
    confidence_high_threshold: float = 0.82
    confidence_medium_threshold: float = 0.55
    confidence_low_threshold: float = 0.0
    large_doc_routing_enabled: bool = False
    async_textract_enabled: bool = False
    parallel_sync_enabled: bool = False
    small_doc_max_pages: int = 5
    medium_doc_max_pages: int = 20
    max_document_pages: int = 150
    max_textract_page_units: int = 150
    max_raster_bytes: int = 943718400
    max_source_bytes: int = 524288000
    parallel_sync_max_workers: int = 4
    estimated_sync_seconds_per_page: float = 3.0
    lambda_timeout_budget_seconds: float = 90.0
    ocr_diagnostics_enabled: bool = True
    dynamic_thresholds_enabled: bool = False
    image_table_filtering_enabled: bool = False
    fallback_field_lines_enabled: bool = False
    checkbox_detection_enabled: bool = False
    debug_artifacts_enabled: bool = False
    baseline_dir: Path | None = None
    pipeline_mode: str = "ocr"

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        return cls(
            preprocessing=PreprocessingConfig.from_env(),
            structural_refinement=StructuralRefinementConfig.from_env(),
            storage=StorageConfig.from_env(),
            widget_registry_enabled=_bool_env("FORM_PARSER_WIDGET_REGISTRY_ENABLED", False),
            comb_detection_enabled=_bool_env("FORM_PARSER_COMB_DETECTION_ENABLED", False),
            comb_min_confidence=max(
                0.0,
                min(1.0, _float_env("FORM_PARSER_COMB_MIN_CONFIDENCE", 0.84)),
            ),
            radio_grouping_enabled=_bool_env("FORM_PARSER_RADIO_GROUPING_ENABLED", False),
            radio_min_confidence=max(
                0.0,
                min(1.0, _float_env("FORM_PARSER_RADIO_MIN_CONFIDENCE", 0.86)),
            ),
            confidence_pipeline_enabled=_bool_env("FORM_PARSER_CONFIDENCE_PIPELINE_ENABLED", False),
            review_queue_enabled=_bool_env("FORM_PARSER_REVIEW_QUEUE_ENABLED", False),
            confidence_high_threshold=max(
                0.0,
                min(1.0, _float_env("FORM_PARSER_CONFIDENCE_HIGH_THRESHOLD", 0.82)),
            ),
            confidence_medium_threshold=max(
                0.0,
                min(1.0, _float_env("FORM_PARSER_CONFIDENCE_MEDIUM_THRESHOLD", 0.55)),
            ),
            confidence_low_threshold=max(
                0.0,
                min(1.0, _float_env("FORM_PARSER_CONFIDENCE_LOW_THRESHOLD", 0.0)),
            ),
            large_doc_routing_enabled=_bool_env("FORM_PARSER_LARGE_DOC_ROUTING_ENABLED", False),
            async_textract_enabled=_bool_env(
                "FORM_PARSER_ASYNC_TEXTRACT_ENABLED",
                _bool_env("FORM_PARSER_TEXTRACT_ASYNC", False),
            ),
            parallel_sync_enabled=_bool_env("FORM_PARSER_PARALLEL_SYNC_ENABLED", False),
            small_doc_max_pages=max(1, _int_env("FORM_PARSER_SMALL_DOC_MAX_PAGES", 5)),
            medium_doc_max_pages=max(1, _int_env("FORM_PARSER_MEDIUM_DOC_MAX_PAGES", 20)),
            max_document_pages=max(1, _int_env("FORM_PARSER_MAX_DOCUMENT_PAGES", 150)),
            max_textract_page_units=max(1, _int_env("FORM_PARSER_MAX_TEXTRACT_PAGE_UNITS", 150)),
            max_raster_bytes=max(1, _int_env("FORM_PARSER_MAX_RASTER_BYTES", 943718400)),
            max_source_bytes=max(1, _int_env("FORM_PARSER_MAX_SOURCE_BYTES", 524288000)),
            parallel_sync_max_workers=max(1, _int_env("FORM_PARSER_PARALLEL_SYNC_MAX_WORKERS", 4)),
            estimated_sync_seconds_per_page=max(
                0.1,
                _float_env("FORM_PARSER_ESTIMATED_SYNC_SECONDS_PER_PAGE", 3.0),
            ),
            lambda_timeout_budget_seconds=max(
                1.0,
                _float_env("FORM_PARSER_LAMBDA_TIMEOUT_BUDGET_SECONDS", 90.0),
            ),
            ocr_diagnostics_enabled=_bool_env("FORM_PARSER_OCR_DIAGNOSTICS_ENABLED", True),
            dynamic_thresholds_enabled=_bool_env("FORM_PARSER_DYNAMIC_THRESHOLDS_ENABLED", False),
            image_table_filtering_enabled=_bool_env("FORM_PARSER_IMAGE_TABLE_FILTERING_ENABLED", False),
            fallback_field_lines_enabled=_bool_env("FORM_PARSER_FALLBACK_FIELD_LINES_ENABLED", False),
            checkbox_detection_enabled=_bool_env("FORM_PARSER_CHECKBOX_DETECTION_ENABLED", False),
            debug_artifacts_enabled=_bool_env("FORM_PARSER_DEBUG_ARTIFACTS_ENABLED", False),
            baseline_dir=_path_env("FORM_PARSER_BASELINE_DIR"),
            pipeline_mode=_pipeline_mode_env(),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.baseline_dir is not None:
            payload["baseline_dir"] = str(self.baseline_dir)
        return payload
