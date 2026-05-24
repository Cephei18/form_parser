from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


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


@dataclass(frozen=True)
class PipelineConfig:
    preprocessing: PreprocessingConfig
    structural_refinement: StructuralRefinementConfig
    ocr_diagnostics_enabled: bool = True
    dynamic_thresholds_enabled: bool = False
    image_table_filtering_enabled: bool = False
    fallback_field_lines_enabled: bool = False
    checkbox_detection_enabled: bool = False
    debug_artifacts_enabled: bool = True
    baseline_dir: Path | None = None

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        return cls(
            preprocessing=PreprocessingConfig.from_env(),
            structural_refinement=StructuralRefinementConfig.from_env(),
            ocr_diagnostics_enabled=_bool_env("FORM_PARSER_OCR_DIAGNOSTICS_ENABLED", True),
            dynamic_thresholds_enabled=_bool_env("FORM_PARSER_DYNAMIC_THRESHOLDS_ENABLED", False),
            image_table_filtering_enabled=_bool_env("FORM_PARSER_IMAGE_TABLE_FILTERING_ENABLED", False),
            fallback_field_lines_enabled=_bool_env("FORM_PARSER_FALLBACK_FIELD_LINES_ENABLED", False),
            checkbox_detection_enabled=_bool_env("FORM_PARSER_CHECKBOX_DETECTION_ENABLED", False),
            debug_artifacts_enabled=_bool_env("FORM_PARSER_DEBUG_ARTIFACTS_ENABLED", True),
            baseline_dir=_path_env("FORM_PARSER_BASELINE_DIR"),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.baseline_dir is not None:
            payload["baseline_dir"] = str(self.baseline_dir)
        return payload
