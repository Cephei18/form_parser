from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.pipeline_config import PipelineConfig
from src.pipelines.ocr_pipeline import run_ocr_pipeline
from src.pipelines.textract_pipeline import run_textract_pipeline

logger = logging.getLogger("form_parser.pipeline.router")


SUPPORTED_MODES = {"ocr", "textract", "hybrid"}


def resolve_pipeline_mode() -> str:
    config = PipelineConfig.from_env()
    mode = str(config.pipeline_mode or "ocr").strip().lower()
    if mode not in SUPPORTED_MODES:
        logger.warning("[pipeline] invalid pipeline mode=%r; falling back to OCR", mode)
        return "ocr"
    return mode


def run_pipeline(
    file_path: str | Path,
    output_dir: str | Path,
    reference_image_path: str | Path | None = None,
    *,
    page_images: list[tuple[int, str | Path]] | None = None,
    document_location: dict[str, str] | None = None,
) -> dict[str, Any]:
    mode = resolve_pipeline_mode()
    logger.info("[pipeline] Pipeline mode: %s", mode)

    if mode == "textract":
        logger.info("ACTIVE PIPELINE: TEXTRACT")
        logger.info("[pipeline] Running Textract pipeline")
        return run_textract_pipeline(
            file_path,
            output_dir,
            reference_image_path=reference_image_path,
            page_images=page_images,
            document_location=document_location,
        )

    if mode == "hybrid":
        logger.warning("[pipeline] Hybrid mode is reserved; falling back to OCR for now")

    logger.info("ACTIVE PIPELINE: OCR")
    logger.info("[pipeline] Running OCR pipeline")
    return run_ocr_pipeline(file_path, output_dir)


run_document_pipeline = run_pipeline
