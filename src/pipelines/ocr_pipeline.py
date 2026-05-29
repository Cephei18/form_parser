from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("form_parser.pipeline.ocr")


def run_ocr_pipeline(file_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Run the existing stable OCR pipeline without changing its behavior."""
    from src.main import run_pipeline as run_legacy_ocr_pipeline

    image_path = Path(file_path)
    destination_dir = Path(output_dir)
    logger.info("[pipeline] Running OCR pipeline")
    output = run_legacy_ocr_pipeline(image_path, destination_dir)
    output.setdefault("pipeline_mode", "ocr")
    return output


run_pipeline = run_ocr_pipeline
