from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from statistics import median
from typing import Any


logger = logging.getLogger(__name__)
ocr = None


class OCRRuntimeError(RuntimeError):
    """Raised when the OCR backend cannot initialize or process an image."""


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
        logger.warning("[ocr] invalid integer env %s=%r; using %s", name, value, default)
        return default


def _get_ocr_model():
    global ocr

    if ocr is not None:
        return ocr

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    try:
        import easyocr
    except ModuleNotFoundError as exc:
        raise OCRRuntimeError(
            "easyocr is not installed in the active environment. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    languages = [
        language.strip()
        for language in os.getenv("FORM_PARSER_OCR_LANGUAGES", "en").split(",")
        if language.strip()
    ]
    if not languages:
        languages = ["en"]

    model_dir = os.getenv("FORM_PARSER_EASYOCR_MODEL_DIR")
    download_enabled = _bool_env("FORM_PARSER_EASYOCR_DOWNLOAD_ENABLED", True)
    thread_count = max(1, _int_env("FORM_PARSER_OCR_THREADS", 1))

    try:
        try:
            import torch

            torch.set_num_threads(thread_count)
            torch.set_num_interop_threads(thread_count)
        except Exception:
            logger.warning("[ocr] unable to set torch thread count", exc_info=True)

        logger.info(
            "[ocr] initializing EasyOCR reader languages=%s gpu=False model_dir=%s download_enabled=%s threads=%s",
            languages,
            model_dir or "<default>",
            download_enabled,
            thread_count,
        )
        kwargs: dict[str, Any] = {
            "gpu": False,
            "download_enabled": download_enabled,
            "verbose": False,
        }
        if model_dir:
            model_path = Path(model_dir)
            model_path.mkdir(parents=True, exist_ok=True)
            user_network_path = model_path / "user_network"
            user_network_path.mkdir(parents=True, exist_ok=True)
            kwargs["model_storage_directory"] = str(model_path)
            kwargs["user_network_directory"] = str(user_network_path)

        ocr = easyocr.Reader(languages, **kwargs)
        logger.info("[ocr] EasyOCR reader ready")
        return ocr
    except Exception as exc:
        logger.exception("[ocr] EasyOCR initialization failed")
        raise OCRRuntimeError(f"OCR initialization failed: {exc}") from exc


def _normalize_bbox(raw_bbox: Any) -> list[list[float]] | None:
    if not raw_bbox or len(raw_bbox) != 4:
        return None

    normalized = []
    for point in raw_bbox:
        if not point or len(point) < 2:
            return None
        try:
            normalized.append([float(point[0]), float(point[1])])
        except (TypeError, ValueError):
            return None

    return normalized


def _normalize_easyocr_result(result: Any) -> dict[str, Any] | None:
    if not isinstance(result, (list, tuple)) or len(result) < 2:
        return None

    bbox = _normalize_bbox(result[0])
    text = str(result[1]).strip() if result[1] is not None else ""
    if bbox is None or not text:
        return None

    confidence = None
    if len(result) > 2 and result[2] is not None:
        try:
            confidence = float(result[2])
        except (TypeError, ValueError):
            confidence = None

    return {"text": text, "bbox": bbox, "confidence": confidence, "source_item_count": 1}


def _ocr_bounds(item: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = item.get("bbox")
    if not bbox:
        return None
    try:
        xs = [float(point[0]) for point in bbox]
        ys = [float(point[1]) for point in bbox]
    except Exception:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _bbox_from_bounds(bounds: tuple[float, float, float, float]) -> list[list[float]]:
    x1, y1, x2, y2 = bounds
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _box_height(item: dict[str, Any]) -> float:
    bounds = _ocr_bounds(item)
    if bounds is None:
        return 0.0
    return max(0.0, bounds[3] - bounds[1])


def _median_height(items: list[dict[str, Any]], default: float = 24.0) -> float:
    heights = [_box_height(item) for item in items]
    heights = [height for height in heights if height > 0]
    if not heights:
        return default
    return float(median(heights))


def _vertical_overlap_ratio(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    overlap = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    min_height = max(min(a[3] - a[1], b[3] - b[1]), 1.0)
    return overlap / min_height


def _should_drop_ocr_item(item: dict[str, Any], median_height: float) -> bool:
    text = str(item.get("text", "")).strip()
    if not text:
        return True

    bounds = _ocr_bounds(item)
    if bounds is None:
        return True

    confidence = item.get("confidence")
    confidence_value = float(confidence) if isinstance(confidence, (int, float)) else 1.0
    alnum_count = sum(1 for char in text if char.isalnum())
    symbol_count = sum(1 for char in text if not char.isalnum() and not char.isspace())
    symbol_ratio = symbol_count / max(len(text), 1)
    height = bounds[3] - bounds[1]

    if confidence_value < 0.15 and (len(text) <= 8 or symbol_ratio >= 0.2):
        return True
    if confidence_value < 0.35 and alnum_count <= 2:
        return True
    if confidence_value < 0.40 and height < max(median_height * 0.45, 8.0):
        return True

    return False


def _join_ocr_text(left: str, right: str) -> str:
    left = left.strip()
    right = right.strip()
    if not left:
        return right
    if not right:
        return left
    if left.endswith(("/", "-", "(", ":")) or right in {":", ".", ",", ";", ")", "/", "-"}:
        return f"{left}{right}"
    return f"{left} {right}"


def _merge_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    bounds = [_ocr_bounds(item) for item in items]
    bounds = [bound for bound in bounds if bound is not None]
    x1 = min(bound[0] for bound in bounds)
    y1 = min(bound[1] for bound in bounds)
    x2 = max(bound[2] for bound in bounds)
    y2 = max(bound[3] for bound in bounds)
    text = ""
    confidences = []
    source_item_count = 0

    for item in sorted(items, key=lambda entry: (_ocr_bounds(entry) or (0, 0, 0, 0))[0]):
        text = _join_ocr_text(text, str(item.get("text", "")))
        confidence = item.get("confidence")
        if isinstance(confidence, (int, float)):
            confidences.append(float(confidence))
        source_item_count += int(item.get("source_item_count", 1) or 1)

    return {
        "text": re.sub(r"\s+", " ", text).strip(),
        "bbox": _bbox_from_bounds((x1, y1, x2, y2)),
        "confidence": round(sum(confidences) / len(confidences), 4) if confidences else None,
        "source_item_count": source_item_count,
    }


def _merge_same_line_tokens(items: list[dict[str, Any]], median_height: float) -> list[dict[str, Any]]:
    row_tolerance = max(8.0, median_height * 0.65)
    merge_gap = max(42.0, median_height * 1.8)
    rows: list[dict[str, Any]] = []

    for item in sorted(items, key=lambda entry: ((_ocr_bounds(entry) or (0, 0, 0, 0))[1], (_ocr_bounds(entry) or (0, 0, 0, 0))[0])):
        bounds = _ocr_bounds(item)
        if bounds is None:
            continue
        center_y = (bounds[1] + bounds[3]) / 2.0
        target_row = None
        for row in rows:
            if abs(center_y - row["center_y"]) <= row_tolerance:
                target_row = row
                break
        if target_row is None:
            rows.append({"center_y": center_y, "items": [item]})
            continue
        target_row["items"].append(item)
        target_row["center_y"] = sum(
            ((_ocr_bounds(entry) or (0, 0, 0, 0))[1] + (_ocr_bounds(entry) or (0, 0, 0, 0))[3]) / 2.0
            for entry in target_row["items"]
        ) / len(target_row["items"])

    normalized: list[dict[str, Any]] = []
    for row in rows:
        current_group: list[dict[str, Any]] = []
        for item in sorted(row["items"], key=lambda entry: (_ocr_bounds(entry) or (0, 0, 0, 0))[0]):
            item_bounds = _ocr_bounds(item)
            if item_bounds is None:
                continue
            if not current_group:
                current_group = [item]
                continue

            previous_bounds = _ocr_bounds(current_group[-1])
            if previous_bounds is None:
                normalized.extend(current_group)
                current_group = [item]
                continue

            horizontal_gap = item_bounds[0] - previous_bounds[2]
            overlaps_vertically = _vertical_overlap_ratio(previous_bounds, item_bounds) >= 0.45
            same_text_line = 0 <= horizontal_gap <= merge_gap and overlaps_vertically

            if same_text_line:
                current_group.append(item)
                continue

            normalized.append(_merge_items(current_group) if len(current_group) > 1 else current_group[0])
            current_group = [item]

        if current_group:
            normalized.append(_merge_items(current_group) if len(current_group) > 1 else current_group[0])

    return sorted(normalized, key=lambda item: ((_ocr_bounds(item) or (0, 0, 0, 0))[1], (_ocr_bounds(item) or (0, 0, 0, 0))[0]))


def normalize_ocr_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not items:
        return []

    median_height = _median_height(items)
    filtered = [item for item in items if not _should_drop_ocr_item(item, median_height)]
    merged = _merge_same_line_tokens(filtered, median_height)

    logger.info(
        "[ocr] normalized items raw=%s filtered=%s merged=%s median_height=%.2f",
        len(items),
        len(filtered),
        len(merged),
        median_height,
    )
    return merged


def extract_text(image_path):
    logger.info("[ocr] extracting text from %s", image_path)

    if not Path(image_path).exists():
        raise OCRRuntimeError(f"OCR input image does not exist: {image_path}")

    model = _get_ocr_model()

    try:
        results = model.readtext(
            str(image_path),
            detail=1,
            paragraph=False,
            batch_size=max(1, _int_env("FORM_PARSER_OCR_BATCH_SIZE", 1)),
        )
    except Exception as exc:
        logger.exception("[ocr] EasyOCR extraction failed image=%s", image_path)
        raise OCRRuntimeError(f"OCR extraction failed: {exc}") from exc

    if not results:
        logger.info("[ocr] no OCR results returned for %s", image_path)
        return []

    extracted = []
    for index, result in enumerate(results):
        item = _normalize_easyocr_result(result)
        if item is None:
            logger.warning("[ocr] skipping malformed OCR result index=%s", index)
            continue
        extracted.append(item)

    normalized = normalize_ocr_items(extracted)
    logger.info("[ocr] extracted items=%s normalized=%s image=%s", len(extracted), len(normalized), image_path)
    return normalized
