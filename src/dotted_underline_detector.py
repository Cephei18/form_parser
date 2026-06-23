from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger("form_parser.dotted_underline")

INLINE_LEADER_RE = re.compile(r"(?P<label>.*?)(?P<leader>(?:[.\-_–—]\s*){5,})\s*$")
DOT_CHARS = {".", "·", "•"}
DASH_CHARS = {"-", "_", "–", "—"}


@dataclass(frozen=True)
class SyntheticUnderline:
    bbox: dict[str, float]
    page: int
    confidence: float
    source_type: str
    segment_count: int = 0
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "bbox": _round_box(self.bbox),
            "page": int(self.page),
            "confidence": round(float(self.confidence), 4),
            "source_type": self.source_type,
            "segment_count": int(self.segment_count),
            "reasons": list(self.reasons),
        }


def _round_box(box: dict[str, float]) -> dict[str, float]:
    return {
        "x": round(float(box["x"]), 6),
        "y": round(float(box["y"]), 6),
        "width": round(float(box["width"]), 6),
        "height": round(float(box["height"]), 6),
    }


def _box_from_pixels(x: float, y: float, w: float, h: float, image_width: int, image_height: int) -> dict[str, float]:
    return _round_box(
        {
            "x": max(0.0, x) / float(image_width),
            "y": max(0.0, y) / float(image_height),
            "width": max(1.0, min(w, image_width - max(0.0, x))) / float(image_width),
            "height": max(1.0, min(h, image_height - max(0.0, y))) / float(image_height),
        }
    )


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _component_records(binary: np.ndarray) -> list[dict[str, float]]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    image_height, image_width = binary.shape[:2]
    page_area = float(image_width * image_height)
    records: list[dict[str, float]] = []
    for idx in range(1, count):
        x, y, w, h, area = stats[idx]
        if area < 2 or area > page_area * 0.01:
            continue
        if h > max(18, image_height * 0.035):
            continue
        if w > image_width * 0.20:
            continue
        if w < 2 or h < 1:
            continue
        fill = float(area) / max(float(w * h), 1.0)
        aspect = float(w) / max(float(h), 1.0)
        if fill < 0.12:
            continue
        if aspect < 0.25:
            continue
        records.append(
            {
                "x": float(x),
                "y": float(y),
                "w": float(w),
                "h": float(h),
                "area": float(area),
                "cx": float(centroids[idx][0]),
                "cy": float(centroids[idx][1]),
                "aspect": aspect,
                "fill": fill,
            }
        )
    return sorted(records, key=lambda item: (item["cy"], item["cx"]))


def _row_groups(components: list[dict[str, float]], image_height: int) -> list[list[dict[str, float]]]:
    groups: list[list[dict[str, float]]] = []
    tolerance = max(7.0, image_height * 0.006)
    for component in components:
        best_index: int | None = None
        best_delta = float("inf")
        for index, group in enumerate(groups):
            center = median([item["cy"] for item in group])
            delta = abs(component["cy"] - center)
            if delta <= tolerance and delta < best_delta:
                best_index = index
                best_delta = delta
        if best_index is None:
            groups.append([component])
        else:
            groups[best_index].append(component)
    return [sorted(group, key=lambda item: item["cx"]) for group in groups]


def _contiguous_runs(group: list[dict[str, float]], image_width: int) -> list[list[dict[str, float]]]:
    if not group:
        return []
    widths = [item["w"] for item in group]
    median_width = median(widths) if widths else 6.0
    max_gap = max(28.0, median_width * 4.0, image_width * 0.035)
    runs: list[list[dict[str, float]]] = [[group[0]]]
    for previous, current in zip(group, group[1:]):
        gap = current["x"] - (previous["x"] + previous["w"])
        if gap <= max_gap:
            runs[-1].append(current)
        else:
            runs.append([current])
    return runs


def _line_residual(run: list[dict[str, float]]) -> float:
    if len(run) < 2:
        return 0.0
    xs = np.array([item["cx"] for item in run], dtype=np.float32)
    ys = np.array([item["cy"] for item in run], dtype=np.float32)
    if float(xs.max() - xs.min()) < 1.0:
        return float("inf")
    coeffs = np.polyfit(xs, ys, 1)
    predicted = coeffs[0] * xs + coeffs[1]
    return float(np.sqrt(np.mean((ys - predicted) ** 2)))


def _classify_run(run: list[dict[str, float]], image_width: int, image_height: int) -> tuple[str | None, float, list[str]]:
    if len(run) < 2:
        return None, 0.0, ["too_few_segments"]

    x1 = min(item["x"] for item in run)
    x2 = max(item["x"] + item["w"] for item in run)
    y1 = min(item["y"] for item in run)
    y2 = max(item["y"] + item["h"] for item in run)
    total_width = x2 - x1
    total_height = y2 - y1
    if total_width < max(65.0, image_width * 0.045):
        return None, 0.0, ["too_short"]
    if total_height > max(18.0, image_height * 0.035):
        return None, 0.0, ["too_tall"]

    widths = [item["w"] for item in run]
    heights = [item["h"] for item in run]
    aspects = [item["aspect"] for item in run]
    gaps = [run[i + 1]["x"] - (run[i]["x"] + run[i]["w"]) for i in range(len(run) - 1)]
    median_width = float(median(widths))
    median_height = float(median(heights))
    median_aspect = float(median(aspects))
    median_gap = float(median(gaps)) if gaps else 0.0
    gap_cv = float(np.std(gaps) / max(median_gap, 1.0)) if len(gaps) > 1 else 0.0
    residual = _line_residual(run)
    if residual > max(5.0, median_height * 1.2):
        return None, 0.0, ["not_collinear"]

    coverage = sum(widths) / max(total_width, 1.0)
    reasons = ["collinear_segments"]

    if len(run) >= 5 and median_aspect >= 2.1 and median_width >= max(7.0, median_height * 2.0):
        confidence = 0.60 + min(len(run), 10) * 0.018 + (0.08 if gap_cv < 0.75 else 0.0)
        return "dashed", _clamp(confidence, 0.58, 0.80), reasons + ["dash_segments"]

    long_segments = [item for item in run if item["w"] >= max(22.0, image_width * 0.018)]
    if len(long_segments) >= 2 and total_width >= max(100.0, image_width * 0.07):
        confidence = 0.62 + min(len(long_segments), 5) * 0.035 + min(coverage, 0.9) * 0.12
        if median_gap <= max(70.0, image_width * 0.07):
            confidence += 0.05
            reasons.append("mergeable_small_gaps")
        return "broken", _clamp(confidence, 0.55, 0.78), reasons + ["long_broken_segments"]

    if len(run) >= 5 and median_width <= max(14.0, image_width * 0.018):
        confidence = 0.62 + min(len(run), 18) * 0.012 + (0.08 if gap_cv < 0.9 else 0.0)
        return "dotted", _clamp(confidence, 0.58, 0.82), reasons + ["dot_sequence"]

    return None, 0.0, ["segment_pattern_not_field_like"]


def detect_dotted_underlines(image_path: str | Path, *, page: int = 1) -> dict[str, Any]:
    """Detect dotted, dashed, and broken horizontal answer guides in one page image."""
    image = cv2.imread(str(image_path))
    if image is None:
        logger.warning("[dotted] unable to read image: %s", image_path)
        return {"detected": [], "rejected": [], "image_size": None}

    image_height, image_width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        9,
    )

    components = _component_records(binary)
    detected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for group in _row_groups(components, image_height):
        for run in _contiguous_runs(group, image_width):
            source_type, confidence, reasons = _classify_run(run, image_width, image_height)
            if source_type is None:
                if len(run) >= 3:
                    x1 = min(item["x"] for item in run)
                    x2 = max(item["x"] + item["w"] for item in run)
                    y1 = min(item["y"] for item in run)
                    y2 = max(item["y"] + item["h"] for item in run)
                    rejected.append(
                        {
                            "bbox": _box_from_pixels(x1, y1, x2 - x1, max(y2 - y1, 1.0), image_width, image_height),
                            "page": page,
                            "segment_count": len(run),
                            "reasons": reasons,
                        }
                    )
                continue

            x1 = min(item["x"] for item in run)
            x2 = max(item["x"] + item["w"] for item in run)
            y1 = min(item["y"] for item in run)
            y2 = max(item["y"] + item["h"] for item in run)
            underline = SyntheticUnderline(
                bbox=_box_from_pixels(x1, y1, x2 - x1, max(y2 - y1, 1.0), image_width, image_height),
                page=page,
                confidence=confidence,
                source_type=source_type,
                segment_count=len(run),
                reasons=reasons,
            )
            detected.append(underline.as_dict())

    return {
        "detected": _dedupe_underlines(detected),
        "rejected": rejected[:80],
        "image_size": {"width": image_width, "height": image_height},
        "source_image": str(image_path),
    }


def _dedupe_underlines(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(items, key=lambda item: (item.get("page", 1), item["bbox"]["y"], item["bbox"]["x"], -item["bbox"]["width"]))
    deduped: list[dict[str, Any]] = []
    for item in ordered:
        box = item["bbox"]
        duplicate = False
        for existing in deduped:
            other = existing["bbox"]
            x1 = max(box["x"], other["x"])
            y1 = max(box["y"], other["y"])
            x2 = min(box["x"] + box["width"], other["x"] + other["width"])
            y2 = min(box["y"] + box["height"], other["y"] + other["height"])
            intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            smaller = min(box["width"] * box["height"], other["width"] * other["height"])
            if intersection / max(smaller, 1e-9) > 0.72:
                duplicate = True
                break
        if not duplicate:
            deduped.append(item)
    return deduped


def split_inline_answer_region(
    text: str,
    bbox: dict[str, float],
    *,
    page: int = 1,
    line_height: float | None = None,
) -> dict[str, Any] | None:
    """Split fused labels such as ``Mobile No.:........`` into label + answer box."""
    raw = str(text or "").strip()
    if not raw or "@" in raw or "://" in raw.lower():
        return None
    match = INLINE_LEADER_RE.match(raw)
    if not match:
        return None

    label = match.group("label").rstrip()
    leader = match.group("leader")
    if not label or not any(ch.isalpha() for ch in label):
        return None
    if re.fullmatch(r"(?:v(?:er(?:sion)?)?\s*)?\d+(?:\.\d+){1,4}", raw, flags=re.IGNORECASE):
        return None

    words = re.findall(r"[A-Za-z0-9]+", label)
    if len(words) > 4 and not label.endswith(":"):
        return None
    if len(label) > 70:
        return None

    leader_chars = [ch for ch in leader if ch in DOT_CHARS or ch in DASH_CHARS]
    if len(leader_chars) < 5:
        return None
    if len(set(leader_chars) - DOT_CHARS) > 0:
        source_type = "inline_dashed"
    else:
        source_type = "inline_dotted"

    try:
        x = float(bbox["x"])
        y = float(bbox["y"])
        width = float(bbox["width"])
        height = float(bbox["height"])
    except (KeyError, TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None

    leader_start = match.start("leader")
    ratio = _clamp(leader_start / max(len(raw), 1), 0.12, 0.92)
    gap = min(max(width * 0.012, 0.002), 0.008)
    answer_x = min(x + width - 0.01, x + width * ratio + gap)
    answer_w = max(0.0001, x + width - answer_x)
    answer_h = max(height * 1.35, float(line_height or 0.0) * 1.45, 0.018)
    answer_y = max(0.0, y + height * 0.5 - answer_h * 0.5)

    label_w = max(0.0001, answer_x - gap - x)
    label_bbox = _round_box({"x": x, "y": y, "width": label_w, "height": height})
    answer_bbox = _round_box({"x": answer_x, "y": answer_y, "width": answer_w, "height": answer_h})
    confidence = 0.84 if source_type == "inline_dotted" else 0.82

    return {
        "label": label,
        "label_bbox": label_bbox,
        "answer_bbox": answer_bbox,
        "page": int(page),
        "confidence": confidence,
        "source_type": source_type,
        "leader_length": len(leader_chars),
        "reasons": ["inline_leader_sequence", source_type],
    }
