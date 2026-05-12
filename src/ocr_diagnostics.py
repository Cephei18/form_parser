from __future__ import annotations

import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean, median
from typing import Any

import cv2


def _read_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Expected OCR list in {path}")
    return [item for item in payload if isinstance(item, dict)]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _bounds(item: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = item.get("bbox")
    if not bbox:
        return None
    try:
        xs = [float(point[0]) for point in bbox]
        ys = [float(point[1]) for point in bbox]
    except Exception:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _text(item: dict[str, Any]) -> str:
    return str(item.get("text", "")).strip()


def _height(item: dict[str, Any]) -> float:
    bounds = _bounds(item)
    if bounds is None:
        return 0.0
    return max(0.0, bounds[3] - bounds[1])


def _width(item: dict[str, Any]) -> float:
    bounds = _bounds(item)
    if bounds is None:
        return 0.0
    return max(0.0, bounds[2] - bounds[0])


def _median(values: list[float], default: float = 0.0) -> float:
    values = [value for value in values if value > 0]
    if not values:
        return default
    return float(median(values))


def _group_rows(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    median_height = _median([_height(item) for item in items], 18.0)
    tolerance = max(8.0, median_height * 0.70)
    rows: list[dict[str, Any]] = []

    for item in sorted(items, key=lambda entry: ((_bounds(entry) or (0, 0, 0, 0))[1], (_bounds(entry) or (0, 0, 0, 0))[0])):
        bounds = _bounds(item)
        if bounds is None:
            continue
        center_y = (bounds[1] + bounds[3]) / 2.0
        target = None
        for row in rows:
            if abs(center_y - row["center_y"]) <= tolerance:
                target = row
                break
        if target is None:
            rows.append({"center_y": center_y, "items": [item]})
        else:
            target["items"].append(item)
            target["center_y"] = mean(
                [((_bounds(entry) or (0, 0, 0, 0))[1] + (_bounds(entry) or (0, 0, 0, 0))[3]) / 2.0 for entry in target["items"]]
            )

    grouped = []
    for row in rows:
        row_items = sorted(row["items"], key=lambda entry: (_bounds(entry) or (0, 0, 0, 0))[0])
        row_bounds = [_bounds(item) for item in row_items]
        row_bounds = [bound for bound in row_bounds if bound is not None]
        grouped.append(
            {
                "center_y": round(float(row["center_y"]), 2),
                "item_count": len(row_items),
                "text": " ".join(_text(item) for item in row_items if _text(item)),
                "items": [{"text": _text(item), "bbox": item.get("bbox"), "confidence": item.get("confidence")} for item in row_items],
                "bounds": [
                    round(min(bound[0] for bound in row_bounds), 2),
                    round(min(bound[1] for bound in row_bounds), 2),
                    round(max(bound[2] for bound in row_bounds), 2),
                    round(max(bound[3] for bound in row_bounds), 2),
                ]
                if row_bounds
                else None,
            }
        )
    return grouped


def _summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    confidences = [float(item["confidence"]) for item in items if isinstance(item.get("confidence"), (int, float))]
    rows = _group_rows(items)
    merged_items = [
        int(item.get("source_item_count", 1))
        for item in items
        if isinstance(item.get("source_item_count", 1), int) and int(item.get("source_item_count", 1)) > 1
    ]
    return {
        "item_count": len(items),
        "row_count": len(rows),
        "median_height": round(_median([_height(item) for item in items], 0.0), 2),
        "median_width": round(_median([_width(item) for item in items], 0.0), 2),
        "confidence_available": bool(confidences),
        "mean_confidence": round(mean(confidences), 4) if confidences else None,
        "low_confidence_count": sum(1 for value in confidences if value < 0.35),
        "merged_item_count": len(merged_items),
        "source_items_merged": sum(merged_items),
        "rows_with_multiple_items": sum(1 for row in rows if row["item_count"] > 1),
    }


def _similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left.lower(), right.lower()).ratio()


def _compare_rows(baseline_rows: list[dict[str, Any]], current_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons = []
    unmatched_current = set(range(len(current_rows)))

    for baseline_row in baseline_rows:
        best_index = None
        best_distance = float("inf")
        for index, current_row in enumerate(current_rows):
            distance = abs(float(baseline_row["center_y"]) - float(current_row["center_y"]))
            if distance < best_distance:
                best_distance = distance
                best_index = index

        if best_index is None or best_distance > 36.0:
            comparisons.append(
                {
                    "baseline_y": baseline_row["center_y"],
                    "current_y": None,
                    "y_delta": None,
                    "baseline_text": baseline_row["text"],
                    "current_text": None,
                    "text_similarity": 0.0,
                    "status": "missing_current_row",
                }
            )
            continue

        current_row = current_rows[best_index]
        unmatched_current.discard(best_index)
        comparisons.append(
            {
                "baseline_y": baseline_row["center_y"],
                "current_y": current_row["center_y"],
                "y_delta": round(best_distance, 2),
                "baseline_item_count": baseline_row["item_count"],
                "current_item_count": current_row["item_count"],
                "baseline_text": baseline_row["text"],
                "current_text": current_row["text"],
                "text_similarity": round(_similarity(baseline_row["text"], current_row["text"]), 4),
                "status": "matched",
            }
        )

    for index in sorted(unmatched_current):
        current_row = current_rows[index]
        comparisons.append(
            {
                "baseline_y": None,
                "current_y": current_row["center_y"],
                "y_delta": None,
                "baseline_text": None,
                "current_text": current_row["text"],
                "text_similarity": 0.0,
                "status": "extra_current_row",
            }
        )

    return comparisons


def _draw_overlay(image_path: Path, baseline: list[dict[str, Any]], current: list[dict[str, Any]], output_path: Path) -> bool:
    image = cv2.imread(str(image_path))
    if image is None:
        return False

    for item in baseline:
        bounds = _bounds(item)
        if bounds is None:
            continue
        x1, y1, x2, y2 = [int(round(value)) for value in bounds]
        cv2.rectangle(image, (x1, y1), (x2, y2), (40, 180, 40), 2)

    for item in current:
        bounds = _bounds(item)
        if bounds is None:
            continue
        x1, y1, x2, y2 = [int(round(value)) for value in bounds]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 140, 255), 2)

    cv2.putText(image, "baseline", (24, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 180, 40), 2, cv2.LINE_AA)
    cv2.putText(image, "current", (24, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), image))


def write_ocr_comparison(
    baseline_path: Path,
    current_path: Path,
    output_dir: Path,
    image_path: Path | None = None,
    name: str = "ocr_comparison",
) -> dict[str, Any]:
    baseline = _read_json(baseline_path)
    current = _read_json(current_path)
    baseline_rows = _group_rows(baseline)
    current_rows = _group_rows(current)

    payload = {
        "baseline_path": str(baseline_path),
        "current_path": str(current_path),
        "baseline_summary": _summarize(baseline),
        "current_summary": _summarize(current),
        "row_comparison": _compare_rows(baseline_rows, current_rows),
        "baseline_rows": baseline_rows,
        "current_rows": current_rows,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / f"{name}.json"
    _write_json(comparison_path, payload)
    payload["comparison_path"] = str(comparison_path)

    if image_path is not None and image_path.exists():
        overlay_path = output_dir / f"{name}_overlay.png"
        payload["overlay_path"] = str(overlay_path) if _draw_overlay(image_path, baseline, current, overlay_path) else None

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare OCR JSON outputs and render bbox diagnostics.")
    parser.add_argument("--baseline", default="output/ocr.json")
    parser.add_argument("--current", default="output/result.json")
    parser.add_argument("--image", default="output/form_page_1.png")
    parser.add_argument("--out-dir", default="output/ocr_diagnostics")
    parser.add_argument("--name", default="ocr_comparison")
    args = parser.parse_args()

    payload = write_ocr_comparison(
        baseline_path=Path(args.baseline),
        current_path=Path(args.current),
        output_dir=Path(args.out_dir),
        image_path=Path(args.image) if args.image else None,
        name=args.name,
    )
    print(json.dumps({key: payload.get(key) for key in ["comparison_path", "overlay_path"]}, indent=2))


if __name__ == "__main__":
    main()
