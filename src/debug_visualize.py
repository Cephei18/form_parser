import json
from pathlib import Path

import cv2
import numpy as np


def resolve_input_image(project_root: Path) -> Path:
    input_dir = project_root / "input"
    candidates = [
        input_dir / "form.png",
        input_dir / "image.png",
        input_dir / "imange.png",
        project_root / "output" / "form_page_1.png",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError("No input image found for debug overlay")


def _draw_text(img, text, point, color):
    cv2.putText(
        img,
        str(text)[:90],
        (int(point[0]), int(point[1])),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        color,
        1,
        cv2.LINE_AA,
    )


def _ocr_bounds(item):
    bbox = item.get("bbox")
    if not bbox:
        return None
    try:
        xs = [float(point[0]) for point in bbox]
        ys = [float(point[1]) for point in bbox]
    except Exception:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _line_center(line):
    x1, y1 = line["start"]
    x2, y2 = line["end"]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def create_debug_overlay(project_root: Path, output_path: Path | None = None) -> Path:
    output_dir = project_root / "output"
    output_path = output_path or output_dir / "debug_reasoning.png"

    result_data = _load_json(output_dir / "result.json", [])
    mappings = _load_json(output_dir / "mappings.json", [])
    diagnostics = _load_json(output_dir / "mapping_diagnostics.json", {})
    layout_structure = _load_json(output_dir / "layout_structure.json", {})

    image_path = resolve_input_image(project_root)
    img = cv2.imread(str(image_path))

    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    for item in result_data:
        bounds = _ocr_bounds(item)
        if bounds is None:
            continue
        x1, y1, x2, y2 = bounds
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (120, 160, 255), 1)
        _draw_text(img, item.get("text", ""), (x1, max(10, y1 - 3)), (90, 120, 255))

    for region in diagnostics.get("semantic_regions", []) or layout_structure.get("semantic_regions", []):
        x = int(region.get("x", 0))
        y = int(region.get("y", 0))
        w = int(region.get("width", 0))
        h = int(region.get("height", 0))
        region_type = region.get("type", "region")
        color = (160, 160, 160)
        if region_type == "multiline_text_region":
            color = (170, 0, 170)
        elif region_type in {"photo_region", "non_text_sparse_region", "non_text_candidate"}:
            color = (0, 120, 255)
        elif region_type == "signature_area":
            color = (200, 0, 180)
        elif region_type == "table_like_region":
            color = (0, 180, 180)
        elif region_type == "checkbox_region":
            color = (0, 180, 255)
        elif region_type == "standard_input":
            color = (80, 180, 80)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        _draw_text(img, f"{region_type} {float(region.get('confidence', 0.0)):.2f}", (x, max(12, y - 6)), color)

    for column in layout_structure.get("label_columns", []):
        bounds = column.get("bounds")
        if not bounds or len(bounds) != 4:
            continue
        x1, y1, x2, y2 = bounds
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 160, 255), 1)
        _draw_text(img, f"label_col {column.get('index', 0)}", (x1, max(12, y1 - 6)), (0, 160, 255))

    for column in layout_structure.get("field_columns", []):
        bounds = column.get("bounds")
        if not bounds or len(bounds) != 4:
            continue
        x1, y1, x2, y2 = bounds
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 120), 1)
        _draw_text(img, f"field_col {column.get('index', 0)}", (x1, max(12, y1 - 6)), (0, 200, 120))

    for band in layout_structure.get("layout_bands", []):
        bounds = band.get("bounds")
        if not bounds or len(bounds) != 4:
            continue
        x1, y1, x2, y2 = bounds
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (160, 160, 0), 1)
        _draw_text(img, f"band {band.get('type', 'band')}", (x1, max(12, y1 - 6)), (160, 160, 0))

    for zone in layout_structure.get("structural_zones", []):
        bounds = zone.get("bounds")
        if not bounds or len(bounds) != 4:
            continue
        x1, y1, x2, y2 = bounds
        color = (200, 80, 0) if zone.get("zone_type") == "label" else (80, 180, 60)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
        _draw_text(img, zone.get("type", "zone"), (x1, max(12, y1 - 6)), color)

    for region in layout_structure.get("excluded_regions", []):
        bounds = region.get("bounds")
        if not bounds or len(bounds) != 4:
            continue
        x1, y1, x2, y2 = bounds
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        _draw_text(img, f"exclude {region.get('type', 'region')}", (x1, max(12, y1 - 6)), (0, 0, 255))

    for region in layout_structure.get("ownership_regions", []):
        bounds = region.get("bounds")
        if not bounds or len(bounds) != 4:
            continue
        x1, y1, x2, y2 = bounds
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (180, 0, 180), 2)
        _draw_text(img, region.get("type", "ownership"), (x1, max(12, y1 - 6)), (180, 0, 180))

    for mapping in mappings:
        label_x, label_y = mapping.get("label_pos", (0, 0))
        state = mapping.get("state", mapping.get("confidence_class", "unknown"))
        score = float(mapping.get("candidate_score", 0.0))

        for line in mapping.get("field_lines", []):
            x1, y1 = line["start"]
            x2, y2 = line["end"]
            field_x, field_y = _line_center(line)
            cv2.line(img, (int(label_x), int(label_y)), (int(field_x), int(field_y)), (0, 0, 255), 2)
            cv2.rectangle(img, (int(min(x1, x2)), int(min(y1, y2) - 8)), (int(max(x1, x2)), int(max(y1, y2) + 10)), (0, 200, 0), 2)
            _draw_text(img, f"{state} {score:.2f}", (min(x1, x2), min(y1, y2) - 12), (0, 120, 0))

        for rejected in mapping.get("rejected_candidates", []):
            line = rejected.get("line")
            if not line:
                continue
            x1, y1 = line["start"]
            x2, y2 = line["end"]
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 140, 255), 1)
            _draw_text(img, f"reject {float(rejected.get('candidate_score', 0.0)):.2f}", (min(x1, x2), max(y1, y2) + 14), (0, 100, 180))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    return output_path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_path = create_debug_overlay(project_root)
    print(f"Saved debug image: {output_path}")


if __name__ == "__main__":
    main()
