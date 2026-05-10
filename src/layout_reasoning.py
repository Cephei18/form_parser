import numpy as np


def _median(values, default: float) -> float:
    clean = [float(value) for value in values if isinstance(value, (int, float)) and value > 0]
    if not clean:
        return default
    return float(np.median(clean))


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


def _line_length(line):
    x1, y1 = line["start"]
    x2, y2 = line["end"]
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def _group_by_y(items, y_getter, tolerance):
    groups = []
    for item in sorted(items, key=y_getter):
        y = y_getter(item)
        if not groups or abs(groups[-1]["center_y"] - y) > tolerance:
            groups.append({"center_y": y, "items": [item]})
            continue
        group = groups[-1]
        group["items"].append(item)
        group["center_y"] = sum(y_getter(entry) for entry in group["items"]) / len(group["items"])
    return groups


def infer_layout_structure(ocr_data, field_lines, semantic_regions):
    text_heights = []
    for item in ocr_data:
        bounds = _ocr_bounds(item)
        if bounds is None:
            continue
        _, y1, _, y2 = bounds
        text_heights.append(y2 - y1)

    avg_text_height = _median(text_heights, 18.0)
    avg_line_length = _median([_line_length(line) for line in field_lines], 220.0)
    row_tolerance = max(18.0, avg_text_height * 1.5)

    text_rows = _group_by_y(ocr_data, lambda item: item.get("center", (0, 0))[1], row_tolerance)
    field_rows = _group_by_y(field_lines, lambda line: _line_center(line)[1], row_tolerance)

    field_clusters = []
    for group in field_rows:
        lines = group["items"]
        xs = [_line_center(line)[0] for line in lines]
        ys = [_line_center(line)[1] for line in lines]
        field_clusters.append(
            {
                "center_y": round(group["center_y"], 2),
                "line_count": len(lines),
                "min_x": round(min(xs), 2) if xs else None,
                "max_x": round(max(xs), 2) if xs else None,
                "min_y": round(min(ys), 2) if ys else None,
                "max_y": round(max(ys), 2) if ys else None,
                "avg_line_length": round(_median([_line_length(line) for line in lines], avg_line_length), 2),
                "type": "repeated_row" if len(lines) >= 3 else "field_row",
            }
        )

    return {
        "metrics": {
            "avg_text_height": round(avg_text_height, 2),
            "avg_line_length": round(avg_line_length, 2),
            "row_tolerance": round(row_tolerance, 2),
            "ocr_count": len(ocr_data),
            "field_line_count": len(field_lines),
            "semantic_region_count": len(semantic_regions or []),
        },
        "text_rows": [
            {
                "center_y": round(group["center_y"], 2),
                "item_count": len(group["items"]),
                "texts": [item.get("text", "") for item in group["items"]],
            }
            for group in text_rows
        ],
        "field_clusters": field_clusters,
        "semantic_regions": semantic_regions or [],
    }
