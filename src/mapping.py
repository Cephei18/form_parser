import math
import logging

import cv2
import numpy as np

logger = logging.getLogger("form_parser.mapping")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def line_center(line):
    x1, y1 = line["start"]
    x2, y2 = line["end"]
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def line_length(line) -> float:
    x1, y1 = line["start"]
    x2, y2 = line["end"]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def line_bounds(line):
    x1, y1 = line["start"]
    x2, y2 = line["end"]
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def ocr_bounds(item):
    bbox = item.get("bbox")
    if not bbox:
        return None
    try:
        xs = [float(point[0]) for point in bbox]
        ys = [float(point[1]) for point in bbox]
    except Exception:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _safe_confidence(item) -> float:
    confidence = item.get("confidence")
    if isinstance(confidence, (int, float)):
        return _clamp01(float(confidence))
    return 0.75


def _median(values, default: float) -> float:
    clean = [float(v) for v in values if isinstance(v, (int, float)) and v > 0]
    if not clean:
        return default
    return float(np.median(clean))


def build_layout_metrics(ocr_data, field_lines):
    line_heights = []
    text_widths = []
    for item in ocr_data:
        bounds = ocr_bounds(item)
        if bounds is None:
            continue
        x1, y1, x2, y2 = bounds
        line_heights.append(y2 - y1)
        text_widths.append(x2 - x1)

    line_lengths = [line_length(line) for line in field_lines]
    avg_line_height = _median(line_heights, 18.0)
    avg_line_width = _median(line_lengths, 220.0)
    ocr_density = len(ocr_data) / max(1.0, len(field_lines) or 1.0)

    return {
        "avg_line_height": avg_line_height,
        "avg_line_width": avg_line_width,
        "avg_text_width": _median(text_widths, 80.0),
        "row_tolerance": max(28.0, avg_line_height * 2.2),
        "relaxed_row_tolerance": max(42.0, avg_line_height * 3.4),
        "distance_limit": max(420.0, avg_line_width * 3.2),
        "multiline_y_gap": max(34.0, avg_line_height * 2.6),
        "indent_tolerance": max(42.0, avg_line_height * 3.0),
        "ocr_density": ocr_density,
    }


def _semantic_hint_score(label: str, line) -> tuple[float, list[str]]:
    source = line.get("field_type", "line")
    score = 0.55
    reasons = []

    if source in {"box", "weak_line", "fallback_text_region"}:
        score += 0.06
        reasons.append(f"field_source_{source}")
    if line_length(line) <= 260:
        score += 0.04
        reasons.append("compact_field_candidate")

    return _clamp01(score), reasons


def _row_overlap_score(label_item, line, metrics) -> tuple[float, list[str]]:
    bounds = ocr_bounds(label_item)
    _, line_y = line_center(line)
    if bounds is None:
        return 0.5, ["missing_label_bounds"]

    _, y1, _, y2 = bounds
    if y1 <= line_y <= y2 + metrics["avg_line_height"] * 0.8:
        return 1.0, ["row_overlap"]

    dy = min(abs(line_y - y1), abs(line_y - y2))
    score = 1.0 - (dy / max(metrics["relaxed_row_tolerance"], 1.0))
    return _clamp01(score), ["row_nearby" if score >= 0.45 else "row_separated"]


def score_candidate(label_item, line, assigned_lines, metrics, table_regions):
    label_center = label_item["center"]
    candidate_center = line_center(line)
    dx = candidate_center[0] - label_center[0]
    dy = abs(candidate_center[1] - label_center[1])
    reasons = []
    rejection_reasons = []

    if id(line) in assigned_lines:
        rejection_reasons.append("already_assigned")
    if dx <= 0:
        rejection_reasons.append("not_right_of_label")
    if any(line in region for region in table_regions):
        rejection_reasons.append("inside_table_region")

    horizontal_score = _clamp01(dx / max(metrics["avg_text_width"] * 1.5, 80.0))
    if dx > metrics["distance_limit"]:
        horizontal_score *= 0.55
        reasons.append("far_horizontal_distance")
    elif dx > 0:
        reasons.append("right_of_label")

    vertical_score = _clamp01(1.0 - (dy / max(metrics["relaxed_row_tolerance"], 1.0)))
    if vertical_score >= 0.72:
        reasons.append("strong_vertical_alignment")
    elif vertical_score >= 0.38:
        reasons.append("weak_vertical_alignment")
    else:
        rejection_reasons.append("poor_vertical_alignment")

    row_score, row_reasons = _row_overlap_score(label_item, line, metrics)
    reasons.extend(row_reasons)

    width = line_length(line)
    field_size_score = _clamp01(1.0 - abs(width - metrics["avg_line_width"]) / max(metrics["avg_line_width"] * 1.6, 1.0))
    if field_size_score >= 0.6:
        reasons.append("field_size_consistent")
    else:
        reasons.append("field_size_outlier")

    semantic_score, semantic_reasons = _semantic_hint_score(label_item.get("text", ""), line)
    reasons.extend(semantic_reasons)

    ocr_score = _safe_confidence(label_item)
    if ocr_score < 0.55:
        reasons.append("low_ocr_confidence")

    candidate_score = (
        horizontal_score * 0.18
        + vertical_score * 0.26
        + row_score * 0.18
        + field_size_score * 0.12
        + semantic_score * 0.14
        + ocr_score * 0.12
    )

    if rejection_reasons:
        candidate_score *= 0.35

    score_breakdown = {
        "horizontal_distance": round(horizontal_score, 4),
        "vertical_alignment": round(vertical_score, 4),
        "row_overlap": round(row_score, 4),
        "field_size_consistency": round(field_size_score, 4),
        "semantic_hints": round(semantic_score, 4),
        "ocr_confidence": round(ocr_score, 4),
    }

    return {
        "line": line,
        "candidate_score": round(candidate_score, 4),
        "confidence": round(_clamp01(candidate_score), 4),
        "reasons": reasons,
        "rejection_reasons": rejection_reasons,
        "score_breakdown": score_breakdown,
        "distance": round(distance(label_center, candidate_center), 2),
        "dx": round(dx, 2),
        "dy": round(dy, 2),
    }


def classify_confidence(best, runner_up=None) -> tuple[str, list[str]]:
    if best is None:
        return "unresolved", ["no_candidate"]

    score = best["candidate_score"]
    margin = score - (runner_up["candidate_score"] if runner_up else 0.0)
    notes = []

    if runner_up and margin < 0.08:
        notes.append("close_second_candidate")
        return "ambiguous", notes
    if best["rejection_reasons"]:
        notes.append("selected_candidate_has_rejections")
        return "weak_match", notes
    if score >= 0.72 and margin >= 0.12:
        return "strong_match", notes
    if score >= 0.48:
        return "weak_match", notes
    return "unresolved", ["candidate_score_below_threshold"]


def infer_column_split(ocr_data, fallback_split: int = 300):
    xs = sorted(item["center"][0] for item in ocr_data if "center" in item)
    if len(xs) < 2:
        return fallback_split

    max_gap = 0
    split = fallback_split
    for left, right in zip(xs, xs[1:]):
        gap = right - left
        if gap > max_gap:
            max_gap = gap
            split = (left + right) / 2

    if split < fallback_split * 0.75:
        return fallback_split

    return split


def _intervening_label_exists(current_y, candidate_y, ocr_data, column_split, avg_line_height):
    upper = min(current_y, candidate_y) + avg_line_height * 0.35
    lower = max(current_y, candidate_y) - avg_line_height * 0.35
    if lower <= upper:
        return False

    for item in ocr_data or []:
        center = item.get("center")
        text = item.get("text")
        if not center or not text:
            continue
        if center[0] >= column_split:
            continue
        if upper <= center[1] <= lower:
            return True
    return False


def _candidate_row_has_label(candidate_y, ocr_data, column_split, avg_line_height):
    for item in ocr_data or []:
        center = item.get("center")
        text = item.get("text")
        if not center or not text:
            continue
        if center[0] >= column_split:
            continue
        if abs(center[1] - candidate_y) <= avg_line_height * 1.15:
            return True
    return False


def expand_multiline_field(
    best_line,
    lines,
    y_threshold=60,
    x_tolerance=80,
    max_group_size=20,
    max_depth=4,
    assigned_lines=None,
    ocr_data=None,
    column_split=300,
    avg_line_height=18.0,
):
    grouped = [best_line]
    grouped_ids = {id(best_line)}
    queue = [(best_line, 0)]
    assigned_lines = assigned_lines or set()

    # Keep expansion local so one noisy label cannot absorb the whole page.
    while queue and len(grouped) < max_group_size:
        current, depth = queue.pop(0)
        if depth >= max_depth:
            logger.info("[mapping] multiline stop max_depth=%s current=%s", max_depth, current)
            continue
        gx1, gy1 = current["start"]
        _, current_y = line_center(current)

        for ln in sorted(lines, key=lambda item: (line_center(item)[1], line_center(item)[0])):
            line_id = id(ln)
            if line_id in grouped_ids:
                logger.info("[mapping] multiline skip visited line=%s", ln)
                continue
            if line_id in assigned_lines:
                logger.info("[mapping] multiline skip owned line=%s", ln)
                continue

            lx1, ly1 = ln["start"]
            _, candidate_y = line_center(ln)
            x_aligned = abs(lx1 - gx1) < x_tolerance
            y_gap = candidate_y - current_y
            y_close = 0 < y_gap <= y_threshold
            length_similar = abs(line_length(ln) - line_length(best_line)) <= max(line_length(best_line) * 0.45, 80)
            blocked_by_label = _intervening_label_exists(
                current_y,
                candidate_y,
                ocr_data,
                column_split,
                avg_line_height,
            ) or _candidate_row_has_label(candidate_y, ocr_data, column_split, avg_line_height)

            if x_aligned and y_close and length_similar and not blocked_by_label:
                logger.info("[mapping] multiline add line=%s from=%s", ln, current)
                grouped.append(ln)
                grouped_ids.add(line_id)
                queue.append((ln, depth + 1))
            elif blocked_by_label:
                logger.info("[mapping] multiline stop intervening_label current=%s candidate=%s", current, ln)

                if len(grouped) >= max_group_size:
                    logger.info(
                        "[mapping] multiline stop max_group_size=%s best_line=%s",
                        max_group_size,
                        best_line,
                    )
                    break

    return sorted(grouped, key=lambda l: l["start"][1])


def lines_in_row_window(lines, y_min, y_max):
    out = []
    for l in lines:
        _, y = line_center(l)
        if y_min <= y <= y_max:
            out.append(l)
    return out


def is_table_row(row_lines, threshold=10):
    return len(row_lines) >= threshold


def detect_table_regions(lines, y_threshold=15, min_lines=5):
    """Detect vertically dense regions of lines (likely tables)."""
    regions = []

    if not lines:
        return regions

    lines_sorted = sorted(lines, key=lambda l: l["start"][1])

    current_group = [lines_sorted[0]]

    for i in range(1, len(lines_sorted)):
        prev = lines_sorted[i - 1]
        curr = lines_sorted[i]

        if abs(curr["start"][1] - prev["start"][1]) < y_threshold:
            current_group.append(curr)
        else:
            if len(current_group) >= min_lines:
                regions.append(current_group)
            current_group = [curr]

    if len(current_group) >= min_lines:
        regions.append(current_group)

    return regions


def _select_weighted_candidate(
    label_item,
    field_lines,
    assigned_lines,
    table_regions,
    metrics,
):
    label = label_item.get("text", "")
    label_center = label_item["center"]
    y_min = label_center[1] - metrics["relaxed_row_tolerance"]
    y_max = label_center[1] + metrics["relaxed_row_tolerance"]
    row_lines = lines_in_row_window(field_lines, y_min, y_max)

    logger.info(
        "[mapping] label=%r row_window_candidates=%s row_tolerance=%.1f",
        label,
        len(row_lines),
        metrics["relaxed_row_tolerance"],
    )

    candidates = [
        score_candidate(label_item, line, assigned_lines, metrics, table_regions)
        for line in field_lines
    ]
    candidates.sort(key=lambda item: item["candidate_score"], reverse=True)

    best = candidates[0] if candidates else None
    runner_up = candidates[1] if len(candidates) > 1 else None
    confidence_class, ambiguity_notes = classify_confidence(best, runner_up)

    for idx, candidate in enumerate(candidates[:5], start=1):
        logger.info(
            "[mapping] candidate label=%r rank=%s score=%.4f confidence=%.4f dx=%.1f dy=%.1f rejects=%s reasons=%s breakdown=%s",
            label,
            idx,
            candidate["candidate_score"],
            candidate["confidence"],
            candidate["dx"],
            candidate["dy"],
            candidate["rejection_reasons"],
            candidate["reasons"],
            candidate["score_breakdown"],
        )

    selected = best if confidence_class != "unresolved" else None
    if selected is None:
        logger.info(
            "[mapping] label unresolved text=%r candidates=%s notes=%s",
            label,
            len(candidates),
            ambiguity_notes,
        )

    return {
        "selected": selected,
        "candidate_count": len(candidates),
        "confidence_class": confidence_class,
        "ambiguity_notes": ambiguity_notes,
        "top_candidates": candidates[:5],
    }


def map_labels_to_fields(ocr_data, lines, row_tolerance: int = 36, fallback_split: int = 300):
    mappings = []
    assigned_lines = set()

    column_split = infer_column_split(ocr_data, fallback_split=fallback_split)

    labels = [item for item in ocr_data if item["center"][0] < column_split]
    field_lines = [line for line in lines if line_center(line)[0] > column_split]

    logger.info(
        "[mapping] column_split=%s labels=%s field_lines=%s",
        column_split,
        len(labels),
        len(field_lines),
    )

    # detect table-like regions among candidate field lines
    table_regions = detect_table_regions(field_lines)
    logger.info("[mapping] table_regions=%s", len(table_regions))
    metrics = build_layout_metrics(ocr_data, field_lines)
    logger.info("[mapping] dynamic_metrics=%s", metrics)

    for label_index, item in enumerate(labels, start=1):
        label = item["text"]
        label_center = item["center"]
        logger.info(
            "[mapping] label start index=%s text=%r center=%s",
            label_index,
            label,
            label_center,
        )

        selection = _select_weighted_candidate(
            item,
            field_lines,
            assigned_lines,
            table_regions,
            metrics,
        )

        match = selection["selected"]
        if match is None:
            logger.info(
                "[mapping] label final unresolved text=%r candidate_count=%s classification=%s notes=%s",
                label,
                selection["candidate_count"],
                selection["confidence_class"],
                selection["ambiguity_notes"],
            )
            continue

        # expand into multiline group
        grouped = expand_multiline_field(
            match["line"],
            field_lines,
            y_threshold=metrics["multiline_y_gap"],
            x_tolerance=metrics["indent_tolerance"],
            max_group_size=8,
            assigned_lines=assigned_lines,
            ocr_data=labels,
            column_split=column_split,
            avg_line_height=metrics["avg_line_height"],
        )
        logger.info(
            "[mapping] label selected label=%r best_line=%s grouped=%s score=%.4f class=%s reasons=%s",
            label,
            match["line"],
            len(grouped),
            match["candidate_score"],
            selection["confidence_class"],
            match["reasons"],
        )

        # mark all grouped lines as assigned
        for g in grouped:
            assigned_lines.add(id(g))

        logger.info(
            "[mapping] label assigned label=%r assigned_total=%s",
            label,
            len(assigned_lines),
        )

        mappings.append(
            {
                "label": label,
                "label_pos": label_center,
                "field_lines": grouped,
                "field_type": grouped[0].get("field_type", "line") if grouped else "line",
                "distance": match["distance"],
                "column_split": column_split,
                "candidate_score": match["candidate_score"],
                "confidence": match["confidence"],
                "confidence_class": selection["confidence_class"],
                "reasons": match["reasons"],
                "score_breakdown": match["score_breakdown"],
                "candidate_count": selection["candidate_count"],
                "ambiguity_notes": selection["ambiguity_notes"],
                "rejected_candidates": [
                    {
                        "line": candidate["line"],
                        "candidate_score": candidate["candidate_score"],
                        "rejection_reasons": candidate["rejection_reasons"],
                        "score_breakdown": candidate["score_breakdown"],
                    }
                    for candidate in selection["top_candidates"]
                    if candidate["line"] is not match["line"]
                ],
                "multiline_group_size": len(grouped),
            }
        )

        logger.info(
            "[mapping] label final accepted class=%s label=%r mapping_count=%s assigned_total=%s",
            selection["confidence_class"],
            label,
            len(mappings),
            len(assigned_lines),
        )

    return mappings


def _draw_text(img, text, point, color):
    cv2.putText(
        img,
        str(text)[:80],
        (int(point[0]), int(point[1])),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_mapping(image_path, mappings, output_path, ocr_data=None, candidate_lines=None, semantic_regions=None):
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    for item in ocr_data or []:
        bounds = ocr_bounds(item)
        if bounds is None:
            continue
        x1, y1, x2, y2 = bounds
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 180, 0), 1)
        _draw_text(img, item.get("text", ""), (x1, max(10, y1 - 4)), (120, 90, 0))

    for line in candidate_lines or []:
        x1, y1 = line["start"]
        x2, y2 = line["end"]
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (160, 160, 160), 1)

    for region in semantic_regions or []:
        x = int(region["x"])
        y = int(region["y"])
        w = int(region["width"])
        h = int(region["height"])
        region_type = region.get("type", "region")
        confidence = float(region.get("confidence", 0.0))
        color = (180, 180, 180)
        if region_type in {"non_text_sparse_region", "non_text_candidate"}:
            color = (0, 120, 255)
        elif region_type == "multiline_text_region":
            color = (180, 0, 180)
        elif region_type == "standard_input":
            color = (80, 180, 80)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        _draw_text(img, f"{region_type} {confidence:.2f}", (x, max(12, y - 6)), color)

    for mapping in mappings:
        label_x, label_y = mapping["label_pos"]
        confidence_class = mapping.get("confidence_class", "unknown")
        score = mapping.get("candidate_score", 0)

        for line in mapping.get("field_lines", []):
            x1, y1 = line["start"]
            x2, y2 = line["end"]

            field_x = (x1 + x2) // 2
            field_y = (y1 + y2) // 2

            cv2.line(
                img,
                (int(label_x), int(label_y)),
                (int(field_x), int(field_y)),
                (0, 0, 255),
                2,
            )
            cv2.rectangle(
                img,
                (int(min(x1, x2)), int(min(y1, y2) - 8)),
                (int(max(x1, x2)), int(max(y1, y2) + 12)),
                (0, 200, 0),
                2,
            )
            _draw_text(
                img,
                f"{confidence_class} {score:.2f}",
                (min(x1, x2), min(y1, y2) - 12),
                (0, 120, 0),
            )

        for rejected in mapping.get("rejected_candidates", []):
            line = rejected.get("line")
            if not line:
                continue
            x1, y1 = line["start"]
            x2, y2 = line["end"]
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 140, 255), 1)
            _draw_text(
                img,
                f"reject {rejected.get('candidate_score', 0):.2f}",
                (min(x1, x2), max(y1, y2) + 14),
                (0, 100, 180),
            )

        if mapping.get("field_type") == "checkbox":
            for box in mapping.get("field_bboxes", []):
                x = int(box["x"])
                y = int(box["y"])
                w = int(box["width"])
                h = int(box["height"])
                field_x = x + w // 2
                field_y = y + h // 2
                cv2.line(
                    img,
                    (int(label_x), int(label_y)),
                    (int(field_x), int(field_y)),
                    (0, 0, 255),
                    2,
                )
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 180, 255), 2)

    cv2.imwrite(output_path, img)
