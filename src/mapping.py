import math
import logging

import cv2

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

    return split


def expand_multiline_field(best_line, lines, y_threshold=60, x_tolerance=80, max_group_size=20):
    grouped = [best_line]
    grouped_ids = {id(best_line)}
    queue = [best_line]

    # Keep expansion local so one noisy label cannot absorb the whole page.
    while queue and len(grouped) < max_group_size:
        current = queue.pop(0)
        gx1, gy1 = current["start"]

        for ln in lines:
            line_id = id(ln)
            if line_id in grouped_ids:
                logger.info("[mapping] multiline skip visited line=%s", ln)
                continue

            lx1, ly1 = ln["start"]
            x_aligned = abs(lx1 - gx1) < x_tolerance
            y_close = abs(ly1 - gy1) < y_threshold

            if x_aligned and y_close:
                logger.info("[mapping] multiline add line=%s from=%s", ln, current)
                grouped.append(ln)
                grouped_ids.add(line_id)
                queue.append(ln)

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


def _best_field_match(
    label_center,
    field_lines,
    assigned_lines,
    row_tolerance: int,
    distance_threshold: float,
):
    closest_line = None
    min_dist = float("inf")
    candidate_count = 0
    rejected_assigned = 0
    rejected_alignment = 0

    for line in field_lines:
        candidate_center = line_center(line)

        if id(line) in assigned_lines:
            rejected_assigned += 1
            continue

        if abs(candidate_center[1] - label_center[1]) >= row_tolerance:
            rejected_alignment += 1
            continue

        if candidate_center[0] <= label_center[0]:
            rejected_alignment += 1
            continue

        candidate_count += 1
        dist = distance(label_center, candidate_center)

        if dist < min_dist:
            min_dist = dist
            closest_line = line

    accepted = closest_line is not None and min_dist <= distance_threshold
    return {
        "closest_line": closest_line,
        "min_dist": min_dist,
        "candidate_count": candidate_count,
        "rejected_assigned": rejected_assigned,
        "rejected_alignment": rejected_alignment,
        "accepted": accepted,
    }


def _match_label_with_passes(
    label,
    label_center,
    field_lines,
    assigned_lines,
    table_regions,
    strict_row_tolerance: int,
    strict_distance_threshold: float,
    relaxed_row_tolerance: int,
    relaxed_distance_threshold: float,
):
    def is_in_table(line):
        for region in table_regions:
            if line in region:
                return True
        return False

    passes = [
        ("strict", strict_row_tolerance, strict_distance_threshold),
        ("fallback", relaxed_row_tolerance, relaxed_distance_threshold),
    ]

    for pass_name, row_tolerance, distance_threshold in passes:
        y_min = label_center[1] - row_tolerance
        y_max = label_center[1] + row_tolerance
        row_lines = lines_in_row_window(field_lines, y_min, y_max)

        logger.info(
            "[mapping] label pass=%s text=%r row_tolerance=%s row_window_candidates=%s distance_threshold=%s",
            pass_name,
            label,
            row_tolerance,
            len(row_lines),
            distance_threshold,
        )

        if is_table_row(row_lines):
            logger.info(
                "[mapping] label pass=%s reject reason=table_row text=%r candidates=%s threshold=%s",
                pass_name,
                label,
                len(row_lines),
                10,
            )
            continue

        selection = _best_field_match(
            label_center,
            field_lines,
            assigned_lines,
            row_tolerance=row_tolerance,
            distance_threshold=distance_threshold,
        )

        logger.info(
            "[mapping] label pass=%s text=%r candidates=%s rejected_assigned=%s rejected_alignment=%s best_line=%s min_dist=%s accepted=%s",
            pass_name,
            label,
            selection["candidate_count"],
            selection["rejected_assigned"],
            selection["rejected_alignment"],
            selection["closest_line"],
            f"{selection['min_dist']:.2f}" if selection["closest_line"] is not None else "inf",
            selection["accepted"],
        )

        if not selection["accepted"]:
            if selection["closest_line"] is None:
                logger.info("[mapping] label pass=%s reject reason=no_candidate text=%r", pass_name, label)
            else:
                logger.info(
                    "[mapping] label pass=%s reject reason=distance text=%r best_line=%s min_dist=%.2f threshold=%.2f",
                    pass_name,
                    label,
                    selection["closest_line"],
                    selection["min_dist"],
                    distance_threshold,
                )
            continue

        best_line = selection["closest_line"]
        if is_in_table(best_line):
            logger.info(
                "[mapping] label pass=%s reject reason=table_region text=%r best_line=%s",
                pass_name,
                label,
                best_line,
            )
            continue

        return {
            "pass_name": pass_name,
            "best_line": best_line,
            "min_dist": selection["min_dist"],
            "row_tolerance": row_tolerance,
            "distance_threshold": distance_threshold,
        }

    return None


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

    for label_index, item in enumerate(labels, start=1):
        label = item["text"]
        label_center = item["center"]
        logger.info(
            "[mapping] label start index=%s text=%r center=%s",
            label_index,
            label,
            label_center,
        )

        match = _match_label_with_passes(
            label,
            label_center,
            field_lines,
            assigned_lines,
            table_regions,
            strict_row_tolerance=row_tolerance,
            strict_distance_threshold=650,
            relaxed_row_tolerance=max(row_tolerance + 12, 48),
            relaxed_distance_threshold=900,
        )

        if match is None:
            logger.info("[mapping] label final reject text=%r reason=no_match", label)
            continue

        # expand into multiline group
        grouped = expand_multiline_field(match["best_line"], field_lines)
        logger.info(
            "[mapping] label selected pass=%s label=%r best_line=%s grouped=%s min_dist=%.2f",
            match["pass_name"],
            label,
            match["best_line"],
            len(grouped),
            match["min_dist"],
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
                "distance": match["min_dist"],
                "column_split": column_split,
            }
        )

        logger.info(
            "[mapping] label final accepted pass=%s label=%r mapping_count=%s assigned_total=%s",
            match["pass_name"],
            label,
            len(mappings),
            len(assigned_lines),
        )

    return mappings


def draw_mapping(image_path, mappings, output_path):
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    for mapping in mappings:
        label_x, label_y = mapping["label_pos"]

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
