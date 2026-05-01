import math

import cv2


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


def expand_multiline_field(best_line, lines, y_threshold=60, x_tolerance=80):
    grouped = [best_line]

    added = True
    while added:
        added = False

        for ln in lines:
            if ln in grouped:
                continue

            for g in grouped:
                gx1, gy1 = g["start"]
                lx1, ly1 = ln["start"]

                x_aligned = abs(lx1 - gx1) < x_tolerance
                y_close = abs(ly1 - gy1) < y_threshold

                if x_aligned and y_close:
                    grouped.append(ln)
                    added = True

    return sorted(grouped, key=lambda l: l["start"][1])


def lines_in_row_window(lines, y_min, y_max):
    out = []
    for l in lines:
        _, y = line_center(l)
        if y_min <= y <= y_max:
            out.append(l)
    return out


def is_table_row(row_lines, threshold=8):
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


def map_labels_to_fields(ocr_data, lines, row_tolerance: int = 25, fallback_split: int = 300):
    mappings = []
    assigned_lines = set()

    column_split = infer_column_split(ocr_data, fallback_split=fallback_split)

    labels = [item for item in ocr_data if item["center"][0] < column_split]
    field_lines = [line for line in lines if line_center(line)[0] > column_split]

    # detect table-like regions among candidate field lines
    table_regions = detect_table_regions(field_lines)

    def is_in_table(line):
        for region in table_regions:
            if line in region:
                return True
        return False

    for item in labels:
        label = item["text"]
        label_center = item["center"]

        closest_line = None
        min_dist = float("inf")

        # avoid mapping rows that clearly look like table rows
        y_min = label_center[1] - row_tolerance
        y_max = label_center[1] + row_tolerance
        row_lines = lines_in_row_window(field_lines, y_min, y_max)

        if is_table_row(row_lines):
            continue

        for line in field_lines:
            candidate_center = line_center(line)

            if id(line) in assigned_lines:
                continue

            # For this form style, fields are on the same row and to the right of labels.
            if (
                abs(candidate_center[1] - label_center[1]) < row_tolerance
                and candidate_center[0] > label_center[0]
            ):
                dist = distance(label_center, candidate_center)

                if dist < min_dist:
                    min_dist = dist
                    closest_line = line


        # safety: skip weak matches or no match
        if closest_line is None or min_dist > 500:
            continue

        # skip if this line is inside a detected table region
        if is_in_table(closest_line):
            continue

        # expand into multiline group
        grouped = expand_multiline_field(closest_line, field_lines)

        # mark all grouped lines as assigned
        for g in grouped:
            assigned_lines.add(id(g))

        mappings.append(
            {
                "label": label,
                "label_pos": label_center,
                "field_lines": grouped,
                "distance": min_dist,
                "column_split": column_split,
            }
        )

    return mappings


def draw_mapping(image_path, mappings, output_path):
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    for mapping in mappings:
        label_x, label_y = mapping["label_pos"]

        # draw to the first field line center for visualization
        if mapping.get("field_lines"):
            x1, y1 = mapping["field_lines"][0]["start"]
            x2, y2 = mapping["field_lines"][0]["end"]

            field_x = (x1 + x2) // 2
            field_y = (y1 + y2) // 2

            cv2.line(
                img,
                (int(label_x), int(label_y)),
                (int(field_x), int(field_y)),
                (0, 0, 255),
                2,
            )

    cv2.imwrite(output_path, img)
