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


def map_labels_to_fields(ocr_data, lines, row_tolerance: int = 25, fallback_split: int = 300):
    mappings = []
    assigned_lines = set()

    column_split = infer_column_split(ocr_data, fallback_split=fallback_split)

    labels = [item for item in ocr_data if item["center"][0] < column_split]
    field_lines = [line for line in lines if line_center(line)[0] > column_split]

    for item in labels:
        label = item["text"]
        label_center = item["center"]

        closest_line = None
        min_dist = float("inf")

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

        if closest_line:
            assigned_lines.add(id(closest_line))
            mappings.append(
                {
                    "label": label,
                    "label_pos": label_center,
                    "field_line": closest_line,
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

        x1, y1 = mapping["field_line"]["start"]
        x2, y2 = mapping["field_line"]["end"]

        field_x = (x1 + x2) // 2
        field_y = (y1 + y2) // 2

        # Draw the explicit label-to-field relationship.
        cv2.line(
            img,
            (int(label_x), int(label_y)),
            (int(field_x), int(field_y)),
            (0, 0, 255),
            2,
        )

    cv2.imwrite(output_path, img)
