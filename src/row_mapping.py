from typing import Any


def row_label_text(row: list[dict[str, Any]]) -> str:
    """Builds label text from the full merged row."""
    return " ".join(item["text"] for item in row).strip()


def row_y_bounds(row: list[dict[str, Any]], pad: int = 20) -> tuple[float, float]:
    """Returns vertical min/max bounds for a row with optional padding."""
    ys = [item["center"][1] for item in row]
    return min(ys) - pad, max(ys) + pad


def lines_in_row_window(
    lines: list[dict[str, Any]], y_min: float, y_max: float
) -> list[dict[str, Any]]:
    """Keeps only lines whose center Y falls within the row window."""
    out: list[dict[str, Any]] = []
    for line in lines:
        y = (line["start"][1] + line["end"][1]) / 2
        if y_min <= y <= y_max:
            out.append(line)
    return out


def score_line(line: dict[str, Any], label_center: tuple[float, float]) -> float:
    """Scores candidate lines by row alignment, right-side distance, and line width."""
    x1, y1 = line["start"]
    x2, y2 = line["end"]

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    width = abs(x2 - x1)

    vertical_diff = abs(cy - label_center[1])
    horizontal_diff = cx - label_center[0]

    if horizontal_diff <= 0:
        return float("inf")

    return vertical_diff * 5 + horizontal_diff - width * 0.5


def is_right_of_row(
    line: dict[str, Any], row: list[dict[str, Any]], min_gap: int = 40
) -> bool:
    """Keeps lines that are clearly to the right of the row text extent."""
    row_right = max(item["center"][0] for item in row)
    line_center_x = (line["start"][0] + line["end"][0]) / 2
    return line_center_x > (row_right + min_gap)


def map_rows_to_fields(
    rows: list[list[dict[str, Any]]],
    lines: list[dict[str, Any]],
    row_pad: int = 25,
) -> list[dict[str, Any]]:
    """Maps each merged row to its most likely field line."""
    mappings: list[dict[str, Any]] = []
    noise_keywords = ["snb", "jspcb", "affix", "contd"]

    for row in rows:
        if not row:
            continue

        label_text = row_label_text(row)
        low = label_text.lower()
        if any(keyword in low for keyword in noise_keywords):
            continue

        label_center = (
            sum(item["center"][0] for item in row) / len(row),
            sum(item["center"][1] for item in row) / len(row),
        )

        y_min, y_max = row_y_bounds(row, pad=row_pad)
        candidates = lines_in_row_window(lines, y_min, y_max)

        # detect table regions once per mapping run (lines are full set)
        # (compute outside the loop would be slightly more efficient; kept here for clarity)
        table_regions = []
        if lines:
            lines_sorted = sorted(lines, key=lambda l: l["start"][1])
            current_group = [lines_sorted[0]]
            for i in range(1, len(lines_sorted)):
                prev = lines_sorted[i - 1]
                curr = lines_sorted[i]
                if abs(curr["start"][1] - prev["start"][1]) < 15:
                    current_group.append(curr)
                else:
                    if len(current_group) >= 5:
                        table_regions.append(current_group)
                    current_group = [curr]
            if len(current_group) >= 5:
                table_regions.append(current_group)

        def is_in_table(line):
            for region in table_regions:
                if line in region:
                    return True
            return False

        # basic table detection: skip rows with many short lines
        if len(candidates) >= 8:
            continue

        best_line = None
        best_score = float("inf")

        for line in candidates:
            if not is_right_of_row(line, row):
                continue

            score = score_line(line, label_center)

            if score < best_score:
                best_score = score
                best_line = line

        # safety: ignore very weak matches
        if best_line is None or best_score > 500:
            continue

        # skip lines that are part of detected table regions
        if is_in_table(best_line):
            continue

        # expand multiline fields (iterative grouping)
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

        grouped = expand_multiline_field(best_line, candidates)

        mappings.append(
            {
                "label": label_text,
                "label_pos": label_center,
                "field_lines": grouped,
            }
        )

    return mappings
