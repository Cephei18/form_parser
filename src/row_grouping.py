from typing import Any


def group_rows(ocr_data: list[dict[str, Any]], threshold: int = 20) -> list[list[dict[str, Any]]]:
    """Groups OCR elements into rows based on Y-coordinate similarity."""
    # Sort top-to-bottom using the Y component of each item's center.
    sorted_data = sorted(ocr_data, key=lambda x: x["center"][1])

    rows: list[list[dict[str, Any]]] = []
    current_row: list[dict[str, Any]] = []

    for item in sorted_data:
        y = item["center"][1]

        if not current_row:
            current_row.append(item)
            continue

        prev_y = current_row[-1]["center"][1]

        # Nearby Y values are considered part of the same visual row.
        if abs(y - prev_y) < threshold:
            current_row.append(item)
        else:
            rows.append(current_row)
            current_row = [item]

    if current_row:
        rows.append(current_row)

    return rows


def looks_like_continuation(row: list[dict[str, Any]]) -> bool:
    """Heuristic that detects short or fragment rows likely continuing a previous label."""
    text = " ".join(item["text"] for item in row).strip()

    if len(text) < 25:
        return True

    if text.endswith(":"):
        return True

    keywords = ["letters", "years", "months", "days", "man"]
    if any(keyword in text.lower() for keyword in keywords):
        return True

    return False


def smart_merge_rows(
    rows: list[list[dict[str, Any]]], y_threshold: int = 40
) -> list[list[dict[str, Any]]]:
    """Merges close rows when the next row looks like continuation text."""
    merged: list[list[dict[str, Any]]] = []
    i = 0

    while i < len(rows):
        current = rows[i]

        if i < len(rows) - 1:
            next_row = rows[i + 1]

            current_y = sum(item["center"][1] for item in current) / len(current)
            next_y = sum(item["center"][1] for item in next_row) / len(next_row)

            if abs(next_y - current_y) < y_threshold:
                if looks_like_continuation(next_row):
                    merged.append(current + next_row)
                    i += 2
                    continue

        merged.append(current)
        i += 1

    return merged


def is_useful_row(row: list[dict[str, Any]]) -> bool:
    """Keep rows unless they are structurally empty."""
    return any(item.get("text", "").strip() for item in row)
