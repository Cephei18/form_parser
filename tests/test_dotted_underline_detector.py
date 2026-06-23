from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.dotted_underline_detector import detect_dotted_underlines, split_inline_answer_region
from src.field_anchor_engine import build_anchored_mappings
from src.pipelines.textract_pipeline import run_textract_pipeline
from src.textract_parser import parse_textract_response


def _bbox(left: float, top: float, width: float, height: float = 0.03) -> dict:
    return {"BoundingBox": {"Left": left, "Top": top, "Width": width, "Height": height}}


def _field_blocks(page: int, field_id: str, text: str, left: float, top: float, width: float = 0.13) -> tuple[list[dict], list[str]]:
    key_id = f"k_{page}_{field_id}"
    word_id = f"w_{page}_{field_id}"
    blocks = [
        {
            "Id": key_id,
            "BlockType": "KEY_VALUE_SET",
            "EntityTypes": ["KEY"],
            "Confidence": 95.0,
            "Page": page,
            "Geometry": _bbox(left, top, width),
            "Relationships": [{"Type": "CHILD", "Ids": [word_id]}],
        },
        {
            "Id": word_id,
            "BlockType": "WORD",
            "Text": text,
            "Confidence": 99.0,
            "Page": page,
            "Geometry": _bbox(left, top, width),
        },
    ]
    return blocks, [key_id, word_id]


def _response(fields: list[dict]) -> dict:
    blocks: list[dict] = []
    by_page: dict[int, list[str]] = {}
    for item in fields:
        field_blocks, child_ids = _field_blocks(
            int(item.get("page", 1)),
            str(item["id"]),
            str(item["text"]),
            float(item.get("left", 0.10)),
            float(item.get("top", 0.20)),
            float(item.get("width", 0.13)),
        )
        blocks.extend(field_blocks)
        by_page.setdefault(int(item.get("page", 1)), []).extend(child_ids)
    page_blocks = [
        {
            "Id": f"page_{page}",
            "BlockType": "PAGE",
            "Page": page,
            "Geometry": _bbox(0, 0, 1, 1),
            "Relationships": [{"Type": "CHILD", "Ids": ids}],
        }
        for page, ids in sorted(by_page.items())
    ]
    return {"Blocks": page_blocks + blocks, "DocumentMetadata": {"Pages": len(by_page) or 1}}


def _blank(path: Path, *, width: int = 1200, height: int = 800) -> Path:
    cv2.imwrite(str(path), np.full((height, width, 3), 255, np.uint8))
    return path


def _draw_dotted(path: Path, *, x1=0.24, x2=0.80, y=0.215, spacing=18, skew=0.0) -> Path:
    width, height = 1200, 800
    image = np.full((height, width, 3), 255, np.uint8)
    start = int(x1 * width)
    end = int(x2 * width)
    base_y = int(y * height)
    span = max(end - start, 1)
    for x in range(start, end + 1, spacing):
        yy = int(base_y + skew * (x - start) / span)
        cv2.circle(image, (x, yy), 3, (0, 0, 0), -1)
    cv2.imwrite(str(path), image)
    return path


def _draw_dashed(path: Path, *, x1=0.24, x2=0.80, y=0.215) -> Path:
    width, height = 1200, 800
    image = np.full((height, width, 3), 255, np.uint8)
    start = int(x1 * width)
    end = int(x2 * width)
    yy = int(y * height)
    for x in range(start, end + 1, 38):
        cv2.line(image, (x, yy), (min(x + 20, end), yy), (0, 0, 0), 2)
    cv2.imwrite(str(path), image)
    return path


def _draw_broken(path: Path, *, x1=0.24, x2=0.80, y=0.215) -> Path:
    width, height = 1200, 800
    image = np.full((height, width, 3), 255, np.uint8)
    start = int(x1 * width)
    end = int(x2 * width)
    yy = int(y * height)
    segments = [(0.00, 0.24), (0.30, 0.52), (0.58, 0.76), (0.82, 1.00)]
    span = end - start
    for left, right in segments:
        cv2.line(image, (start + int(span * left), yy), (start + int(span * right), yy), (0, 0, 0), 2)
    cv2.imwrite(str(path), image)
    return path


def _anchored(raw: dict, image: Path, *, page_images: dict[int, Path] | None = None) -> dict:
    parsed = parse_textract_response(raw)
    return build_anchored_mappings(raw, parsed, image, page_images=page_images)


def test_detector_recovers_dotted_dashed_broken_and_skewed(tmp_path):
    dotted = detect_dotted_underlines(_draw_dotted(tmp_path / "dotted.png"))["detected"]
    dashed = detect_dotted_underlines(_draw_dashed(tmp_path / "dashed.png"))["detected"]
    broken = detect_dotted_underlines(_draw_broken(tmp_path / "broken.png"))["detected"]
    skewed = detect_dotted_underlines(_draw_dotted(tmp_path / "skewed.png", skew=7.0))["detected"]

    assert dotted and dotted[0]["source_type"] == "dotted"
    assert dashed and dashed[0]["source_type"] == "dashed"
    assert broken and broken[0]["source_type"] == "broken"
    assert skewed and skewed[0]["source_type"] == "dotted"


def test_inline_field_splitting_avoids_obvious_false_positives():
    box = {"x": 0.10, "y": 0.20, "width": 0.42, "height": 0.03}
    split = split_inline_answer_region("Mobile No.:..............", box, line_height=0.03)
    assert split
    assert split["label"] == "Mobile No.:"
    assert split["answer_bbox"]["x"] > split["label_bbox"]["x"]
    assert split_inline_answer_region("Version 1.2.3", box, line_height=0.03) is None
    assert split_inline_answer_region("Please read the instructions carefully..............", box, line_height=0.03) is None


def test_inline_dotted_fields_anchor_healthcare_and_banking_examples(tmp_path):
    raw = _response(
        [
            {"id": "name", "text": "Patient Name....................", "width": 0.42, "top": 0.20},
            {"id": "acct", "text": "Account No.:..............", "width": 0.36, "top": 0.32},
            {"id": "pin", "text": "Pin:.....................", "width": 0.30, "top": 0.44},
        ]
    )
    out = _anchored(raw, _blank(tmp_path / "blank.png"))
    mappings = {mapping["label"]: mapping for mapping in out["mappings"]}

    assert mappings["Patient Name"]["answer_region"]["type"] == "inline_dotted_field"
    assert mappings["Account No.:"]["answer_region"]["type"] == "inline_dotted_field"
    assert mappings["Pin:"]["answer_region"]["type"] == "inline_dotted_field"
    assert out["diagnostics"]["answer_regions"]["selected_type_counts"]["inline_dotted_field"] == 3


def test_dotted_dashed_and_broken_underlines_become_answer_regions(tmp_path):
    cases = [
        (_draw_dotted(tmp_path / "dots.png"), "dotted_underline"),
        (_draw_dashed(tmp_path / "dashes.png"), "dotted_underline"),
        (_draw_broken(tmp_path / "broken.png"), "broken_underline"),
    ]
    for image, expected_type in cases:
        raw = _response([{"id": image.stem, "text": "Name", "left": 0.10, "top": 0.20, "width": 0.10}])
        out = _anchored(raw, image)
        assert len(out["mappings"]) == 1
        mapping = out["mappings"][0]
        assert mapping["answer_region"]["type"] == expected_type
        assert mapping["bbox"]["width"] > 0.45


def test_broken_underline_does_not_select_tiny_fragments(tmp_path):
    raw = _response([{"id": "broken", "text": "Name", "left": 0.10, "top": 0.20, "width": 0.10}])
    out = _anchored(raw, _draw_broken(tmp_path / "broken.png"))
    mapping = out["mappings"][0]
    assert mapping["answer_region"]["type"] == "broken_underline"
    assert mapping["bbox"]["width"] > 0.45


def test_multiple_fields_on_same_line_choose_nearest_dotted_region(tmp_path):
    width, height = 1200, 800
    image = np.full((height, width, 3), 255, np.uint8)
    y = int(0.215 * height)
    for start, end in [(0.20, 0.43), (0.62, 0.90)]:
        for x in range(int(start * width), int(end * width), 18):
            cv2.circle(image, (x, y), 3, (0, 0, 0), -1)
    image_path = tmp_path / "same_line.png"
    cv2.imwrite(str(image_path), image)
    raw = _response(
        [
            {"id": "first", "text": "First", "left": 0.08, "top": 0.20, "width": 0.08},
            {"id": "last", "text": "Last", "left": 0.52, "top": 0.20, "width": 0.08},
        ]
    )

    out = _anchored(raw, image_path)
    mappings = {mapping["label"]: mapping for mapping in out["mappings"]}

    assert mappings["First"]["bbox"]["x"] < 0.30
    assert mappings["Last"]["bbox"]["x"] > 0.55


def test_paragraph_leader_dots_do_not_create_inline_field(tmp_path):
    raw = _response(
        [
            {
                "id": "paragraph",
                "text": "Please read the instructions carefully..............",
                "left": 0.10,
                "top": 0.20,
                "width": 0.60,
            }
        ]
    )
    out = _anchored(raw, _blank(tmp_path / "blank.png"))
    assert out["mappings"] == []


def test_multipage_dotted_and_inline_compatibility(tmp_path):
    page1 = _blank(tmp_path / "page1.png")
    page2 = _draw_dotted(tmp_path / "page2.png")
    raw = _response(
        [
            {"page": 1, "id": "inline", "text": "Mobile No.:..............", "left": 0.10, "top": 0.20, "width": 0.38},
            {"page": 2, "id": "visual", "text": "Name", "left": 0.10, "top": 0.20, "width": 0.10},
        ]
    )

    out = _anchored(raw, page1, page_images={1: page1, 2: page2})
    by_page = {mapping["page"]: mapping for mapping in out["mappings"]}

    assert by_page[1]["answer_region"]["type"] == "inline_dotted_field"
    assert by_page[2]["answer_region"]["type"] == "dotted_underline"
    assert out["diagnostics"]["fields_per_page"] == {"1": 1, "2": 1}


def test_pipeline_writes_phase_f_debug_artifacts(tmp_path):
    image = _draw_dotted(tmp_path / "page.png")
    raw = _response([{"id": "name", "text": "Name", "left": 0.10, "top": 0.20, "width": 0.10}])
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    output_dir = tmp_path / "out"

    result = run_textract_pipeline(raw_path, output_dir, reference_image_path=image)

    dotted_path = output_dir / "dotted_underline_debug.json"
    answer_path = output_dir / "answer_region_debug.json"
    diagnostics = json.loads(Path(result["diagnostics_path"]).read_text(encoding="utf-8"))

    assert dotted_path.exists()
    assert answer_path.exists()
    assert json.loads(dotted_path.read_text(encoding="utf-8"))["detected_count"] >= 1
    assert json.loads(answer_path.read_text(encoding="utf-8"))["selected_type_counts"]["dotted_underline"] == 1
    assert diagnostics["answer_regions"]["selected_type_counts"]["dotted_underline"] == 1
