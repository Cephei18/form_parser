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


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    json_path = project_root / "output" / "result.json"
    output_path = project_root / "output" / "debug.png"

    if not json_path.exists():
        raise FileNotFoundError("Missing output/result.json. Run src/main.py first.")

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    image_path = resolve_input_image(project_root)
    img = cv2.imread(str(image_path))

    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    for item in data:
        bbox = item.get("bbox", [])
        text = item.get("text", "")

        if len(bbox) != 4:
            continue

        pts = [(int(p[0]), int(p[1])) for p in bbox]

        # Draw OCR polygon and label for visual debugging.
        cv2.polylines(img, [np.array(pts, dtype=np.int32)], True, (0, 255, 0), 2)
        cv2.putText(
            img,
            text,
            pts[0],
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    print(f"Saved debug image: {output_path}")


if __name__ == "__main__":
    main()
