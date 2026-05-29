from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .textract_parser import parse_textract_response


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def parse_response_file(input_path: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    p = Path(input_path)
    response = load_json(p)
    parsed = parse_textract_response(response)
    if output_path:
        save_json(Path(output_path), parsed)
    return parsed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Wrap Textract parser for file-based workflows")
    parser.add_argument("input", help="Path to raw Textract response JSON")
    parser.add_argument("--output", help="Optional output path for parsed JSON")
    args = parser.parse_args()

    out = parse_response_file(args.input, args.output)
    print(json.dumps({
        "summary": {
            "fields": len(out.get("fields", {})),
            "tables": len(out.get("tables", [])),
            "checkboxes": len(out.get("checkboxes", [])),
        }
    }, indent=2, ensure_ascii=False))
