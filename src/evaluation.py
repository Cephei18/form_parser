import json
from pathlib import Path
from statistics import mean


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_mappings(mappings: list[dict], labels: list[dict] | None = None) -> dict:
    scores = [
        float(mapping.get("candidate_score", 0.0))
        for mapping in mappings
        if isinstance(mapping.get("candidate_score"), (int, float))
    ]
    confidence_classes: dict[str, int] = {}
    for mapping in mappings:
        key = mapping.get("confidence_class", "unknown")
        confidence_classes[key] = confidence_classes.get(key, 0) + 1

    label_texts = {item.get("text") for item in labels or [] if item.get("text")}
    mapped_labels = {mapping.get("label") for mapping in mappings if mapping.get("label")}
    unmatched_labels = sorted(label_texts - mapped_labels)

    return {
        "mapping_count": len(mappings),
        "average_candidate_score": round(mean(scores), 4) if scores else 0.0,
        "confidence_classes": confidence_classes,
        "unmatched_label_count": len(unmatched_labels),
        "unmatched_labels": unmatched_labels,
    }


def evaluate_mapping_file(mappings_path: Path, result_path: Path | None = None) -> dict:
    mappings = _load_json(mappings_path)
    labels = _load_json(result_path) if result_path and result_path.exists() else []
    return summarize_mappings(mappings, labels)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "output"
    summary = evaluate_mapping_file(output_dir / "mappings.json", output_dir / "result.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
