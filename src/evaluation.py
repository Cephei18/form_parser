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
    region_types: dict[str, int] = {}
    for mapping in mappings:
        key = mapping.get("confidence_class", "unknown")
        confidence_classes[key] = confidence_classes.get(key, 0) + 1
        region_key = mapping.get("region_support") or mapping.get("field_type") or "unknown"
        region_types[region_key] = region_types.get(region_key, 0) + 1

    label_texts = {item.get("text") for item in labels or [] if item.get("text")}
    mapped_labels = {mapping.get("label") for mapping in mappings if mapping.get("label")}
    unmatched_labels = sorted(label_texts - mapped_labels)
    multiline_count = sum(1 for mapping in mappings if int(mapping.get("multiline_group_size", 1)) > 1)
    ambiguous_count = confidence_classes.get("ambiguous", 0)
    unresolved_count = confidence_classes.get("unresolved", 0)

    return {
        "mapping_count": len(mappings),
        "average_candidate_score": round(mean(scores), 4) if scores else 0.0,
        "confidence_classes": confidence_classes,
        "region_types": region_types,
        "unmatched_label_count": len(unmatched_labels),
        "unmatched_labels": unmatched_labels,
        "multiline_mapping_count": multiline_count,
        "ambiguous_mapping_count": ambiguous_count,
        "unresolved_mapping_count": unresolved_count,
        "strong_match_count": confidence_classes.get("strong_match", 0),
        "weak_match_count": confidence_classes.get("weak_match", 0),
    }


def compare_mapping_summaries(baseline: dict, candidate: dict) -> dict:
    baseline_count = int(baseline.get("mapping_count", 0))
    candidate_count = int(candidate.get("mapping_count", 0))
    return {
        "mapping_count_delta": candidate_count - baseline_count,
        "average_candidate_score_delta": round(
            float(candidate.get("average_candidate_score", 0.0)) - float(baseline.get("average_candidate_score", 0.0)),
            4,
        ),
        "unmatched_label_delta": int(candidate.get("unmatched_label_count", 0)) - int(baseline.get("unmatched_label_count", 0)),
        "strong_match_delta": int(candidate.get("strong_match_count", 0)) - int(baseline.get("strong_match_count", 0)),
        "weak_match_delta": int(candidate.get("weak_match_count", 0)) - int(baseline.get("weak_match_count", 0)),
        "ambiguous_delta": int(candidate.get("ambiguous_mapping_count", 0)) - int(baseline.get("ambiguous_mapping_count", 0)),
        "unresolved_delta": int(candidate.get("unresolved_mapping_count", 0)) - int(baseline.get("unresolved_mapping_count", 0)),
    }


def evaluate_mapping_file(mappings_path: Path, result_path: Path | None = None, baseline_path: Path | None = None) -> dict:
    mappings = _load_json(mappings_path)
    labels = _load_json(result_path) if result_path and result_path.exists() else []
    summary = summarize_mappings(mappings, labels)

    if baseline_path and baseline_path.exists():
        baseline_mappings = _load_json(baseline_path)
        baseline_summary = summarize_mappings(baseline_mappings, labels)
        summary["comparison"] = compare_mapping_summaries(baseline_summary, summary)

    return summary


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "output"
    summary = evaluate_mapping_file(
        output_dir / "mappings.json",
        output_dir / "result.json",
        baseline_path=output_dir / "mappings_rule.json",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
