"""Task 5 — offline replay of a cached Textract response (no AWS).

Re-runs the anchor/render pipeline on a saved Textract response + page rasters,
producing mappings, diagnostics, the curated benchmark report, an HTML
dashboard, and per-page visualizations. Zero AWS calls, zero production impact —
it writes only into ``--out``.

    python scripts/replay.py --textract-response response.json --pages ./pages --flags diagnostics --out ./replay_out
    python scripts/replay.py --textract-response response.json --pdf form.pdf --flags diagnostics

``--flags diagnostics`` turns on every additive observe/diagnostic flag (all
non-destructive); ``--flags none`` replays with production defaults.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# All additive, non-destructive. CHECKBOX validation runs in OBSERVE mode (never
# drops). No enforce flag is ever set by this preset.
DIAGNOSTIC_FLAGS = {
    "FORM_PARSER_TABLE_INTELLIGENCE_ENABLED": "true",
    "FORM_PARSER_COMB_DIAGNOSTICS_ENABLED": "true",
    "FORM_PARSER_DOTTED_LEADER_DIAGNOSTICS_ENABLED": "true",
    "FORM_PARSER_CONFIDENCE_CALIBRATION_ENABLED": "true",
    "FORM_PARSER_CONFIDENCE_PIPELINE_ENABLED": "true",
    "FORM_PARSER_CHECKBOX_VALIDATION_OBSERVE": "true",
}


def _discover_pages(pages_dir: Path) -> list[tuple[int, Path]]:
    found: dict[int, Path] = {}
    for p in sorted(pages_dir.glob("*.png")):
        stem = p.stem
        digits = "".join(ch for ch in stem.split("_")[-1] if ch.isdigit())
        if digits:
            found[int(digits)] = p
    return sorted(found.items())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Replay a cached Textract response offline (no AWS).")
    ap.add_argument("--textract-response", required=True, help="Cached Textract AnalyzeDocument JSON.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pages", help="Directory of page rasters (page_<n>.png / input_page_<n>.png).")
    src.add_argument("--pdf", help="Source PDF to rasterize locally (Poppler or PyMuPDF).")
    ap.add_argument("--flags", choices=["diagnostics", "none"], default="diagnostics")
    ap.add_argument("--out", default="replay_out", help="Output directory.")
    ap.add_argument("--no-viz", action="store_true", help="Skip page visualizations.")
    args = ap.parse_args(argv)

    response = Path(args.textract_response)
    if not response.is_file():
        print(f"error: response not found: {response}", file=sys.stderr)
        return 2

    # Configure flags BEFORE importing the pipeline (flags are read at call time,
    # but artifact backend / mode are read from env too).
    os.environ.setdefault("FORM_PARSER_PIPELINE_MODE", "textract")
    os.environ["FORM_PARSER_ARTIFACT_BACKEND"] = "local"
    os.environ.setdefault("FORM_PARSER_MULTIPAGE", "true")
    if args.flags == "diagnostics":
        for key, value in DIAGNOSTIC_FLAGS.items():
            os.environ[key] = value

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from src.pipelines.textract_pipeline import run_textract_pipeline  # noqa: E402

    if args.pdf:
        from src.document_render import render_pdf_pages  # noqa: E402

        pages = render_pdf_pages(Path(args.pdf), out_dir / "_pages")
    else:
        pages = _discover_pages(Path(args.pages))
    if not pages:
        print("error: no page rasters found", file=sys.stderr)
        return 2

    print(f"replaying {response.name} over {len(pages)} page(s) with flags={args.flags} ...")
    run_textract_pipeline(response, out_dir, reference_image_path=pages[0][1], page_images=pages)

    summary_path = out_dir / "benchmark_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    # Dashboard
    from src.diagnostics_dashboard import build_dashboard_html  # noqa: E402

    (out_dir / "dashboard.html").write_text(build_dashboard_html(summary), encoding="utf-8")

    # Visualizations
    if not args.no_viz:
        from src.diagnostics_visualizer import render_diagnostics_pages  # noqa: E402

        page_map = {n: p for n, p in pages}
        mappings = json.loads((out_dir / "mappings.json").read_text(encoding="utf-8"))
        render_diagnostics_pages(page_map, mappings, summary, out_dir / "viz")

    fds = summary.get("form_diagnostics_summary", {})
    print("\n=== form diagnostics summary ===")
    print(json.dumps(fds, indent=2)[:1400])
    print(f"\nartifacts in {out_dir}: mappings.json, benchmark_summary.json, dashboard.html, viz/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
