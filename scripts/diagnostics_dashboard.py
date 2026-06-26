"""CLI: turn a benchmark_summary.json into a self-contained dashboard.html.

    python scripts/diagnostics_dashboard.py --summary out/benchmark_summary.json --out out/dashboard.html
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.diagnostics_dashboard import build_dashboard_html  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Render an HTML diagnostics dashboard from a benchmark summary.")
    ap.add_argument("--summary", required=True, help="Path to benchmark_summary.json")
    ap.add_argument("--out", default=None, help="Output HTML path (default: <summary dir>/dashboard.html)")
    args = ap.parse_args(argv)

    summary_path = Path(args.summary)
    if not summary_path.is_file():
        print(f"error: summary not found: {summary_path}", file=sys.stderr)
        return 2
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    out = Path(args.out) if args.out else summary_path.with_name("dashboard.html")
    out.write_text(build_dashboard_html(summary), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
