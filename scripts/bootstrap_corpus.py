"""Bootstrap a benchmark corpus scaffold from the forms in ``input/``.

Pure orchestration around the EXISTING pipeline — no parser logic, no production
code changes. For each input file it runs the parser exactly as production does
(default flags), then assembles:

    benchmarks/corpus/<form_name>/
      input.<ext>                 # original
      textract_raw_response.json  # if produced
      mappings.json               # if produced
      benchmark_summary.json      # if produced
      ground_truth.json           # {"widgets": []}
      notes.md                    # template
      artifacts/input_page_<n>.png

Failures on one form are logged and skipped; the rest continue.

    python scripts/bootstrap_corpus.py [--input input] [--out benchmarks/corpus]
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Run exactly as production: textract mode, local artifacts, DEFAULT flags (no
# diagnostic/observe flags set). AWS profile/region match the project defaults.
os.environ.setdefault("AWS_PROFILE", "form-pdf-poc")
os.environ.setdefault("AWS_DEFAULT_REGION", "ap-south-1")
os.environ.setdefault("AWS_REGION", "ap-south-1")
os.environ.setdefault("FORM_PARSER_AWS_REGION", "ap-south-1")
os.environ["FORM_PARSER_PIPELINE_MODE"] = "textract"
os.environ["FORM_PARSER_ARTIFACT_BACKEND"] = "local"

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

GROUND_TRUTH_TEMPLATE = '{\n  "widgets": []\n}\n'
NOTES_TEMPLATE = """# Form Notes

Form Type:
Pages:

Contains Tables:
Contains Comb Fields:
Contains Repeating Sections:
Contains Checkbox Matrix:

Main Failure:
Root Cause Hypothesis:

Observations:
"""


def _rasterize(input_path: Path, artifacts_dir: Path) -> list[tuple[int, Path]]:
    """Produce artifacts/input_page_<n>.png and return [(page_no, path)]."""
    from PIL import Image
    from src.document_render import render_pdf_pages

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    suffix = input_path.suffix.lower()
    if suffix == ".pdf":
        raw_dir = artifacts_dir / "_raw"
        raw_dir.mkdir(exist_ok=True)
        pages = render_pdf_pages(input_path, raw_dir)
        out: list[tuple[int, Path]] = []
        for n, p in pages:
            target = artifacts_dir / f"input_page_{n}.png"
            shutil.copyfile(p, target)
            out.append((n, target))
        shutil.rmtree(raw_dir, ignore_errors=True)
        return out
    if suffix in IMAGE_SUFFIXES:
        target = artifacts_dir / "input_page_1.png"
        Image.open(input_path).convert("RGB").save(target)
        return [(1, target)]
    raise ValueError(f"unsupported input type {suffix!r}")


def _process_form(input_path: Path, out_root: Path) -> dict:
    name = input_path.stem
    dest = out_root / name
    dest.mkdir(parents=True, exist_ok=True)
    artifacts = dest / "artifacts"

    # 1) Rasterize pages into artifacts/.
    pages = _rasterize(input_path, artifacts)

    # 2) Run the parser exactly as production does, into a throwaway dir.
    from src.pipelines.textract_pipeline import run_textract_pipeline

    with tempfile.TemporaryDirectory(prefix="corpus_run_") as run_dir:
        run_dir_path = Path(run_dir)
        run_textract_pipeline(
            input_path,
            run_dir_path,
            reference_image_path=pages[0][1],
            page_images=pages,
        )
        # 3) Copy the wanted artifacts (if the pipeline produced them).
        copied = []
        for fname in ("textract_raw_response.json", "mappings.json", "benchmark_summary.json"):
            src = run_dir_path / fname
            if src.is_file():
                shutil.copyfile(src, dest / fname)
                copied.append(fname)

    # 4) Copy the original input alongside the scaffold.
    shutil.copyfile(input_path, dest / f"input{input_path.suffix.lower()}")

    # 5) Empty ground-truth template + 6) notes template (don't clobber edits).
    gt = dest / "ground_truth.json"
    if not gt.exists():
        gt.write_text(GROUND_TRUTH_TEMPLATE, encoding="utf-8")
    notes = dest / "notes.md"
    if not notes.exists():
        notes.write_text(NOTES_TEMPLATE, encoding="utf-8")

    return {"name": name, "dest": dest, "pages": len(pages), "artifacts_copied": copied}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Bootstrap a benchmark corpus scaffold from input/ forms.")
    ap.add_argument("--input", default="input", help="Directory of source forms (default: input)")
    ap.add_argument("--out", default="benchmarks/corpus", help="Corpus output dir (default: benchmarks/corpus)")
    args = ap.parse_args(argv)

    in_dir = Path(args.input)
    out_root = Path(args.out)
    if not in_dir.is_dir():
        print(f"error: input dir not found: {in_dir}", file=sys.stderr)
        return 2

    forms = sorted(p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in (IMAGE_SUFFIXES | {".pdf"}))
    succeeded: list[dict] = []
    failed: list[tuple[str, str]] = []

    for form in forms:
        print(f"\n=== processing {form.name} ===")
        try:
            result = _process_form(form, out_root)
            succeeded.append(result)
            print(f"  OK  {result['name']}  pages={result['pages']}  artifacts={result['artifacts_copied']}")
        except Exception as exc:  # noqa: BLE001 — skip-and-continue by design
            failed.append((form.name, f"{type(exc).__name__}: {exc}"))
            print(f"  FAIL {form.name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()

    print("\n================ SUMMARY ================")
    print(f"Processed: {len(forms)}")
    print(f"Succeeded: {len(succeeded)}")
    print(f"Failed: {len(failed)}")
    print("Output locations:")
    for r in succeeded:
        print(f"  {r['dest']}")
    for name, err in failed:
        print(f"  [FAILED] {name}: {err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
