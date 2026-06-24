# Evaluation corpus

Hand-annotated forms used by the Phase 0 evaluation harness
(`scripts/evaluate.py`). Each form pairs a **source document** with a
**ground-truth** annotation of its *logical* widgets, so the parser's output can
be scored objectively.

## Layout

```
benchmarks/corpus/
  manifest.json                 # ordered list of forms + planning metadata
  ground_truth.template.json    # copy this to start a new annotation
  <form_dir>/
    ground_truth.json           # REQUIRED — the annotation (see schema below)
    source.pdf | source.png     # optional — the original document
    pages/
      page_1.png                # optional — page rasters (diff-image backgrounds)
      page_2.png
    predictions/
      mappings.json             # optional — a cached pipeline prediction to score
```

A form is **discovered** only if it contains `ground_truth.json`. Forms listed
in `manifest.json` with `"status": "unannotated"` are placeholders — they are
skipped until annotated, so the manifest doubles as a backlog.

## How predictions are supplied

The harness never runs the model. It scores an existing `mappings.json`, sourced
either from:

1. `<form_dir>/predictions/mappings.json` (committed alongside the form), or
2. a parallel tree via `--predictions-root <root>` → `<root>/<form_dir>/mappings.json`
   (point this at a fresh `output/` batch to score a new pipeline run).

To regenerate predictions offline and for free, replay a cached Textract
response through the existing pipeline (it accepts a `.json` input — no AWS
call), then copy the resulting `mappings.json` into `predictions/`.

## Ground-truth schema (v1.0)

See `ground_truth.template.json` for a copy-paste starting point. Key rules:

- **`bbox`** is page-local **fractions** `{x, y, width, height}` in `0..1`,
  top-left origin — the *same* convention as the pipeline's `mappings.json`.
- **`type`** ∈ `text, multiline, date, comb, checkbox, checkbox_group,
  radio_group, signature, table_cell`.
- **`page_types`** marks non-fillable pages
  (`instructions | cover | terms | signature_only | blank`); any prediction on
  those pages is counted as a false positive.
- **`cell_count` > 1** (or a group `type`) marks a widget as a *group*, enabling
  fragmentation scoring (e.g. a 9-box SSN comb should be **one** widget).
- **`repeat_group_id`** ties together copies of a repeated section so
  repeated-section recall can be measured.

## Annotating a new form

1. `cp ground_truth.template.json <form_dir>/ground_truth.json`
2. Drop the source in `<form_dir>/source.pdf` and rasters in `pages/`.
3. Annotate every fillable region as a logical widget (group comb cells and
   checkbox matrices into **single** widgets — that is the whole point).
4. Run `python scripts/evaluate.py --corpus benchmarks/corpus --out eval_runs/local`
   and inspect `eval_runs/local/<form>/diff_page_*.png`.
