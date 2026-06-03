# PDF / preview polish pass (Task 4)

Minimal-risk, flag-gated. The generated **`output.pdf` was already clean** — it
contains only AcroForm widgets (text fields + checkboxes) over the background
image; there are no debug labels or overlay clutter baked into it. So no change
was made to the PDF rendering path, checkbox rendering, multiline logic, alignment,
or field anchoring.

## What changed (one isolated lever)

`src/pipelines/textract_pipeline.py :: _draw_mapping_preview` — the per-field text
labels drawn on the **`mapping.png` diagnostic preview** (the "Field detection
preview" shown under Advanced details) are now gated behind a flag:

```
FORM_PARSER_PREVIEW_LABELS = true   (default — unchanged, validated behavior)
FORM_PARSER_PREVIEW_LABELS = false  (clean preview: colored boxes only, no label text)
```

Rationale: the label text can overlap adjacent boxes and reads as clutter on dense
forms. Boxes-only is cleaner for demos/screenshots. Default preserves the exact
prior output, so the 14/14-validated behavior is byte-identical unless the flag is set.

| | Labels on (default) | Labels off (`=false`) |
|---|---|---|
| `output.pdf` | unchanged | unchanged |
| `mapping.png` | boxes + field-name text | colored boxes only |
| checkbox color | orange box | orange box (unchanged) |
| debug overlay (`textract_mapping_debug.png`) | unchanged | unchanged |

## What was deliberately NOT touched

- `create_pdf_with_fields` (widgets, borders, fonts, checkbox style) — zero edits.
- `draw_anchor_debug_overlay` / `textract_mapping_debug.png` — already a separate
  debug artifact, **not surfaced to users** (the result page only shows `mapping.png`).
- Field anchoring, multiline detection, Textract orchestration — untouched.

## Regression risk

- **Default (flag unset):** none — identical pixels to the validated baseline.
- **Flag = false:** affects only `mapping.png`; cannot alter the PDF, the overlay
  geometry (the result page draws its own overlay from `result.json` coordinates),
  checkbox/multiline rendering, or alignment.
- Takes effect only after a worker container rebuild (the file lives in the image):
  `pwsh scripts/deploy_lambda_worker.ps1`. The currently-deployed worker is
  unaffected until then.

## Validation

- `python -m py_compile src/pipelines/textract_pipeline.py` → OK.
- Logic: when `draw_labels` is False only the `cv2.putText` call is skipped; the
  `cv2.rectangle` calls and image write are unchanged.
- To confirm visually before relying on it: rebuild the worker, run
  `scripts/e2e_async_validate.py`, and open the new `mapping.png` (both flag states).

## Rollback

Unset `FORM_PARSER_PREVIEW_LABELS` (or set `true`) → original preview. To revert
the code entirely, restore the single `cv2.putText` block in `_draw_mapping_preview`
(see git diff for this file). No data migration, no PDF impact.
