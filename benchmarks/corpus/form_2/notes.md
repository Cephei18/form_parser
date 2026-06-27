# Form Notes

Form Type: HDFC Mutual Fund — Application-cum-Transaction Form for Market Makers & Large Investors (April 2023)
Pages: 8 — pages 1-4 fillable; page 5 CHECKLIST (office use); pages 6-7 Scheme/Benchmark Riskometer; page 8 intentionally blank.

Contains Tables: Yes (Section 7 ETF scheme grid, Section 6 FATCA matrix, Section 12 / page-4 beneficiary lists, page-5 document checklist)
Contains Comb Fields: Yes (PAN/PEKRN, KYC Number, DP ID, Beneficiary A/c No, PIN, IFSC, Account Number, DOB) — NOTE: Textract returns these as single merged value regions, so they are NOT fragmented here (comb detection stays off without harm).
Contains Repeating Sections: Yes (First/Second/Third applicant blocks; email-belongs-to and mobile-belongs-to option rows)
Contains Checkbox Matrix: Yes (ETF Cash/Portfolio grid ~80 cells; FATCA Yes/No per applicant; account-type/payment-mode/belongs-to groups)

Pipeline output (current engine, default flags): 549 mappings (was 661 before page-gating fix).
per page: {1:92, 2:123, 3:261, 4:73, 5-8:0}.

Main Failure (MAJOR — FIXED): pages 5-8 are non-fillable (checklist / riskometer / blank) yet the engine emitted 112 widgets there — Textract reports KEY/VALUE pairs, pre-printed ticks (page 5 has 61 SELECTION_ELEMENTs that are checkmarks in an office-use table) and table cells on them.
Root cause: page-type gating existed only in the EVALUATION layer (uses ground-truth page_types); the PIPELINE had no page classifier and emitted on every page.
Fix: src/page_classifier.py (FORM_PARSER_PAGE_GATING_ENABLED, default ON) detects non-fillable pages from a SHORT top banner ("checklist"/"riskometer"/"intentionally left blank"/title markers) + a blank-page check (no words/fields/checkboxes/cells). The engine skips ALL widget emission (fields, checkboxes, photos, table-fill, signature-table) on those pages. Single-page forms are never flagged; a page is "blank" only when it also has no parsed fields/checkboxes (keeps sparse real pages safe). Result: non_fillable_false_positives 112 -> 0.

Page-1 bugs (user-flagged) — fixed:
- EUIN "Sign Here" signature grid: a 2x3 table (R1 "Sign Here" boxes, R2 applicant labels). Previously the labels became text fields and the wide signing boxes were lost/deduped. FIX: src/table_fill.emit_signature_grid_cells detects tables containing "Sign Here"/"signature"/"sign" cells, emits one signature widget per box (the full wide cell) labelled from the adjacent label row, and suppresses the table's KEY fields. emit_signature_table_cells now skips sign-here tables (avoids double emission). "Sign. Guardian" (form_5 dotted-leader labels) is NOT matched.
- "Received from Mr. / Ms. / M/s." (acknowledgement slip, y~0.952) was wrongly dropped as margin_furniture. FIX: margin-furniture suppression now requires a SHORT label (<=3 words); a multi-word bottom-of-page field is preserved (form_5's terse "downlost 5"/"SampleWords" still suppressed).

Comb character-box fields — FIXED (two detection methods):
- src/comb_field_detector.py (FORM_PARSER_COMB_RUN_ENABLED, default ON). The engine WIDENS an existing under-sized text field whose answer falls inside a detected run into a proper comb widget (widget_type=comb, comb_cells=N). Never creates new widgets; skips non-fillable pages. Verified 0 runs / 0 widenings on form_3/4/5 -> zero regression.
  (a) Separate-box combs: interior-ink filter (a comb cell is an EMPTY box, ink<0.20; a glyph is ink-filled) -> handles DP ID, Beneficiary A/c.
  (b) Continuous combs (cells share walls -> one contour): borders are LIGHT GREY (gray~200) that OTSU drops, so detect a regular run of short light-grey vertical DIVIDERS (>=7, spacing CV<0.20, dark-ink-empty interior) -> handles Folio No (13 cells), email comb, etc.
- FIXED page 1: Folio No (-> 13-cell comb x0.183 w0.352), DP ID (7), Beneficiary A/c (12), eDocs Email (7). Both detectors yield 0 on comb-free forms (naive size/spacing detectors caught text: form_4=85, form_5=25 false runs — the ink/regularity gates are what make it safe).

False checkboxes amid text/numbers — FIXED:
- Bug: 124 checkboxes rendered as WIDE boxes (w 0.05-0.30, anchor_type table_cell) across row text/numbers (page-4 collection table, page-1 belongs-to row). Root cause: a KEY with a "[ ]" value classified as checkbox then anchored to a wide table_cell/value_block instead of the small glyph.
- Fix (field_anchor_engine._select_answer_region): a checkbox-typed field now anchors to its VALUE-block glyph (the small "[ ]", ~0.012 wide), not wide visual candidates. Result: 0 wide checkboxes; page-4 row checkboxes are now glyphs at the row start (x0.058), page-1 belongs-to options at their glyphs. form_3/4/5 checkboxes unchanged (they come from SELECTION_ELEMENTs, not the token path).

Still NOT fixed (deferred):
- Mailing Address blank lines: Textract emits NO KEY for the address writing area (only CITY/STATE/PIN below have fields). Synthesizing a field from a label without a Textract KEY is a new capability (risk of spurious fields).
- Section 7 ETF grid / Section 6 FATCA matrix: dense matrices, not exhaustively annotated in ground truth.

Ground truth: page_types COMPLETE (8 pages). Widgets = representative high-confidence subset (37) of discrete fields (mode-of-holding, market-maker/large-investor, tax-status, payable/receivable, bank account-type groups, signatures, principal text/comb fields). The dense matrices are intentionally not cell-annotated yet, so P/R understate; the headline metric for the page-gating fix is non_fillable_false_positives (now 0).
