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

Still NOT fixed (need dedicated comb/CV work — higher risk, deferred):
- Comb character-box fields (Folio No, DP ID, PIN, Beneficiary A/c, etc.): Textract returns a degenerate first-box value (w~0.007-0.009, just above the 0.006 degenerate cutoff) and the CV rectangle detector finds only 3 boxes on page 1 (the character cells are below its min-width; raw contours are noisy, ~91 mixed text/box strokes). Rendering the full comb run needs real comb-run detection (group equal-spaced small boxes) — a substantial, regression-risky feature.
- Mailing Address blank lines: Textract emits NO KEY for the address writing area (only CITY/STATE/PIN below have fields). Synthesizing a field from a label without a Textract KEY is a new capability (risk of spurious fields).
- Section 7 ETF grid / Section 6 FATCA matrix: dense matrices, not exhaustively annotated in ground truth.

Ground truth: page_types COMPLETE (8 pages). Widgets = representative high-confidence subset (37) of discrete fields (mode-of-holding, market-maker/large-investor, tax-status, payable/receivable, bank account-type groups, signatures, principal text/comb fields). The dense matrices are intentionally not cell-annotated yet, so P/R understate; the headline metric for the page-gating fix is non_fillable_false_positives (now 0).
