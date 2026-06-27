# Form Notes

Form Type: Paras Health "Registration Form" (hospital patient registration, Ver 2.0/April'23)
Pages: 1 (fillable)

Contains Tables: Yes (signature block at bottom: Sign Guardian / Sign Patient, Name / Relationship with Patient)
Contains Comb Fields: No
Contains Repeating Sections: No (two-column layout repeats some labels: Pin, Mobile No, Alternate No, Email, Incase of Emergency contact, Name, Relationship with Patient appear on both left and right)
Contains Checkbox Matrix: Yes — option groups:
  - Sex: Male / Female
  - Marital Status: Single / Married / Divorced / Widowed
  - "I came to know about Paras Hospital from": Radio / Friend-Relative / Internet (Google/Facebook) / My Doctor / Hoarding-Banner / Newspaper / Flyer-Leaflet
  - Consent: "get healthcare information through SMS" / "& or E-mail"

Pipeline output (current code, default flags): 44 mappings — 26 text, 16 checkbox, 2 multiline.
Anchor types: 26 value_block, 16 checkbox_region, 2 dotted_underline.

Main Failure: multiple — see root causes below. Score vs ground truth (48 widgets): F1 1.00 (P 1.00 / R 1.00, tp 48/48) after all fixes.

Root Causes (analyzed) + fix status — ALL FIXED:
1. Present/Permanent Address dropped — tall multiline value region geometrically contains the small Pin field; dedupe dropped the big region. FIXED: dedupe requires comparable box sizes (DUPLICATE_MIN_AREA_RATIO).
2. Mobile No (both) not rendering — "Mobile No. :........" colon touches dotted leader -> Textract emits a ~1px (degenerate) value block; drop fell to a distant full-width dotted line. FIXED: degenerate value anchors are REPAIRED (extend right to next text).
3. Sign. Patient mis-rendered as a tall block — degenerate value block. FIXED by the repair + signature-table emission below.
4. Consent "I ...... the undersigned" blank — single-char key substring-matched a distant signature TABLE cell. FIXED: table-cell text-match needs label length >= 3; "I" then repairs to the consent line.
5. Footer "FO/Reg." / "Form/Ver." version codes. FIXED: field-hygiene boilerplate patterns.
6. "Any other please specify" write-in — value "[ ]" made it a checkbox that token-dedup removed. FIXED: a write-in cue ("specify") in the label overrides checkbox classification; the field anchors to the fill line below (clamped to one line); the checkbox glyph is still emitted from its SELECTION_ELEMENT.
7. Bottom signature table (Sign. Guardian / Sign. Patient / Name / Relation) — labels + dotted answer lines share one cell; Textract KV mis-links them. FIXED: signature-table emitter detects an all-populated table whose cells carry dotted leaders, emits one inline answer per cell (label-right -> next label), and suppresses the unreliable KEY fields in that table. Checkbox matrices (cells with "[ ]") are excluded.
8. Present/Permanent Address single-line — FIXED: a tall (>=3 line) multiline value block is widened left to the label margin so wrapped lines are inside the widget.

Observations:
- Most text fields use dotted-leader fill lines ("First Name : ........"); these anchor as value_block / dotted_underline.
- Repeated left/right labels (Pin, Mobile No, Email, Name, Relationship with Patient) risk duplicate/ambiguous mapping — verify left vs right disambiguation when authoring ground truth.
- Present Address / Permanent Address are multiline (two dotted rows each).
- Bottom signature block is a Textract table — candidate for table-fill input cells; confirm Sign/Name/Relationship cells render fillable.
- Footer "Note: For any change..." and "FO/Reg. Form/Ver. 2.0/April'23" are non-fillable furniture — confirm field-hygiene suppresses them.

ground_truth.json left as the blank template ({"widgets": []}); author manually to enable scoring via scripts/evaluate.py.
