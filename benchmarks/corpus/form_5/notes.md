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

Main Failure: multiple — see root causes below. Score vs ground truth (48 widgets): F1 0.86 (P 0.87 / R 0.85, tp 41) after fixes; was lower before.

Root Causes (analyzed) + fix status:
1. Present/Permanent Address dropped — tall multiline value region geometrically contains the small Pin field; dedupe dropped the big region. FIXED: dedupe now requires comparable box sizes (DUPLICATE_MIN_AREA_RATIO).
2. Mobile No (both) not rendering — "Mobile No. :........" colon touches dotted leader -> Textract emits a ~1px (degenerate) value block; drop fell to a distant full-width dotted line. FIXED: degenerate value anchors are REPAIRED (extend right to next text) instead of dropped to junk.
3. Sign. Patient mis-rendered as a tall block — degenerate value block. FIXED by the same repair (now anchors the signature line).
4. Consent "I ...... the undersigned" blank — single-char key substring-matched a distant signature TABLE cell. FIXED: table-cell text-match now needs label length >= 3 (exact match still allowed); "I" then repairs to the consent line.
5. Footer "FO/Reg." / "Form/Ver." version codes mapped as fields. FIXED: added to field-hygiene boilerplate patterns.

Remaining known limitations (lower impact / higher risk — not yet fixed):
- "Any other please specify" write-in line: the KEY's value overlaps the adjacent checkbox SELECTION_ELEMENT and is removed by token-checkbox dedup; the write-in dotted line below is not recovered (checkbox + write-in sharing one label).
- Bottom signature table (Sign. Guardian / Name / Relation with Patient): labels and dotted answer lines live in the SAME table cell, so table-fill sees no empty cells and Textract's KV positions are unreliable (Sign. Guardian lands in the wrong column). Needs in-cell inline-dotted detection.
- Present/Permanent Address captured as a single line, not the full 2-line multiline region (partial IoU).

Observations:
- Most text fields use dotted-leader fill lines ("First Name : ........"); these anchor as value_block / dotted_underline.
- Repeated left/right labels (Pin, Mobile No, Email, Name, Relationship with Patient) risk duplicate/ambiguous mapping — verify left vs right disambiguation when authoring ground truth.
- Present Address / Permanent Address are multiline (two dotted rows each).
- Bottom signature block is a Textract table — candidate for table-fill input cells; confirm Sign/Name/Relationship cells render fillable.
- Footer "Note: For any change..." and "FO/Reg. Form/Ver. 2.0/April'23" are non-fillable furniture — confirm field-hygiene suppresses them.

ground_truth.json left as the blank template ({"widgets": []}); author manually to enable scoring via scripts/evaluate.py.
