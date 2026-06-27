# Form Notes

Form Type: Application for Employment (SampleWords single-page template)
Pages: 1 (fillable)

Contains Tables: Yes (EDUCATION grid: 4 rows x 3 data columns)
Contains Comb Fields: No (phone uses "(  )" paren groups, not combs)
Contains Repeating Sections: Yes-ish (the EDUCATION table rows repeat the same 3 columns)
Contains Checkbox Matrix: No

Main Failure:
The Contact Information / telephone line and the EDUCATION table are both mishandled.
Pipeline produced 18 fields; ground truth has 25 logical widgets.

Root Cause Hypothesis (CONFIRMED - brackets):
The phone area-code parentheses on the "Contact Information:" line are the core
problem. Textract emits "(", ")", and "( )" as standalone tokens and even as
KEY-VALUE *value* blocks (textract_raw_response.json lines 540, 581, 622).
The anchor stage then:
  - merged the whole contact line into one field whose value is the literal
    "( ) ( ) Mobile Telephone" (anchored_field_18), and
  - spawned two degenerate 1px value_blocks labeled "Email" (field_1) and
    "Home Telephone" (field_2, bbox width 0.001) instead of real fill regions.
So Home Telephone / Mobile Telephone / Email - three distinct fields - collapse
into one bad field plus two zero-area ghosts. The brackets are the trigger.

Observations:
- SPURIOUS (non-fillable false positives):
    * anchored_field_12 "downlost 5"  -> garbled OCR of the footer/copyright line.
    * anchored_field_17 "SampleWords" -> the watermark/logo, not a field.
- DEGENERATE answer regions (1px wide, type value_block):
    * anchored_field_1  "Email"            (width 0.001)
    * anchored_field_2  "Home Telephone"   (width 0.001)
    * anchored_field_4  "Street City/State Zip" (width 0.001)
- LABEL-AS-FIELD / duplicates (sub-captions promoted to their own widgets):
    * anchored_field_4  "Street City/State Zip"  = caption under Alternate Address
    * anchored_field_15 "Street (Apt) City/State Zip" = caption under Address
      (duplicate of anchored_field_13 "Address:")
- WRONG ANCHOR:
    * anchored_field_5 "Desired Pay Range" anchored to a dotted region at y~0.59
      (down in the table area) instead of the underline right of the label at y~0.45.
- TABLE UNDER-SEGMENTATION (biggest miss):
    * Only "High School" produced a single table_cell (anchored_field_3).
    * Missing the other 11 input cells: College/University, Specialized Training,
      Other Education rows x (Name and Location | Graduate | Degree Major) columns.
- CORRECT detections: Name, Address, Alternate Address, How did you learn,
  POSITION SOUGHT, Available Start Date, Are you currently employed,
  DATE OF APPLICATION, "Please list your areas..." (multiline).

ground_truth.json contains the 25 corrected widgets (paren phone fields split
back into 3, full 12-cell EDUCATION table, spurious logo/footer dropped).
