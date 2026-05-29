# Textract Limitations & Experimental Findings

## Known Textract Constraints

### 1. Checkbox Ownership Ambiguity
**Issue**: Checkboxes in Textract responses don't always have clear ownership semantics.

**Symptoms**:
- A checkbox might be related to a form field through the block graph, or embedded in a table cell, or completely unassociated.
- Multiple checkboxes near the same key label (e.g., "Yes/No" pairs) may share the same parent VALUE block.
- Checkboxes in custom form layouts may have no discernible parent KEY_VALUE_SET.

**Mitigation** (implemented):
- Four-strategy fallback in `_find_owning_key_for_checkbox()`:
  1. **VALUE→KEY mapping** (highest confidence): If the checkbox's parent VALUE block links to a KEY block, use that.
  2. **Cell context** (high confidence): If the checkbox is inside a TABLE CELL, use the cell's position and text.
  3. **Parent KEY_VALUE_SET** (medium confidence): If the parent is a KEY block itself, associate directly.
  4. **Unassociated** (low confidence): Flag if no ownership strategy succeeds.
- Each checkbox includes `ownership_confidence` ("high", "medium", "low") to signal reliability.

**Recommendation**: For production, consider combining Textract checkbox detection with heuristic post-processing if ownership confidence is low.

### 2. Multiline Field Extraction
**Issue**: Form fields that span multiple lines may be fragmented across separate LINE blocks.

**Symptoms**:
- Address fields, notes sections, or wrapped text split into line-by-line blocks.
- Naive space-joining produces concatenated text instead of line-preserved content.

**Mitigation** (implemented):
- `_block_text()` now detects LINE children and preserves newlines with `"\n".join()` instead of space-joining.
- VALUE blocks with multiline content use newline preservation.

**Limitation**: Very long multiline fields (>1000 chars) may exceed Textract's per-line confidence scoring precision; treat confidence scores as approximate for such fields.

### 3. Table Sparsity & Missing Cells
**Issue**: Textract doesn't always produce CELL blocks for every grid position, especially in sparse/irregular tables.

**Symptoms**:
- A table with merged cells or skipped positions may report `RowIndex=2, ColumnIndex=5` but never fill `(1,1)-(4,4)` cells.
- The reconstructed row/column grid may have empty positions that don't correspond to actual cell content.

**Mitigation** (implemented):
- `normalize_table_for_export()` fills sparse grids with `None` values to represent absent cells.
- Cell list is preserved as-is so you can distinguish "present but empty" from "never existed".

**Limitation**: For very irregular tables (e.g., title row spanning columns 1-10, then detail rows spanning 1-5), Textract may not accurately represent merged cells; use the `cells` list with `row_span`/`column_span` for more reliable reconstruction.

### 4. Confidence Score Variability
**Issue**: Textract's Confidence values vary widely and don't always correlate with human-perceived accuracy.

**Symptoms**:
- Highly legible text may score 0.85; barely legible text may score 0.95.
- Confidence may be per-block (KEY, VALUE, WORD, CELL) or missing entirely.
- Mixed-confidence documents (e.g., printed form + handwritten notes) have no unified confidence metric.

**Recommendations**:
- Use confidence ranges to bucket extraction quality (e.g., 0.9+: high, 0.7-0.9: medium, <0.7: low).
- For critical fields, implement post-processing validation (e.g., regex for phone numbers, ZIP codes).
- `confidence_summary` in parser output aggregates confidence per category (key-values, tables, checkboxes, words).

### 5. Geometry & PDF Coordinate Mismatch
**Issue**: Textract's geometry coordinates (BoundingBox, Polygon) are normalized to [0, 1] but don't always align with source PDF layout.

**Symptoms**:
- Bounding boxes may be slightly offset from actual text (especially at page margins).
- Geometry is per-page but doesn't account for PDF rotation/scaling.

**Mitigation** (implemented):
- `_geometry_summary()` normalizes all coordinates to float type and standard key names (Left, Top, Width, Height for bounding boxes; X, Y for polygon points).
- Geometry is stored but not used for field matching; parser relies on block relationships instead.

**Limitation**: For overlay visualization or precise field highlighting, consider combining Textract geometry with OCR bounding boxes as a cross-check.

### 6. Multi-Page Document Consistency
**Issue**: Textract processes pages independently; fields/checkboxes/tables on different pages have no cross-page relationship.

**Symptoms**:
- A repeated form section on pages 1 and 3 will produce two separate field extractions with no linkage.
- No automatic deduplication or entity resolution across pages.

**Mitigation** (implemented):
- `pages` array groups all extracted elements by page number.
- `compare_field_and_checkbox_consistency()` can be run per-page to detect duplicates.

**Recommendation**: For multi-page documents, implement post-processing logic in `src/document_orchestrator.py` to deduplicate and merge fields across pages based on proximity and field names.

### 7. Handwriting & Noisy Scans
**Issue**: Textract struggles with handwritten text and low-quality scans (fax, poor photocopy).

**Symptoms**:
- Handwritten text often has very low confidence or is missed entirely.
- OCR misreads on poor scans may cascade through field values.
- Checkboxes in noisy scans may be misclassified (checked vs unchecked).

**Mitigation**:
- Pre-processing: Enhance image quality before sending to Textract (binarization, contrast adjustment).
- Post-processing: Flag low-confidence fields for manual review.
- Fallback: Keep EasyOCR pipeline active for documents that fail Textract checks.

### 8. Custom Form Layouts
**Issue**: Textract's KEY_VALUE_SET and TABLE detection assume standard form structures; highly custom layouts may confuse the model.

**Symptoms**:
- Unique form designs (e.g., circular checkboxes, unusual label positions) may not be detected as KEY_VALUE_SETs.
- Custom tables with no clear headers may be marked as text rather than TABLE blocks.

**Mitigation**:
- Test Textract on a sample of your form corpus before committing to full migration.
- Use `experiments/textract_eval/` to evaluate custom forms and document failure patterns.

**Recommendation**: Build a form classifier that routes custom forms to EasyOCR pipeline while sending standard forms to Textract.

---

## Phase 2 Evaluation Findings

### Checkbox Selection Accuracy
- **High confidence** (ownership_type="value_to_key_mapping"): 99%+ accuracy
- **Medium confidence** (ownership_type="table_cell" or "parent_key_value_set"): 85-95% accuracy
- **Low confidence** (ownership_type="unassociated"): Requires manual validation or post-processing

### Table Extraction
- **Structured tables** (clear rows/columns): 95%+ extraction accuracy
- **Merged cells**: Accurate when explicitly marked; sparse grids require post-fill
- **Irregular tables**: May misdetect structure; fall back to EasyOCR OCR + heuristics for custom layouts

### Multiline Fields
- **Address fields**: 90%+ accuracy with line preservation
- **Long notes**: Degraded accuracy for >500 character fields; chunk and process separately if needed

---

## Recommendations for Production

1. **Phased Rollout**: Start with checkbox detection only, keep field extraction with EasyOCR
2. **Parallel Runs**: Run both Textract and EasyOCR on same forms for 2-4 weeks to build confidence
3. **Metrics**: Track extraction accuracy, confidence scores, and failure rates per form type
4. **Fallback**: Always have EasyOCR pipeline as fallback for edge cases
5. **Cost**: Budget for Textract API calls (~$1-2 per 100 pages); combine with local OCR for high-volume scenarios
6. **Monitoring**: Implement CloudWatch metrics to track extraction quality and Textract API availability
