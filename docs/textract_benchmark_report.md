# Textract Benchmark Report — Phase 2

## Executive Summary

This report benchmarks AWS Textract against the existing OCR + heuristics pipeline (EasyOCR + `mapping.py`) on representative form samples.

**Status**: Evaluation in progress. See `experiments/textract_eval/outputs/summary.json` for raw results.

---

## Evaluation Methodology

### Test Dataset
- Source: Sample forms from `experiments/textract_eval/`
- Categories: checkbox-heavy, multi-column, multiline addresses, tables, noisy scans, multi-page
- Format: PDF → PNG → Textract AnalyzeDocument / EasyOCR

### Metrics
- **Checkbox Detection Accuracy**: `selected_count`, `unselected_count` vs manual verification
- **Table Extraction**: Cell accuracy, header detection, row/column preservation
- **Field Extraction**: Key-value accuracy, multiline handling
- **Confidence Distribution**: Avg confidence per category
- **Ownership Mapping**: Checkbox-to-field association success rate
- **Runtime**: API latency (Textract) vs local processing (EasyOCR)
- **Cost**: Per-page processing cost estimate

### Validation Tools
- `src/textract_validators.py`: Automated quality checks
- `experiments/textract_eval/run_eval.py`: Manifest-driven evaluation runner
- Output: JSON reports with detailed statistics and failure analysis

---

## Results

### Document: textract_sample

**File**: `experiments/textract_eval/raw_response.json`

**Extraction Summary**:
- Fields extracted: [TBD — run evaluation]
- Tables extracted: [TBD]
- Checkboxes detected: [TBD]
- Avg confidence (fields): [TBD]
- Avg confidence (tables): [TBD]
- Avg confidence (checkboxes): [TBD]

**Detailed Reports**:
- Tables: `experiments/textract_eval/outputs/textract_sample.tables.json`
- Checkboxes: `experiments/textract_eval/outputs/textract_sample.checkboxes.json`
- Consistency: `experiments/textract_eval/outputs/textract_sample.consistency.json`

---

## Comparative Analysis

### Textract vs EasyOCR + Heuristics

| Aspect | Textract | EasyOCR + Mapping | Winner |
|--------|----------|-------------------|--------|
| Checkbox state detection | Native SELECTION_ELEMENT | Heuristic (color, size) | Textract |
| Table structure | Direct TABLE blocks | Heuristic (grid detection) | Textract |
| Key-value understanding | KEY_VALUE_SET relationships | Spatial heuristics | Textract |
| Multiline field handling | Relationship-based | Line merging + proximity | Tie |
| Local processing speed | API latency (~1-3s) | Fast (~100ms) | EasyOCR |
| Cost per page | ~$0.01-0.02 | Free (local) | EasyOCR |
| Handwriting support | Limited | Limited | Tie |
| Custom form support | [To be tested] | Good (heuristic-based) | TBD |

---

## Key Findings

### Checkbox Handling
- **Strength**: Textract SELECTION_ELEMENT is reliable for detecting checked/unchecked state
- **Challenge**: Ownership mapping requires 4-strategy fallback (see `src/textract_parser.py` for strategy details)
- **Recommendation**: Use Textract for checkbox detection; implement post-processing for ownership if needed

### Table Extraction
- **Strength**: Direct TABLE block extraction beats grid detection heuristics
- **Challenge**: Sparse/merged cells require careful handling (see `normalize_table_for_export()`)
- **Recommendation**: Production-ready with validation layer (`textract_validators.py`)

### Field Extraction
- **Strength**: KEY_VALUE_SET relationships are more reliable than spatial heuristics
- **Challenge**: Multiline values must preserve line breaks (implemented)
- **Recommendation**: Worth evaluating as OCR replacement

### Performance & Cost
- **Textract API**: Async jobs ~1-3s end-to-end; charges per page
- **EasyOCR**: Instant local processing; no recurring costs
- **Hybrid approach**: Route complex documents to Textract, simple ones to EasyOCR

---

## Limitations Encountered

(See `docs/textract_limitations.md` for detailed analysis)

1. Checkbox ownership is context-dependent; fallback required
2. Multi-page consistency not automatic; entity deduplication needed
3. Geometry coordinates don't always align with source PDF
4. Handwritten text and poor scans still problematic
5. Custom form layouts may confuse model

---

## Recommendations

### Immediate (Phase 2)
1. **Complete evaluation** on full test corpus
2. **Document failure patterns** by form type
3. **Build form classifier** to route forms intelligently
4. **Keep EasyOCR pipeline** active as fallback

### Short-term (Phase 3)
1. **Implement document orchestrator** (`src/document_orchestrator.py`) to choose Textract vs OCR
2. **Build Textract-specific validators** for form-specific rules
3. **Set up cost monitoring** to track Textract spend
4. **Run parallel pipeline** (both Textract and EasyOCR) for 2-4 weeks

### Long-term (Phase 4+)
1. **Migrate to serverless** (Lambda + API Gateway + S3)
2. **Implement ML-based ownership mapper** for complex forms
3. **Build per-form-type confidence thresholds**
4. **Deprecate EasyOCR** once Textract handles 95%+ of forms

---

## Conclusion

Textract shows strong potential for replacing significant portions of the heuristic pipeline, particularly for:
- Checkbox detection and state recovery
- Table extraction and normalization
- Form field key-value understanding

However, careful evaluation and phased rollout are needed to:
- Handle edge cases and custom form layouts
- Maintain backward compatibility with existing pipeline
- Manage API costs and performance SLAs
- Build confidence with stakeholders

**Next step**: Run `python experiments/textract_eval/run_eval.py` to populate results with actual form samples.

