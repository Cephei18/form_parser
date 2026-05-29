# form-parser

## Structure

- `input/` - put test forms here
- `output/` - future outputs (pdf/json)
- `src/` - source code
  - `ocr.py` - OCR logic
  - `utils.py` - helper functions
  - `main.py` - entry point

## Getting started

1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Add one input file:
   - `input/form.png` (preferred), or
   - `input/form.pdf` (first page is auto-converted to `output/form_page_1.png`)
   - PDF conversion uses `pdf2image` when Poppler is available, with PyMuPDF fallback.
4. Run the batch script:
   - `python src/main.py`

## Run API server

Start the FastAPI backend used by the Next.js frontend:

- Local development:
  - `uvicorn src.api:app --host 0.0.0.0 --port 8000`
- EC2 / Ubuntu deployment:
  - `uvicorn src.api:app --host 0.0.0.0 --port 8000`

Environment variables:

- `FORM_PARSER_OUTPUT_DIR` sets the root output directory.
- `FORM_PARSER_UPLOAD_DIR` sets the temporary upload directory.
- `FORM_PARSER_RUNS_DIR` sets the per-request run directory.
- `CORS_ORIGINS` sets allowed frontend origins as a comma-separated list.
- `FORM_PARSER_OCR_LANGUAGES` sets EasyOCR languages as a comma-separated list, defaulting to `en`.
- `FORM_PARSER_EASYOCR_MODEL_DIR` sets an optional model cache directory.
- `FORM_PARSER_EASYOCR_DOWNLOAD_ENABLED` controls whether EasyOCR may download missing models, defaulting to `true`.
- `FORM_PARSER_OCR_THREADS` controls EasyOCR/PyTorch CPU threads, defaulting to `1`.
- `FORM_PARSER_OCR_BATCH_SIZE` controls EasyOCR read batch size, defaulting to `1`.
- `FORM_PARSER_OCR_DIAGNOSTICS_ENABLED` writes `ocr_raw.json` and `ocr_diagnostics.json`, defaulting to `true`.
- `FORM_PARSER_PREPROCESSING_ENABLED` enables the optional preprocessing stage, defaulting to `false`.
- `FORM_PARSER_PREPROCESS_DENOISE`, `FORM_PARSER_PREPROCESS_CONTRAST`, `FORM_PARSER_PREPROCESS_SHARPEN`, `FORM_PARSER_PREPROCESS_ADAPTIVE_THRESHOLD`, `FORM_PARSER_PREPROCESS_SKEW`, and `FORM_PARSER_PREPROCESS_DPI_NORMALIZE` control individual preprocessing steps.
- `FORM_PARSER_DYNAMIC_THRESHOLDS_ENABLED` enables page-relative field filtering thresholds, defaulting to `false` for production-safe behavior.
- `FORM_PARSER_IMAGE_TABLE_FILTERING_ENABLED` enables dense table-region exclusion, defaulting to `false`.
- `FORM_PARSER_FALLBACK_FIELD_LINES_ENABLED` enables OCR-derived field candidates when normal filtering finds no fields, defaulting to `false`.
- `FORM_PARSER_CHECKBOX_DETECTION_ENABLED` enables checkbox mapping candidates, defaulting to `false`.
- `FORM_PARSER_STRUCTURAL_REFINEMENT_ENABLED` enables optional structural reasoning refinements, defaulting to `false`.
- `FORM_PARSER_FIELD_QUALITY_REFINEMENT_ENABLED` controls candidate quality scoring and decorative/table-line removal when structural refinement is enabled.
- `FORM_PARSER_SECTION_GROUPING_ENABLED` controls section and logical-block formation when structural refinement is enabled.
- `FORM_PARSER_OWNERSHIP_PROPAGATION_ENABLED` controls ownership-chain scoring when structural refinement is enabled.
- `FORM_PARSER_TABLE_AWARE_REFINEMENT_ENABLED` controls table-structure penalties and exclusions when structural refinement is enabled.
- `FORM_PARSER_MIN_FIELD_QUALITY_SCORE` controls the optional field-candidate quality cutoff, defaulting to `0.32`.
- `FORM_PARSER_BASELINE_DIR` points at a previous run directory and writes `before_after_comparison.json` for the current run.

Available endpoints:

- `GET /` health message
- `POST /process-form` multipart file upload (`file` field)
- `GET /files/...` generated output files (PDF and mapping preview)

Successful `POST /process-form` responses include:

- `status`
- `pdf_url`
- `mapping_preview`
- `result_url`
- `stats.mapping_count`
- `pipeline_mode`
- `processing_time_ms`
- `response_metadata`

Errors return a structured JSON response with `status`, `detail`, and `error`.

## Local Textract frontend testing

To run the Next.js frontend against the local backend, keep the API contract unchanged and switch the backend pipeline with an environment variable only.

Backend:

```bash
$env:CORS_ORIGINS="http://localhost:3000,http://127.0.0.1:3000"
$env:FORM_PARSER_PIPELINE_MODE="textract"
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

Frontend:

```bash
cd frontend
$env:NEXT_PUBLIC_API_BASE_URL="http://localhost:8000"
npm run dev
```

When the backend is in Textract mode, logs should include:

- `ACTIVE PIPELINE: TEXTRACT`
- `Running Textract pipeline`
- runtime metadata such as processing time and detected table/checkbox counts

The frontend remains intelligence-agnostic and continues to use the same upload flow, result page, and file URLs.

## Phase 1 backend diagnostics

The default pipeline remains production-safe: behavior-changing improvements are behind flags, while diagnostic artifacts are additive.

Capture a baseline run:

```bash
python -c "from pathlib import Path; from src.main import resolve_input_image, run_pipeline; root=Path.cwd(); run_pipeline(resolve_input_image(root), root/'output'/'baselines'/'phase1_pre')"
```

Run a current comparison against that baseline:

```bash
FORM_PARSER_BASELINE_DIR=output/baselines/phase1_pre python -c "from pathlib import Path; from src.main import resolve_input_image, run_pipeline; root=Path.cwd(); run_pipeline(resolve_input_image(root), root/'output'/'baselines'/'phase1_after')"
```

Or compare any two saved run directories:

```bash
python src/pipeline_compare.py --baseline-dir output/baselines/phase1_pre --current-dir output/baselines/phase1_after --output output/baselines/phase1_after/before_after_comparison.json
```

Run isolated structural reasoning experiments:

```bash
python src/structural_evaluation.py
```

Run one structural experiment against one representative sample:

```bash
python src/structural_evaluation.py --samples multiline_reference --experiments section_grouping_only
```

## Ubuntu / EC2 notes

Install Python dependencies from `requirements.txt`. For best PDF conversion support, also install Poppler:

```bash
sudo apt-get update
sudo apt-get install -y poppler-utils
pip install -r requirements.txt
```

EasyOCR loads lazily in CPU mode on the first request and is cached for later requests. For production containers, prewarm or persist the EasyOCR model cache so the runtime does not depend on downloading models during the first upload.
