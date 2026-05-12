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

Errors return a structured JSON response with `status`, `detail`, and `error`.

## Ubuntu / EC2 notes

Install Python dependencies from `requirements.txt`. For best PDF conversion support, also install Poppler:

```bash
sudo apt-get update
sudo apt-get install -y poppler-utils
pip install -r requirements.txt
```

EasyOCR loads lazily in CPU mode on the first request and is cached for later requests. For production containers, prewarm or persist the EasyOCR model cache so the runtime does not depend on downloading models during the first upload.
