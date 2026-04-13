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
   - On Windows, PDF conversion requires Poppler available in `PATH`.
4. Run the app:
   - `python src/main.py`
