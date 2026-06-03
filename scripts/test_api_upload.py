import os
import requests
from pathlib import Path

API = os.getenv("FORM_PARSER_TEST_API", "http://127.0.0.1:8000/process-form")
PDF = Path("input/form.pdf")
if not PDF.exists():
    raise SystemExit("input/form.pdf not found")

with PDF.open("rb") as fh:
    files = {"file": (PDF.name, fh, "application/pdf")}
    data = {"mode": "ml"}
    resp = requests.post(API, files=files, data=data)
    print(resp.status_code)
    try:
        print(resp.json())
    except Exception:
        print(resp.text)
