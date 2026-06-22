import sys
import requests
from pathlib import Path

API = "http://localhost:8000/process-form"
INPUT = Path(__file__).resolve().parents[1] / "input"

def main():
    files = list(INPUT.glob("*"))
    if not files:
        print("No files found in input/ to upload.")
        sys.exit(1)
    sample = files[0]
    print(f"Uploading {sample.name} to {API}")
    with open(sample, "rb") as fh:
        # Explicitly set content type to ensure FastAPI detects the uploaded file type
        mime = "application/pdf" if sample.suffix.lower() == ".pdf" else "image/png"
        resp = requests.post(API, files={"file": (sample.name, fh, mime)}, data={"mode": "rule"}, timeout=600)
    try:
        print(resp.status_code)
        print(resp.json())
    except Exception:
        print(resp.text)

if __name__ == "__main__":
    main()
