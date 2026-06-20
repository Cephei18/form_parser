import os
import requests
import sys


def main():
    # URL can be provided via env FORM_PARSER_BASE_URL or as first CLI arg
    base = os.getenv("FORM_PARSER_BASE_URL")
    if len(sys.argv) > 1:
        base = sys.argv[1]
    if not base:
        base = "http://127.0.0.1:8000"

    url = base.rstrip("/") + "/process-form"
    filepath = "input/form.pdf"
    try:
        with open(filepath, "rb") as fh:
            files = {"file": ("form.pdf", fh, "application/pdf")}
            data = {"mode": "rule"}
            print(f"Posting {filepath} -> {url}")
            resp = requests.post(url, files=files, data=data, timeout=300)
            print("Status:", resp.status_code)
            print("Response headers:\n", resp.headers)
            print("Body:\n", resp.text)
            if resp.status_code != 200:
                sys.exit(1)
    except Exception as exc:
        print("E2E test failed:", exc)
        sys.exit(2)


if __name__ == "__main__":
    main()
