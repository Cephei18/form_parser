Textract evaluation scripts and outputs.

Files:
- check_textract_access.py: small script to verify AWS credentials and Textract access using an in-memory image. Writes `access_result.json` and `raw_response.json` in this folder.

Usage:
Use the project virtualenv (.venv) and run:

```powershell
& .\.venv\Scripts\Activate.ps1
python experiments/textract_eval/check_textract_access.py
```
