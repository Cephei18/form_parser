# Stop any existing uvicorn processes and start uvicorn with logs redirected to server.log
Get-Process -Name uvicorn -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Milliseconds 300

# Run backend using Textract pipeline on port 8001 for local testing.
# Set the pipeline mode environment variable so the running process picks up Textract.
$env:FORM_PARSER_PIPELINE_MODE = "textract"

$python = "d:\form_parser\\.venv\\Scripts\\python.exe"
$args = "-m uvicorn src.api:app --reload --host 0.0.0.0 --port 8001"
Start-Process -FilePath $python -ArgumentList $args -RedirectStandardOutput "server.log" -RedirectStandardError "server.err.log" -NoNewWindow | Out-Null
Write-Output "uvicorn started on port 8001 with FORM_PARSER_PIPELINE_MODE=textract, logs -> server.log / server.err.log"
