# Stop any existing uvicorn processes and start uvicorn with logs redirected to server.log
Get-Process -Name uvicorn -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Milliseconds 300
& d:\form_parser\.venv\Scripts\python.exe -m uvicorn src.api:app --reload --host 0.0.0.0 --port 8000 > server.log 2>&1 &
Write-Output "uvicorn started, logs -> server.log"
