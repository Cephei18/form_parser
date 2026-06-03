<#
.SYNOPSIS
  Package and deploy the four async API Lambda handlers (code-only).

.DESCRIPTION
  Zips each handler in lambda/<name>/lambda_function.py and pushes it with
  lambda:UpdateFunctionCode. The functions already exist as placeholder stubs
  (py3.12 Zip, handler lambda_function.lambda_handler, shared exec role), so
  this is a pure code swap — no create, no config change, no IAM.

  boto3 ships in the py3.12 runtime, so the handlers need no bundled deps.

  REVERSIBLE: run with -Backup first to snapshot the current deployed zips into
  scripts/.lambda_backup/, then -Restore to roll any/all functions back to the
  saved stub.

  PREREQUISITES:
    * The deploy identity has lambda:UpdateFunctionCode on the 4 functions.
      (As intern.form-pdf-poc this is the one deploy action likely permitted;
       all wiring — routes/triggers/IAM — is in wire_async_infra.ps1 and needs
       an elevated identity.)

.EXAMPLE
  pwsh scripts/deploy_async_handlers.ps1 -Backup        # snapshot current code
  pwsh scripts/deploy_async_handlers.ps1                # deploy all 4 handlers
  pwsh scripts/deploy_async_handlers.ps1 -Only presigned-url
  pwsh scripts/deploy_async_handlers.ps1 -Restore       # roll back to snapshots
#>
param(
  [string]$Profile = "form-pdf-poc",
  [string]$Region  = "ap-south-1",
  [string]$Only    = "",
  [switch]$Backup,
  [switch]$Restore
)

$ErrorActionPreference = "Stop"
$RepoRoot   = Split-Path -Parent $PSScriptRoot
$LambdaRoot = Join-Path $RepoRoot "lambda"
$BackupDir  = Join-Path $PSScriptRoot ".lambda_backup"
$WorkDir    = Join-Path ([System.IO.Path]::GetTempPath()) "form_parser_lambda_build"

# function name  ->  source directory under lambda/
$Handlers = @{
  "form-pdf-poc-dev-presigned-url"   = "presigned_url"
  "form-pdf-poc-dev-upload-handler"  = "upload_handler"
  "form-pdf-poc-dev-job-status"      = "job_status"
  "form-pdf-poc-dev-result-handler"  = "result_handler"
}

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }

$targets = $Handlers.GetEnumerator()
if ($Only) {
  $targets = $targets | Where-Object { $_.Key -eq "form-pdf-poc-dev-$Only" -or $_.Key -eq $Only }
  if (-not $targets) { throw "No handler matches -Only '$Only'." }
}

if ($Backup) {
  New-Item -ItemType Directory -Force -Path $BackupDir | Out-Null
  foreach ($h in $targets) {
    $fn = $h.Key
    Step "Backing up current code for $fn"
    $loc = aws lambda get-function --function-name $fn --query "Code.Location" --output text --profile $Profile --region $Region
    Invoke-WebRequest -Uri $loc -OutFile (Join-Path $BackupDir "$fn.zip")
    Write-Host "  saved $BackupDir\$fn.zip" -ForegroundColor Gray
  }
  Write-Host "`nBackup complete. Roll back later with: -Restore" -ForegroundColor Yellow
  return
}

if ($Restore) {
  foreach ($h in $targets) {
    $fn = $h.Key
    $zip = Join-Path $BackupDir "$fn.zip"
    if (-not (Test-Path $zip)) { Write-Host "  no backup for $fn; skipping" -ForegroundColor DarkYellow; continue }
    Step "Restoring $fn from backup"
    aws lambda update-function-code --function-name $fn --zip-file "fileb://$zip" `
      --profile $Profile --region $Region | Out-Null
    aws lambda wait function-updated --function-name $fn --profile $Profile --region $Region
  }
  Write-Host "`nRestore complete." -ForegroundColor Yellow
  return
}

# Deploy ---------------------------------------------------------------------
New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
foreach ($h in $targets) {
  $fn  = $h.Key
  $src = Join-Path $LambdaRoot $h.Value
  $py  = Join-Path $src "lambda_function.py"
  if (-not (Test-Path $py)) { throw "Missing handler source: $py" }

  $zip = Join-Path $WorkDir "$fn.zip"
  if (Test-Path $zip) { Remove-Item $zip -Force }
  Step "Packaging $fn  <-  lambda/$($h.Value)/lambda_function.py"
  Compress-Archive -Path $py -DestinationPath $zip -Force

  Step "Updating function code: $fn"
  aws lambda update-function-code --function-name $fn --zip-file "fileb://$zip" `
    --profile $Profile --region $Region | Out-Null
  aws lambda wait function-updated --function-name $fn --profile $Profile --region $Region
  Write-Host "  deployed $fn" -ForegroundColor Green
}

Write-Host "`nDone. Next: wire routes/triggers/IAM via scripts/wire_async_infra.ps1 (admin identity)." -ForegroundColor Yellow
