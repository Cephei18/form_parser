<#
.SYNOPSIS
  Build, push, and deploy the Textract Lambda worker container image.

.DESCRIPTION
  First real deployment of the serverless Textract worker:
    1. docker build  (Dockerfile.lambda, linux/amd64)
    2. ECR login + push to the existing form-pdf-poc-dev-worker repo
    3. create-or-update the form-pdf-poc-dev-worker Lambda from the image

  Idempotent: if the function exists it updates code (+ config unless -CodeOnly);
  otherwise it creates it. Reversible: see -Delete to remove the function (the
  ECR image is left intact).

  PREREQUISITES (confirm before running):
    * Docker Desktop running.
    * The AWS principal can: ecr push, lambda:CreateFunction (first run),
      lambda:UpdateFunction*, and iam:PassRole on the execution role.
    * The execution role allows: s3:GetObject (raw), s3:PutObject (processed),
      textract:AnalyzeDocument, and CloudWatch Logs.

.EXAMPLE
  pwsh scripts/deploy_lambda_worker.ps1                 # full build + push + deploy
  pwsh scripts/deploy_lambda_worker.ps1 -SkipBuild      # reuse local image, push + deploy
  pwsh scripts/deploy_lambda_worker.ps1 -CodeOnly       # update image only (keep config)
  pwsh scripts/deploy_lambda_worker.ps1 -Delete         # tear down the function
#>
param(
  [string]$Profile      = "form-pdf-poc",
  [string]$Region       = "ap-south-1",
  [string]$AccountId    = "637423601842",
  [string]$Repo         = "form-pdf-poc-dev-worker",
  [string]$Tag          = "textract-lambda",
  [string]$FunctionName = "form-pdf-poc-dev-worker",
  [string]$RoleArn      = "arn:aws:iam::637423601842:role/FormPdfPocLambdaExecutionRole",
  [int]$TimeoutSec      = 120,
  [int]$MemoryMb        = 3008,
  [int]$EphemeralMb     = 1024,
  [switch]$SkipBuild,
  [switch]$CodeOnly,
  [switch]$Delete
)

$ErrorActionPreference = "Stop"
$RegistryHost = "$AccountId.dkr.ecr.$Region.amazonaws.com"
$EcrUri       = "$RegistryHost/$Repo"
$ImageUri     = "${EcrUri}:${Tag}"

# Environment variables for the function (no commas in any value --- shorthand-safe).
$EnvVars = "Variables={" + (@(
  "FORM_PARSER_PIPELINE_MODE=textract",
  "FORM_PARSER_ARTIFACT_BACKEND=s3",
  "FORM_PARSER_PROCESSED_BUCKET=form-pdf-poc-dev-processed-documents",
  "FORM_PARSER_ARTIFACT_PREFIX=textract",
  "FORM_PARSER_AWS_REGION=$Region",
  "FORM_PARSER_WORK_ROOT=/tmp/form_parser_jobs",
  "FORM_PARSER_WORKER_CLEANUP=true"
) -join ",") + "}"

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }

if ($Delete) {
  Step "Deleting Lambda function $FunctionName (ECR image left intact)"
  aws lambda delete-function --function-name $FunctionName --profile $Profile --region $Region
  Write-Host "Deleted. Re-run without -Delete to recreate." -ForegroundColor Yellow
  return
}

# 1) Build ------------------------------------------------------------------
if (-not $SkipBuild) {
  Step "Building image $ImageUri (linux/amd64)"
  # --provenance=false : BuildKit otherwise emits an attestation manifest LIST,
  # which Lambda rejects ("image manifest ... not supported"). Lambda needs a
  # single-arch image manifest.
  docker build --platform linux/amd64 --provenance=false -f Dockerfile.lambda -t $ImageUri .
  if ($LASTEXITCODE -ne 0) { throw "docker build failed" }
} else {
  Step "Skipping build (-SkipBuild); using local $ImageUri"
}

# 2) Login + push -----------------------------------------------------------
Step "Authenticating Docker to ECR ($RegistryHost)"
aws ecr get-login-password --profile $Profile --region $Region | docker login --username AWS --password-stdin $RegistryHost
if ($LASTEXITCODE -ne 0) { throw "ECR login failed" }

Step "Pushing $ImageUri"
docker push $ImageUri
if ($LASTEXITCODE -ne 0) { throw "docker push failed (check ecr:PutImage / layer-upload permissions)" }

# 3) Create or update -------------------------------------------------------
$exists = $true
try {
  aws lambda get-function-configuration --function-name $FunctionName --profile $Profile --region $Region 2>$null | Out-Null
  if ($LASTEXITCODE -ne 0) { $exists = $false }
} catch { $exists = $false }

if ($exists) {
  Step "Function exists --- updating image"
  aws lambda update-function-code --function-name $FunctionName --image-uri $ImageUri `
    --profile $Profile --region $Region | Out-Null
  aws lambda wait function-updated --function-name $FunctionName --profile $Profile --region $Region

  if (-not $CodeOnly) {
    Step "Updating configuration (timeout=$TimeoutSec memory=$MemoryMb ephemeral=$EphemeralMb)"
    aws lambda update-function-configuration --function-name $FunctionName `
      --timeout $TimeoutSec --memory-size $MemoryMb --ephemeral-storage "Size=$EphemeralMb" `
      --environment $EnvVars --profile $Profile --region $Region | Out-Null
    aws lambda wait function-updated --function-name $FunctionName --profile $Profile --region $Region
  }
} else {
  Step "Creating function $FunctionName from image"
  aws lambda create-function --function-name $FunctionName `
    --package-type Image --code "ImageUri=$ImageUri" `
    --role $RoleArn `
    --architectures x86_64 `
    --timeout $TimeoutSec --memory-size $MemoryMb --ephemeral-storage "Size=$EphemeralMb" `
    --environment $EnvVars `
    --profile $Profile --region $Region | Out-Null
  aws lambda wait function-active --function-name $FunctionName --profile $Profile --region $Region
}

Step "Deployed. Validate the function with:"
Write-Host "  aws lambda invoke --function-name $FunctionName --payload fileb://scripts/lambda_test_event.json --profile $Profile --region $Region out.json ; Get-Content out.json" -ForegroundColor Gray
Write-Host "  aws logs tail /aws/lambda/$FunctionName --since 10m --profile $Profile --region $Region" -ForegroundColor Gray
