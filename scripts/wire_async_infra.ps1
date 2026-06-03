<#
.SYNOPSIS
  Wire the async Textract orchestration: IAM, worker env, SQS ESM, S3 trigger,
  API Gateway routes, and bucket CORS. REQUIRES AN ELEVATED (admin/CI) IDENTITY.

.DESCRIPTION
  Every action here is gated off the deploy identity (intern.form-pdf-poc is
  denied iam:*, apigateway:*, s3:PutBucketNotification/Cors, and most likely
  lambda:CreateEventSourceMapping/AddPermission — see docs/iam_permission_matrix.md).
  Run this with a principal that holds those permissions.

  Each slice is a switch so it can be applied and verified independently. With
  no switch, -All runs every slice in dependency order. The script is written to
  be idempotent (safe to re-run); creation calls tolerate "already exists".

  Slices (matches docs/orchestration_implementation_plan.md sequencing):
    -Role       put the runtime policy on the shared exec role (infra/lambda_exec_role_policy.json)
    -WorkerEnv  add DDB_TABLE to the worker so it drives the DDB state machine
    -ApiRoutes  POST /uploads, GET /jobs/{job_id}, GET /result/{job_id} + invoke perms
    -S3Notify   raw-bucket ObjectCreated(uploads/) -> upload-handler + invoke perm
    -Esm        SQS -> worker event-source mapping (batch=1, max-concurrency)
    -Cors       raw (POST/PUT) + processed (GET) bucket CORS for the frontend origin
    -All        all of the above

.EXAMPLE
  pwsh scripts/wire_async_infra.ps1 -All -FrontendOrigin https://app.example.com
  pwsh scripts/wire_async_infra.ps1 -ApiRoutes
  pwsh scripts/wire_async_infra.ps1 -Esm

.NOTES
  Rollback pointers are printed by each slice. Master user-facing rollback is the
  frontend flag NEXT_PUBLIC_TEXTRACT_ASYNC=false (no infra teardown needed).
#>
param(
  [string]$Profile         = "form-pdf-poc-admin",
  [string]$Region          = "ap-south-1",
  [string]$AccountId       = "637423601842",
  [string]$ApiId           = "58is64i9kb",
  [string]$RoleName        = "FormPdfPocLambdaExecutionRole",
  [string]$RawBucket       = "form-pdf-poc-dev-raw-documents",
  [string]$ProcessedBucket = "form-pdf-poc-dev-processed-documents",
  [string]$QueueArn        = "arn:aws:sqs:ap-south-1:637423601842:form-pdf-poc-dev-processing-queue",
  [string]$DdbTable        = "form-pdf-poc-dev-jobs",
  [string]$FrontendOrigin  = "https://REPLACE_WITH_FRONTEND_ORIGIN",
  [int]$MaxConcurrency     = 5,
  [switch]$Role,
  [switch]$WorkerEnv,
  [switch]$ApiRoutes,
  [switch]$S3Notify,
  [switch]$Esm,
  [switch]$Cors,
  [switch]$All
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Worker        = "form-pdf-poc-dev-worker"
$UploadHandler = "form-pdf-poc-dev-upload-handler"
$Presigned     = "form-pdf-poc-dev-presigned-url"
$JobStatus     = "form-pdf-poc-dev-job-status"
$ResultHandler = "form-pdf-poc-dev-result-handler"

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Info($msg) { Write-Host "    $msg" -ForegroundColor Gray }
function Tolerate($block) { try { & $block } catch { Write-Host "    (tolerated) $($_.Exception.Message)" -ForegroundColor DarkYellow } }

function FnArn($name) { "arn:aws:lambda:${Region}:${AccountId}:function:$name" }

if (-not ($Role -or $WorkerEnv -or $ApiRoutes -or $S3Notify -or $Esm -or $Cors)) { $All = $true }

# --- Slice: exec role policy -------------------------------------------------
if ($Role -or $All) {
  Step "Attaching runtime policy to $RoleName"
  $policy = Join-Path $RepoRoot "infra/lambda_exec_role_policy.json"
  aws iam put-role-policy --role-name $RoleName --policy-name "form-pdf-poc-async-runtime" `
    --policy-document "file://$policy" --profile $Profile
  Info "Rollback: aws iam delete-role-policy --role-name $RoleName --policy-name form-pdf-poc-async-runtime"
}

# --- Slice: worker DDB_TABLE env --------------------------------------------
if ($WorkerEnv -or $All) {
  Step "Adding DDB_TABLE to $Worker (enables the worker's DDB state machine)"
  # update-function-configuration REPLACES the env map, so merge with existing.
  $envJson = aws lambda get-function-configuration --function-name $Worker `
    --query "Environment.Variables" --output json --profile $Profile --region $Region
  $vars = $envJson | ConvertFrom-Json
  $map = @{}
  foreach ($p in $vars.PSObject.Properties) { $map[$p.Name] = $p.Value }
  $map["DDB_TABLE"] = $DdbTable
  $pairs = ($map.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) -join ","
  aws lambda update-function-configuration --function-name $Worker `
    --environment "Variables={$pairs}" --profile $Profile --region $Region | Out-Null
  aws lambda wait function-updated --function-name $Worker --profile $Profile --region $Region
  Info "Rollback: re-run update-function-configuration without DDB_TABLE (worker then skips DDB writes)."
}

# --- Slice: API Gateway routes ----------------------------------------------
if ($ApiRoutes -or $All) {
  $routes = @(
    @{ Key = "POST /uploads";            Fn = $Presigned     },
    @{ Key = "GET /jobs/{job_id}";       Fn = $JobStatus     },
    @{ Key = "GET /result/{job_id}";     Fn = $ResultHandler }
  )
  foreach ($r in $routes) {
    Step "Wiring route '$($r.Key)' -> $($r.Fn)"
    $intId = aws apigatewayv2 create-integration --api-id $ApiId `
      --integration-type AWS_PROXY --integration-uri (FnArn $r.Fn) `
      --payload-format-version "2.0" --integration-method POST `
      --query "IntegrationId" --output text --profile $Profile --region $Region
    Info "integration $intId"
    Tolerate { aws apigatewayv2 create-route --api-id $ApiId --route-key $r.Key `
      --target "integrations/$intId" --profile $Profile --region $Region | Out-Null }

    # Allow API Gateway to invoke the function for this route. Path parameters
    # ({job_id}) MUST be a wildcard in the source ARN — the literal token does
    # not match at invoke time and API Gateway would return 500 (not authorized).
    $method   = ($r.Key -split " ")[0]
    $path     = ($r.Key -split " ")[1]
    $arnPath  = [regex]::Replace($path, "\{[^}]+\}", "*")
    $stmtId   = "apigw-" + ($r.Fn -replace "[^A-Za-z0-9]", "-")
    Tolerate { aws lambda add-permission --function-name $r.Fn --statement-id $stmtId `
      --action lambda:InvokeFunction --principal apigateway.amazonaws.com `
      --source-arn "arn:aws:execute-api:${Region}:${AccountId}:${ApiId}/*/$method$arnPath" `
      --profile $Profile --region $Region | Out-Null }
  }
  Step "Configuring API CORS for $FrontendOrigin"
  # JSON (not shorthand): AllowMethods is a comma-containing list and would
  # collide with the Key=val,Key=val shorthand parser.
  $corsCfg = @{
    AllowOrigins = @($FrontendOrigin)
    AllowMethods = @("GET", "POST", "OPTIONS")
    AllowHeaders = @("content-type")
    MaxAge       = 300
  } | ConvertTo-Json -Compress
  $corsTmp = Join-Path ([System.IO.Path]::GetTempPath()) "api_cors.json"
  $corsCfg | Out-File -FilePath $corsTmp -Encoding utf8
  aws apigatewayv2 update-api --api-id $ApiId --cors-configuration "file://$corsTmp" `
    --profile $Profile --region $Region | Out-Null
  Info "Rollback: aws apigatewayv2 delete-route / delete-integration for the IDs above."
}

# --- Slice: S3 -> upload-handler notification --------------------------------
if ($S3Notify -or $All) {
  Step "Granting S3 permission to invoke $UploadHandler"
  Tolerate { aws lambda add-permission --function-name $UploadHandler --statement-id "s3-raw-invoke" `
    --action lambda:InvokeFunction --principal s3.amazonaws.com `
    --source-arn "arn:aws:s3:::$RawBucket" --source-account $AccountId `
    --profile $Profile --region $Region | Out-Null }

  Step "Configuring raw-bucket ObjectCreated(uploads/) -> $UploadHandler"
  $cfg = @{
    LambdaFunctionConfigurations = @(
      @{
        LambdaFunctionArn = (FnArn $UploadHandler)
        Events            = @("s3:ObjectCreated:*")
        Filter            = @{ Key = @{ FilterRules = @(@{ Name = "prefix"; Value = "uploads/" }) } }
      }
    )
  } | ConvertTo-Json -Depth 10
  $tmp = Join-Path ([System.IO.Path]::GetTempPath()) "raw_notify.json"
  $cfg | Out-File -FilePath $tmp -Encoding utf8
  aws s3api put-bucket-notification-configuration --bucket $RawBucket `
    --notification-configuration "file://$tmp" --profile $Profile --region $Region
  Info "Rollback: put-bucket-notification-configuration with an empty {} (stops enqueues instantly)."
}

# --- Slice: SQS -> worker event-source mapping -------------------------------
if ($Esm -or $All) {
  Step "Creating SQS -> $Worker event-source mapping (batch=1, max-concurrency=$MaxConcurrency)"
  Tolerate {
    aws lambda create-event-source-mapping --function-name $Worker `
      --event-source-arn $QueueArn --batch-size 1 `
      --scaling-config "MaximumConcurrency=$MaxConcurrency" `
      --profile $Profile --region $Region | Out-Null
  }
  Info "Verify: aws lambda list-event-source-mappings --function-name $Worker --profile $Profile --region $Region"
  Info "Rollback: aws lambda delete-event-source-mapping --uuid <UUID> (stops consumption; in-flight ages to DLQ)."
}

# --- Slice: bucket CORS ------------------------------------------------------
if ($Cors -or $All) {
  if ($FrontendOrigin -like "*REPLACE_WITH*") { throw "Pass -FrontendOrigin <https://your-app> for CORS." }
  Step "Applying raw-bucket CORS (browser POST/PUT)"
  $rawCors = (Get-Content (Join-Path $RepoRoot "infra/raw_bucket_cors.json") -Raw).Replace("https://REPLACE_WITH_FRONTEND_ORIGIN", $FrontendOrigin)
  $rawTmp = Join-Path ([System.IO.Path]::GetTempPath()) "raw_cors.json"
  $rawCors | Out-File -FilePath $rawTmp -Encoding utf8
  aws s3api put-bucket-cors --bucket $RawBucket --cors-configuration "file://$rawTmp" --profile $Profile --region $Region

  Step "Applying processed-bucket CORS (browser GET)"
  $procCors = (Get-Content (Join-Path $RepoRoot "infra/processed_bucket_cors.json") -Raw).Replace("https://REPLACE_WITH_FRONTEND_ORIGIN", $FrontendOrigin)
  $procTmp = Join-Path ([System.IO.Path]::GetTempPath()) "proc_cors.json"
  $procCors | Out-File -FilePath $procTmp -Encoding utf8
  aws s3api put-bucket-cors --bucket $ProcessedBucket --cors-configuration "file://$procTmp" --profile $Profile --region $Region
  Info "Rollback: aws s3api delete-bucket-cors --bucket <bucket>"
}

Write-Host "`nWiring step(s) complete. Validate end-to-end with a real upload, then flip NEXT_PUBLIC_TEXTRACT_ASYNC=true." -ForegroundColor Green
