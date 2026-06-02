# Textract Worker — first Lambda deployment runbook

First serverless deployment of the Textract worker as a **Lambda container image**.
Additive and reversible; does not touch the OCR pipeline or the existing Zip
Lambdas (`upload-handler`, `presigned-url`, `result-handler`, `job-status`).

## Targets
| | |
|---|---|
| ECR repo | `637423601842.dkr.ecr.ap-south-1.amazonaws.com/form-pdf-poc-dev-worker` |
| Function | `form-pdf-poc-dev-worker` (does not exist yet) |
| Region | `ap-south-1` |
| Execution role | `arn:aws:iam::637423601842:role/FormPdfPocLambdaExecutionRole` (existing) |
| Image | `Dockerfile.lambda` → CMD `src.lambda_worker.handler` |

## Recommended config
- Timeout **120s** (Textract ~4–6s + cold start + render/publish headroom)
- Memory **3008 MB** (OpenCV/ReportLab/PDF render are CPU-bound; CPU scales with memory)
- Ephemeral `/tmp` **1024 MB** (artifacts ~1.3 MB/job; headroom for large scans)
- Architecture **x86_64** (image built on Docker Desktop / Windows AMD64)
- Env: `FORM_PARSER_PIPELINE_MODE=textract`, `FORM_PARSER_ARTIFACT_BACKEND=s3`,
  `FORM_PARSER_PROCESSED_BUCKET=form-pdf-poc-dev-processed-documents`,
  `FORM_PARSER_ARTIFACT_PREFIX=textract`, `FORM_PARSER_AWS_REGION=ap-south-1`,
  `FORM_PARSER_WORK_ROOT=/tmp/form_parser_jobs`, `FORM_PARSER_WORKER_CLEANUP=true`

## Prerequisites (verify first)
1. **Docker Desktop running** (`docker version` must reach the daemon).
2. **Principal permissions** for the deploy identity:
   - ECR push: `ecr:GetAuthorizationToken`, `BatchCheckLayerAvailability`,
     `InitiateLayerUpload`, `UploadLayerPart`, `CompleteLayerUpload`, `PutImage`.
     *(Auth + layer-check verified for `intern.form-pdf-poc`; full push perms to confirm on first push.)*
   - Lambda: `lambda:CreateFunction` (first run) + `lambda:UpdateFunctionCode/Configuration`.
   - `iam:PassRole` on `FormPdfPocLambdaExecutionRole`.
   *(These admin actions appear restricted for `intern.form-pdf-poc` — `GetAccountSettings`,
   `ListRoles`, `ListImages` are denied. If `CreateFunction`/`PassRole` are denied, run the
   deploy with an admin/CI role.)*
3. **Execution role policy** must include `textract:AnalyzeDocument` (plus the S3
   get/put it already has for the other functions). If missing, the first invocation
   fails with AccessDenied on Textract — add the action to the role.

## Deploy
```powershell
# one command (build + push + create/update):
pwsh scripts/deploy_lambda_worker.ps1

# iterate on code only later:
pwsh scripts/deploy_lambda_worker.ps1 -CodeOnly
```

## Validate the deployed function
```powershell
aws lambda invoke --function-name form-pdf-poc-dev-worker `
  --payload fileb://scripts/lambda_test_event.json `
  --profile form-pdf-poc --region ap-south-1 out.json
Get-Content out.json        # expect status":"succeeded", checkboxes_detected, artifacts base_uri

# CloudWatch logs
aws logs tail /aws/lambda/form-pdf-poc-dev-worker --since 10m --profile form-pdf-poc --region ap-south-1

# Confirm artifacts + checkbox rendering in S3 output
aws s3 ls s3://form-pdf-poc-dev-processed-documents/textract/lambda-smoke-1/ --profile form-pdf-poc --region ap-south-1
```
The smoke object `uploads/mapping.png` (a checkbox form) is already in the raw bucket
from local validation; the deployed run should reproduce **15 `/Btn` checkbox widgets**.

## Rollback
```powershell
pwsh scripts/deploy_lambda_worker.ps1 -Delete   # remove the function; ECR image kept
```
Nothing else is mutated; the OCR pipeline and other Lambdas are untouched.

## IAM verification (ADMIN — required before first deploy)

The scoped `intern.form-pdf-poc` user has **no IAM read/simulate permissions**
(`iam:GetRole`, `ListAttachedRolePolicies`, `ListRolePolicies`,
`SimulatePrincipalPolicy` all denied), so the two gates below must be confirmed by
an **admin / IAM-as-code owner** with an elevated profile.

```bash
ADMIN=<admin-profile>; REGION=ap-south-1; ACCT=637423601842
ROLE=arn:aws:iam::$ACCT:role/FormPdfPocLambdaExecutionRole

# Gate 1 — does the EXECUTION ROLE allow Textract + the S3 it needs?
aws iam simulate-principal-policy --profile $ADMIN \
  --policy-source-arn $ROLE \
  --action-names textract:AnalyzeDocument s3:GetObject s3:PutObject \
  --resource-arns "*" \
    arn:aws:s3:::form-pdf-poc-dev-raw-documents/* \
    arn:aws:s3:::form-pdf-poc-dev-processed-documents/* \
  --query 'EvaluationResults[*].[EvalActionName,EvalDecision]' --output text

# Gate 2 — can the DEPLOY identity create the function + pass the role + push?
aws iam simulate-principal-policy --profile $ADMIN \
  --policy-source-arn <deploy-identity-arn> \
  --action-names lambda:CreateFunction iam:PassRole ecr:PutImage \
  --resource-arns $ROLE "*" \
  --query 'EvaluationResults[*].[EvalActionName,EvalDecision]' --output text
```

If Gate 1 shows `textract:AnalyzeDocument = implicitDeny`, attach the bundled policy
(idempotent; S3 statements are no-ops if already granted):

```bash
aws iam put-role-policy --profile $ADMIN \
  --role-name FormPdfPocLambdaExecutionRole \
  --policy-name worker-textract \
  --policy-document file://infra/worker_role_textract_policy.json
```

If Gate 2 denies `lambda:CreateFunction` / `iam:PassRole`, run the deploy with the
admin/CI role instead of the intern user (the ECR push itself is permitted for the
intern user). Only once both gates pass should the image be built and deployed.

## Not in this slice (deferred by design)
S3-trigger wiring, SQS consumption, API Gateway, DynamoDB job-status writes. The
worker's `JobResult` is already shaped to drop into a DynamoDB `PutItem` later.
