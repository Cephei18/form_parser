# IAM & permission matrix — form-pdf-poc Textract serverless

Complete permission/trust map for the async Textract architecture so rollout isn't
repeatedly blocked. **Analysis only — no infrastructure changed.** Account
`637423601842`, region `ap-south-1`.

Grounded in verified probes of the current deploy identity (`intern.form-pdf-poc`):
- **Allowed:** `sts:GetCallerIdentity`; `lambda:Get*/List*/GetPolicy`; `ecr:DescribeRepositories`,
  `ecr:GetAuthorizationToken`, `ecr:BatchCheckLayerAvailability`; `s3:ListBucket/GetObject/PutObject`
  (both buckets); `dynamodb:DescribeTable`; `sqs:GetQueueAttributes`; `textract:AnalyzeDocument` (proven).
- **Denied:** all `iam:*` (incl. `PassRole` simulate, `GetRole`, `SimulatePrincipalPolicy`);
  `lambda:ListEventSourceMappings`, `lambda:GetAccountSettings`; `ecr:ListImages`,
  `ecr:GetRepositoryPolicy`; `s3:GetBucketNotification/GetLifecycleConfiguration/GetBucketWebsite`;
  `apigateway:GET`.
- Lambda exec role **`FormPdfPocLambdaExecutionRole`** (shared by all functions); contents unreadable.

---

## 1. Principals (who acts)

| Principal | Identity | Used for |
|---|---|---|
| **Deploy identity** | today `user/intern.form-pdf-poc`; **recommend a deploy role/CI** | build/push, create/update functions, wire routes/triggers |
| **Lambda execution role** | `role/FormPdfPocLambdaExecutionRole` | the runtime principal for **all** functions (presigned-url, upload-handler, job-status, result-handler, worker) |
| **`lambda.amazonaws.com`** | AWS service principal | assumes the exec role; **pulls the container image from ECR** |
| **`s3.amazonaws.com`** | AWS service principal | invokes upload-handler on ObjectCreated |
| **`apigateway.amazonaws.com`** | AWS service principal | invokes the 3 API Lambdas |
| **Browser / presigned holder** | no IAM identity | uses presigned URLs minted with the *exec role's* authority |

Note: **SQS does not invoke Lambda.** The Lambda service *polls* SQS for the worker using the
**worker's execution role** (event-source mapping). So SQS read perms live on the exec role.

---

## 2. Deployment-time permissions (on the DEPLOY identity)

| Action | Permissions | Resource | Verified state |
|---|---|---|---|
| `docker push` to ECR | `ecr:GetAuthorizationToken` (account), `ecr:BatchCheckLayerAvailability`, `InitiateLayerUpload`, `UploadLayerPart`, `CompleteLayerUpload`, `PutImage` | repo `form-pdf-poc-dev-worker` | auth + layer-check **allowed**; full push unverified but likely OK |
| Create worker (container) | **`lambda:CreateFunction`**, **`iam:PassRole`** (on exec role), `ecr:GetDownloadUrlForLayer`, `ecr:BatchGetImage`, **`ecr:GetRepositoryPolicy`**, **`ecr:SetRepositoryPolicy`** | function + role + repo | **GetRepositoryPolicy DENIED**; CreateFunction/PassRole **almost certainly denied** |
| Update 3 zip handlers | `lambda:UpdateFunctionCode`, `lambda:UpdateFunctionConfiguration` | the 3 functions | unverified |
| Add API GW invoke perm | `lambda:AddPermission` (writes the Lambda resource policy) | each API function | unverified |
| Add S3→Lambda invoke perm | `lambda:AddPermission` | upload-handler | unverified |
| Create SQS→worker mapping | `lambda:CreateEventSourceMapping` (+ `GetEventSourceMapping`) | worker + queue | `List*` denied → likely restricted |
| Wire API routes | `apigateway:POST/PATCH` (manage routes/integrations) | API `58is64i9kb` | `apigateway:GET` denied |
| S3 notification | `s3:PutBucketNotification` | raw bucket | `GetBucketNotification` denied |
| S3 CORS | `s3:PutBucketCors` | raw (+processed) | unverified |
| Update exec role policy | `iam:PutRolePolicy`/`AttachRolePolicy` | exec role | **all IAM denied** → admin only |
| Set ECR repo policy | `ecr:SetRepositoryPolicy` | worker repo | denied (Get denied) |

→ **Deployment is gated on an admin/CI identity.** The intern user can push images but
cannot create functions, pass the role, set the ECR repo policy, or edit the exec role.

## 3. Runtime permissions (on the Lambda EXECUTION ROLE)

| Function | DynamoDB | SQS | S3 | Textract | Logs |
|---|---|---|---|---|---|
| **presigned-url** | `PutItem` (jobs) | – | `PutObject` (raw/uploads/*) — *signs the upload URL* | – | ✓ |
| **upload-handler** | `UpdateItem` (jobs) | `SendMessage` (queue) | `GetObject`/Head (raw) | – | ✓ |
| **worker** | `UpdateItem` (jobs) | `ReceiveMessage`,`DeleteMessage`,`GetQueueAttributes` (queue) | `GetObject` (raw), `PutObject` (processed) | `AnalyzeDocument` (`*`) | ✓ |
| **job-status** | `GetItem` (jobs) | – | – | – | ✓ |
| **result-handler** | `GetItem` (jobs) | – | `GetObject` (processed) — *signs download URLs* | – | ✓ |

- **Logs (all):** `logs:CreateLogGroup`,`CreateLogStream`,`PutLogEvents` scoped to
  `arn:aws:logs:ap-south-1:637423601842:log-group:/aws/lambda/form-pdf-poc-dev-*`.
- **GSI queries** (optional, ops/history): `dynamodb:Query` on
  `…table/form-pdf-poc-dev-jobs/index/{status-created_at-index,user_id-created_at-index}`.
- Today **one shared role** holds the union of the above (simplest, current setup). The role
  must trust `lambda.amazonaws.com` (see §4).

## 4. Cross-service trust & resource policies (NOT identity policies)

| Edge | Policy type | Where it lives | Statement |
|---|---|---|---|
| Lambda assumes exec role | **Trust policy** | on `FormPdfPocLambdaExecutionRole` | `Principal: Service lambda.amazonaws.com`, `sts:AssumeRole` |
| **Lambda pulls image** | **ECR repository policy** | on the ECR repo | `Principal: Service lambda.amazonaws.com`; `ecr:BatchGetImage`,`ecr:GetDownloadUrlForLayer`; `Condition: aws:SourceArn = arn:…:function:form-pdf-poc-dev-worker` |
| API GW → Lambda | **Lambda resource policy** | on each API function | `Principal: apigateway.amazonaws.com`, `lambda:InvokeFunction`, `Condition: aws:SourceArn = arn:…:{apiId}/*/{METHOD}{ROUTE}` |
| S3 → Lambda | **Lambda resource policy** | on upload-handler | `Principal: s3.amazonaws.com`, `lambda:InvokeFunction`, `Condition: aws:SourceArn = raw-bucket ARN`, `aws:SourceAccount = 637423601842` |
| Browser → S3 (CORS) | **Bucket CORS** (not IAM) | raw (PUT/POST), processed (GET if fetched via JS) | allow frontend origin + methods/headers |

Currently the 4 functions have **no resource policy** (verified) → API GW / S3 cannot invoke
them yet; these `AddPermission` statements are part of the rollout.

## 5. Frontend / browser (presigned URLs)

- The browser holds **no IAM identity**. A presigned URL carries the **signer's** authority
  (the Lambda exec role). So: upload works only if the role has `s3:PutObject` on the raw
  key; download works only if the role has `s3:GetObject` on the processed key.
- **AccessDenied surfaces at the browser** as a `403` when it uses the URL (not at the Lambda).
- **CORS is required** on the raw bucket for browser PUT/POST (separate from IAM — a 403/CORS
  error if missing).

## 6. Per-interaction matrix (who → whom, principal, AccessDenied mode)

| # | Caller → Callee | Acting principal | Perm location | AccessDenied looks like |
|---|---|---|---|---|
| 1 | deploy → ECR (push) | deploy id | identity | "not authorized: ecr:PutImage" |
| 2 | deploy → Lambda (CreateFunction) | deploy id | identity (+PassRole on role) | "not authorized: lambda:CreateFunction" / "cannot perform iam:PassRole" |
| 3 | **Lambda → ECR (image pull)** | `lambda.amazonaws.com` | **ECR repo policy** | "The image manifest … cannot be accessed" / "Lambda was unable to access the image" |
| 4 | API GW → Lambda | `apigateway.amazonaws.com` | Lambda resource policy | API returns 500; logs "not authorized: lambda:InvokeFunction" |
| 5 | S3 → upload-handler | `s3.amazonaws.com` | Lambda resource policy | event silently dropped; no invocation |
| 6 | Lambda(worker) ← SQS (poll) | **worker exec role** | identity (role) | ESM state "problem"; "insufficient permissions … sqs" |
| 7 | upload-handler → SQS send | exec role | identity | "not authorized: sqs:SendMessage" |
| 8 | worker → Textract | exec role | identity | "not authorized: textract:AnalyzeDocument" |
| 9 | worker → S3 r/w | exec role | identity (+ no bucket-policy deny) | 403 on Get/Put |
| 10 | handlers → DynamoDB | exec role | identity | "not authorized: dynamodb:…Item" |
| 11 | presigned-url signs PUT | exec role | identity (s3:PutObject) | browser PUT → 403 |
| 12 | result-handler signs GET | exec role | identity (s3:GetObject) | browser GET → 403 |
| 13 | all Lambdas → Logs | exec role | identity | logs missing; function still runs |

---

## 7. The container-image pull model (current blocker, in detail)

Lambda container images need **two** independent grants:

1. **Deploy-time (identity):** `CreateFunction` calls validate the image and **auto-attach a
   pull statement to the ECR repo policy**. For that the caller needs `ecr:GetRepositoryPolicy`
   + `ecr:SetRepositoryPolicy` (and `BatchGetImage`,`GetDownloadUrlForLayer`). The intern user
   is **denied `ecr:GetRepositoryPolicy`** → `CreateFunction` will fail to configure pull.
2. **Runtime (resource):** the **ECR repository policy** must allow `lambda.amazonaws.com` to
   `BatchGetImage` + `GetDownloadUrlForLayer` (ideally `Condition aws:SourceArn = worker fn ARN`).
   If absent, even a created function fails at first invoke with an image-access error.

**Exact permissions likely causing the current ECR/image-access failure:**
- On the **deploy identity**: missing `lambda:CreateFunction`, `iam:PassRole`
  (on `FormPdfPocLambdaExecutionRole`), `ecr:GetRepositoryPolicy`, `ecr:SetRepositoryPolicy`,
  `ecr:BatchGetImage`, `ecr:GetDownloadUrlForLayer`.
- On the **ECR repo policy**: missing the `lambda.amazonaws.com` pull statement
  (unverifiable — `GetRepositoryPolicy` denied — therefore must be checked/added by an admin).

Either gap manifests as "deployment blocked / image cannot be accessed."

---

## 8. Permission categories (separated)

- **Deployment-time** (deploy identity): §2 — ECR push, Lambda create/update, `PassRole`,
  `AddPermission`, `CreateEventSourceMapping`, API GW route mgmt, `PutBucketNotification/Cors`,
  `SetRepositoryPolicy`; (admin) `PutRolePolicy`, `PutMetricAlarm`.
- **Runtime** (exec role): §3 — DynamoDB, SQS, S3, Textract, Logs.
- **Cross-service trust / resource policies**: §4 — role trust, ECR repo policy, Lambda
  resource policies (API GW + S3), bucket CORS.
- **Optional observability**: `cloudwatch:PutMetricData` (custom/EMF metrics),
  `xray:PutTraceSegments`+`PutTelemetryRecords` (tracing), `cloudwatch:PutMetricAlarm` +
  `logs:PutRetentionPolicy` (deploy-time alarms/retention).

---

## 9. Deployment-readiness checklist

- [ ] Deploy identity (admin/CI, not intern) has: ECR push, `lambda:CreateFunction`/`Update*`,
      `iam:PassRole` (exec role), `lambda:AddPermission`, `lambda:CreateEventSourceMapping`,
      API GW route mgmt, `s3:PutBucketNotification`/`PutBucketCors`,
      `ecr:GetRepositoryPolicy`/`SetRepositoryPolicy`/`BatchGetImage`/`GetDownloadUrlForLayer`.
- [ ] **ECR repo policy** allows `lambda.amazonaws.com` pull (or `CreateFunction` can set it).
- [ ] Exec-role **trust** allows `lambda.amazonaws.com`.
- [ ] Exec-role **identity policy** includes Textract + DynamoDB(Put/Get/Update[/Query]) +
      SQS(Send/Receive/Delete/GetQueueAttributes) + S3(Get raw / Put processed / Put raw for
      presign / Get processed for presign) + Logs.
- [ ] Lambda **resource policies** added for API GW (3 fns) and S3 (upload-handler).
- [ ] Raw-bucket **CORS** for browser PUT/POST.
- [ ] (Optional) alarms + log retention.

## 10. Recommended least-privilege structure

- **Split deploy vs runtime:** deploy via a dedicated **deploy role** assumed by CI/admin;
  never deploy with a broad user. The intern user keeps push-only ECR for local builds.
- **Scope every runtime action to exact ARNs** (jobs table + its 2 indexes; the specific
  queue; `raw/uploads/*` and `processed/textract/*` prefixes; `/aws/lambda/form-pdf-poc-dev-*`
  logs). Textract `AnalyzeDocument` is `Resource:*` (no resource-level support).
- **Per-function roles (hardening, optional):** worker = Textract+SQS+S3rw+DDB:Update;
  presigned-url = S3:PutObject(raw)+DDB:PutItem; upload-handler = SQS:Send+DDB:Update;
  job-status = DDB:GetItem; result-handler = DDB:GetItem+S3:GetObject(processed). Today's
  single shared role is acceptable for first rollout if its actions are ARN-scoped.
- **ECR repo policy** conditioned on `aws:SourceArn = worker function ARN` (not blanket Lambda).
- **Resource-policy conditions:** API GW invoke conditioned on the API/route `SourceArn`;
  S3 invoke conditioned on bucket `SourceArn` + `SourceAccount`.

## 11. Admin verification checklist (elevated profile)

```bash
ADMIN=<admin>; A=637423601842; ROLE=arn:aws:iam::$A:role/FormPdfPocLambdaExecutionRole
# role can do the runtime actions?
aws iam simulate-principal-policy --profile $ADMIN --policy-source-arn $ROLE \
  --action-names textract:AnalyzeDocument dynamodb:PutItem dynamodb:GetItem dynamodb:UpdateItem \
    sqs:SendMessage sqs:ReceiveMessage s3:GetObject s3:PutObject logs:PutLogEvents
# deploy identity can create + pass role + pull image?
aws iam simulate-principal-policy --profile $ADMIN --policy-source-arn <deploy-arn> \
  --action-names lambda:CreateFunction iam:PassRole ecr:SetRepositoryPolicy ecr:BatchGetImage
# ECR repo policy grants Lambda pull?
aws ecr get-repository-policy --profile $ADMIN --region ap-south-1 --repository-name form-pdf-poc-dev-worker
# role trust allows lambda?
aws iam get-role --profile $ADMIN --role-name FormPdfPocLambdaExecutionRole --query 'Role.AssumeRolePolicyDocument'
```

## 12. Future-rollout permission checklist (avoid repeat blocks)

Each new slice, confirm BEFORE building:
- New runtime action → add to exec role (ARN-scoped) and re-simulate.
- New trigger (S3/EventBridge/API route) → add the corresponding **Lambda resource policy**
  (`AddPermission`) + the source config.
- New Lambda from a container image → confirm ECR repo policy + deploy `PassRole`/`CreateFunction`.
- New queue/table/bucket → extend exec-role resource ARNs.
- DLQ consumer (later) → `sqs:ReceiveMessage/DeleteMessage` on the DLQ + `dynamodb:UpdateItem`.
- Always run the two `simulate-principal-policy` gates (role + deploy id) as the pre-flight.
