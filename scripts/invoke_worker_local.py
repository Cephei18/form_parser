#!/usr/bin/env python
"""
Local smoke test for the serverless Textract worker against REAL AWS.

Runs ``src.lambda_worker.process_job`` (or ``handler``) on your machine using the
project's ``form-pdf-poc`` profile / ``ap-south-1`` region, so you can validate
the full event-driven path — S3 download -> /tmp execution -> Textract ->
S3 artifact publishing -> structured JobResult — before deploying the Lambda
container image.

WARNING: a non-dry run calls AWS Textract (billable) and writes objects to the
processed S3 bucket. Use ``--dry-run`` first to verify access without spending.

Examples
--------
  # Verify credentials + bucket/object access only (no Textract, no cost):
  python scripts/invoke_worker_local.py --key uploads/form.png --dry-run

  # Upload a local file to the raw bucket, then process it end to end:
  python scripts/invoke_worker_local.py --upload input/form.png

  # Process an object already in the raw bucket:
  python scripts/invoke_worker_local.py --key uploads/form.png

  # Exercise the Lambda handler event shape instead of process_job directly:
  python scripts/invoke_worker_local.py --key uploads/form.png --via-handler

  # Dry of S3 publishing: run Textract but keep artifacts local only:
  python scripts/invoke_worker_local.py --key uploads/form.png --local-artifacts
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

# Project defaults (overridable via flags / env). See repo memory.
DEFAULT_PROFILE = "form-pdf-poc"
DEFAULT_REGION = "ap-south-1"
DEFAULT_RAW_BUCKET = "form-pdf-poc-dev-raw-documents"
DEFAULT_PROCESSED_BUCKET = "form-pdf-poc-dev-processed-documents"

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-AWS smoke test for the Textract Lambda worker.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--key", help="Existing S3 object key in the raw bucket to process.")
    src.add_argument("--upload", help="Local file to upload to the raw bucket, then process.")

    parser.add_argument("--bucket", default=DEFAULT_RAW_BUCKET, help="Raw documents bucket (input).")
    parser.add_argument("--processed-bucket", default=DEFAULT_PROCESSED_BUCKET, help="Processed artifacts bucket (output).")
    parser.add_argument("--upload-prefix", default="uploads", help="Key prefix used when --upload is given.")
    parser.add_argument("--job-id", help="Explicit job id (default: derived from the object key).")
    parser.add_argument("--profile", default=os.getenv("AWS_PROFILE", DEFAULT_PROFILE), help="AWS profile.")
    parser.add_argument("--region", default=os.getenv("AWS_REGION", DEFAULT_REGION), help="AWS region.")
    parser.add_argument("--artifact-prefix", default="textract", help="S3 key prefix for published artifacts.")

    parser.add_argument("--render-page", action="store_true", help="If --upload is a PDF, rasterise page 1 locally and upload the PNG instead.")
    parser.add_argument("--via-handler", action="store_true", help="Invoke handler() with an S3-event shape instead of process_job().")
    parser.add_argument("--local-artifacts", action="store_true", help="Run Textract but keep artifacts local (no S3 publish).")
    parser.add_argument("--dry-run", action="store_true", help="Only verify credentials and bucket/object access; do NOT run Textract.")
    parser.add_argument("--keep-workdir", action="store_true", help="Do not delete the local /tmp working dir after the run.")
    parser.add_argument("--work-root", help="Override the working directory root (default: <tempdir>/form_parser_jobs).")
    parser.add_argument("--no-output-check", action="store_true", help="Skip downloading + inspecting the produced PDF for checkbox widgets.")
    return parser.parse_args(argv)


def _configure_environment(args: argparse.Namespace) -> None:
    """Set AWS + pipeline env BEFORE importing boto3 / the worker, so lazily
    created clients and module-level config pick these up."""
    os.environ.setdefault("AWS_PROFILE", args.profile)
    os.environ["AWS_DEFAULT_REGION"] = args.region
    os.environ["AWS_REGION"] = args.region
    os.environ["FORM_PARSER_AWS_REGION"] = args.region
    os.environ["FORM_PARSER_PIPELINE_MODE"] = "textract"

    if args.local_artifacts:
        os.environ["FORM_PARSER_ARTIFACT_BACKEND"] = "local"
    else:
        os.environ["FORM_PARSER_ARTIFACT_BACKEND"] = "s3"
        os.environ["FORM_PARSER_PROCESSED_BUCKET"] = args.processed_bucket
        os.environ["FORM_PARSER_ARTIFACT_PREFIX"] = args.artifact_prefix

    # Keep the staged artifacts around for inspection unless told otherwise.
    # In local-artifacts mode the "published" artifacts ARE the workdir files, so
    # cleanup must stay off or there would be nothing left to inspect.
    keep = args.keep_workdir or args.local_artifacts
    os.environ["FORM_PARSER_WORKER_CLEANUP"] = "false" if keep else "true"

    work_root = args.work_root or str(Path(tempfile.gettempdir()) / "form_parser_jobs")
    os.environ["FORM_PARSER_WORK_ROOT"] = work_root
    args._work_root = work_root  # stash for later


def _make_session(region: str):
    import boto3

    # Honour AWS_PROFILE if present; otherwise default chain (env/role).
    profile = os.environ.get("AWS_PROFILE")
    if profile:
        return boto3.Session(profile_name=profile, region_name=region)
    return boto3.Session(region_name=region)


def _preflight(session, args: argparse.Namespace) -> bool:
    """Verify identity + access before doing anything billable."""
    ok = True
    try:
        ident = session.client("sts").get_caller_identity()
        print(f"  identity   : {ident.get('Arn')}")
    except Exception as exc:
        print(f"  identity   : FAILED - {type(exc).__name__}: {exc}")
        return False

    s3 = session.client("s3")
    for label, bucket in (("raw bucket", args.bucket), ("processed", args.processed_bucket if not args.local_artifacts else None)):
        if not bucket:
            continue
        try:
            s3.head_bucket(Bucket=bucket)
            print(f"  {label:11}: s3://{bucket}  OK")
        except Exception as exc:
            print(f"  {label:11}: s3://{bucket}  FAILED - {type(exc).__name__}: {exc}")
            ok = False
    return ok


def _maybe_upload(session, args: argparse.Namespace) -> str:
    """Upload a local file to the raw bucket if --upload given; return the key."""
    if not args.upload:
        return args.key

    local = Path(args.upload)
    if not local.is_file():
        raise SystemExit(f"--upload file not found: {local}")

    upload_path = local
    if args.render_page and local.suffix.lower() == ".pdf":
        sys.path.insert(0, str(REPO_ROOT))
        from src.document_render import render_pdf_first_page

        rendered = Path(tempfile.gettempdir()) / f"{local.stem}_page_1.png"
        print(f"  rendering  : {local} -> {rendered} (page 1)")
        upload_path = render_pdf_first_page(local, rendered)

    key = f"{args.upload_prefix.strip('/')}/{upload_path.name}"
    print(f"  uploading  : {upload_path} -> s3://{args.bucket}/{key}")
    session.client("s3").upload_file(str(upload_path), args.bucket, key)
    return key


def _verify_object(session, bucket: str, key: str) -> None:
    meta = session.client("s3").head_object(Bucket=bucket, Key=key)
    size = meta.get("ContentLength")
    print(f"  object     : s3://{bucket}/{key}  ({size} bytes, {meta.get('ContentType')})")


def _split_s3_uri(uri: str) -> tuple[str, str]:
    without = uri[len("s3://"):]
    bucket, _, key = without.partition("/")
    return bucket, key


def _fetch_artifact(session, manifest: dict, name: str) -> bytes | None:
    artifacts = (manifest or {}).get("artifacts") or {}
    uri = artifacts.get(name)
    if not uri:
        return None
    try:
        if str(uri).startswith("s3://"):
            bucket, key = _split_s3_uri(uri)
            dest = Path(tempfile.gettempdir()) / f"_worker_check_{name}"
            session.client("s3").download_file(bucket, key, str(dest))
            return dest.read_bytes()
        return Path(uri).read_bytes()
    except Exception as exc:
        print(f"  (could not fetch {name}: {type(exc).__name__}: {exc})")
        return None


def _verify_output_artifacts(session, manifest: dict) -> None:
    """Download the produced PDF + diagnostics and confirm checkbox rendering in
    the real AWS-generated output."""
    print("-- output verification --")
    pdf = _fetch_artifact(session, manifest, "output.pdf")
    if pdf:
        btn = pdf.count(b"/Btn")      # AcroForm checkbox (button) widgets
        tx = pdf.count(b"/Tx")        # AcroForm text fields
        print(f"  output.pdf : {len(pdf)} bytes | checkbox widgets(/Btn)={btn} | text fields(/Tx)={tx}")
    else:
        print("  output.pdf : not found in manifest")

    diag = _fetch_artifact(session, manifest, "mapping_diagnostics.json")
    if diag:
        try:
            anchoring = json.loads(diag).get("anchoring", {})
            print(
                "  checkboxes : detected={} deduped_token={} unassociated={}".format(
                    anchoring.get("checkbox_field_count"),
                    anchoring.get("deduped_token_checkbox_count"),
                    anchoring.get("unassociated_checkbox_count"),
                )
            )
            validation = json.loads(diag).get("validation", {})
            if isinstance(validation, dict):
                print(f"  validation : passed={validation.get('passed')}")
        except Exception as exc:
            print(f"  (could not parse diagnostics: {type(exc).__name__}: {exc})")


def _report_cleanup(work_root: str, job_id: str, kept: bool) -> None:
    job_dir = Path(work_root) / job_id
    exists = job_dir.exists()
    if kept:
        print(f"  workdir    : kept at {job_dir} (exists={exists})")
    else:
        status = "OK (removed)" if not exists else "WARNING (still present)"
        print(f"  cleanup    : {status} {job_dir}")


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    _configure_environment(args)

    # Make `import src...` work when run from anywhere.
    sys.path.insert(0, str(REPO_ROOT))

    import logging
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s %(message)s")

    try:
        session = _make_session(args.region)
    except ModuleNotFoundError:
        print("boto3 is not installed in this environment.", file=sys.stderr)
        return 2

    print("== Worker smoke test ==")
    print(f"  profile    : {os.environ.get('AWS_PROFILE')}   region: {args.region}")
    print(f"  artifacts  : {'LOCAL (no publish)' if args.local_artifacts else 's3://' + args.processed_bucket + '/' + args.artifact_prefix}")
    print(f"  work root  : {args._work_root}")
    print("-- preflight --")
    if not _preflight(session, args):
        print("preflight FAILED — fix access before running. (No Textract was called.)")
        return 1

    key = _maybe_upload(session, args)
    try:
        _verify_object(session, args.bucket, key)
    except Exception as exc:
        print(f"  object     : s3://{args.bucket}/{key}  FAILED - {type(exc).__name__}: {exc}")
        return 1

    if args.dry_run:
        print("\nDRY RUN complete — credentials and object access verified. No Textract was called.")
        return 0

    # Import the worker only now (after env is configured).
    from src.lambda_worker import process_job, handler, _job_id_from_key
    from src.lambda_worker import WORK_ROOT  # resolved from FORM_PARSER_WORK_ROOT

    job_id = args.job_id or _job_id_from_key(key)
    print(f"\n-- running worker (job_id={job_id}) --")

    if args.via_handler:
        event = {"Records": [{"s3": {"bucket": {"name": args.bucket}, "object": {"key": key}}}]}
        result = handler(event)
    else:
        result = process_job(args.bucket, key, job_id, work_root=WORK_ROOT, region=args.region)

    print("\n== JobResult ==")
    print(json.dumps(result, indent=2, default=str))

    # Determine success + surface where outputs landed.
    jobs = result.get("results", [result]) if isinstance(result, dict) and "results" in result else [result]
    failed = [j for j in jobs if j.get("status") != "succeeded"]
    kept = args.keep_workdir or args.local_artifacts
    for j in jobs:
        manifest = j.get("artifacts") or {}
        if manifest.get("base_uri"):
            print(f"\n  artifacts for {j.get('job_id')}: {manifest.get('base_uri')}  ({manifest.get('artifact_count')} files)")
            for name, uri in sorted((manifest.get("artifacts") or {}).items()):
                print(f"      {name:28} {uri}")
        if j.get("status") == "succeeded" and not args.no_output_check:
            _verify_output_artifacts(session, manifest)
        _report_cleanup(args._work_root, j.get("job_id", ""), kept)

    if failed:
        print(f"\nRESULT: {len(failed)}/{len(jobs)} job(s) FAILED")
        return 1
    print(f"\nRESULT: {len(jobs)} job(s) succeeded [OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
