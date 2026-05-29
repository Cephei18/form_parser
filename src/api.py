import gc
import logging
import os
import shutil
import time
import uuid
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from src.main import resolve_uploaded_input
from src.pipelines.pipeline_router import run_pipeline
from src.ocr import OCRRuntimeError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(os.getenv("FORM_PARSER_OUTPUT_DIR", PROJECT_ROOT / "output"))
UPLOAD_ROOT = Path(os.getenv("FORM_PARSER_UPLOAD_DIR", OUTPUT_ROOT / "uploads"))
RUNS_ROOT = Path(os.getenv("FORM_PARSER_RUNS_DIR", OUTPUT_ROOT / "runs"))

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
RUNS_ROOT.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf"}
ALLOWED_MODES = {"rule", "ml"}
SERVABLE_FILENAMES = {
    "output.pdf",
    "mapping.png",
    "mappings.json",
    "result.json",
}
BLOCKED_FILE_PATH_PARTS = {"uploads", "easyocr-models"}
READ_CHUNK_SIZE = 1024 * 1024

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def _parse_origins(value: str | None) -> list[str]:
    if not value:
        return []

    return [origin.strip() for origin in value.split(",") if origin.strip()]


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("[api] invalid integer env %s=%r; using %s", name, value, default)
        return default


MAX_UPLOAD_SIZE_MB = max(1, _int_env("FORM_PARSER_MAX_UPLOAD_SIZE_MB", 20))
MAX_UPLOAD_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
RUN_RETENTION_DAYS = max(0, _int_env("FORM_PARSER_RUN_RETENTION_DAYS", 7))
MAX_RUN_DIRS = max(0, _int_env("FORM_PARSER_MAX_RUN_DIRS", 200))


async def _read_upload_limited(file: UploadFile) -> bytes:
    chunks: list[bytes] = []
    total_size = 0

    while True:
        chunk = await file.read(READ_CHUNK_SIZE)
        if not chunk:
            break
        total_size += len(chunk)
        if total_size > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Uploaded file must be smaller than {MAX_UPLOAD_SIZE_MB}MB.",
            )
        chunks.append(chunk)

    return b"".join(chunks)


def _sniff_upload_type(contents: bytes) -> str | None:
    if contents.startswith(b"%PDF-"):
        return ".pdf"
    if contents.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if contents.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    return None


def _verify_image_bytes(contents: bytes) -> None:
    try:
        from PIL import Image

        with Image.open(BytesIO(contents)) as image:
            image.verify()
    except HTTPException:
        raise
    except Exception as exc:
        logger.info("[api] uploaded image failed validation: %s", exc)
        raise HTTPException(status_code=400, detail="Uploaded image could not be decoded.") from exc


def _validate_upload_content(extension: str, contents: bytes) -> None:
    detected_extension = _sniff_upload_type(contents)
    normalized_extension = ".jpg" if extension == ".jpeg" else extension

    if detected_extension is None:
        raise HTTPException(status_code=400, detail="Uploaded file content is not a supported PDF, PNG, or JPEG.")

    if normalized_extension == ".jpeg":
        normalized_extension = ".jpg"

    if detected_extension != normalized_extension:
        raise HTTPException(status_code=400, detail="Uploaded file extension does not match its content.")

    if normalized_extension in {".png", ".jpg"}:
        _verify_image_bytes(contents)


def _safe_remove_dir(path: Path) -> None:
    try:
        resolved = path.resolve()
        runs_root = RUNS_ROOT.resolve()
        if runs_root == resolved or runs_root not in resolved.parents:
            logger.warning("[api] skip cleanup outside runs root: %s", path)
            return
        shutil.rmtree(resolved, ignore_errors=True)
    except Exception:
        logger.warning("[api] failed to remove run directory: %s", path, exc_info=True)


def _cleanup_old_run_dirs(current_run_id: str | None = None) -> None:
    if not RUNS_ROOT.exists():
        return

    run_dirs = [path for path in RUNS_ROOT.iterdir() if path.is_dir()]
    now = time.time()
    retention_seconds = RUN_RETENTION_DAYS * 24 * 60 * 60

    if retention_seconds > 0:
        for run_dir in run_dirs:
            if current_run_id and run_dir.name == current_run_id:
                continue
            try:
                if now - run_dir.stat().st_mtime > retention_seconds:
                    _safe_remove_dir(run_dir)
            except Exception:
                logger.warning("[api] failed to inspect run directory: %s", run_dir, exc_info=True)

    if MAX_RUN_DIRS <= 0:
        return

    remaining = [path for path in RUNS_ROOT.iterdir() if path.is_dir()]
    overflow = len(remaining) - MAX_RUN_DIRS
    if overflow <= 0:
        return

    for run_dir in sorted(remaining, key=lambda path: path.stat().st_mtime)[:overflow]:
        if current_run_id and run_dir.name == current_run_id:
            continue
        _safe_remove_dir(run_dir)


def _resolve_served_file(file_path: str) -> Path:
    requested_path = (OUTPUT_ROOT / file_path).resolve()
    output_root = OUTPUT_ROOT.resolve()

    if output_root != requested_path and output_root not in requested_path.parents:
        raise HTTPException(status_code=404, detail="File not found.")
    if any(part in BLOCKED_FILE_PATH_PARTS for part in requested_path.relative_to(output_root).parts):
        raise HTTPException(status_code=404, detail="File not found.")
    if requested_path.name not in SERVABLE_FILENAMES:
        raise HTTPException(status_code=404, detail="File not found.")
    if not requested_path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")

    return requested_path


DEFAULT_CORS_ORIGINS = [
    "http://form-pdf-poc-dev-frontend.s3-website.ap-south-1.amazonaws.com",
]

app = FastAPI(title="Form Parser API", version="1.0.0")

cors_origins = list(dict.fromkeys(DEFAULT_CORS_ORIGINS + _parse_origins(os.getenv("CORS_ORIGINS"))))

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event() -> None:
    logger.info(
        "[api] startup project_root=%s output_root=%s upload_root=%s runs_root=%s max_upload_mb=%s cors_origins=%s",
        PROJECT_ROOT,
        OUTPUT_ROOT,
        UPLOAD_ROOT,
        RUNS_ROOT,
        MAX_UPLOAD_SIZE_MB,
        cors_origins or "<disabled>",
    )
    _cleanup_old_run_dirs()


@app.get("/files/{file_path:path}")
def get_generated_file(file_path: str) -> FileResponse:
    requested_path = _resolve_served_file(file_path)
    return FileResponse(
        requested_path,
        headers={
            "Cache-Control": "private, max-age=300",
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    error_code = "http_error"
    message = exc.detail if isinstance(exc.detail, str) else "Request failed."
    if isinstance(exc.detail, dict):
        error_code = str(exc.detail.get("code") or error_code)
        message = str(exc.detail.get("message") or message)

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": message,
            "detail": message,
            "error": {
                "code": error_code,
                "message": message,
            },
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled API error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error while processing form.",
            "detail": "Internal server error while processing form.",
            "error": {
                "code": "internal_error",
                "message": "Internal server error while processing form.",
            },
        },
    )


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Form Parser API is running.",
        "process_endpoint": "POST /process-form",
    }


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    logger.info("[api] request start method=%s path=%s", request.method, request.url.path)
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("[api] request failed method=%s path=%s", request.method, request.url.path)
        raise

    logger.info(
        "[api] request end method=%s path=%s status=%s",
        request.method,
        request.url.path,
        response.status_code,
    )
    return response


def _stats(output: dict) -> dict:
    mappings = output.get("mappings") or []
    confidence_classes: dict[str, int] = {}
    scores = []
    multiline_count = 0
    checkbox_count = 0
    for mapping in mappings:
        confidence_class = mapping.get("confidence_class", "unknown")
        confidence_classes[confidence_class] = confidence_classes.get(confidence_class, 0) + 1
        if isinstance(mapping.get("candidate_score"), (int, float)):
            scores.append(float(mapping["candidate_score"]))
        if int(mapping.get("multiline_group_size", 1)) > 1:
            multiline_count += 1
        if mapping.get("field_type") == "checkbox":
            checkbox_count += 1

    return {
        "mapping_count": len(mappings),
        "line_count": int(output.get("lines_count", 0)),
        "field_candidate_count": int(output.get("filtered_lines_count", 0)),
        "multiline_count": multiline_count,
        "multi_line_count": multiline_count,
        "checkbox_count": checkbox_count,
        "average_candidate_score": round(sum(scores) / len(scores), 4) if scores else 0,
        "confidence_classes": confidence_classes,
    }


@app.post("/process-form")
async def process_form(file: UploadFile = File(...), mode: str = Form("rule")) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing uploaded filename.")

    extension = Path(file.filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only PNG, JPG, JPEG, and PDF are supported.")
    if mode not in ALLOWED_MODES:
        raise HTTPException(status_code=400, detail="Mode must be either 'rule' or 'ml'.")

    contents = await _read_upload_limited(file)
    uploaded_size = len(contents)
    logger.info("[api] upload received filename=%s size=%s mode=%s", file.filename, uploaded_size, mode)

    if uploaded_size <= 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    _validate_upload_content(extension, contents)

    run_id = uuid.uuid4().hex
    upload_path = UPLOAD_ROOT / f"{run_id}{extension}"
    run_output_dir = RUNS_ROOT / run_id
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    run_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("[api] saving upload: %s", file.filename)
        upload_path.write_bytes(contents)

        started = time.perf_counter()
        logger.info("[api] processing run_id=%s mode=%s", run_id, mode)
        source_image = resolve_uploaded_input(upload_path, run_output_dir)
        output = run_pipeline(source_image, run_output_dir)
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        logger.info(
            "[api] run complete run_id=%s pipeline_mode=%s elapsed_ms=%.2f",
            run_id,
            output.get("pipeline_mode", "ocr"),
            elapsed_ms,
        )
    except HTTPException:
        raise
    except OCRRuntimeError as exc:
        logger.exception("[api] OCR failed run_id=%s: %s", run_id, exc)
        raise HTTPException(
            status_code=503,
            detail={
                "code": "ocr_failed",
                "message": "OCR processing failed. Verify the OCR model is available and the uploaded document can be read.",
            },
        ) from exc
    except Exception as exc:
        logger.exception("[api] processing failed run_id=%s: %s", run_id, exc)
        raise HTTPException(
            status_code=500,
            detail={
                "code": "processing_failed",
                "message": "Form processing failed. Please try another document or retry later.",
            },
        ) from exc
    finally:
        file.file.close()
        try:
            upload_path.unlink(missing_ok=True)
        except Exception:
            logger.warning("[api] failed to remove temp upload: %s", upload_path)
        _cleanup_old_run_dirs(current_run_id=run_id)
        gc.collect()

    pdf_rel = output["pdf_output_path"].relative_to(OUTPUT_ROOT).as_posix()
    map_rel = output["mapping_image_path"].relative_to(OUTPUT_ROOT).as_posix()
    result_rel = output["result_path"].relative_to(OUTPUT_ROOT).as_posix()
    base_url = "/files"

    pipeline_mode = str(output.get("pipeline_mode") or "ocr")
    processing_time_ms = float(output.get("processing_time_ms") or elapsed_ms if "elapsed_ms" in locals() else 0.0)
    tables_detected = int(output.get("tables_detected") or len(output.get("tables") or []))
    checkboxes_detected = int(output.get("checkboxes_detected") or len(output.get("checkboxes") or []))

    return {
        "status": "success",
        "message": "Form processed successfully.",
        "mode": mode,
        "pipeline_mode": pipeline_mode,
        "processing_time_ms": processing_time_ms,
        "response_metadata": {
            "pipeline_mode": pipeline_mode,
            "processing_time_ms": processing_time_ms,
            "tables_detected": tables_detected,
            "checkboxes_detected": checkboxes_detected,
        },
        "pdf_url": f"{base_url}/{pdf_rel}",
        "mapping_preview": f"{base_url}/{map_rel}",
        "result_url": f"{base_url}/{result_rel}",
        "stats": _stats(output),
    }
