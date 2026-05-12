import gc
import logging
import os
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.main import resolve_uploaded_input, run_pipeline
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


DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
    "http://localhost:3002",
    "http://127.0.0.1:3002",
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
        "[api] startup project_root=%s output_root=%s upload_root=%s runs_root=%s cors_origins=%s",
        PROJECT_ROOT,
        OUTPUT_ROOT,
        UPLOAD_ROOT,
        RUNS_ROOT,
        cors_origins or "<disabled>",
    )

app.mount("/files", StaticFiles(directory=str(OUTPUT_ROOT)), name="files")


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
    for mapping in mappings:
        confidence_class = mapping.get("confidence_class", "unknown")
        confidence_classes[confidence_class] = confidence_classes.get(confidence_class, 0) + 1
        if isinstance(mapping.get("candidate_score"), (int, float)):
            scores.append(float(mapping["candidate_score"]))
        if int(mapping.get("multiline_group_size", 1)) > 1:
            multiline_count += 1

    return {
        "mapping_count": len(mappings),
        "line_count": int(output.get("lines_count", 0)),
        "field_candidate_count": int(output.get("filtered_lines_count", 0)),
        "multiline_count": multiline_count,
        "average_candidate_score": round(sum(scores) / len(scores), 4) if scores else 0,
        "confidence_classes": confidence_classes,
    }


@app.post("/process-form")
async def process_form(file: UploadFile = File(...), mode: str = Form("rule")) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing uploaded filename.")

    contents = await file.read()
    uploaded_size = len(contents)
    logger.info("[api] upload received filename=%s size=%s mode=%s", file.filename, uploaded_size, mode)

    if uploaded_size <= 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    extension = Path(file.filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only PNG, JPG, JPEG, and PDF are supported.")
    if mode not in ALLOWED_MODES:
        raise HTTPException(status_code=400, detail="Mode must be either 'rule' or 'ml'.")

    run_id = uuid.uuid4().hex
    upload_path = UPLOAD_ROOT / f"{run_id}{extension}"
    run_output_dir = RUNS_ROOT / run_id
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    run_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("[api] saving upload: %s", file.filename)
        upload_path.write_bytes(contents)

        logger.info("[api] processing run_id=%s mode=%s", run_id, mode)
        source_image = resolve_uploaded_input(upload_path, run_output_dir)
        output = run_pipeline(source_image, run_output_dir)
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
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        file.file.close()
        try:
            upload_path.unlink(missing_ok=True)
        except Exception:
            logger.warning("[api] failed to remove temp upload: %s", upload_path)
        gc.collect()

    pdf_rel = output["pdf_output_path"].relative_to(OUTPUT_ROOT).as_posix()
    map_rel = output["mapping_image_path"].relative_to(OUTPUT_ROOT).as_posix()
    result_rel = output["result_path"].relative_to(OUTPUT_ROOT).as_posix()
    base_url = "/files"

    return {
        "status": "success",
        "message": "Form processed successfully.",
        "mode": mode,
        "pdf_url": f"{base_url}/{pdf_rel}",
        "mapping_preview": f"{base_url}/{map_rel}",
        "result_url": f"{base_url}/{result_rel}",
        "stats": _stats(output),
    }
