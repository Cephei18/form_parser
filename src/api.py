import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.main import resolve_uploaded_input, run_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "output"
UPLOAD_ROOT = OUTPUT_ROOT / "uploads"
RUNS_ROOT = OUTPUT_ROOT / "runs"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
RUNS_ROOT.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf"}

app = FastAPI(title="Form Parser API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    # Allow local frontend dev servers (ports 3000 and 3001) during development.
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/files", StaticFiles(directory=str(OUTPUT_ROOT)), name="files")


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Form Parser API is running.",
        "process_endpoint": "POST /process-form",
    }


@app.post("/process-form")
async def process_form(file: UploadFile = File(...)) -> dict[str, str]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing uploaded filename.")

    extension = Path(file.filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only PNG, JPG, JPEG, and PDF are supported.")

    run_id = uuid.uuid4().hex
    upload_path = UPLOAD_ROOT / f"{run_id}{extension}"
    run_output_dir = RUNS_ROOT / run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with upload_path.open("wb") as out_file:
            shutil.copyfileobj(file.file, out_file)

        source_image = resolve_uploaded_input(upload_path, run_output_dir)
        output = run_pipeline(source_image, run_output_dir)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        file.file.close()

    pdf_rel = output["pdf_output_path"].relative_to(OUTPUT_ROOT).as_posix()
    map_rel = output["mapping_image_path"].relative_to(OUTPUT_ROOT).as_posix()
    base_url = "/files"

    return {
        "pdf_url": f"{base_url}/{pdf_rel}",
        "mapping_preview": f"{base_url}/{map_rel}",
    }
