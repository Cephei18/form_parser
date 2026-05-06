from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import uuid
import os

from src.main import resolve_uploaded_input, run_pipeline

app = FastAPI(title="Form Parser API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/process")
async def process_form(file: UploadFile = File(...)):
    tmp_dir = Path("uploads")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    upload_path = tmp_dir / f"{uuid.uuid4().hex}_{file.filename}"
    try:
        with upload_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}")

    output_dir = Path("output") / upload_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # convert PDFs if needed, and run the existing pipeline
        image_path = resolve_uploaded_input(upload_path, output_dir)
        result = run_pipeline(image_path, output_dir)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

    return {
        "message": "Processed",
        "mappings": str(result["mappings_path"]),
        "pdf": str(result["pdf_output_path"]),
    }
