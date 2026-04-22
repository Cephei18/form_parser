try:
    from paddleocr import PaddleOCR
except ModuleNotFoundError:
    PaddleOCR = None  # type: ignore[assignment]

ocr = None


def _get_ocr_model():
    global ocr

    if PaddleOCR is None:
        raise RuntimeError(
            "paddleocr is not installed in the active environment. "
            "Install dependencies with: pip install -r requirements.txt"
        )

    # Initialize OCR model once so repeated calls stay fast.
    if ocr is None:
        ocr = PaddleOCR(use_angle_cls=True)

    return ocr


def extract_text(image_path):
    model = _get_ocr_model()
    results = model.ocr(image_path)

    extracted = []

    for line in results[0]:
        bbox = line[0]
        text = line[1][0]

        extracted.append({"text": text, "bbox": bbox})

    return extracted
