from paddleocr import PaddleOCR

# Initialize OCR model once so repeated calls are fast.
ocr = PaddleOCR(use_angle_cls=True)


def extract_text(image_path):
    results = ocr.ocr(image_path)

    extracted = []

    for line in results[0]:
        bbox = line[0]
        text = line[1][0]

        extracted.append({
            "text": text,
            "bbox": bbox
        })

    return extracted
