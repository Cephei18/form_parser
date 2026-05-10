"""
Lightweight ML inference helpers using sentence embeddings.

This is a simple, demo-ready module — it uses a transformer encoder to
produce embeddings and matches labels to values by cosine similarity.

Note: This requires `transformers` and `torch` to be installed. Keep
`USE_ML = False` in `src/main.py` until you've installed deps.
"""

from typing import List, Dict

# Lazy import / load: avoid heavy downloads or import errors at module import time.
_TOKENIZER = None
_MODEL = None


def _ensure_model_loaded():
    """Ensure `torch` + transformer model/tokenizer are available and loaded.

    This does the imports at call-time so earlier module-imports don't lock
    in a negative availability check. Returns True when transformer+torch
    are usable, False when not (caller should fallback).
    """
    global _TOKENIZER, _MODEL

    try:
        import torch as _torch
        from transformers import AutoTokenizer, AutoModel

        globals()["torch"] = _torch

        if _TOKENIZER is None or _MODEL is None:
            _TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")
            _MODEL = AutoModel.from_pretrained("bert-base-uncased")

        return True
    except Exception:
        # transformer or torch unavailable — caller should fall back
        return False


def get_embedding(text: str):
    # Only call when transformers are loaded; caller must check availability.
    if _TOKENIZER is None or _MODEL is None:
        raise RuntimeError("Transformer model not loaded — cannot compute embeddings.")

    inputs = _TOKENIZER(text, return_tensors="pt", truncation=True)
    outputs = _MODEL(**inputs)
    # mean pooling over sequence length
    return outputs.last_hidden_state.mean(dim=1)


def ml_pipeline(ocr_results: List[Dict]):
    """
    ocr_results: list of {"text": str, "bbox": [...]}

    Returns list of mappings: {"label": str, "value": str, "bbox": bbox}
    """
    labels = []
    values = []

    for item in ocr_results:
        text = (item.get("text") or "").lower()

        if ":" in text or text.endswith("?"):
            labels.append(item)
        else:
            values.append(item)

    mappings = []

    # Try transformer embeddings first; if unavailable, fall back to RapidFuzz
    use_transformer = _ensure_model_loaded()

    if use_transformer:
        # precompute embeddings for values to avoid repeated work
        value_embs = []
        for val in values:
            emb = get_embedding(val.get("text", ""))
            value_embs.append((val, emb))

        for label in labels:
            best_val = None
            best_score = -1.0

            emb1 = get_embedding(label.get("text", ""))

            for val, emb2 in value_embs:
                score = float(torch.cosine_similarity(emb1, emb2).item())

                if score > best_score:
                    best_score = score
                    best_val = val

            if best_val:
                mappings.append({
                    "label": label.get("text", ""),
                    "value": best_val.get("text", ""),
                    "bbox": best_val.get("bbox"),
                    "score": best_score,
                })
    else:
        # Fallback: use RapidFuzz token-set ratio as a cheap semantic proxy.
        try:
            from rapidfuzz import fuzz
        except Exception:
            # Last-resort: simple substring / equality heuristic
            for label in labels:
                best_val = None
                best_score = -1
                ltxt = (label.get("text", "") or "").lower()
                for val in values:
                    vtxt = (val.get("text", "") or "").lower()
                    score = 100 if ltxt == vtxt else (100 if ltxt in vtxt or vtxt in ltxt else 0)
                    if score > best_score:
                        best_score = score
                        best_val = val

                if best_val:
                    mappings.append({
                        "label": label.get("text", ""),
                        "value": best_val.get("text", ""),
                        "bbox": best_val.get("bbox"),
                        "score": best_score / 100.0,
                    })
        else:
            for label in labels:
                best_val = None
                best_score = -1.0
                ltxt = (label.get("text", "") or "")

                for val in values:
                    vtxt = (val.get("text", "") or "")
                    score = float(fuzz.token_set_ratio(ltxt, vtxt)) / 100.0

                    if score > best_score:
                        best_score = score
                        best_val = val

                if best_val:
                    mappings.append({
                        "label": label.get("text", ""),
                        "value": best_val.get("text", ""),
                        "bbox": best_val.get("bbox"),
                        "score": best_score,
                    })

    return mappings
