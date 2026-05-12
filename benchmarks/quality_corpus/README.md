# Quality Corpus

This corpus is a small validation set built from preserved pipeline runs in `output/` and `output/runs/`.

It is intentionally compact. The goal is to compare OCR normalization and structural reasoning across a few representative layouts:

- multiline-heavy reference
- dense headers with weak lines
- balanced line-rich scans
- mixed layouts
- noisy fragmented OCR
- sparse-output stability check

Regenerate the report with:

```bash
python src/quality_validation.py --manifest benchmarks/quality_corpus/manifest.json --output-dir output/quality_validation
```

The generated per-sample summaries live under `output/quality_validation/samples/`, with the top-level report in `output/quality_validation/report.json` and `output/quality_validation/report.md`.