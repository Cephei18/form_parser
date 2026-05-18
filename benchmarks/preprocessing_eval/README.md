# Preprocessing Evaluation Corpus

This manifest groups preserved representative form images into:

- simple structured forms
- multiline forms
- table-heavy forms
- noisy/low-quality scans
- dense layouts

Run isolated preprocessing experiments with:

```bash
python src/preprocessing_evaluation.py --manifest benchmarks/preprocessing_eval/manifest.json --output-dir output/preprocessing_eval
```

Each experiment toggles exactly one preprocessing stage and writes preserved artifacts, per-sample comparisons, `report.json`, and `report.md`.
