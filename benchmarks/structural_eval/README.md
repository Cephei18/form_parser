# Structural Reasoning Evaluation Corpus

This corpus is used for isolated structural reasoning experiments after preprocessing evaluation.

The runner preserves production-safe defaults by default and only enables one structural experiment at a time:

- `field_quality_only`
- `section_grouping_only`
- `ownership_propagation_only`
- `table_aware_only`

Run the full isolated matrix:

```bash
python src/structural_evaluation.py
```

Run one sample or one experiment:

```bash
python src/structural_evaluation.py --samples multiline_reference --experiments section_grouping_only
```

Outputs are written to `output/structural_eval/` and include each run's preserved artifacts, `before_after_comparison.json`, `sample_report.json`, `report.json`, and `report.md`.
