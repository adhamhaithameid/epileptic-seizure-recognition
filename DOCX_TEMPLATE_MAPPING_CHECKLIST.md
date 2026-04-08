# DOCX Template Mapping Checklist

Use this checklist to fill:
- `paper/template/Fill your Project Information in this document.docx`

## Section mapping
- `Project Title` -> from `RESEARCH_PAPER_FINAL_DRAFT.md` top section
- `Abstract` -> `## Abstract`
- `Introduction` -> `## 1. Introduction`
- `Related Work` -> `## 2. Related Work` (complete with 10+ papers)
- `Methodology` -> `## 3. Methodology`
- `Proposed Model` -> `## 4. Proposed Model`
- `Results and Discussion` -> `## 5. Results and Discussion`
- `Conclusion and Future Work` -> `## 6. Conclusion and Future Work`
- `References` -> `## 7. References`

## Insert these result artifacts
- Dataset statistics: `results/tables/dataset_descriptive_stats.csv`
- Correlation/Covariance: `results/tables/correlation_matrix.csv`, `results/tables/covariance_matrix.csv`
- Full Cartesian summary: `results/tables/cartesian_summary_by_combo.csv`
- Binary ranking table: `results/tables/cartesian_rankings_binary.csv`
- Multiclass ranking table: `results/tables/cartesian_rankings_multiclass.csv`
- Comparison narrative: `results/reports/cartesian_comparison_report.md`
- Figures: `results/figures/cartesian_*.png`

## Final checks before submission
- Fill student metadata (name, ID, section).
- Replace Related Work placeholders with real citations.
- Ensure in-text citations and reference list are consistent.
- Ensure reported combination counts are correct:
  - `expected_combos = 1536`
  - `expected_fold_evals = 4608`
- Ensure skip/failure interpretation is explained using `status` and `skip_reason` columns.
