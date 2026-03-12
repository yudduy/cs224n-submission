# Data Manifest

## Source Dataset
- **MATH-500**: 500 problems from the MATH benchmark (Hendrycks et al., 2021)
  - Loaded via HuggingFace datasets: `HuggingFaceH4/MATH-500`

## Generated Data (from pipeline.py)
All data was generated on Mithril A100 80GB using Qwen3-8B.

### Pipeline Outputs
- `rollouts/` — N=8 rollouts per MATH-500 problem (Step 1)
- `goldilocks/` — Filtered problems with mixed correct/incorrect rollouts (Step 2)
- `pivots/` — DEEP-GRPO pivot positions with Q(t) scores (Step 3)
- `corrections/` — Generated corrections at pivot positions (Step 5)
- `eval_summary.json` — Causal validation results (Step 6)

### Figures (in figures/)
- `fig1_pivot_depth_dist.png` — Distribution of pivot depths across problems
- `fig2_yield_by_depth.png` — Correction yield rate by normalized depth
- `fig3_p_phi_calibration.png` — P_phi recoverability calibration curve
- `fig4_per_level_yield.png` — Yield rate by MATH difficulty level
- `fig5_per_problem_yield_dist.png` — Per-problem yield distribution
- `fig6_per_subject_yield.png` — Yield rate by MATH subject
- `fig7_accuracy_by_level.png` — Accuracy by difficulty level

## Reproduction
Run `code/pipeline.py` to regenerate all data. Total compute cost: ~$47 on A100 80GB.
