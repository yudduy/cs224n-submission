# Where Reasoning Branches: How Preference Pair Construction Shapes DPO for Mathematical Reasoning

**Duy Nguyen** (duynguy@stanford.edu)
Stanford CS224N Custom Project, Winter 2026
Mentor: Mirac Suzgun

## Abstract

We compare two ways of building preference pairs for Direct Preference Optimization on mathematical reasoning: correction-based pairs that splice self-corrections at identified error positions, and rollout-based pairs that use natural correct vs. incorrect solutions from the model's own distribution. Across 11 experimental conditions on MATH-500 with Qwen3-8B, correction-based pairs consistently degrade performance while rollout-based pairs consistently improve it. We trace the correction failure to the failed-reasoning prefix carried into the chosen response, not to the self-repair phrasing itself, and offer a gradient-level account of why DPO on shared-prefix pairs behaves differently from SFT on the same data.

## Repository Structure

```
paper.tex                  Main paper (NeurIPS format)
references.bib             Bibliography
neurips_2019.sty           Style file
figures/
  main_results.png         Delta pass@1 across all 11 conditions (Figure 1)
  pipeline_diagram.png     Correction vs rollout pipeline comparison (Figure 2)
  gradient_cancellation.png  SFT vs DPO gradient behavior on shared prefixes (Figure 3)
code/
  pipeline.py              ExIt data pipeline (rollouts, pivot identification, corrections)
  evaluate.py              Pipeline evaluation and summary statistics
  sft_ablation.py          SFT ablation experiment (corrections hurt SFT too)
  divergence_dpo.py        Correction-based DPO data preparation (3 conditions)
  rollout_dpo.py           Rollout-based DPO data preparation (5 conditions)
  rollout_dpo_eval.py      DPO model evaluation (pass@1, pass@4, McNemar)
  make_main_results_figure.py    Figure 1 generation
  make_pipeline_diagram.py       Figure 2 generation
  make_gradient_diagram.py       Figure 3 generation
  utils/                   Shared utilities (answer extraction, ChatML formatting, I/O)
data/
  correction_dpo_results.json        Correction-based DPO results (3 conditions, beta=0.2)
  rollout_dpo_results.json           Rollout-based DPO results (5 conditions)
  matched_correction_results/        Matched replication (3 conditions, beta=0.1)
  ablation_results.json              Rhetoric neutralization ablation
  statistical_analysis.md            Full statistical analysis with McNemar tests
```

## Reproduction

All experiments were run on a single A100 80GB GPU with Qwen3-8B.

### Requirements
- Python 3.10+
- PyTorch 2.x with CUDA
- vLLM (for rollout generation)
- transformers, peft, trl (for DPO/SFT training)

### Pipeline Steps

1. **Generate rollouts**: `pipeline.py` generates N=8 rollouts per MATH-500 problem using vLLM
2. **Pivot identification**: `pipeline.py` identifies DEEP-GRPO pivot positions in Goldilocks problems
3. **Correction generation**: `pipeline.py` generates self-corrections at pivot positions
4. **SFT ablation**: `sft_ablation.py` tests correction-based SFT (result: corrections hurt)
5. **Correction DPO**: `divergence_dpo.py` prepares correction-based pairs (3 conditions by divergence depth)
6. **Rollout DPO**: `rollout_dpo.py` prepares rollout-based pairs (full-trajectory vs prefix-sharing)
7. **Evaluation**: `rollout_dpo_eval.py` evaluates all conditions on MATH-500 with McNemar paired tests

See individual code files for detailed usage and configuration.

## Data

Summary results are in `data/`. Raw rollouts and model weights are not included due to size constraints. The pipeline scripts can regenerate all data from scratch given GPU access.
