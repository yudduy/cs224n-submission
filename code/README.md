# Reproduction Instructions

## Requirements
- Python 3.10+
- vLLM (`pip install vllm`)
- PyTorch 2.x with CUDA
- transformers, datasets, scipy, sklearn, matplotlib
- trl, unsloth (for SFT/DPO training)

## Hardware
All experiments were run on a single A100 80GB GPU via Mithril (SkyPilot).

## Scripts

### pipeline.py — DEEP-GRPO Pivot Pipeline (Steps 0-6)
1. Load MATH-500, generate N=8 rollouts via vLLM (Qwen3-8B)
2. Goldilocks filter (mixed correct/incorrect)
3. DEEP-GRPO pivot identification with online P_phi bootstrap
4. Correction generation at pivots + baselines (random, root, first-divergence)
5. Causal validation via symbolic answer equivalence

```bash
python pipeline.py --model Qwen/Qwen3-8B --dataset math500 --output /path/to/results
```

### evaluate.py — Figure and Summary Generation
```bash
python evaluate.py --results /path/to/results --output figures/
```

### sft_ablation.py — SFT Kill-Test (Finding 3)
Three-condition ablation: clean traces (A), clean + corrections (B), volume control (C).
```bash
python sft_ablation.py train --condition a --data-dir /path/to/sft_data --output-dir /path/to/models
python sft_ablation.py eval --model-dir /path/to/models/model_a --data-dir /path/to/sft_data
```

### divergence_dpo.py — Bifurcation-Aware DPO
DPO training with divergence-aligned preference pairs from natural rollout bifurcations.
```bash
python divergence_dpo.py train --condition 1 --data-dir /path/to/dpo_data --output-dir /path/to/models
```

## Models
- **Qwen3-8B** (`Qwen/Qwen3-8B`) — pipeline and SFT/DPO base
- **DeepSeek-R1-Distill-Qwen** (3B, 7B) — scaffold kill-test (separate repo: cs224n-scaf)
