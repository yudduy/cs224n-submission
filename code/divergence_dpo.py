#!/usr/bin/env python3
"""E-104: Step-DPO with Divergence-Depth Selection.

Key insight: Divergence depth as zero-cost proxy for step-level pair quality.
Late-divergence pairs share longer correct prefixes → more concentrated
gradient signal at the actual decision point (Step-DPO prefix cancellation).

3-condition ablation (all Step-DPO, varying PAIR SELECTION):
  Cond 1: Random subset of all pairs (dataset size control)
  Cond 2: Late-divergence pairs only (E-102 fuzzy depth > 10%) — OUR METHOD
  Cond 3: Early-divergence pairs only (E-102 fuzzy depth ≤ 10%) — expected-worse

Step-DPO format:
  prompt  = ChatML(problem) + shared_prefix[:div_point]
  chosen  = correct_suffix (from correction_text)
  rejected = incorrect_suffix (from original rollout)

β is scaled by median suffix length: β_eff = β_base * (median_full_len / median_suffix_len)
This compensates for shorter sequences having less room for log-prob divergence.

Pipeline:
  Step 0: Score (GPU) — measure model uncertainty per pair → kill test
  Step 1: Prep (CPU) — build Step-DPO pairs for 3 conditions from E-102 data
  Step 2: Train (GPU) — Step-DPO on E-103 Model A checkpoint, 3 conditions
  Step 3: Eval (GPU) — MATH-500 pass@1 + diagnostics

Usage:
  # Step 1: Prepare Step-DPO pairs (CPU, local)
  python e104_divergence_dpo.py prep \
    --rollouts-file data/exit_pipeline_results/rollouts.jsonl \
    --corrections-file data/analysis/e102_results/corrections_max.jsonl \
    --output-dir data/dpo_data/e104

  # Step 2: Train (GPU, Mithril) — requires E-103 Model A checkpoint
  python e104_divergence_dpo.py train \
    --data-dir data/dpo_data/e104 \
    --sft-model-dir /mnt/local/e103_models/model_a/merged \
    --condition 2 \
    --beta 0.3 \
    --output-dir /mnt/local/e104_models

  # Step 3: Evaluate (GPU, Mithril)
  python e104_divergence_dpo.py eval \
    --model-dir /mnt/local/e104_models/cond2_beta0.3 \
    --data-dir data/dpo_data/e104 \
    --output-dir /mnt/local/e104_eval
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

SEED = 42
MAX_SEQ_LENGTH = 4096


def format_chatml_prompt(problem: str) -> str:
    """Format problem as Qwen3 ChatML prompt (up to assistant header)."""
    return f"<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n"


def format_chatml_response(text: str) -> str:
    """Format response with ChatML end token."""
    return f"{text}<|im_end|>"

# ═══════════════════════════════════════════════════════════════════
# Answer Extraction (shared with E-102, E-103)
# ═══════════════════════════════════════════════════════════════════

def extract_boxed(text: str) -> str | None:
    """Extract last \\boxed{...} content, handling nested braces."""
    i = text.rfind("\\boxed{")
    if i == -1:
        return None
    depth, start = 0, i + 7
    for j in range(start, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            if depth == 0:
                return text[start:j].strip()
            depth -= 1
    return None


def extract_clean_answer(text: str) -> str | None:
    ans = extract_boxed(text)
    if ans is not None:
        return ans
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    return numbers[-1] if numbers else None


def answers_match(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    a_clean = re.sub(r"\s+", "", a.lower().strip())
    b_clean = re.sub(r"\s+", "", b.lower().strip())
    if a_clean == b_clean:
        return True
    try:
        return abs(float(a_clean) - float(b_clean)) < 1e-6
    except (ValueError, OverflowError):
        return False



# ═══════════════════════════════════════════════════════════════════
# Step 1: DPO Pair Construction
# ═══════════════════════════════════════════════════════════════════

def build_dpo_pairs(rollouts_data: list[dict],
                    corrections_data: list[dict],
                    uncertainty_scores: dict[str, dict] | None = None,
                    ) -> dict[str, list[dict]]:
    """Build Step-DPO pairs for all conditions.

    DESIGN (Step-DPO with divergence-guided pair selection):
    Each pair uses the prefix up to the divergence point as part of the prompt.
    The chosen response is the correction suffix (from E-102), and the rejected
    response is the wrong suffix from the original rollout. This concentrates
    the DPO gradient on the decision boundary at the divergence point.

    Conditions:
      Cond1: Random subset of all pairs (size control)
      Cond2: Late-divergence pairs only (E-102 fuzzy depth > 10%) — OUR METHOD
      Cond3: If uncertainty_scores provided: top-576 by μ(1-μ) (Cond3u).
             Otherwise: random subset (backward compat).
      Cond4: Early-divergence pairs only (E-102 fuzzy depth ≤ 10%)
      Cond5: All annotated pairs with continuous divergence-depth weighting
             (sigmoid-weighted resampling to target_size)

    All conditions subsampled to the SAME pair count for fair comparison.

    Returns dict with keys: cond1, cond2, cond3, cond4, cond5
    """
    random.seed(SEED)
    np.random.seed(SEED)

    # Index corrections by (problem_idx, rollout_id)
    # corrections_max.jsonl has multiple entries per key (one per reference rollout).
    # Select the best CORRECT correction per key (prefer higher div_depth for richer prefix).
    corrections_by_key = {}
    for c in corrections_data:
        key = (c["problem_idx"], c["rollout_id"])
        if not c.get("correction_correct", False):
            # Only consider correct corrections
            if key not in corrections_by_key:
                corrections_by_key[key] = c  # fallback: keep even incorrect if no other
            continue
        prev = corrections_by_key.get(key)
        if prev is None or not prev.get("correction_correct", False):
            # First correct correction for this key
            corrections_by_key[key] = c
        elif c["div_depth"] > prev["div_depth"]:
            # Prefer deeper divergence (richer prefix)
            corrections_by_key[key] = c

    # Index rollout text by (problem_idx, rollout_id)
    rollout_text_by_key = {}
    for prob in rollouts_data:
        for r in prob["rollouts"]:
            rollout_text_by_key[(prob["problem_idx"], r["rollout_id"])] = r["text"]

    # Filter to Goldilocks problems (1-7 correct out of 8)
    goldilocks = [d for d in rollouts_data if 1 <= d["n_correct"] <= 7]
    print(f"Goldilocks problems: {len(goldilocks)}")

    # Build Step-DPO pairs from corrections data
    # Each pair: prefix (shared) → prompt, correction → chosen, wrong suffix → rejected
    all_pairs = []
    for prob in goldilocks:
        problem_idx = prob["problem_idx"]
        problem_text = prob["problem"]

        incorrect_rollouts = [r for r in prob["rollouts"] if not r["is_correct"]]

        if not incorrect_rollouts:
            continue

        for inc_r in incorrect_rollouts:
            inc_text = inc_r["text"]
            inc_id = inc_r["rollout_id"]

            # Get E-102 correction data (required for Step-DPO)
            corr_key = (problem_idx, inc_id)
            corr_info = corrections_by_key.get(corr_key)
            if corr_info is None:
                continue  # No correction data → skip
            if not corr_info.get("correction_correct", False):
                continue  # Correction itself is wrong → skip

            div_pos = corr_info["div_pos"]
            e102_div_depth = corr_info["div_depth"]

            # Step-DPO pair construction:
            # - prefix = rollout text up to divergence point
            # - chosen = correction_text (correct continuation from div point)
            # - rejected = rollout text after divergence point (wrong continuation)
            prefix = inc_text[:div_pos]
            wrong_suffix = inc_text[div_pos:]
            correct_suffix = corr_info["correction_text"]

            # Skip if either suffix is empty
            if not wrong_suffix.strip() or not correct_suffix.strip():
                continue

            pair = {
                "problem_idx": problem_idx,
                "problem": problem_text,
                "prompt": problem_text,
                "prefix": prefix,
                "chosen": correct_suffix,
                "rejected": wrong_suffix,
                "metadata": {
                    "incorrect_rollout_id": inc_id,
                    "e102_div_depth": e102_div_depth,
                    "e102_div_pos": div_pos,
                    "n_refs": corr_info["n_refs"],
                    "prefix_len": len(prefix),
                    "chosen_len": len(correct_suffix),
                    "rejected_len": len(wrong_suffix),
                }
            }
            all_pairs.append(pair)

    print(f"Total candidate pairs: {len(all_pairs)}")

    # E-102 divergence depth distribution
    depths = [p["metadata"]["e102_div_depth"] for p in all_pairs]
    print(f"E-102 fuzzy divergence depth:")
    print(f"  Mean: {np.mean(depths):.3f}, Median: {np.median(depths):.3f}")
    print(f"  25th: {np.percentile(depths, 25):.3f}, "
          f"75th: {np.percentile(depths, 75):.3f}")

    # ─── Filter out unannotated pairs (no E-102 data) ───
    annotated_pairs = [p for p in all_pairs if p["metadata"]["n_refs"] > 0]
    n_unannotated = len(all_pairs) - len(annotated_pairs)
    if n_unannotated > 0:
        print(f"WARNING: {n_unannotated} pairs have no E-102 annotation — "
              f"excluded from cond2/cond4")

    # ─── Condition 2: Late divergence pairs (depth > 10%) ───
    LATE_THRESHOLD = 0.10
    cond2_pairs = [p for p in annotated_pairs
                   if p["metadata"]["e102_div_depth"] > LATE_THRESHOLD]
    for p in cond2_pairs:
        p["condition"] = 2

    # ─── Condition 4: Early divergence pairs (depth ≤ 10%) ───
    cond4_pairs = [p for p in annotated_pairs
                   if p["metadata"]["e102_div_depth"] <= LATE_THRESHOLD]
    for p in cond4_pairs:
        p["condition"] = 4

    print(f"\nLate divergence (>{LATE_THRESHOLD:.0%}): {len(cond2_pairs)} pairs")
    print(f"Early divergence (≤{LATE_THRESHOLD:.0%}): {len(cond4_pairs)} pairs")

    # ─── Equalize dataset sizes ───
    target_size = min(len(cond2_pairs), len(cond4_pairs))
    print(f"Target size (equalized): {target_size} pairs")

    # Subsample cond2 and cond4 to target_size
    # Each condition gets its own RNG to avoid cross-condition state contamination
    rng2 = random.Random(SEED + 2)
    rng4 = random.Random(SEED + 4)
    if len(cond2_pairs) > target_size:
        cond2_pairs = rng2.sample(cond2_pairs, target_size)
    if len(cond4_pairs) > target_size:
        cond4_pairs = rng4.sample(cond4_pairs, target_size)

    # ─── Condition 1: Random subset of annotated pairs ───
    # Draw from annotated_pairs (not all_pairs) to avoid mixing unannotated
    # pairs with e102_div_depth=0.0, which would confound Cond1 vs Cond2
    rng1 = random.Random(SEED + 1)
    cond1_pairs = [dict(p, condition=1)
                   for p in rng1.sample(annotated_pairs, min(target_size, len(annotated_pairs)))]

    # ─── Condition 3: Uncertainty-sampled (Cond3u) or random fallback ───
    if uncertainty_scores is not None:
        # Cond3u: top pairs by μ(1-μ) uncertainty proxy
        # Build pair key → uncertainty mapping
        scored_pairs = []
        for p in all_pairs:
            pair_key = f"{p['problem_idx']}_{p['metadata']['correct_rollout_id']}_{p['metadata']['incorrect_rollout_id']}"
            score_info = uncertainty_scores.get(pair_key)
            if score_info is not None:
                mu = score_info.get("mu_tau1.0", 0.5)  # default tau=1.0
                informativeness = mu * (1 - mu)
                scored_pairs.append((informativeness, p))
        scored_pairs.sort(key=lambda x: x[0], reverse=True)  # most uncertain first
        cond3_pairs = [dict(p, condition=3)
                       for _, p in scored_pairs[:target_size]]
        cond3_label = "Uncertainty-sampled (Cond3u)"
        print(f"\nCond3u: selected top-{len(cond3_pairs)} by μ(1-μ) from "
              f"{len(scored_pairs)} scored pairs")
        if cond3_pairs:
            uncertainties = [s for s, _ in scored_pairs[:target_size]]
            print(f"  μ(1-μ) range: [{min(uncertainties):.4f}, {max(uncertainties):.4f}]")
    else:
        # Fallback: random subset from annotated pairs (backward compat)
        rng3 = random.Random(SEED + 3)
        cond3_pairs = [dict(p, condition=3)
                       for p in rng3.sample(annotated_pairs, min(target_size, len(annotated_pairs)))]
        cond3_label = "Random subset (control)"

    # ─── Condition 5: Continuous divergence-depth weighting ───
    # All annotated pairs, weighted by sigmoid of divergence depth
    # Weight = σ(10(d - 0.10)) — high weight for late divergence
    weighted_pairs = []
    for p in annotated_pairs:
        d = p["metadata"]["e102_div_depth"]
        weight = 1.0 / (1.0 + math.exp(-10 * (d - 0.10)))
        weighted_pairs.append((weight, p))

    # Weighted resampling to target_size WITH replacement (pool ≈ target_size,
    # so replace=False would nullify the weights)
    if weighted_pairs:
        weights_arr = np.array([w for w, _ in weighted_pairs])
        weights_arr = weights_arr / weights_arr.sum()  # normalize to probabilities
        indices = np.random.choice(
            len(weighted_pairs), size=target_size,
            replace=True, p=weights_arr,
        )
        cond5_pairs = []
        for idx in indices:
            w, p = weighted_pairs[idx]
            cond5_pairs.append(dict(p, condition=5,
                                    **{"metadata": {**p["metadata"], "dpo_weight": w}}))
    else:
        cond5_pairs = []

    # Print stats
    print(f"\n=== DPO Pair Construction Stats (equalized) ===")
    print(f"Cond 1 (Full-trace, random subset):  {len(cond1_pairs)} pairs")
    print(f"Cond 2 (Late divergence, >10%):      {len(cond2_pairs)} pairs")
    print(f"Cond 3 ({cond3_label}):     {len(cond3_pairs)} pairs")
    print(f"Cond 4 (Early divergence, ≤10%):     {len(cond4_pairs)} pairs")
    print(f"Cond 5 (Weighted resampling):         {len(cond5_pairs)} pairs")

    # Divergence depth stats per condition
    for name, pairs in [("cond2", cond2_pairs), ("cond4", cond4_pairs),
                         ("cond5", cond5_pairs)]:
        if pairs:
            d = [p["metadata"]["e102_div_depth"] for p in pairs]
            print(f"  {name} div depth: mean={np.mean(d):.3f}, "
                  f"median={np.median(d):.3f}")

    if cond5_pairs:
        weights = [p["metadata"]["dpo_weight"] for p in cond5_pairs]
        print(f"  cond5 weights: mean={np.mean(weights):.3f}, "
              f"median={np.median(weights):.3f}")

    return {
        "cond1": cond1_pairs,
        "cond2": cond2_pairs,
        "cond3": cond3_pairs,
        "cond4": cond4_pairs,
        "cond5": cond5_pairs,
    }


def split_by_problem(pairs: list[dict], val_ratio: float = 0.15,
                     seed: int = SEED) -> tuple[list[dict], list[dict]]:
    """Problem-level train/val split — NOT pair-level."""
    rng = random.Random(seed)
    problem_ids = sorted(set(p["problem_idx"] for p in pairs))
    rng.shuffle(problem_ids)

    n_val = max(1, int(len(problem_ids) * val_ratio))
    val_ids = set(problem_ids[:n_val])

    train = [p for p in pairs if p["problem_idx"] not in val_ids]
    val = [p for p in pairs if p["problem_idx"] in val_ids]
    return train, val


def save_jsonl(data: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")


def cmd_prep(args):
    """Build DPO pairs for all 5 conditions."""
    print("=" * 60)
    print("E-104 Divergence-DPO — Data Preparation")
    print("=" * 60)

    # Load data
    print(f"Loading rollouts from {args.rollouts_file}...")
    rollouts = [json.loads(l) for l in open(args.rollouts_file)]
    print(f"  {len(rollouts)} problems")

    print(f"Loading corrections from {args.corrections_file}...")
    corrections = [json.loads(l) for l in open(args.corrections_file)]
    print(f"  {len(corrections)} corrections")

    # Load uncertainty scores if provided
    uncertainty_scores = None
    if hasattr(args, 'uncertainty_scores') and args.uncertainty_scores:
        print(f"Loading uncertainty scores from {args.uncertainty_scores}...")
        uncertainty_scores = {}
        for line in open(args.uncertainty_scores):
            rec = json.loads(line)
            uncertainty_scores[rec["pair_key"]] = rec
        print(f"  {len(uncertainty_scores)} scored pairs")

    # Build pairs
    all_pairs = build_dpo_pairs(rollouts, corrections,
                                uncertainty_scores=uncertainty_scores)

    # Save with problem-level train/val split
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for cond_name, pairs in all_pairs.items():
        if not pairs:
            print(f"WARNING: {cond_name} has no pairs!")
            continue

        train, val = split_by_problem(pairs)

        save_jsonl(train, output_dir / f"{cond_name}_train.jsonl")
        save_jsonl(val, output_dir / f"{cond_name}_val.jsonl")

        summary[cond_name] = {
            "total": len(pairs),
            "train": len(train),
            "val": len(val),
            "n_problems": len(set(p["problem_idx"] for p in pairs)),
            "n_train_problems": len(set(p["problem_idx"] for p in train)),
            "n_val_problems": len(set(p["problem_idx"] for p in val)),
        }
        print(f"  {cond_name}: {len(train)} train + {len(val)} val "
              f"({summary[cond_name]['n_train_problems']} + "
              f"{summary[cond_name]['n_val_problems']} problems)")

    # Save held-out eval problems (not in any training set)
    all_train_problems = set()
    for pairs in all_pairs.values():
        train, _ = split_by_problem(pairs)
        all_train_problems.update(p["problem_idx"] for p in train)

    eval_problems = [d for d in rollouts if d["problem_idx"] not in all_train_problems
                     and d["n_correct"] == 0]  # Include all-wrong for harder eval
    # Also include all-correct (ceiling check)
    eval_problems += [d for d in rollouts if d["n_correct"] == 8]
    # And the val split of Goldilocks
    goldilocks_val_ids = set()
    for pairs in all_pairs.values():
        _, val = split_by_problem(pairs)
        goldilocks_val_ids.update(p["problem_idx"] for p in val)
    eval_problems += [d for d in rollouts if d["problem_idx"] in goldilocks_val_ids]

    # Deduplicate
    seen = set()
    eval_unique = []
    for d in eval_problems:
        if d["problem_idx"] not in seen:
            eval_unique.append({
                "problem_idx": d["problem_idx"],
                "problem": d["problem"],
                "gt_answer": d["gt_answer"],
            })
            seen.add(d["problem_idx"])

    save_jsonl(eval_unique, output_dir / "eval_problems.jsonl")
    summary["eval_problems"] = len(eval_unique)

    # Save full MATH-500 for pass@1 eval
    all_problems = [{
        "problem_idx": d["problem_idx"],
        "problem": d["problem"],
        "gt_answer": d["gt_answer"],
    } for d in rollouts]
    save_jsonl(all_problems, output_dir / "math500_all.jsonl")
    summary["math500_total"] = len(all_problems)

    # Save summary
    with open(output_dir / "prep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {output_dir / 'prep_summary.json'}")
    print(json.dumps(summary, indent=2))


# ═══════════════════════════════════════════════════════════════════
# Step 0: Score — Measure model uncertainty per pair (KILL TEST)
# ═══════════════════════════════════════════════════════════════════

def cmd_score(args):
    """Score DPO pairs with SFT model to measure uncertainty.

    For each pair, compute:
      s_i = (1/|y⁺|)·log π(y⁺|x) - (1/|y⁻|)·log π(y⁻|x)
      μ_i = σ(τ·s_i) for τ ∈ {0.5, 1.0, 2.0}
      informativeness = μ_i·(1 - μ_i)

    KILL TEST: If E[μ(1-μ) | Cond2] ≤ E[μ(1-μ) | Cond4], the Active DPO
    reframing fails — divergence depth does NOT proxy for informativeness.
    """
    print("=" * 60)
    print("E-104 Divergence-DPO — Score (Kill Test)")
    print("=" * 60)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all condition files to score
    all_pairs = []
    for cond in [1, 2, 3, 4]:
        train_file = data_dir / f"cond{cond}_train.jsonl"
        val_file = data_dir / f"cond{cond}_val.jsonl"
        for f in [train_file, val_file]:
            if f.exists():
                pairs = [json.loads(l) for l in open(f)]
                for p in pairs:
                    p.setdefault("condition", cond)
                all_pairs.append((cond, pairs))

    # Deduplicate pairs by (problem_idx, correct_id, incorrect_id)
    unique_pairs = {}
    for cond, pairs in all_pairs:
        for p in pairs:
            key = (f"{p['problem_idx']}_"
                   f"{p['metadata']['correct_rollout_id']}_"
                   f"{p['metadata']['incorrect_rollout_id']}")
            if key not in unique_pairs:
                unique_pairs[key] = p
    pair_list = list(unique_pairs.items())
    print(f"Unique pairs to score: {len(pair_list)}")

    # Load SFT model
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(f"Missing dependency: {e}")
        sys.exit(1)

    print(f"\nLoading SFT model from {args.sft_model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    # Apply chat template consistent with SFT training and eval
    CHAT_TEMPLATE = "<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n"

    def compute_logprob(formatted_prompt: str, response_text: str) -> tuple[float, int]:
        """Compute total log-prob of response given formatted prompt.

        Args:
            formatted_prompt: Chat-templated prompt (with special tokens).
            response_text: Raw response text.

        Returns (total_logprob, n_response_tokens).
        """
        prompt_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
        response_ids = tokenizer.encode(response_text, add_special_tokens=False)
        input_ids = prompt_ids + response_ids

        # Compute response_len before truncation
        if len(input_ids) > MAX_SEQ_LENGTH:
            input_ids = input_ids[:MAX_SEQ_LENGTH]
        response_len = max(0, len(input_ids) - len(prompt_ids))

        if response_len == 0:
            return 0.0, 0

        input_tensor = torch.tensor([input_ids], device=model.device)
        with torch.no_grad():
            outputs = model(input_tensor)
            logits = outputs.logits[0]  # (seq_len, vocab_size)

        # Log-probs for response tokens (shifted by 1)
        prompt_len = len(prompt_ids)
        total_logprob = 0.0
        for i in range(prompt_len, len(input_ids)):
            token_logprobs = torch.log_softmax(logits[i - 1], dim=-1)
            total_logprob += token_logprobs[input_ids[i]].item()

        return total_logprob, response_len

    # Score all pairs
    taus = [0.5, 1.0, 2.0]
    results = []
    n_skipped_long = 0
    for idx, (pair_key, p) in enumerate(pair_list):
        if idx % 50 == 0:
            print(f"  Scoring pair {idx}/{len(pair_list)}...")

        # Format prompt with chat template (consistent with SFT and eval)
        formatted_prompt = CHAT_TEMPLATE.format(problem=p["prompt"])
        chosen = p["chosen"]
        rejected = p["rejected"]

        logprob_chosen, n_chosen = compute_logprob(formatted_prompt, chosen)
        logprob_rejected, n_rejected = compute_logprob(formatted_prompt, rejected)

        # Length-normalized log-prob gap
        if n_chosen > 0 and n_rejected > 0:
            s_i = (logprob_chosen / n_chosen) - (logprob_rejected / n_rejected)
        else:
            s_i = 0.0
            n_skipped_long += 1

        record = {
            "pair_key": pair_key,
            "problem_idx": p["problem_idx"],
            "e102_div_depth": p["metadata"]["e102_div_depth"],
            "condition": p.get("condition", 0),
            "logprob_chosen": logprob_chosen,
            "logprob_rejected": logprob_rejected,
            "n_chosen_tokens": n_chosen,
            "n_rejected_tokens": n_rejected,
            "s_i": s_i,
        }

        # Compute μ and informativeness for each temperature
        for tau in taus:
            mu = 1.0 / (1.0 + math.exp(-tau * s_i)) if abs(tau * s_i) < 500 else (1.0 if s_i > 0 else 0.0)
            record[f"mu_tau{tau}"] = mu
            record[f"informativeness_tau{tau}"] = mu * (1 - mu)

        results.append(record)

    if n_skipped_long > 0:
        print(f"\nWARNING: {n_skipped_long}/{len(pair_list)} pairs had prompt too long "
              f"to score (>={MAX_SEQ_LENGTH} tokens). Scored as s_i=0.")

    # Save scores
    scores_file = output_dir / "uncertainty_scores.jsonl"
    save_jsonl(results, scores_file)
    print(f"\nScores saved to {scores_file}")

    # ─── Kill test analysis ───
    print("\n" + "=" * 60)
    print("KILL TEST: E[μ(1-μ) | Cond2] vs E[μ(1-μ) | Cond4]")
    print("=" * 60)

    # Group strictly by condition field to avoid mixing Cond1/3 pairs
    # into late/early buckets
    for tau in taus:
        key = f"informativeness_tau{tau}"
        cond2_vals = [r[key] for r in results if r["condition"] == 2]
        cond4_vals = [r[key] for r in results if r["condition"] == 4]

        if not cond2_vals or not cond4_vals:
            print(f"\n  τ={tau}: insufficient data (cond2={len(cond2_vals)}, cond4={len(cond4_vals)})")
            continue

        mean_cond2 = np.mean(cond2_vals)
        mean_cond4 = np.mean(cond4_vals)
        diff = mean_cond2 - mean_cond4

        print(f"\n  τ={tau}:")
        print(f"    Cond2 (late div):  E[μ(1-μ)] = {mean_cond2:.4f} "
              f"(n={len(cond2_vals)}, std={np.std(cond2_vals):.4f})")
        print(f"    Cond4 (early div): E[μ(1-μ)] = {mean_cond4:.4f} "
              f"(n={len(cond4_vals)}, std={np.std(cond4_vals):.4f})")
        print(f"    Δ = {diff:+.4f}")

        if diff > 0:
            print(f"    ✓ PASS — late divergence IS more uncertain (Active DPO viable)")
        else:
            print(f"    ✗ FAIL — late divergence NOT more uncertain (Active DPO reframing dead)")

    # Distribution stats per condition
    print(f"\n--- Distribution Summary (τ=1.0) ---")
    for cond_label, cond_id in [("Cond1 (random)", 1), ("Cond2 (late)", 2),
                                 ("Cond3 (control)", 3), ("Cond4 (early)", 4)]:
        vals = [r["informativeness_tau1.0"] for r in results if r["condition"] == cond_id]
        if vals:
            print(f"  {cond_label}: mean={np.mean(vals):.4f}, "
                  f"median={np.median(vals):.4f}, "
                  f"std={np.std(vals):.4f}, n={len(vals)}")


# ═══════════════════════════════════════════════════════════════════
# Step 2: DPO Training
# ═══════════════════════════════════════════════════════════════════

class GradientDiagnosticsCallback:
    """Lightweight gradient logging: norm + direction consistency between steps.

    Uses Trainer's logged grad_norm (computed before zero_grad) and tracks
    direction consistency via accumulated parameter deltas between steps.
    Saved to gradient_diagnostics.jsonl in the output directory.

    NOTE: TrainerCallback.on_log does NOT receive model kwarg, and gradients
    are zeroed before on_log fires. We extract grad_norm from Trainer's own
    logs (which computes it pre-zero) and approximate direction consistency
    from parameter deltas instead.
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.prev_params = None
        self.prev_delta = None
        self.records = []

    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        """Extract grad_norm from Trainer's logs + compute param delta cosine."""
        if logs is None:
            return

        # Trainer already logs grad_norm (computed before zero_grad)
        grad_norm = logs.get("grad_norm")

        record = {
            "step": state.global_step,
            "grad_norm": grad_norm,
            "train_loss": logs.get("loss"),
            "param_delta_cos": None,
        }
        self.records.append(record)

    def compute_param_delta(self, model, step):
        """Compute parameter delta direction consistency.

        Called manually at checkpoints to compare parameter movement direction
        between consecutive intervals. Cheaper than per-step grad capture.
        """
        try:
            import torch
            current_params = torch.cat([
                p.detach().flatten() for p in model.parameters()
                if p.requires_grad
            ])

            if self.prev_params is not None:
                delta = current_params - self.prev_params
                if self.prev_delta is not None:
                    cos_sim = (torch.dot(delta, self.prev_delta) /
                               (delta.norm() * self.prev_delta.norm() + 1e-8)).item()
                    # Update the last record with cosine similarity
                    if self.records:
                        self.records[-1]["param_delta_cos"] = cos_sim
                self.prev_delta = delta.clone()

            self.prev_params = current_params.clone()
        except Exception as e:
            print(f"  [GradDiag] param delta error at step {step}: {e}")

    def save(self):
        """Save collected records to JSONL."""
        if self.records:
            out_file = self.output_dir / "gradient_diagnostics.jsonl"
            save_jsonl(self.records, out_file)
            print(f"Gradient diagnostics saved to {out_file} ({len(self.records)} records)")


def cmd_train(args):
    """Train DPO on a specific condition."""
    condition = args.condition
    beta = args.beta
    print("=" * 60)
    print(f"E-104 Divergence-DPO — Training Condition {condition} (β={beta})")
    print("=" * 60)

    data_dir = Path(args.data_dir)
    sft_model_dir = args.sft_model_dir

    # Load DPO pairs
    cond_name = f"cond{condition}"
    train_file = data_dir / f"{cond_name}_train.jsonl"
    val_file = data_dir / f"{cond_name}_val.jsonl"

    if not train_file.exists():
        print(f"ERROR: {train_file} not found. Run 'prep' first.")
        sys.exit(1)

    train_data = [json.loads(l) for l in open(train_file)]
    val_data = [json.loads(l) for l in open(val_file)]
    print(f"Train: {len(train_data)} pairs, Val: {len(val_data)} pairs")

    # Import training dependencies
    try:
        from unsloth import FastLanguageModel, PatchDPOTrainer
        PatchDPOTrainer()  # CRITICAL: must patch before constructing DPOTrainer
        from trl import DPOConfig, DPOTrainer
        from datasets import Dataset
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install: pip install unsloth trl datasets")
        sys.exit(1)

    # Load SFT model as base (E-103 Model A)
    print(f"\nLoading SFT base model from {sft_model_dir}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        sft_model_dir,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,
    )

    # Apply LoRA for DPO fine-tuning
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        lora_alpha=128,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # Format data for Step-DPO
    # DPOTrainer expects: prompt, chosen, rejected
    # Step-DPO: prompt = ChatML(problem) + prefix, chosen/rejected = suffixes only
    # CRITICAL: Must apply ChatML formatting since model was SFT'd on ChatML
    def format_for_dpo(data: list[dict]) -> Dataset:
        formatted = []
        for d in data:
            prefix = d.get("prefix", "")
            if prefix:
                # Step-DPO: prefix becomes part of the prompt
                prompt = format_chatml_prompt(d["prompt"]) + prefix
                chosen = format_chatml_response(d["chosen"])
                rejected = format_chatml_response(d["rejected"])
            else:
                # Fallback: full-trace (no divergence info)
                prompt = format_chatml_prompt(d["prompt"])
                chosen = format_chatml_response(d["chosen"])
                rejected = format_chatml_response(d["rejected"])
            formatted.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            })
        return Dataset.from_list(formatted)

    train_ds = format_for_dpo(train_data)
    val_ds = format_for_dpo(val_data)

    # Output directory
    output_dir = Path(args.output_dir) / f"cond{condition}_beta{beta}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # DPO config
    lr = args.learning_rate
    n_epochs = args.epochs

    config = DPOConfig(
        output_dir=str(output_dir),
        beta=beta,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,  # effective batch = 8 (2×4)
        num_train_epochs=n_epochs,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        max_length=MAX_SEQ_LENGTH,
        max_prompt_length=MAX_SEQ_LENGTH // 2,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,  # Unsloth LoRA reload unreliable for DPO
        logging_steps=10,
        save_total_limit=2,
        weight_decay=0.01,
        max_grad_norm=1.0,
        seed=SEED,
        report_to="none",
        remove_unused_columns=False,
    )

    # Verify Unsloth LoRA model supports adapter toggling for implicit reference
    assert hasattr(model, 'disable_adapter_layers'), \
        "Model missing disable_adapter_layers — ref_model=None requires PEFT adapter toggling"
    print("  ✓ Adapter toggling verified (ref_model=None will use frozen base as reference)")

    # Gradient diagnostics callback — extracts grad_norm from Trainer logs
    from transformers import TrainerCallback

    grad_diag = GradientDiagnosticsCallback(str(output_dir))

    class _GradCallback(TrainerCallback):
        def on_step_end(self, args, state, control, model=None, **kwargs):
            grad_diag.on_log(args, state, control, logs=None, model=model, **kwargs)

        def on_evaluate(self, args, state, control, model=None, **kwargs):
            # Compute param delta at eval boundaries (end of each epoch)
            if model is not None:
                grad_diag.compute_param_delta(model, state.global_step)

    # TRL API changed: processing_class (≥0.12) vs tokenizer (older)
    import inspect
    dpo_init_params = inspect.signature(DPOTrainer.__init__).parameters
    if "processing_class" in dpo_init_params:
        tokenizer_kwarg = {"processing_class": tokenizer}
    else:
        tokenizer_kwarg = {"tokenizer": tokenizer}

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Use implicit reference (LoRA base = reference)
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=config,
        callbacks=[_GradCallback()],
        **tokenizer_kwarg,
    )

    print(f"\nStarting DPO training: {n_epochs} epochs, β={beta}, "
          f"lr={lr}, batch=2x4")
    print(f"  Train pairs: {len(train_data)}")
    print(f"  Val pairs: {len(val_data)}")

    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    print(f"\nTraining complete in {elapsed/60:.1f} min")

    # Save gradient diagnostics
    grad_diag.save()

    # Save final model
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    print(f"Model saved to {output_dir / 'final'}")

    # Save training log (comprehensive for HF sharing)
    log = {
        "condition": condition,
        "beta": beta,
        "learning_rate": lr,
        "epochs": n_epochs,
        "n_train": len(train_data),
        "n_val": len(val_data),
        "training_time_s": elapsed,
        "final_train_loss": trainer.state.log_history[-1].get("train_loss"),
        "best_eval_loss": trainer.state.best_metric,
        "loss_type": "sigmoid",  # standard DPO
        "format": "step_dpo",
        "lora_r": 64,
        "lora_alpha": 128,
        "per_device_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "effective_batch_size": 8,
        "max_seq_length": MAX_SEQ_LENGTH,
    }
    with open(output_dir / "train_log.json", "w") as f:
        json.dump(log, f, indent=2)

    # Save full training curve (step-level loss/grad for plotting)
    with open(output_dir / "training_curve.jsonl", "w") as f:
        for entry in trainer.state.log_history:
            f.write(json.dumps(entry) + "\n")
    print(f"Training curve: {len(trainer.state.log_history)} entries → {output_dir / 'training_curve.jsonl'}")

    # Save data distribution metadata (for reproducibility)
    depth_stats = {
        "depths": [d["metadata"]["e102_div_depth"] for d in train_data],
        "prefix_lens": [d["metadata"]["prefix_len"] for d in train_data],
        "chosen_lens": [d["metadata"]["chosen_len"] for d in train_data],
        "rejected_lens": [d["metadata"]["rejected_len"] for d in train_data],
    }
    with open(output_dir / "data_distribution.json", "w") as f:
        json.dump({k: {"mean": float(np.mean(v)), "median": float(np.median(v)),
                        "std": float(np.std(v)), "min": float(np.min(v)),
                        "max": float(np.max(v)), "n": len(v)}
                   for k, v in depth_stats.items()}, f, indent=2)

    print(json.dumps(log, indent=2))


# ═══════════════════════════════════════════════════════════════════
# Step 3: Evaluation
# ═══════════════════════════════════════════════════════════════════

def cmd_eval(args):
    """Evaluate a DPO-trained model on MATH-500."""
    print("=" * 60)
    print(f"E-104 Divergence-DPO — Evaluation")
    print("=" * 60)

    model_dir = Path(args.model_dir)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load eval problems (full MATH-500)
    eval_file = data_dir / "math500_all.jsonl"
    eval_data = [json.loads(l) for l in open(eval_file)]
    print(f"Eval problems: {len(eval_data)}")

    # Merge LoRA weights for vLLM
    # load_in_4bit=True matches QLoRA training; Unsloth handles dequant in merged_16bit
    merged_dir = model_dir / "merged"
    if not merged_dir.exists():
        print("Merging LoRA weights for vLLM...")
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            str(model_dir / "final"),
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=True,
        )
        model.save_pretrained_merged(str(merged_dir), tokenizer,
                                     save_method="merged_16bit")
        print(f"Merged model saved to {merged_dir}")

    from vllm import LLM, SamplingParams

    llm = LLM(
        str(merged_dir),
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=True,
    )

    # MATH-500 pass@1
    print("\n--- MATH-500 pass@1 ---")
    prompts = [
        f"<|im_start|>user\n{d['problem']}<|im_end|>\n<|im_start|>assistant\n"
        for d in eval_data
    ]
    sampling = SamplingParams(temperature=0, top_p=1.0, max_tokens=2048)
    outputs = llm.generate(prompts, sampling)

    correct = 0
    results_per_problem = []
    generation_lengths = []
    for d, out in zip(eval_data, outputs):
        text = out.outputs[0].text
        ans = extract_clean_answer(text)
        hit = answers_match(ans, d["gt_answer"])
        correct += int(hit)
        gen_len = len(text)
        generation_lengths.append(gen_len)
        results_per_problem.append({
            "problem_idx": d["problem_idx"],
            "correct": hit,
            "answer": ans,
            "gt_answer": d["gt_answer"],
            "generation_length": gen_len,
            "generation": text,  # full text for analysis
        })

    accuracy = correct / len(eval_data) if eval_data else 0
    print(f"  Accuracy: {accuracy:.3f} ({correct}/{len(eval_data)})")
    print(f"  Generation length: mean={np.mean(generation_lengths):.0f}, "
          f"median={np.median(generation_lengths):.0f}")

    results = {
        "math500_pass_at_1": accuracy,
        "n_correct": correct,
        "n_total": len(eval_data),
        "model_dir": str(model_dir),
        "mean_generation_length": float(np.mean(generation_lengths)),
        "median_generation_length": float(np.median(generation_lengths)),
    }

    # Save
    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    save_jsonl(results_per_problem, output_dir / "per_problem_results.jsonl")
    print(f"Results saved to {output_dir}")
    print(json.dumps(results, indent=2))


def cmd_compare(args):
    """Compare all conditions."""
    print("=" * 60)
    print("E-104 Divergence-DPO — Comparison")
    print("=" * 60)

    eval_dir = Path(args.eval_dir)
    results = {}

    # Collect all eval results
    for d in sorted(eval_dir.iterdir()):
        if d.is_dir():
            eval_file = d / "eval_results.json"
            if eval_file.exists():
                with open(eval_file) as f:
                    r = json.load(f)
                results[d.name] = r
                print(f"  {d.name}: {r['math500_pass_at_1']:.3f}")

    if not results:
        print("No results found!")
        return

    # Key comparisons
    print("\n=== Key Comparisons ===")
    cond_names = sorted(results.keys())
    for i, a in enumerate(cond_names):
        for b in cond_names[i+1:]:
            diff = results[a]["math500_pass_at_1"] - results[b]["math500_pass_at_1"]
            print(f"  {a} - {b} = {diff:+.3f}")

    # Load per-problem results for paired McNemar test
    per_problem = {}
    for d in sorted(eval_dir.iterdir()):
        if d.is_dir():
            pp_file = d / "per_problem_results.jsonl"
            if pp_file.exists():
                per_problem[d.name] = {
                    r["problem_idx"]: r["correct"]
                    for r in (json.loads(l) for l in open(pp_file))
                }

    # Significance helper: McNemar paired test (preferred — more powerful)
    def mcnemar_test(name_a, name_b):
        """McNemar test on paired per-problem results."""
        pp_a = per_problem.get(name_a, {})
        pp_b = per_problem.get(name_b, {})
        if not pp_a or not pp_b:
            return None
        shared = set(pp_a.keys()) & set(pp_b.keys())
        if not shared:
            return None
        from scipy.stats import binomtest
        n_10 = sum(1 for k in shared if pp_a[k] and not pp_b[k])
        n_01 = sum(1 for k in shared if not pp_a[k] and pp_b[k])
        n_disc = n_10 + n_01
        if n_disc == 0:
            return {"p_value": 1.0, "n_discordant": 0, "n_10": 0, "n_01": 0,
                    "delta": 0.0, "n_shared": len(shared)}
        result = binomtest(n_10, n_disc, 0.5, alternative='two-sided')
        return {
            "p_value": result.pvalue,
            "n_discordant": n_disc,
            "n_10": n_10, "n_01": n_01,
            "delta": (n_10 - n_01) / len(shared),
            "n_shared": len(shared),
        }

    # Fallback: two-proportion z-test (unpaired)
    def prop_test(acc_a, n_a, acc_b, n_b):
        """Two-proportion z-test. Returns (z_stat, p_value, significant_at_05)."""
        from scipy import stats
        p_pool = (acc_a * n_a + acc_b * n_b) / (n_a + n_b)
        if p_pool == 0 or p_pool == 1:
            return 0.0, 1.0, False
        se = (p_pool * (1 - p_pool) * (1/n_a + 1/n_b)) ** 0.5
        z = (acc_a - acc_b) / se
        p_val = 2 * (1 - stats.norm.cdf(abs(z)))
        return z, p_val, p_val < 0.05

    # Kill criteria
    print("\n=== Kill Criteria ===")
    c1_keys = [k for k in results if k.startswith("cond1")]
    c2_keys = [k for k in results if k.startswith("cond2")]
    c3_keys = [k for k in results if k.startswith("cond3")]

    def best_acc(keys):
        return max(results[k]["math500_pass_at_1"] for k in keys) if keys else 0

    def report_comparison(name_a, acc_a, name_b, acc_b, msg_win, msg_lose):
        n = 500  # MATH-500
        # Prefer McNemar paired test (more powerful for same-problem eval)
        mc = mcnemar_test(name_a, name_b)
        if mc is not None:
            p_val = mc["p_value"]
            sig = p_val < 0.05
            sig_str = (f" (McNemar: Δ={mc['delta']:+.3f}, "
                       f"disc={mc['n_discordant']}, p={p_val:.3f}"
                       f"{'*' if sig else ''})")
        else:
            try:
                z, p_val, sig = prop_test(acc_a, n, acc_b, n)
                sig_str = f" (z={z:.2f}, p={p_val:.3f}{'*' if sig else ''})"
            except ImportError:
                sig_str = ""
        if acc_a > acc_b:
            print(f"  ✓ {name_a} ({acc_a:.3f}) > {name_b} ({acc_b:.3f}): "
                  f"{msg_win}{sig_str}")
        else:
            print(f"  ✗ {name_a} ({acc_a:.3f}) ≤ {name_b} ({acc_b:.3f}): "
                  f"{msg_lose}{sig_str}")

    # Use actual directory names for McNemar test (must match per_problem keys)
    if c2_keys and c3_keys:
        report_comparison(c2_keys[0], best_acc(c2_keys), c3_keys[0], best_acc(c3_keys),
                          "late-div > early-div (depth matters)",
                          "depth selection doesn't matter")

    if c2_keys and c1_keys:
        report_comparison(c2_keys[0], best_acc(c2_keys), c1_keys[0], best_acc(c1_keys),
                          "curated > random (quality > quantity)",
                          "random pairs sufficient")

    # Step-DPO divergence depth diagnostics
    sft_keys = [k for k in results if k.startswith("sft_base")]
    if sft_keys and c2_keys:
        report_comparison(c2_keys[0], best_acc(c2_keys), sft_keys[0], best_acc(sft_keys),
                          "DPO improves over SFT base",
                          "DPO does NOT improve over SFT base")

    # Full pairwise McNemar results for all condition pairs
    mcnemar_results = {}
    for i, a in enumerate(cond_names):
        for b in cond_names[i+1:]:
            mc = mcnemar_test(a, b)
            if mc is not None:
                mcnemar_results[f"{a}_vs_{b}"] = mc
                print(f"\n  McNemar {a} vs {b}: Δ={mc['delta']:+.3f}, "
                      f"disc={mc['n_discordant']}, p={mc['p_value']:.4f}")

    # Save comprehensive comparison (for HF sharing / plotting)
    comparison = {
        "conditions": {k: v for k, v in results.items()},
        "mcnemar_tests": mcnemar_results,
        "experiment": {
            "name": "e104-step-dpo-divergence-depth",
            "description": "Step-DPO with divergence-depth-guided pair selection",
            "beta": 0.2,
            "loss_type": "sigmoid",
            "eval_set": "MATH-500",
            "n_eval": 500,
            "statistical_test": "McNemar paired",
        },
    }
    with open(eval_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nComparison saved to {eval_dir / 'comparison.json'}")


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="E-104: Divergence-Aligned DPO")
    subparsers = parser.add_subparsers(dest="command")

    # Score (kill test)
    p_score = subparsers.add_parser("score",
                                     help="Score DPO pairs with SFT model (kill test)")
    p_score.add_argument("--data-dir", required=True,
                         help="Directory with cond*_train/val.jsonl files")
    p_score.add_argument("--sft-model-dir", required=True,
                         help="E-103 Model A checkpoint directory")
    p_score.add_argument("--output-dir", required=True,
                         help="Where to write uncertainty_scores.jsonl")

    # Prep
    p_prep = subparsers.add_parser("prep", help="Build DPO pairs")
    p_prep.add_argument("--rollouts-file", required=True)
    p_prep.add_argument("--corrections-file", required=True)
    p_prep.add_argument("--output-dir", required=True)
    p_prep.add_argument("--uncertainty-scores", default=None,
                        help="Path to uncertainty_scores.jsonl from 'score' subcommand. "
                             "If provided, Cond3 becomes Cond3u (uncertainty-sampled).")

    # Train
    p_train = subparsers.add_parser("train", help="Train DPO")
    p_train.add_argument("--data-dir", required=True)
    p_train.add_argument("--sft-model-dir", required=True,
                         help="E-103 Model A checkpoint directory")
    p_train.add_argument("--condition", type=int, required=True, choices=[1,2,3,4,5])
    p_train.add_argument("--beta", type=float, default=0.2)
    p_train.add_argument("--learning-rate", type=float, default=5e-6)
    p_train.add_argument("--epochs", type=int, default=3)
    p_train.add_argument("--output-dir", required=True)

    # Eval
    p_eval = subparsers.add_parser("eval", help="Evaluate model")
    p_eval.add_argument("--model-dir", required=True)
    p_eval.add_argument("--data-dir", required=True)
    p_eval.add_argument("--output-dir", required=True)

    # Compare
    p_compare = subparsers.add_parser("compare", help="Compare conditions")
    p_compare.add_argument("--eval-dir", required=True)

    args = parser.parse_args()

    if args.command == "score":
        cmd_score(args)
    elif args.command == "prep":
        cmd_prep(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "compare":
        cmd_compare(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
