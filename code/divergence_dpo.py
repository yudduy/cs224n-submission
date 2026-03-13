#!/usr/bin/env python3
"""Correction-Based DPO Data Preparation.

Converts correction data into Step-DPO format:
  prompt  = problem_text
  prefix  = incorrect_rollout[:prefix_len]  (shared correct reasoning up to error)
  chosen  = correction_text                  (correct continuation)
  rejected = incorrect_rollout[prefix_len:]  (wrong continuation)

3 conditions (same pair count, varying selection):
  Cond 1: Random subset (control)
  Cond 2: Late divergence (div_depth > 10%)
  Cond 3: Early divergence (div_depth ≤ 10%)

Usage:
  python divergence_dpo.py \
    --rollouts-file data/rollouts.jsonl \
    --corrections-file data/corrections_max.jsonl \
    --output-dir data/dpo_pairs
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np

from utils.io import save_jsonl

SEED = 42
MIN_SUFFIX_CHARS = 20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts-file", required=True)
    parser.add_argument("--corrections-file", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)

    # Load data
    rollouts = [json.loads(l) for l in open(args.rollouts_file)]
    corrections = [json.loads(l) for l in open(args.corrections_file)]
    print(f"Loaded {len(rollouts)} problems, {len(corrections)} corrections")

    # Index rollout texts
    rollout_texts = {}
    for prob in rollouts:
        for r in prob["rollouts"]:
            rollout_texts[(prob["problem_idx"], r["rollout_id"])] = r["text"]

    # Index problems
    problems = {d["problem_idx"]: d for d in rollouts}

    # Build Step-DPO pairs
    all_pairs = []
    stats = {"no_rollout": 0, "wrong_correction": 0, "short_suffix": 0}

    for c in corrections:
        if not c.get("correction_correct", False):
            stats["wrong_correction"] += 1
            continue

        key = (c["problem_idx"], c["rollout_id"])
        inc_text = rollout_texts.get(key)
        if inc_text is None:
            stats["no_rollout"] += 1
            continue

        prefix_len = c["prefix_len"]
        prefix = inc_text[:prefix_len]
        rejected_suffix = inc_text[prefix_len:]
        chosen_suffix = c["correction_text"]

        if len(rejected_suffix.strip()) < MIN_SUFFIX_CHARS or len(chosen_suffix.strip()) < MIN_SUFFIX_CHARS:
            stats["short_suffix"] += 1
            continue

        prob = problems.get(c["problem_idx"])
        if prob is None:
            continue

        pair = {
            "problem_idx": c["problem_idx"],
            "prompt": prob["problem"],
            "prefix": prefix,
            "chosen": chosen_suffix,
            "rejected": rejected_suffix,
            "metadata": {
                "rollout_id": c["rollout_id"],
                "div_depth": c["div_depth"],
                "prefix_len": prefix_len,
                "chosen_len": len(chosen_suffix),
                "rejected_len": len(rejected_suffix),
            }
        }
        all_pairs.append(pair)

    print(f"\nStep-DPO pairs: {len(all_pairs)}")
    print(f"  Skipped (wrong correction): {stats['wrong_correction']}")
    print(f"  Skipped (no rollout): {stats['no_rollout']}")
    print(f"  Skipped (short suffix): {stats['short_suffix']}")

    if not all_pairs:
        print("ERROR: No valid pairs!")
        return

    # Stats
    depths = [p["metadata"]["div_depth"] for p in all_pairs]
    chosen_lens = [p["metadata"]["chosen_len"] for p in all_pairs]
    rejected_lens = [p["metadata"]["rejected_len"] for p in all_pairs]
    prefix_lens = [p["metadata"]["prefix_len"] for p in all_pairs]

    print(f"\nDivergence depth: mean={np.mean(depths):.3f}, median={np.median(depths):.3f}")
    print(f"Prefix length (chars): mean={np.mean(prefix_lens):.0f}, median={np.median(prefix_lens):.0f}")
    print(f"Chosen suffix (chars): mean={np.mean(chosen_lens):.0f}, median={np.median(chosen_lens):.0f}")
    print(f"Rejected suffix (chars): mean={np.mean(rejected_lens):.0f}, median={np.median(rejected_lens):.0f}")

    # Split by divergence depth
    LATE_THRESHOLD = 0.10
    late = [p for p in all_pairs if p["metadata"]["div_depth"] > LATE_THRESHOLD]
    early = [p for p in all_pairs if p["metadata"]["div_depth"] <= LATE_THRESHOLD]

    print(f"\nLate divergence (>{LATE_THRESHOLD:.0%}): {len(late)}")
    print(f"Early divergence (≤{LATE_THRESHOLD:.0%}): {len(early)}")

    # Equalize
    target = min(len(late), len(early))
    if target == 0:
        print("WARNING: One group empty, using all from both")
        target = max(len(late), len(early))

    rng1, rng2, rng3 = random.Random(SEED+1), random.Random(SEED+2), random.Random(SEED+3)

    cond1 = [dict(p, condition=1) for p in rng1.sample(all_pairs, min(target, len(all_pairs)))]
    cond2 = [dict(p, condition=2) for p in rng2.sample(late, min(target, len(late)))]
    cond3 = [dict(p, condition=3) for p in rng3.sample(early, min(target, len(early)))]

    print(f"\n=== Equalized Step-DPO pairs ===")
    print(f"Cond 1 (Random):    {len(cond1)}")
    print(f"Cond 2 (Late div):  {len(cond2)}")
    print(f"Cond 3 (Early div): {len(cond3)}")

    # Save with problem-level train/val split
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for name, pairs in [("cond1", cond1), ("cond2", cond2), ("cond3", cond3)]:
        train, val = split_by_problem(pairs)
        save_jsonl(train, output_dir / f"{name}_train.jsonl")
        save_jsonl(val, output_dir / f"{name}_val.jsonl")
        summary[name] = {"total": len(pairs), "train": len(train), "val": len(val),
                         "n_problems": len(set(p["problem_idx"] for p in pairs))}
        print(f"  {name}: {len(train)} train + {len(val)} val")

    # Save eval problems (MATH-500 for pass@1)
    eval_probs = [{"problem_idx": d["problem_idx"], "problem": d["problem"],
                   "gt_answer": d["gt_answer"]} for d in rollouts]
    save_jsonl(eval_probs, output_dir / "math500_all.jsonl")
    summary["math500_total"] = len(eval_probs)

    # β scaling recommendation
    median_suffix = np.median(chosen_lens + rejected_lens)
    median_full = np.median([len(r["text"]) for d in rollouts for r in d["rollouts"]])
    beta_scale = median_full / median_suffix if median_suffix > 0 else 1.0
    print(f"\nβ scaling: median_full={median_full:.0f}ch, median_suffix={median_suffix:.0f}ch")
    print(f"  Recommended β_eff = β_base × {beta_scale:.1f}")
    print(f"  If β_base=0.1 → β_eff={0.1*beta_scale:.2f}")
    summary["beta_scale_factor"] = float(beta_scale)

    with open(output_dir / "prep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {output_dir}")


def split_by_problem(pairs, val_ratio=0.15, seed=SEED):
    rng = random.Random(seed)
    pids = sorted(set(p["problem_idx"] for p in pairs))
    rng.shuffle(pids)
    n_val = max(1, int(len(pids) * val_ratio))
    val_ids = set(pids[:n_val])
    return ([p for p in pairs if p["problem_idx"] not in val_ids],
            [p for p in pairs if p["problem_idx"] in val_ids])


if __name__ == "__main__":
    main()
