#!/usr/bin/env python3
"""Rollout-Based DPO — Data Preparation (CPU only, runs locally).

Builds TWO conditions of DPO pairs from the same rollout data,
plus a scaling curve at 4 dataset sizes.

Conditions:
  Full-Trajectory — correct vs incorrect full rollouts, no prefix sharing
      prompt   = ChatML(problem)  (no prefix)
      chosen   = a correct rollout (full text)
      rejected = the incorrect rollout (full text)

  Prefix-Sharing — error-localized, natural rollout text (no correction)
      prompt   = ChatML(problem) + prefix[:div_pos]
      chosen   = correct_rollout[div_pos:]  (natural continuation after fork)
      rejected = incorrect_rollout[div_pos:]  (wrong continuation after fork)

Key design constraints:
  - Same problems, same rejected rollouts → only TOPOLOGY varies
  - Both conditions use natural rollout text (no correction text) → no source confound
  - 85/15 problem-level train/val split, consistent across sizes
  - Scaling curve: 150, 300, ALL pairs

Source files:
  data/dpo_data/source_data/rollouts.jsonl
  data/dpo_data/source_data/corrections_max.jsonl

Output: data/dpo_data/rollout_dpo/
  condB_{size}_train.jsonl, condB_{size}_val.jsonl  (full-trajectory)
  condC_{size}_train.jsonl, condC_{size}_val.jsonl  (natural bifurcation)
  math500_all.jsonl
  prep_summary.json

Budget note: this script is CPU-only and runs in <2 minutes locally.

Usage:
  python EXPERIMENTS/rollout_dpo_data_prep.py \
    --rollouts-file data/dpo_data/source_data/rollouts.jsonl \
    --corrections-file data/dpo_data/source_data/corrections_max.jsonl \
    --output-dir data/dpo_data/rollout_dpo
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from utils.math_eval import extract_boxed, extract_clean_answer, answers_match
from utils.formatting import format_chatml_prompt, format_chatml_response
from utils.io import load_jsonl, save_jsonl

SEED = 42
SUFFIX_MIN_CHARS = 20
SCALE_SIZES = [150, 300]  # plus "all"
VAL_RATIO = 0.15


# ════════════════════════════════════════════════════════════════════
# Pair construction
# ════════════════════════════════════════════════════════════════════

def build_best_correction_index(corrections_data: list[dict]) -> dict[tuple, dict]:
    """Index best CORRECT correction per (problem_idx, rollout_id).

    Preference: correction_correct=True, then highest div_depth.
    Falls back to first available if no correct correction exists for a key
    (those get filtered out downstream).
    """
    index: dict[tuple, dict] = {}
    for c in corrections_data:
        key = (c["problem_idx"], c["rollout_id"])
        prev = index.get(key)
        if prev is None:
            index[key] = c
            continue
        # Prefer correct over incorrect
        if c.get("correction_correct", False) and not prev.get("correction_correct", False):
            index[key] = c
        elif c.get("correction_correct", False) == prev.get("correction_correct", False):
            # Both same correctness — prefer deeper divergence
            if c.get("div_depth", 0.0) > prev.get("div_depth", 0.0):
                index[key] = c
    return index


def build_rollout_index(rollouts_data: list[dict]) -> dict[int, dict]:
    """Index problem records by problem_idx."""
    return {d["problem_idx"]: d for d in rollouts_data}


def build_pairs(rollouts_data: list[dict],
                corrections_data: list[dict],
                rng: random.Random,
                ) -> list[dict]:
    """Build matched (Cond A, Cond B) pair tuples.

    Returns list of dicts, each with both condB and condC variants sharing:
      - same problem_idx
      - same rejected rollout (same incorrect_rollout_id)

    Only pairs where BOTH conditions are constructable are kept.
    """
    correction_index = build_best_correction_index(corrections_data)
    problem_index = build_rollout_index(rollouts_data)

    # Goldilocks: 1–7 correct out of N rollouts
    goldilocks = [d for d in rollouts_data if 1 <= d["n_correct"] < d["n_rollouts"]]
    print(f"  Goldilocks problems: {len(goldilocks)}")

    matched_pairs: list[dict] = []
    stats = {
        "no_correction": 0,
        "correction_wrong": 0,
        "suffix_too_short_c": 0,
        "suffix_too_short_b": 0,
        "no_correct_rollout": 0,
        "ok": 0,
    }

    for prob in goldilocks:
        problem_idx = prob["problem_idx"]
        problem_text = prob["problem"]
        gt_answer = prob["gt_answer"]
        subject = prob.get("subject", "")
        level = prob.get("level", 0)

        correct_rollouts = [r for r in prob["rollouts"] if r["is_correct"]]
        incorrect_rollouts = [r for r in prob["rollouts"] if not r["is_correct"]]

        if not correct_rollouts:
            stats["no_correct_rollout"] += len(incorrect_rollouts)
            continue

        for inc_r in incorrect_rollouts:
            inc_text = inc_r["text"]
            inc_id = inc_r["rollout_id"]

            # We need div_pos from correction corrections to know where the fork is
            corr_key = (problem_idx, inc_id)
            corr_info = correction_index.get(corr_key)
            if corr_info is None:
                stats["no_correction"] += 1
                continue
            if not corr_info.get("correction_correct", False):
                stats["correction_wrong"] += 1
                continue

            div_pos = corr_info["div_pos"]
            div_depth = corr_info["div_depth"]

            # Pick a correct rollout for this problem
            available_correct = list(correct_rollouts)
            rng.shuffle(available_correct)
            chosen_correct = available_correct[0]
            chosen_text = chosen_correct["text"]

            # ── Condition C (Natural Bifurcation DPO) ─────────────────
            # Same topology as old Cond A, but chosen = natural rollout suffix
            prefix = inc_text[:div_pos]
            wrong_suffix = inc_text[div_pos:]
            correct_suffix_natural = chosen_text[div_pos:]  # natural continuation after fork

            if len(wrong_suffix.strip()) < SUFFIX_MIN_CHARS or \
               len(correct_suffix_natural.strip()) < SUFFIX_MIN_CHARS:
                stats["suffix_too_short_c"] += 1
                continue

            # ── Condition B (Full-Trajectory DPO) ──────────────────────
            if len(chosen_text.strip()) < SUFFIX_MIN_CHARS or \
               len(inc_text.strip()) < SUFFIX_MIN_CHARS:
                stats["suffix_too_short_b"] += 1
                continue

            # ── Both conditions OK — build matched pair ─────────────────
            pair = {
                "problem_idx": problem_idx,
                "problem": problem_text,
                "gt_answer": gt_answer,
                "subject": subject,
                "level": level,
                "incorrect_rollout_id": inc_id,
                "correct_rollout_id": chosen_correct["rollout_id"],
                # Condition B: Full-Trajectory DPO (no prefix, natural rollouts)
                "condB": {
                    "prompt": problem_text,
                    "prefix": "",                # empty → full-trajectory format
                    "chosen": chosen_text,       # full correct rollout
                    "rejected": inc_text,        # full incorrect rollout
                    "metadata": {
                        "condition": "B",
                        "div_depth": 0.0,   # sentinel — required by source_data logging
                        "prefix_len": 0,
                        "chosen_len": len(chosen_text),
                        "rejected_len": len(inc_text),
                    },
                },
                # Condition C: Natural Bifurcation DPO (prefix + natural suffix)
                "condC": {
                    "prompt": problem_text,
                    "prefix": prefix,            # non-empty → Step-DPO format
                    "chosen": correct_suffix_natural,  # natural correct rollout suffix
                    "rejected": wrong_suffix,    # wrong continuation from fork
                    "metadata": {
                        "condition": "C",
                        "div_pos": div_pos,
                        "div_depth": div_depth,
                        "div_depth": div_depth,  # alias: required by source_data logging
                        "prefix_len": len(prefix),
                        "chosen_len": len(correct_suffix_natural),
                        "rejected_len": len(wrong_suffix),
                    },
                },
            }
            matched_pairs.append(pair)
            stats["ok"] += 1

    print(f"  Pair construction stats:")
    for k, v in stats.items():
        print(f"    {k}: {v}")
    print(f"  Total matched pairs (intersection): {len(matched_pairs)}")
    return matched_pairs


# ════════════════════════════════════════════════════════════════════
# Train/val split (problem-level, identical across sizes)
# ════════════════════════════════════════════════════════════════════

def problem_level_split(pairs: list[dict], val_ratio: float = VAL_RATIO,
                        seed: int = SEED) -> tuple[set[int], set[int]]:
    """Return (train_problem_ids, val_problem_ids) for a given pair list."""
    rng = random.Random(seed)
    problem_ids = sorted(set(p["problem_idx"] for p in pairs))
    rng.shuffle(problem_ids)
    n_val = max(1, int(len(problem_ids) * val_ratio))
    val_ids = set(problem_ids[:n_val])
    train_ids = set(problem_ids[n_val:])
    return train_ids, val_ids


def flatten_condition(pairs: list[dict], cond: str) -> list[dict]:
    """Flatten matched pair list to single-condition records."""
    out = []
    for p in pairs:
        rec = {
            "problem_idx": p["problem_idx"],
            "problem": p["problem"],
            "gt_answer": p["gt_answer"],
            "subject": p["subject"],
            "level": p["level"],
            "incorrect_rollout_id": p["incorrect_rollout_id"],
            "correct_rollout_id": p["correct_rollout_id"],
            **p[cond],  # prompt, prefix, chosen, rejected, metadata
        }
        out.append(rec)
    return out


# ════════════════════════════════════════════════════════════════════
# Stats printing
# ════════════════════════════════════════════════════════════════════

def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _median(vals: list[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    return (s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2)


def print_condition_stats(label: str, flat_pairs: list[dict]) -> None:
    if not flat_pairs:
        print(f"  {label}: 0 pairs")
        return
    prefix_lens = [len(p.get("prefix", "")) for p in flat_pairs]
    chosen_lens = [len(p.get("chosen", "")) for p in flat_pairs]
    rejected_lens = [len(p.get("rejected", "")) for p in flat_pairs]
    n_problems = len(set(p["problem_idx"] for p in flat_pairs))
    n_with_prefix = sum(1 for l in prefix_lens if l > 0)
    print(f"  {label}: {len(flat_pairs)} pairs, {n_problems} problems")
    print(f"    prefix_len:   mean={_mean(prefix_lens):.0f}, "
          f"median={_median(prefix_lens):.0f}, "
          f"n_with_prefix={n_with_prefix}")
    print(f"    chosen_len:   mean={_mean(chosen_lens):.0f}, "
          f"median={_median(chosen_lens):.0f}")
    print(f"    rejected_len: mean={_mean(rejected_lens):.0f}, "
          f"median={_median(rejected_lens):.0f}")


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Rollout DPO data preparation (CPU, local)")
    parser.add_argument("--rollouts-file", required=True,
                        help="Path to data/dpo_data/source_data/rollouts.jsonl")
    parser.add_argument("--corrections-file", required=True,
                        help="Path to data/dpo_data/source_data/corrections_max.jsonl")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory (data/dpo_data/rollout_dpo/)")
    args = parser.parse_args()

    print("=" * 65)
    print("Rollout DPO Data Preparation — Contrastive Bifurcation DPO")
    print("=" * 65)

    rollouts_path = Path(args.rollouts_file)
    corrections_path = Path(args.corrections_file)
    output_dir = Path(args.output_dir)

    if not rollouts_path.exists():
        print(f"ERROR: rollouts file not found: {rollouts_path}")
        sys.exit(1)
    if not corrections_path.exists():
        print(f"ERROR: corrections file not found: {corrections_path}")
        sys.exit(1)

    print(f"\nLoading rollouts from {rollouts_path}...")
    rollouts = load_jsonl(rollouts_path)
    print(f"  {len(rollouts)} problems")

    print(f"Loading corrections from {corrections_path}...")
    corrections = load_jsonl(corrections_path)
    print(f"  {len(corrections)} corrections")

    print("\n--- Building matched pairs (Cond B ∩ Cond C) ---")
    rng = random.Random(SEED)
    matched_pairs = build_pairs(rollouts, corrections, rng)

    if not matched_pairs:
        print("ERROR: No valid matched pairs found. Check source data paths.")
        sys.exit(1)

    # Flatten into per-condition lists
    all_condC = flatten_condition(matched_pairs, "condC")
    all_condB = flatten_condition(matched_pairs, "condB")
    assert len(all_condC) == len(all_condB) == len(matched_pairs)

    print(f"\n--- Overall stats (ALL size) ---")
    print_condition_stats("Cond C (Natural Bifurcation)", all_condC)
    print_condition_stats("Cond B (Full-Traj)", all_condB)

    # Compute problem-level train/val split on the full set (same split for ALL sizes)
    train_ids, val_ids = problem_level_split(matched_pairs)
    print(f"\nProblem split (85/15): {len(train_ids)} train, {len(val_ids)} val problems")

    # Separate train/val matched pairs
    train_matched = [p for p in matched_pairs if p["problem_idx"] in train_ids]
    val_matched = [p for p in matched_pairs if p["problem_idx"] in val_ids]

    # Determine scaling sizes
    n_all = len(matched_pairs)
    scale_configs = []
    for size in SCALE_SIZES:
        if size < n_all:
            scale_configs.append((str(size), size))
    scale_configs.append(("all", n_all))

    print(f"\nScaling curve sizes: {[s for s, _ in scale_configs]}")
    print(f"  (sampling from train split only; val is fixed)")

    summary: dict = {
        "n_matched_pairs_total": n_all,
        "n_train_problems": len(train_ids),
        "n_val_problems": len(val_ids),
        "n_train_pairs_full": len(train_matched),
        "n_val_pairs": len(val_matched),
        "scale_sizes": {},
    }

    # Save each (cond × size) combination
    output_dir.mkdir(parents=True, exist_ok=True)

    for size_label, n_target in scale_configs:
        # Sample n_target pairs from train_matched (problem-balanced subsample)
        # Use deterministic seed (no hash() — Python hash randomization makes it non-reproducible)
        size_salt = {"150": 1, "300": 2, "600": 3, "all": 4}.get(size_label, 99)
        rng_size = random.Random(SEED + size_salt)
        if n_target < len(train_matched):
            # Subsample at problem level first to avoid over-representing problems
            # with many incorrect rollouts. Then fill from per-problem pool.
            # Simple approach: random pair-level subsample (consistent with correction DPO)
            train_subset = rng_size.sample(train_matched, n_target)
        else:
            train_subset = list(train_matched)

        # Combine with val (val is always the same, regardless of size)
        # Build per-condition train/val
        train_C = flatten_condition(train_subset, "condC")
        train_B = flatten_condition(train_subset, "condB")
        val_C = flatten_condition(val_matched, "condC")
        val_B = flatten_condition(val_matched, "condB")

        # Save
        save_jsonl(train_C, output_dir / f"condC_{size_label}_train.jsonl")
        save_jsonl(val_C,   output_dir / f"condC_{size_label}_val.jsonl")
        save_jsonl(train_B, output_dir / f"condB_{size_label}_train.jsonl")
        save_jsonl(val_B,   output_dir / f"condB_{size_label}_val.jsonl")

        n_train_problems = len(set(p["problem_idx"] for p in train_subset))
        n_val_problems = len(set(p["problem_idx"] for p in val_matched))
        summary["scale_sizes"][size_label] = {
            "n_train": len(train_C),
            "n_val": len(val_C),
            "n_train_problems": n_train_problems,
            "n_val_problems": n_val_problems,
        }
        print(f"\n  Size={size_label}: train={len(train_C)} pairs "
              f"({n_train_problems} problems), val={len(val_C)} pairs "
              f"({n_val_problems} problems)")

    # Save math500_all.jsonl (full eval set — all 500 problems)
    math500 = [
        {
            "problem_idx": d["problem_idx"],
            "problem": d["problem"],
            "gt_answer": d["gt_answer"],
        }
        for d in rollouts
    ]
    save_jsonl(math500, output_dir / "math500_all.jsonl")
    summary["math500_total"] = len(math500)
    print(f"\nSaved math500_all.jsonl: {len(math500)} problems")

    # Verify matched pair integrity: Cond C rejected suffix should be a tail of Cond B rejected
    n_checked = 0
    n_mismatch = 0
    for p in matched_pairs:
        inc_c = p["condC"]["rejected"]
        inc_b = p["condB"]["rejected"]
        # Cond B rejected = full incorrect rollout; Cond C rejected = suffix from div_pos
        if inc_c not in inc_b:
            n_mismatch += 1
        n_checked += 1
    if n_mismatch > 0:
        print(f"\nWARNING: {n_mismatch}/{n_checked} pairs have Cond C suffix "
              f"not found in Cond B rejected text (possible text mismatch)")
    else:
        print(f"\n  Integrity check: all {n_checked} pairs have Cond C suffix "
              f"contained in Cond B rejected text")

    # Additional stats: overlap
    print("\n--- Cond C vs Cond B comparison ---")
    c_prefix_lens = [len(p["condC"]["prefix"]) for p in matched_pairs]
    c_chosen_lens = [len(p["condC"]["chosen"]) for p in matched_pairs]
    b_chosen_lens = [len(p["condB"]["chosen"]) for p in matched_pairs]
    c_rej_lens = [len(p["condC"]["rejected"]) for p in matched_pairs]
    b_rej_lens = [len(p["condB"]["rejected"]) for p in matched_pairs]
    print(f"  Cond C prefix length:      mean={_mean(c_prefix_lens):.0f}, "
          f"median={_median(c_prefix_lens):.0f} chars")
    print(f"  Cond C chosen (nat suffix): mean={_mean(c_chosen_lens):.0f}, "
          f"median={_median(c_chosen_lens):.0f} chars")
    print(f"  Cond B chosen (full traj):  mean={_mean(b_chosen_lens):.0f}, "
          f"median={_median(b_chosen_lens):.0f} chars")
    print(f"  Cond C rejected (suffix):   mean={_mean(c_rej_lens):.0f}, "
          f"median={_median(c_rej_lens):.0f} chars")
    print(f"  Cond B rejected (full):     mean={_mean(b_rej_lens):.0f}, "
          f"median={_median(b_rej_lens):.0f} chars")

    # Distribution of div_depth for Cond C pairs
    div_depths = [p["condC"]["metadata"]["div_depth"] for p in matched_pairs]
    q25 = sorted(div_depths)[int(len(div_depths) * 0.25)]
    q75 = sorted(div_depths)[int(len(div_depths) * 0.75)]
    print(f"\n  Cond C div_depth: mean={_mean(div_depths):.3f}, "
          f"median={_median(div_depths):.3f}, "
          f"25th={q25:.3f}, 75th={q75:.3f}")

    # Save summary
    with open(output_dir / "prep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {output_dir / 'prep_summary.json'}")
    print(json.dumps(summary, indent=2))

    print("\n" + "=" * 65)
    print("Rollout DPO data preparation complete.")
    print(f"  Output: {output_dir}/")
    print(f"  Conditions: B (Full-Trajectory), C (Natural Bifurcation)")
    print(f"  Sizes: {[s for s, _ in scale_configs]}")
    print("=" * 65)


if __name__ == "__main__":
    main()
