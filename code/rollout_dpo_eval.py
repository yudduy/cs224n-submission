#!/usr/bin/env python3
"""DPO Model Evaluation — pass@1 (greedy) and pass@4 (T=0.7) on MATH-500.

Standalone eval script that takes a merged model directory and evaluates it
on the full MATH-500 set. Supports both greedy pass@1 and stochastic pass@4.

pass@4: generate 4 rollouts per problem at T=0.7, check if ANY is correct.
pass@1: greedy (T=0, n=1).

Usage:
  # Evaluate a single model (produces JSON results)
  python rollout_dpo_eval.py eval \
    --model-dir /path/to/merged_model \
    --data-file data/math500_all.jsonl \
    --output-dir /path/to/eval_output \
    --model-name model_name

  # Compare all models after all evals are done
  python rollout_dpo_eval.py compare \
    --eval-dir /path/to/eval_output
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from utils.math_eval import extract_boxed, extract_clean_answer, answers_match
from utils.formatting import format_chatml_prompt
from utils.io import load_jsonl, save_jsonl

SEED = 42
MAX_TOKENS = 2048
GPU_MEM_UTIL = 0.92  # A100 80GB can handle 0.92 safely with 8B model
MAX_MODEL_LEN = 8192
PASS_K = 4
TEMPERATURE_STOCHASTIC = 0.7
N_SAMPLES_PASS1 = 1
TEMPERATURE_GREEDY = 0.0


# ════════════════════════════════════════════════════════════════════
# Evaluation logic
# ════════════════════════════════════════════════════════════════════

def evaluate_model(model_dir: str,
                   eval_data: list[dict],
                   output_dir: Path,
                   model_name: str) -> dict:
    """Run pass@1 (greedy) and pass@4 (T=0.7) evaluation.

    Loads the model once, runs both sampling modes, saves results.
    Returns summary dict.
    """
    from vllm import LLM, SamplingParams

    model_dir = str(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"Evaluating: {model_name}")
    print(f"  Model dir: {model_dir}")
    print(f"  Problems: {len(eval_data)}")
    print(f"{'='*65}")

    # Load model (vLLM)
    t_load_start = time.time()
    llm = LLM(
        model_dir,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM_UTIL,
        enable_prefix_caching=True,
        seed=SEED,
    )
    t_load = time.time() - t_load_start
    print(f"  Model loaded in {t_load:.1f}s")

    # Format prompts (ChatML — matches SFT/DPO training format)
    prompts = [format_chatml_prompt(d['problem']) for d in eval_data]

    # ── pass@1 (greedy) ─────────────────────────────────────────────
    print(f"\n--- pass@1 (greedy, T=0) ---")
    t0 = time.time()
    sampling_greedy = SamplingParams(
        temperature=TEMPERATURE_GREEDY,
        top_p=1.0,
        max_tokens=MAX_TOKENS,
        n=N_SAMPLES_PASS1,
    )
    outputs_greedy = llm.generate(prompts, sampling_greedy)
    t_greedy = time.time() - t0

    pass1_results = []
    pass1_correct = 0
    for d, out in zip(eval_data, outputs_greedy):
        text = out.outputs[0].text
        ans = extract_clean_answer(text)
        hit = answers_match(ans, d["gt_answer"])
        pass1_correct += int(hit)
        pass1_results.append({
            "problem_idx": d["problem_idx"],
            "gt_answer": d["gt_answer"],
            "predicted": ans,
            "correct_pass1": hit,
            "generation": text,
            "gen_len": len(text),
        })

    pass1_acc = pass1_correct / len(eval_data) if eval_data else 0.0
    print(f"  pass@1: {pass1_acc:.4f} ({pass1_correct}/{len(eval_data)}) "
          f"[{t_greedy:.1f}s]")

    # ── pass@4 (stochastic, T=0.7) ───────────────────────────────────
    print(f"\n--- pass@{PASS_K} (stochastic, T={TEMPERATURE_STOCHASTIC}) ---")
    t0 = time.time()
    sampling_stochastic = SamplingParams(
        temperature=TEMPERATURE_STOCHASTIC,
        top_p=0.95,
        max_tokens=MAX_TOKENS,
        n=PASS_K,
    )
    outputs_stochastic = llm.generate(prompts, sampling_stochastic)
    t_stochastic = time.time() - t0

    pass4_results = []
    pass4_correct = 0
    for d, out in zip(eval_data, outputs_stochastic):
        sample_texts = [s.text for s in out.outputs]
        sample_hits = []
        for text in sample_texts:
            ans = extract_clean_answer(text)
            hit = answers_match(ans, d["gt_answer"])
            sample_hits.append({
                "text": text,
                "predicted": ans,
                "correct": hit,
                "gen_len": len(text),
            })
        any_correct = any(s["correct"] for s in sample_hits)
        mean_acc = sum(s["correct"] for s in sample_hits) / len(sample_hits)
        pass4_correct += int(any_correct)
        pass4_results.append({
            "problem_idx": d["problem_idx"],
            "gt_answer": d["gt_answer"],
            "pass_at_4": any_correct,
            "mean_accuracy_4": mean_acc,
            "samples": sample_hits,
        })

    pass4_acc = pass4_correct / len(eval_data) if eval_data else 0.0
    print(f"  pass@{PASS_K}: {pass4_acc:.4f} ({pass4_correct}/{len(eval_data)}) "
          f"[{t_stochastic:.1f}s]")

    # ── Generation length stats ──────────────────────────────────────
    greedy_lens = [r["gen_len"] for r in pass1_results]
    mean_len = sum(greedy_lens) / len(greedy_lens) if greedy_lens else 0.0
    median_len = sorted(greedy_lens)[len(greedy_lens) // 2] if greedy_lens else 0.0

    # ── Save results ────────────────────────────────────────────────
    import datetime
    summary = {
        "model_name": model_name,
        "model_dir": model_dir,
        "n_problems": len(eval_data),
        "pass_at_1": pass1_acc,
        "pass_at_4": pass4_acc,
        "n_correct_pass1": pass1_correct,
        "n_correct_pass4": pass4_correct,
        "mean_gen_len_greedy": mean_len,
        "median_gen_len_greedy": median_len,
        "eval_time_load_s": t_load,
        "eval_time_greedy_s": t_greedy,
        "eval_time_pass4_s": t_stochastic,
        "eval_time_total_s": t_load + t_greedy + t_stochastic,
        "temperature_stochastic": TEMPERATURE_STOCHASTIC,
        "n_samples_pass4": PASS_K,
        "timestamp_utc": datetime.datetime.utcnow().isoformat(),
    }

    with open(output_dir / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    save_jsonl(pass1_results, output_dir / "pass1_per_problem.jsonl")
    save_jsonl(pass4_results, output_dir / "pass4_per_problem.jsonl")

    print(f"\n  Results saved to {output_dir}/")
    print(json.dumps(summary, indent=2))
    return summary


# ════════════════════════════════════════════════════════════════════
# Comparison mode
# ════════════════════════════════════════════════════════════════════

def cmd_compare(eval_dir: Path) -> None:
    """Aggregate and compare all eval results in eval_dir."""
    print("=" * 65)
    print("DPO Model Evaluation — Comparison")
    print("=" * 65)

    results = {}
    per_problem_pass1: dict[str, dict[int, bool]] = {}
    per_problem_pass4: dict[str, dict[int, bool]] = {}

    for sub in sorted(eval_dir.iterdir()):
        if not sub.is_dir():
            continue
        summary_file = sub / "eval_summary.json"
        if not summary_file.exists():
            continue
        with open(summary_file) as f:
            r = json.load(f)
        results[sub.name] = r

        p1_file = sub / "pass1_per_problem.jsonl"
        if p1_file.exists():
            per_problem_pass1[sub.name] = {
                rec["problem_idx"]: rec["correct_pass1"]
                for rec in (json.loads(l) for l in open(p1_file))
            }
        p4_file = sub / "pass4_per_problem.jsonl"
        if p4_file.exists():
            per_problem_pass4[sub.name] = {
                rec["problem_idx"]: rec["pass_at_4"]
                for rec in (json.loads(l) for l in open(p4_file))
            }

    if not results:
        print("No eval results found!")
        return

    # Print table
    print(f"\n{'Model':<30} {'pass@1':>8} {'pass@4':>8} {'len_med':>8}")
    print("-" * 58)

    # Sort: base first, then condB/condC by size
    def sort_key(name: str) -> tuple:
        if "base" in name:
            return (0, "", 0)
        cond = "B" if "condB" in name else ("C" if "condC" in name else "?")
        size_str = name.replace("condB_", "").replace("condC_", "").replace("_merged", "")
        try:
            size = int(size_str)
        except ValueError:
            size = 999999
        return (1, cond, size)

    for name in sorted(results.keys(), key=sort_key):
        r = results[name]
        print(f"  {name:<28} {r['pass_at_1']:>8.4f} {r['pass_at_4']:>8.4f} "
              f"{r.get('median_gen_len_greedy', 0):>8.0f}")

    # Scaling curve data
    print("\n--- Scaling Curve ---")
    for cond in ["B", "C"]:
        cond_results = {
            k: v for k, v in results.items()
            if f"cond{cond}" in k
        }
        if not cond_results:
            continue
        print(f"\n  Condition {cond}:")

        def parse_size(name: str) -> int:
            s = name.replace(f"cond{cond}_", "").replace("_merged", "")
            try:
                return int(s)
            except ValueError:
                return 999999

        for name in sorted(cond_results.keys(), key=parse_size):
            r = cond_results[name]
            print(f"    {name}: pass@1={r['pass_at_1']:.4f}, "
                  f"pass@4={r['pass_at_4']:.4f}")

    # Head-to-head: Cond C (Bifurcation) vs Cond B (Full-Traj) at matched sizes
    print("\n--- Cond C (Bifurcation) vs Cond B (Full-Traj) (matched sizes) ---")
    c_names = {
        k.replace("condC_", "").replace("_merged", ""): k
        for k in results if "condC" in k
    }
    b_names = {
        k.replace("condB_", "").replace("_merged", ""): k
        for k in results if "condB" in k
    }
    shared_sizes = set(c_names.keys()) & set(b_names.keys())
    for size in sorted(shared_sizes):
        c_key = c_names[size]
        b_key = b_names[size]
        c_r = results[c_key]
        b_r = results[b_key]
        delta_p1 = c_r["pass_at_1"] - b_r["pass_at_1"]
        delta_p4 = c_r["pass_at_4"] - b_r["pass_at_4"]
        print(f"  size={size}: Δpass@1={delta_p1:+.4f}, Δpass@4={delta_p4:+.4f} "
              f"(C={c_r['pass_at_1']:.4f}, B={b_r['pass_at_1']:.4f})")

    # ── McNemar paired significance tests ──────────────────────────
    from scipy.stats import binomtest

    print("\n--- McNemar Paired Significance Tests (pass@1) ---")
    mcnemar_results = {}
    raw_pvals = []
    pval_keys = []

    # Test each condC vs condB pair, plus each model vs base
    test_pairs = []
    for size in sorted(shared_sizes):
        test_pairs.append((c_names[size], b_names[size], f"C_vs_B_{size}"))
    # Each model vs base (if base exists)
    if "base" in per_problem_pass1:
        for name in sorted(per_problem_pass1.keys()):
            if name != "base":
                test_pairs.append((name, "base", f"{name}_vs_base"))

    for name_a, name_b, label in test_pairs:
        if name_a not in per_problem_pass1 or name_b not in per_problem_pass1:
            continue
        pp_a = per_problem_pass1[name_a]
        pp_b = per_problem_pass1[name_b]
        shared_idxs = set(pp_a.keys()) & set(pp_b.keys())
        if not shared_idxs:
            continue
        # Discordant pairs: a_correct & b_wrong, a_wrong & b_correct
        n_ab = sum(1 for i in shared_idxs if pp_a[i] and not pp_b[i])  # a right, b wrong
        n_ba = sum(1 for i in shared_idxs if not pp_a[i] and pp_b[i])  # b right, a wrong
        n_disc = n_ab + n_ba
        if n_disc == 0:
            p_val = 1.0
        else:
            p_val = binomtest(n_ab, n_disc, 0.5).pvalue
        mcnemar_results[label] = {
            "model_a": name_a,
            "model_b": name_b,
            "n_shared": len(shared_idxs),
            "a_right_b_wrong": n_ab,
            "a_wrong_b_right": n_ba,
            "n_discordant": n_disc,
            "p_value": p_val,
        }
        raw_pvals.append(p_val)
        pval_keys.append(label)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"  {label}: disc=({n_ab},{n_ba}), p={p_val:.4f} {sig}")

    # Benjamini-Hochberg correction
    if raw_pvals:
        m = len(raw_pvals)
        sorted_indices = sorted(range(m), key=lambda i: raw_pvals[i])
        bh_adjusted = [0.0] * m
        for rank_idx, orig_idx in enumerate(sorted_indices):
            rank = rank_idx + 1
            bh_adjusted[orig_idx] = raw_pvals[orig_idx] * m / rank
        # Enforce monotonicity (step-up)
        for i in range(m - 2, -1, -1):
            idx = sorted_indices[i]
            idx_next = sorted_indices[i + 1]
            bh_adjusted[idx] = min(bh_adjusted[idx], bh_adjusted[idx_next])
        bh_adjusted = [min(p, 1.0) for p in bh_adjusted]

        print("\n  Benjamini-Hochberg adjusted p-values:")
        for i, key in enumerate(pval_keys):
            mcnemar_results[key]["p_value_bh"] = bh_adjusted[i]
            sig = "***" if bh_adjusted[i] < 0.001 else "**" if bh_adjusted[i] < 0.01 else "*" if bh_adjusted[i] < 0.05 else ""
            print(f"    {key}: p_bh={bh_adjusted[i]:.4f} {sig}")

    # Save comparison table
    comparison = {
        "models": results,
        "head_to_head": {},
        "mcnemar": mcnemar_results,
    }
    for size in shared_sizes:
        c_key = c_names[size]
        b_key = b_names[size]
        comparison["head_to_head"][size] = {
            "condC": results[c_key],
            "condB": results[b_key],
            "delta_pass1": results[c_key]["pass_at_1"] - results[b_key]["pass_at_1"],
            "delta_pass4": results[c_key]["pass_at_4"] - results[b_key]["pass_at_4"],
        }

    with open(eval_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nComparison saved to {eval_dir / 'comparison.json'}")


# ════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="DPO Model evaluation (pass@1 + pass@4)")
    subparsers = parser.add_subparsers(dest="cmd")

    # eval subcommand
    eval_parser = subparsers.add_parser("eval", help="Evaluate a single model")
    eval_parser.add_argument("--model-dir", required=True,
                             help="Path to merged model directory")
    eval_parser.add_argument("--data-file", required=True,
                             help="Path to math500_all.jsonl")
    eval_parser.add_argument("--output-dir", required=True,
                             help="Where to save results")
    eval_parser.add_argument("--model-name", default=None,
                             help="Human-readable name for this model")

    # compare subcommand
    cmp_parser = subparsers.add_parser("compare", help="Compare all eval results")
    cmp_parser.add_argument("--eval-dir", required=True,
                            help="Directory containing per-model eval subdirs")

    # Allow bare invocation as eval (backward compat with task.yaml calling pattern)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--eval-dir", default=None)
    parser.add_argument("--compare", action="store_true")

    args = parser.parse_args()

    # Dispatch
    if args.cmd == "compare" or args.compare:
        eval_dir = Path(args.eval_dir)
        cmd_compare(eval_dir)
        return

    if args.cmd == "eval" or args.model_dir:
        model_dir = Path(args.model_dir)
        data_file = Path(args.data_file)
        output_dir = Path(args.output_dir)
        model_name = args.model_name or model_dir.parent.name

        if not model_dir.exists():
            print(f"ERROR: model directory not found: {model_dir}")
            sys.exit(1)
        if not data_file.exists():
            print(f"ERROR: data file not found: {data_file}")
            sys.exit(1)

        eval_data = [json.loads(l) for l in open(data_file) if l.strip()]
        print(f"Loaded {len(eval_data)} eval problems from {data_file}")

        evaluate_model(str(model_dir), eval_data, output_dir, model_name)
        return

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
