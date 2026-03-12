#!/usr/bin/env python3
"""ExIt Pipeline Evaluation & Figure Generation
=================================================

Reads output from exit_pipeline.py and generates:
1. Summary statistics table
2. Yield rate comparison (DEEP-GRPO vs baselines)
3. P_phi calibration curve
4. Per-difficulty breakdown
5. Pivot depth distribution
6. Diversity analysis
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_results(output_dir: Path) -> dict:
    """Load all pipeline output files."""
    data = {}
    summary_path = output_dir / "eval_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            data["summary"] = json.load(f)

    rollouts_path = output_dir / "rollouts.jsonl"
    if rollouts_path.exists():
        data["rollouts"] = []
        with open(rollouts_path) as f:
            for line in f:
                data["rollouts"].append(json.loads(line))

    pivots_path = output_dir / "pivots.jsonl"
    if pivots_path.exists():
        data["pivots"] = []
        with open(pivots_path) as f:
            for line in f:
                data["pivots"].append(json.loads(line))

    corrections_path = output_dir / "corrections.jsonl"
    if corrections_path.exists():
        data["corrections"] = []
        with open(corrections_path) as f:
            for line in f:
                data["corrections"].append(json.loads(line))

    divergence_path = output_dir / "divergence_analysis.json"
    if divergence_path.exists():
        with open(divergence_path) as f:
            data["divergence"] = json.load(f)

    pilot_path = output_dir / "pilot_results.json"
    if pilot_path.exists():
        with open(pilot_path) as f:
            data["pilot"] = json.load(f)

    return data


def print_summary_table(data: dict):
    """Print formatted summary of pipeline results."""
    s = data.get("summary", {})
    print("\n" + "=" * 70)
    print("ExIt Pipeline — Results Summary")
    print("=" * 70)

    # Config
    cfg = s.get("config", {})
    print(f"\nModel: {cfg.get('model', '?')}")
    print(f"Dataset: {cfg.get('dataset', '?')}")
    print(f"Rollouts/problem: {cfg.get('n_rollouts', '?')}")
    print(f"Branches/pivot: {cfg.get('n_branches', '?')}")
    print(f"Thinking mode: {'ON' if cfg.get('enable_thinking') else 'OFF'}")
    print(f"Gamma (depth bias): {cfg.get('gamma', '?')}")

    # Step 1
    s1 = s.get("step1", {})
    print(f"\n── Step 1: Rollout Generation ──")
    print(f"  Problems: {s1.get('n_problems', '?')}")
    print(f"  Mean accuracy: {s1.get('mean_accuracy', 0):.1%}")

    # Step 2
    s2 = s.get("step2", {})
    print(f"\n── Step 2: Goldilocks Filtering ──")
    print(f"  Goldilocks problems: {s2.get('n_goldilocks', '?')}")
    print(f"  Goldilocks rate: {s2.get('goldilocks_rate', 0):.1%}")

    # Step 3
    s3 = s.get("step3", {})
    print(f"\n── Step 3: DEEP-GRPO Pivots ──")
    print(f"  Pivots identified: {s3.get('n_pivots', '?')}")
    print(f"  Mean pivot depth: {s3.get('mean_pivot_depth', 0):.3f}")
    phi = s3.get("p_phi_params", {})
    print(f"  P_phi: w={phi.get('w', 0):.3f}, b={phi.get('b', 0):.3f}, "
          f"bootstrapped={phi.get('is_bootstrapped', False)}")

    # Step 4
    s4 = s.get("step4", {})
    print(f"\n── Step 4: Divergence Analysis ──")
    print(f"  Pairs analyzed: {s4.get('n_pairs', '?')}")
    nd = s4.get("mean_normalized_distance")
    print(f"  Mean |pivot - diff| / len: {nd:.3f}" if nd is not None else "  N/A")

    # Step 6
    s6 = s.get("step6", {})
    print(f"\n── Step 6: Causal Validation ──")
    print(f"  Yield rate: {s6.get('yield_rate', 0):.1%}")
    print(f"  Correct / Total: {s6.get('n_correct', 0)} / {s6.get('n_total', 0)}")
    print(f"  Mean per-problem yield: {s6.get('mean_per_problem_yield', 0):.1%}")
    print(f"  Problems with any correct: {s6.get('problems_with_any_correct', 0)}")

    # Baselines
    bl = s.get("baselines", {})
    print(f"\n── Baseline Comparison ──")
    print(f"  {'Method':<25} {'Yield':>8} {'n_correct':>10} {'n_total':>8}")
    print(f"  {'-' * 51}")
    print(f"  {'DEEP-GRPO pivot':<25} {s6.get('yield_rate', 0):>8.1%} "
          f"{s6.get('n_correct', 0):>10} {s6.get('n_total', 0):>8}")
    for name, vals in bl.items():
        yr = vals.get("yield_rate", 0)
        nc = vals.get("n_correct", 0)
        nt = vals.get("n_total", 0)
        print(f"  {name:<25} {yr:>8.1%} {nc:>10} {nt:>8}")

    # Delta vs baselines
    deep_yield = s6.get("yield_rate", 0)
    rand_yield = bl.get("random_position", {}).get("yield_rate", 0)
    root_yield = bl.get("root_resample", {}).get("yield_rate", 0)
    if rand_yield > 0:
        print(f"\n  DEEP-GRPO vs random: {deep_yield / rand_yield:.2f}x")
    if root_yield > 0:
        print(f"  DEEP-GRPO vs root (BoN): {deep_yield / root_yield:.2f}x")

    # Diversity
    div = s.get("diversity", {})
    if div.get("mean_pairwise_diversity") is not None:
        print(f"\n── Diversity ──")
        print(f"  Mean pairwise diversity: {div['mean_pairwise_diversity']:.3f}")
        print(f"  Problems measured: {div['n_problems_measured']}")

    # Per-level
    pl = s.get("per_level", {})
    if pl:
        print(f"\n── Per-Difficulty Breakdown ──")
        print(f"  {'Level':<8} {'Goldilocks':>10} {'Yield':>8}")
        print(f"  {'-' * 26}")
        for lev in sorted(pl.keys()):
            v = pl[lev]
            nt = v.get("n_total_corr", 0)
            nc = v.get("n_correct_corr", 0)
            yr = nc / max(nt, 1)
            print(f"  {lev:<8} {v.get('n_gold', 0):>10} {yr:>8.1%} ({nc}/{nt})")

    # Kill criteria
    kc = s.get("kill_criteria", {})
    print(f"\n── Kill Criteria ──")
    for k, v in kc.items():
        status = "FAIL" if v else ("PASS" if v is not None else "N/A")
        print(f"  {k}: {status} ({v})")

    print(f"\n── Status: {s.get('status', '?')} ──")
    print(f"  Runtime: {s.get('runtime_s', 0):.0f}s ({s.get('runtime_s', 0) / 60:.1f} min)")


def analyze_pivots(data: dict):
    """Detailed pivot analysis."""
    pivots = data.get("pivots", [])
    corrections = data.get("corrections", [])
    if not pivots:
        print("\nNo pivot data to analyze.")
        return

    print("\n" + "=" * 70)
    print("Pivot Analysis")
    print("=" * 70)

    depths = [p["pivot_depth"] for p in pivots]
    q_scores = [p["q_score"] for p in pivots]
    print(f"\n  Depth distribution:")
    print(f"    Mean: {np.mean(depths):.3f}")
    print(f"    Std: {np.std(depths):.3f}")
    print(f"    Min: {np.min(depths):.3f}")
    print(f"    Max: {np.max(depths):.3f}")
    # Histogram (text-based)
    bins = np.linspace(0, 1, 11)
    counts, _ = np.histogram(depths, bins)
    print(f"\n  Depth histogram:")
    for i in range(len(counts)):
        bar = "█" * counts[i]
        print(f"    [{bins[i]:.1f}-{bins[i+1]:.1f}): {counts[i]:>3} {bar}")

    print(f"\n  Q-score distribution:")
    print(f"    Mean: {np.mean(q_scores):.4f}")
    print(f"    Std: {np.std(q_scores):.4f}")

    # P_phi calibration: bin by depth, compare predicted vs actual yield
    if corrections:
        print(f"\n  P_phi Calibration (predicted vs actual recoverability):")
        depth_bins = np.linspace(0, 1, 6)
        for i in range(len(depth_bins) - 1):
            lo, hi = depth_bins[i], depth_bins[i + 1]
            bin_corr = [c for c in corrections if lo <= c["pivot_depth"] < hi]
            if bin_corr:
                actual = np.mean([c["correction_correct"] for c in bin_corr])
                print(f"    depth [{lo:.1f}-{hi:.1f}): "
                      f"n={len(bin_corr)}, actual_yield={actual:.3f}")


def analyze_diversity(data: dict):
    """Diversity analysis of correct corrections."""
    corrections = data.get("corrections", [])
    if not corrections:
        print("\nNo correction data for diversity analysis.")
        return

    print("\n" + "=" * 70)
    print("Diversity Analysis")
    print("=" * 70)

    # Group by problem
    by_problem = defaultdict(list)
    for c in corrections:
        if c["correction_correct"]:
            by_problem[c["problem_idx"]].append(c["correction_text"])

    n_diverse = 0
    for pid, texts in by_problem.items():
        if len(texts) >= 2:
            # Check if corrections are actually different
            unique_texts = set(t.strip() for t in texts)
            if len(unique_texts) > 1:
                n_diverse += 1
            print(f"  Problem {pid}: {len(texts)} correct, "
                  f"{len(unique_texts)} unique")

    print(f"\n  Problems with diverse correct corrections: {n_diverse}")
    print(f"  Problems with any correct: {len(by_problem)}")


def main():
    parser = argparse.ArgumentParser(description="ExIt Pipeline Evaluation")
    parser.add_argument("--output-dir", default="/mnt/local/results",
                        help="Directory with pipeline output files")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Error: {output_dir} does not exist")
        sys.exit(1)

    data = load_results(output_dir)
    if not data:
        print(f"Error: No result files found in {output_dir}")
        sys.exit(1)

    if args.format == "json":
        print(json.dumps(data.get("summary", {}), indent=2))
    else:
        print_summary_table(data)
        analyze_pivots(data)
        analyze_diversity(data)


if __name__ == "__main__":
    main()
