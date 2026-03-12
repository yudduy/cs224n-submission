#!/usr/bin/env python3
"""ExIt Pipeline: DEEP-GRPO Pivot Identification + Correction Generation
=========================================================================

Steps 1-6 of the 9-step ExIt pipeline for CS224N project.

Step 0: Pilot (20 problems, thinking on vs off) — pick winner
Step 1: Generate N=8 rollouts per problem (vLLM, Qwen3-8B)
Step 2: Filter goldilocks: 1 ≤ correct ≤ 7
Step 3: DEEP-GRPO pivot identification on failed traces
Step 4: Token-level diff: pivots vs actual divergence points
Step 5: Correction generation: prefix[:t*] + "Wait..." + resample K=8
Step 6: Causal validation: keep only correct resamples

Infra: Mithril A100 80GB via SkyPilot (`ml launch task.yaml`)
Model: Qwen/Qwen3-8B on MATH-500
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


# Answer Extraction & Verification

def extract_boxed(text: str) -> str | None:
    """Extract content from \\boxed{...} with balanced brace matching."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    depth = 0
    start = idx + len("\\boxed{")
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            if depth == 0:
                return text[start:i]
            depth -= 1
    return text[start:]  # unclosed — take what we have


def extract_answer(text: str) -> str | None:
    """Extract final answer with cascading fallbacks."""
    # 1. \boxed{}
    ans = extract_boxed(text)
    if ans:
        return ans.strip()
    # 2. "#### answer"
    m = re.search(r"####\s*([^\n]+)", text)
    if m:
        return m.group(1).strip()
    # 3. "answer is X" / "answer: X"
    m = re.search(r"(?:answer is|answer:)\s*([^\n.]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # 4. Last number
    nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if nums:
        return nums[-1].replace(",", "")
    return None


def normalize_latex(s: str) -> str:
    """Strip common LaTeX wrappers for comparison."""
    s = s.strip()
    # Remove $...$
    s = re.sub(r"^\$(.+)\$$", r"\1", s)
    # Remove \text{}, \mathrm{}, \textbf{}
    s = re.sub(r"\\(?:text|mathrm|textbf|mathbf)\{([^}]*)\}", r"\1", s)
    # Remove \left, \right
    s = s.replace("\\left", "").replace("\\right", "")
    # \frac{a}{b} → a/b
    s = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", s)
    # \sqrt{x} → sqrt(x)
    s = re.sub(r"\\sqrt\{([^}]*)\}", r"sqrt(\1)", s)
    # Remove \, \; \: \! (spacing)
    s = re.sub(r"\\[,;:!]", "", s)
    # Remove remaining backslashes before known functions
    s = re.sub(r"\\(pi|infty|cdot|times|div|pm|mp)", r"\1", s)
    s = s.replace(",", "").replace(" ", "")
    return s


def is_equiv(pred: str | None, gt: str | None) -> bool:
    """Check mathematical equivalence. Handles numeric, fraction, LaTeX forms."""
    if pred is None or gt is None:
        return False
    p, g = pred.strip(), gt.strip()
    if p == g:
        return True
    pn, gn = normalize_latex(p), normalize_latex(g)
    if pn == gn:
        return True
    def to_float(s):
        s = s.replace(",", "").strip()
        m = re.match(r"^\(?(-?\d+(?:\.\d+)?)\)?\s*/\s*\(?(-?\d+(?:\.\d+)?)\)?$", s)
        if m:
            num, den = float(m.group(1)), float(m.group(2))
            return num / den if den != 0 else None
        try:
            return float(s)
        except (ValueError, OverflowError):
            return None

    pf, gf = to_float(pn), to_float(gn)
    if pf is not None and gf is not None:
        if abs(pf - gf) < 1e-6:
            return True
        if gf != 0 and abs((pf - gf) / gf) < 1e-4:
            return True
    try:
        from sympy import simplify, sympify
        from sympy.parsing.latex import parse_latex
        try:
            pe = parse_latex(p)
            ge = parse_latex(g)
        except Exception:
            pe = sympify(pn)
            ge = sympify(gn)
        if simplify(pe - ge) == 0:
            return True
    except Exception:
        pass
    return False


# Text Segmentation (semantic, not fixed-token)

def segment_text(text: str) -> list[tuple[int, int]]:
    """Segment by \\n\\n boundaries, falling back to sentence punctuation."""
    segments = []
    parts = re.split(r"(\n\n)", text)
    if len(parts) <= 1:
        parts = re.split(r"(\.\s)", text)
    pos = 0
    current_start = 0
    for part in parts:
        pos += len(part)
        if part in ("\n\n", ". "):
            segments.append((current_start, pos))
            current_start = pos
    if current_start < len(text):
        segments.append((current_start, len(text)))
    if not segments:
        segments = [(0, len(text))]
    return segments


# DEEP-GRPO Pivot Identification

class DeepGRPOPivotSelector:
    """DEEP-GRPO pivot selection per arXiv:2602.14169.

    Q(t) = P_phi(success | r_t) * (t/T)^gamma
    P_phi = sigmoid(w * r_t + b) — logistic regression on normalized depth
    """

    def __init__(self, gamma: float = 2.0, refit_interval: int = 10):
        self.gamma = gamma
        self.refit_interval = refit_interval
        # Logistic regression params: P_phi(success|depth) = sigmoid(w*depth + b)
        self.w = 0.0  # uniform prior before fitting
        self.b = 0.0
        self.experience: list[tuple[float, int]] = []  # (depth, success)
        self.n_since_refit = 0
        self.is_bootstrapped = False

    def q_score(self, t: int, T: int) -> float:
        """Q(t) = P_phi(success|depth) * (depth)^gamma, using midpoint (t+0.5)/T."""
        r_t = (t + 0.5) / max(T, 1)
        p_phi = 1.0 / (1.0 + math.exp(-(self.w * r_t + self.b)))
        depth_bias = r_t ** self.gamma
        return p_phi * depth_bias

    def select_pivot(self, T: int) -> int:
        if not self.is_bootstrapped:
            return random.randint(0, max(T - 1, 0))
        scores = [self.q_score(t, T) for t in range(T)]
        total = sum(scores)
        if total < 1e-12:
            return random.randint(0, max(T - 1, 0))
        r = random.random() * total
        cumsum = 0.0
        for t, s in enumerate(scores):
            cumsum += s
            if cumsum >= r:
                return t
        return T - 1

    def add_experience(self, depth: float, success: int, defer_refit: bool = False):
        self.experience.append((depth, success))
        self.n_since_refit += 1
        if not defer_refit and self.n_since_refit >= self.refit_interval and len(self.experience) >= 5:
            self._refit()

    def _refit(self):
        from sklearn.linear_model import LogisticRegression

        X = np.array([e[0] for e in self.experience]).reshape(-1, 1)
        y = np.array([e[1] for e in self.experience])
        if len(np.unique(y)) < 2:
            self.n_since_refit = 0
            return
        clf = LogisticRegression(max_iter=200, C=1.0)
        clf.fit(X, y)
        self.w = float(clf.coef_[0, 0])
        self.b = float(clf.intercept_[0])
        self.is_bootstrapped = True
        self.n_since_refit = 0

    def force_refit(self):
        if len(self.experience) >= 5:
            self._refit()

    def get_params(self) -> dict:
        return {
            "w": self.w, "b": self.b,
            "is_bootstrapped": self.is_bootstrapped,
            "n_experience": len(self.experience),
        }


# Step 0: Thinking Mode Pilot

def run_pilot(llm, tokenizer, problems: list[dict], n_rollouts: int = 8) -> dict:
    """Compare thinking-on vs thinking-off on 20 problems."""
    from vllm import SamplingParams

    pilot_problems = random.sample(problems, min(20, len(problems)))
    results = {"thinking_off": [], "thinking_on": []}

    for mode_name, enable_thinking in [("thinking_off", False), ("thinking_on", True)]:
        if enable_thinking:
            sp = SamplingParams(
                n=n_rollouts, temperature=0.6, top_p=0.95, top_k=20,
                max_tokens=2048,
            )
        else:
            sp = SamplingParams(
                n=n_rollouts, temperature=0.7, top_p=0.8, top_k=20,
                max_tokens=2048,
            )

        prompts = []
        for p in pilot_problems:
            if enable_thinking:
                msgs = [{"role": "user", "content": p["problem"]}]
            else:
                msgs = [{"role": "user", "content": p["problem"]}]
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            prompts.append(text)

        outputs = llm.generate(prompts, sp)

        for p, out in zip(pilot_problems, outputs):
            gt = p["answer"]
            n_correct = 0
            for o in out.outputs:
                text = o.text
                # Strip thinking tags if present
                if "<think>" in text:
                    think_end = text.rfind("</think>")
                    if think_end != -1:
                        text = text[think_end + len("</think>"):]
                ans = extract_answer(text)
                if is_equiv(ans, gt):
                    n_correct += 1
            results[mode_name].append({
                "problem": p["problem"][:100],
                "gt": gt,
                "n_correct": n_correct,
                "accuracy": n_correct / n_rollouts,
                "goldilocks": 1 <= n_correct <= (n_rollouts - 1),
            })

    # Summarize
    for mode in ["thinking_off", "thinking_on"]:
        accs = [r["accuracy"] for r in results[mode]]
        golds = [r["goldilocks"] for r in results[mode]]
        results[f"{mode}_summary"] = {
            "mean_accuracy": float(np.mean(accs)),
            "goldilocks_rate": float(np.mean(golds)),
            "n_problems": len(accs),
        }
        print(f"  {mode}: accuracy={np.mean(accs):.3f}, "
              f"goldilocks={np.mean(golds):.1%}")

    # Pick winner: want ~40-60% accuracy and high goldilocks rate
    off_acc = results["thinking_off_summary"]["mean_accuracy"]
    on_acc = results["thinking_on_summary"]["mean_accuracy"]
    off_gold = results["thinking_off_summary"]["goldilocks_rate"]
    on_gold = results["thinking_on_summary"]["goldilocks_rate"]

    # Score: goldilocks rate * (1 - |accuracy - 0.5|)
    off_score = off_gold * (1 - abs(off_acc - 0.5))
    on_score = on_gold * (1 - abs(on_acc - 0.5))

    winner = "thinking_off" if off_score >= on_score else "thinking_on"
    results["winner"] = winner
    results["scores"] = {"thinking_off": off_score, "thinking_on": on_score}
    print(f"  Winner: {winner} (off={off_score:.3f}, on={on_score:.3f})")
    return results


# Steps 1-6: Main Pipeline

def run_pipeline(args):
    from collections import Counter, defaultdict

    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    print(f"[pipeline] model={args.model} dataset={args.dataset} "
          f"n={args.n_rollouts} K={args.n_branches} gamma={args.gamma} seed={args.seed}")

    print("\n[Step 0] Loading dataset...")
    ds = load_dataset(args.dataset, split="test")
    problems = []
    for i, row in enumerate(ds):
        problems.append({
            "idx": i,
            "problem": row["problem"],
            "solution": row.get("solution", ""),
            "answer": row.get("answer", ""),
            "subject": row.get("subject", ""),
            "level": row.get("level", ""),
        })
    if args.smoke_test:
        problems = problems[:5]
    print(f"  Loaded {len(problems)} problems")

    print("\n[Step 0] Loading model...")
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        max_model_len=8192,
        enable_prefix_caching=True,
    )
    tokenizer = llm.get_tokenizer()

    if not args.skip_pilot:
        print("\n[Step 0] Thinking mode pilot (20 problems)")
        pilot_results = run_pilot(llm, tokenizer, problems, args.n_rollouts)
        with open(output_dir / "pilot_results.json", "w") as f:
            json.dump(pilot_results, f, indent=2)
        enable_thinking = pilot_results["winner"] == "thinking_on"
    else:
        print("\n[Step 0] Skipping pilot, using thinking_off")
        enable_thinking = args.enable_thinking
        pilot_results = {"winner": "thinking_on" if enable_thinking else "thinking_off", "skipped": True}

    print(f"  Mode: {'thinking_on' if enable_thinking else 'thinking_off'}")

    if enable_thinking:
        gen_sp = SamplingParams(
            n=args.n_rollouts, temperature=0.6, top_p=0.95, top_k=20,
            max_tokens=args.max_tokens,
        )
        branch_sp = SamplingParams(
            n=1, temperature=0.6, top_p=0.95, top_k=20,
            max_tokens=args.max_tokens,
        )
    else:
        gen_sp = SamplingParams(
            n=args.n_rollouts, temperature=0.7, top_p=0.8, top_k=20,
            max_tokens=args.max_tokens,
        )
        branch_sp = SamplingParams(
            n=1, temperature=0.7, top_p=0.8, top_k=20,
            max_tokens=args.max_tokens,
        )

    # Step 1: Rollout Generation
    print("[Step 1] Generating rollouts")

    rollout_path = output_dir / "rollouts.jsonl"
    if rollout_path.exists() and not args.force:
        print(f"  Loading existing rollouts from {rollout_path}")
        rollout_data = []
        with open(rollout_path) as f:
            for line in f:
                rollout_data.append(json.loads(line))
    else:
        # Build prompts
        prompts = []
        for p in problems:
            msgs = [{"role": "user", "content": p["problem"]}]
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            prompts.append(text)

        # Generate in batches (vLLM handles batching internally)
        t1 = time.time()
        outputs = llm.generate(prompts, gen_sp)
        t_gen = time.time() - t1
        print(f"  Generated {len(outputs)} × {args.n_rollouts} rollouts in {t_gen:.1f}s")

        rollout_data = []
        for p, out in zip(problems, outputs):
            gt = p["answer"]
            rollouts = []
            for j, o in enumerate(out.outputs):
                text = o.text
                # Strip thinking content for answer extraction
                clean_text = text
                if "<think>" in text:
                    think_end = text.rfind("</think>")
                    if think_end != -1:
                        clean_text = text[think_end + len("</think>"):]
                ans = extract_answer(clean_text)
                correct = is_equiv(ans, gt)
                rollouts.append({
                    "rollout_id": j,
                    "text": text,
                    "extracted_answer": ans,
                    "is_correct": correct,
                })
            n_correct = sum(r["is_correct"] for r in rollouts)
            record = {
                "problem_idx": p["idx"],
                "problem": p["problem"],
                "gt_answer": gt,
                "subject": p["subject"],
                "level": str(p["level"]),
                "n_correct": n_correct,
                "n_rollouts": len(rollouts),
                "rollouts": rollouts,
            }
            rollout_data.append(record)

        # Save
        with open(rollout_path, "w") as f:
            for r in rollout_data:
                f.write(json.dumps(r) + "\n")

    # Stats
    accs = [r["n_correct"] / r["n_rollouts"] for r in rollout_data]
    print(f"  Mean accuracy: {np.mean(accs):.3f} ({np.std(accs):.3f})")
    print(f"  Problems with 0 correct: {sum(1 for a in accs if a == 0)}")
    print(f"  Problems with all correct: {sum(1 for a in accs if a == 1.0)}")

    # Per-level accuracy (for Table 1)
    level_acc = defaultdict(lambda: {"correct": 0, "total": 0})
    subject_acc = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in rollout_data:
        lev = str(r.get("level", "?"))
        subj = r.get("subject", "?")
        level_acc[lev]["correct"] += r["n_correct"]
        level_acc[lev]["total"] += r["n_rollouts"]
        subject_acc[subj]["correct"] += r["n_correct"]
        subject_acc[subj]["total"] += r["n_rollouts"]
    print(f"  Per-level accuracy:")
    for lev in sorted(level_acc):
        la = level_acc[lev]
        print(f"    Level {lev}: {la['correct']}/{la['total']} = {la['correct']/max(la['total'],1):.3f}")

    # Step 2: Goldilocks Filtering
    print("[Step 2] Goldilocks filtering")

    goldilocks = [r for r in rollout_data
                  if 1 <= r["n_correct"] <= (r["n_rollouts"] - 1)]
    print(f"  Goldilocks: {len(goldilocks)}/{len(rollout_data)} "
          f"({100 * len(goldilocks) / len(rollout_data):.1f}%)")
    dist = Counter(r["n_correct"] for r in goldilocks)
    print(f"  n_correct distribution: {dict(sorted(dist.items()))}")

    def strip_thinking_from_prefix(text: str) -> str:
        """Remove <think>...</think> blocks from prefix text."""
        if not enable_thinking:
            return text
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        if "<think>" in cleaned:
            cleaned = cleaned[:cleaned.rfind("<think>")]
        return cleaned.strip()

    def build_correction_prompt(problem_text: str, prefix: str) -> str:
        """Build correction prompt: ChatML prefix + partial response + transition."""
        msgs = [{"role": "user", "content": problem_text}]
        chat_prefix = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        clean_prefix = strip_thinking_from_prefix(prefix)
        transition = "\n\nWait, let me reconsider this step.\n\n"
        return chat_prefix + clean_prefix + transition

    def extract_clean_answer(text: str) -> str | None:
        clean = text
        if "<think>" in text:
            think_end = text.rfind("</think>")
            if think_end != -1:
                clean = text[think_end + len("</think>"):]
        return extract_answer(clean)

    branch_gen_sp = SamplingParams(
        n=args.n_branches,  # generate K branches per prompt in one call
        temperature=gen_sp.temperature,
        top_p=gen_sp.top_p,
        top_k=gen_sp.top_k,
        max_tokens=args.max_tokens,
    )
    single_sp = SamplingParams(
        n=1,
        temperature=gen_sp.temperature,
        top_p=gen_sp.top_p,
        top_k=gen_sp.top_k,
        max_tokens=args.max_tokens,
    )

    print("\n[Steps 3-6] Pivot identification + correction generation")

    pivot_selector = DeepGRPOPivotSelector(gamma=args.gamma)
    pivot_data = []
    corrections = []
    n_correct_corrections = 0
    bootstrap_batch_size = 10
    batch_size = 10  # process in batches of 10 goldilocks problems

    # Build problem index for fast lookup
    goldilocks_map = {g["problem_idx"]: g for g in goldilocks}

    for batch_start in range(0, len(goldilocks), batch_size):
        batch = goldilocks[batch_start:batch_start + batch_size]
        batch_idx = batch_start // batch_size
        is_bootstrap = batch_start < bootstrap_batch_size
        print(f"\n  --- Batch {batch_idx} ({len(batch)} problems, "
              f"{'BOOTSTRAP' if is_bootstrap else 'ONLINE'}) ---")

        # Step 3: Select pivots for this batch
        batch_pivots = []
        for g in batch:
            incorrect_rollouts = [r for r in g["rollouts"] if not r["is_correct"]]
            for r in incorrect_rollouts:
                text = r["text"]
                segments = segment_text(text)
                T = len(segments)
                if T < 2:
                    continue
                pivot_idx = pivot_selector.select_pivot(T)
                pivot_char_end = segments[pivot_idx][1]
                pv = {
                    "problem_idx": g["problem_idx"],
                    "rollout_id": r["rollout_id"],
                    "text": text,
                    "gt_answer": g["gt_answer"],
                    "subject": g.get("subject", "?"),
                    "level": str(g.get("level", "?")),
                    "n_segments": T,
                    "pivot_segment": pivot_idx,
                    "pivot_char_pos": pivot_char_end,
                    "pivot_depth": (pivot_idx + 0.5) / max(T, 1),
                    "q_score": pivot_selector.q_score(pivot_idx, T),
                    "segments": [(s, e) for s, e in segments],
                }
                batch_pivots.append(pv)
        pivot_data.extend(batch_pivots)

        # Step 5: Generate corrections for this batch
        # Use n=n_branches per unique prompt (GPU batch optimization)
        batch_prompts = []
        batch_pivot_refs = []  # one per unique prompt
        for pv in batch_pivots:
            prefix = pv["text"][:pv["pivot_char_pos"]]
            if not prefix.strip():
                continue
            orig = goldilocks_map.get(pv["problem_idx"])
            if orig is None:
                continue
            full_prompt = build_correction_prompt(orig["problem"], prefix)
            batch_prompts.append(full_prompt)
            batch_pivot_refs.append({
                "problem_idx": pv["problem_idx"],
                "rollout_id": pv["rollout_id"],
                "pivot_segment": pv["pivot_segment"],
                "pivot_depth": pv["pivot_depth"],
                "q_score": pv["q_score"],
                "gt_answer": pv["gt_answer"],
                "subject": pv.get("subject", "?"),
                "level": pv.get("level", "?"),
                "prefix_len": len(prefix),
            })

        if not batch_prompts:
            print(f"    No valid prompts in batch")
            continue

        # Generate n_branches completions per prompt in one call
        batch_outputs = llm.generate(batch_prompts, branch_gen_sp)

        # Step 6: Validate and update P_phi
        batch_correct = 0
        batch_total = 0
        for meta, out in zip(batch_pivot_refs, batch_outputs):
            for branch_out in out.outputs:
                text = branch_out.text
                ans = extract_clean_answer(text)
                correct = is_equiv(ans, meta["gt_answer"])
                n_tokens = len(branch_out.token_ids) if hasattr(branch_out, 'token_ids') else len(text.split())
                if correct:
                    n_correct_corrections += 1
                    batch_correct += 1
                batch_total += 1
                corrections.append({
                    **meta,
                    "correction_text": text,
                    "correction_answer": ans,
                    "correction_correct": correct,
                    "n_tokens": n_tokens,
                })
            # Online P_phi update: one experience per pivot (any branch correct?)
            any_correct = any(
                is_equiv(extract_clean_answer(o.text), meta["gt_answer"])
                for o in out.outputs
            )
            pivot_selector.add_experience(meta["pivot_depth"], int(any_correct),
                                          defer_refit=is_bootstrap)

        batch_yield = batch_correct / max(batch_total, 1)
        print(f"    Pivots: {len(batch_pivots)}, Corrections: {batch_total}, "
              f"Correct: {batch_correct} ({batch_yield:.1%})")

        # After bootstrap batch, force P_phi refit
        if is_bootstrap:
            pivot_selector.force_refit()
            print(f"    P_phi refitted: {pivot_selector.get_params()}")

        # Checkpoint after each batch
        ckpt = {
            "batch_idx": batch_idx,
            "n_pivots": len(pivot_data),
            "n_corrections": len(corrections),
            "n_correct": n_correct_corrections,
            "p_phi_params": pivot_selector.get_params(),
        }
        with open(output_dir / "checkpoint.json", "w") as f:
            json.dump(ckpt, f, indent=2)
        # Incremental save of corrections
        with open(output_dir / "corrections.jsonl", "w") as f:
            for c in corrections:
                f.write(json.dumps(c) + "\n")

    # Final P_phi refit
    pivot_selector.force_refit()

    # Save P_phi experience buffer for calibration curve (Fig 3)
    p_phi_experience = [{"depth": d, "success": int(s)}
                        for d, s in pivot_selector.experience]
    with open(output_dir / "p_phi_experience.json", "w") as f:
        json.dump(p_phi_experience, f, indent=2)

    yield_rate = n_correct_corrections / max(len(corrections), 1)
    print(f"\n  === Steps 3-6 Complete ===")
    print(f"  Total pivots: {len(pivot_data)}")
    print(f"  Mean pivot depth: {np.mean([p['pivot_depth'] for p in pivot_data]):.3f}" if pivot_data else "  No pivots")
    print(f"  Total corrections: {len(corrections)}")
    print(f"  Correct corrections: {n_correct_corrections}")
    print(f"  Yield rate: {yield_rate:.3f}")
    print(f"  Final P_phi: {pivot_selector.get_params()}")

    # Per-problem yield
    problem_yields = defaultdict(lambda: {"correct": 0, "total": 0})
    for c in corrections:
        pid = c["problem_idx"]
        problem_yields[pid]["total"] += 1
        if c["correction_correct"]:
            problem_yields[pid]["correct"] += 1
    yields_per_problem = [v["correct"] / max(v["total"], 1)
                          for v in problem_yields.values()]
    if yields_per_problem:
        print(f"  Mean per-problem yield: {np.mean(yields_per_problem):.3f}")
        print(f"  Problems with any correct: "
              f"{sum(1 for y in yields_per_problem if y > 0)}/{len(yields_per_problem)}")

    # Instrumentation: depth histogram, yield-by-depth, yield-per-token
    depth_bins = [(i / 10, (i + 1) / 10) for i in range(10)]  # 0.0-0.1, ..., 0.9-1.0
    depth_histogram = {f"{lo:.1f}-{hi:.1f}": 0 for lo, hi in depth_bins}
    yield_by_depth = {f"{lo:.1f}-{hi:.1f}": {"correct": 0, "total": 0} for lo, hi in depth_bins}
    total_generated_tokens = 0

    for c in corrections:
        d = c.get("pivot_depth", 0)
        total_generated_tokens += c.get("n_tokens", 0)
        for lo, hi in depth_bins:
            if lo <= d < hi or (hi == 1.0 and d == 1.0):
                bucket = f"{lo:.1f}-{hi:.1f}"
                depth_histogram[bucket] += 1
                yield_by_depth[bucket]["total"] += 1
                if c["correction_correct"]:
                    yield_by_depth[bucket]["correct"] += 1
                break

    yield_per_token = n_correct_corrections / max(total_generated_tokens, 1)
    mean_pivot_depth = float(np.mean([p["pivot_depth"] for p in pivot_data])) if pivot_data else 0
    effective_yield = yield_rate * mean_pivot_depth

    print(f"  Depth histogram: {depth_histogram}")
    print(f"  Yield per 1k tokens: {yield_per_token * 1000:.3f}")
    print(f"  Effective yield (yield × depth): {effective_yield:.4f}")
    print(f"  Total generated tokens: {total_generated_tokens}")

    # Step 4: Divergence Analysis (post-hoc, uses all data)
    print("[Step 4] Divergence analysis (pivot vs actual diff)")

    diff_data = []
    for g in goldilocks:
        correct_rollouts = [r for r in g["rollouts"] if r["is_correct"]]
        incorrect_rollouts = [r for r in g["rollouts"] if not r["is_correct"]]
        if not correct_rollouts or not incorrect_rollouts:
            continue
        ref_text = correct_rollouts[0]["text"]
        for inc in incorrect_rollouts:
            inc_text = inc["text"]
            min_len = min(len(ref_text), len(inc_text))
            diff_pos = min_len
            for c_idx in range(min_len):
                if ref_text[c_idx] != inc_text[c_idx]:
                    diff_pos = c_idx
                    break
            pivot_entry = next(
                (p for p in pivot_data
                 if p["problem_idx"] == g["problem_idx"] and p["rollout_id"] == inc["rollout_id"]),
                None
            )
            if pivot_entry:
                total_len = max(len(inc_text), 1)
                diff_data.append({
                    "problem_idx": g["problem_idx"],
                    "rollout_id": inc["rollout_id"],
                    "diff_char_pos": diff_pos,
                    "pivot_char_pos": pivot_entry["pivot_char_pos"],
                    "total_len": total_len,
                    "normalized_distance": abs(pivot_entry["pivot_char_pos"] - diff_pos) / total_len,
                })

    divergence_stats = {}
    if diff_data:
        dists = [d["normalized_distance"] for d in diff_data]
        print(f"  Analyzed {len(diff_data)} (correct, incorrect) pairs")
        print(f"  Mean |pivot - diff| / len: {np.mean(dists):.3f} ± {np.std(dists):.3f}")
        divergence_stats["n_pairs"] = len(diff_data)
        divergence_stats["mean_normalized_distance"] = float(np.mean(dists))
        divergence_stats["std_normalized_distance"] = float(np.std(dists))
        pivot_pos = [d["pivot_char_pos"] for d in diff_data]
        diff_pos_list = [d["diff_char_pos"] for d in diff_data]
        if len(set(pivot_pos)) > 1 and len(set(diff_pos_list)) > 1:
            from scipy.stats import spearmanr
            rho, p_val = spearmanr(pivot_pos, diff_pos_list)
            divergence_stats["spearman_rho"] = float(rho)
            divergence_stats["spearman_p"] = float(p_val)
            print(f"  Spearman(pivot_pos, diff_pos): ρ={rho:.3f}, p={p_val:.4f}")
    else:
        print("  No pairs to analyze")

    # Baseline Comparisons (skip with --skip-baselines)
    baseline_results = {}

    if args.skip_baselines:
        print("\n[Eval] Skipping baselines (--skip-baselines)")

    if not args.skip_baselines:
        print("[Eval] Baseline comparisons")

        # Random position baseline: resample from random segment positions
        random_prompts = []
        random_meta = []
        for pv in pivot_data:
            T = pv["n_segments"]
            rand_seg = random.randint(0, max(T - 1, 0))
            rand_char_pos = pv["segments"][rand_seg][1]
            prefix = pv["text"][:rand_char_pos]
            if not prefix.strip():
                continue
            orig = goldilocks_map.get(pv["problem_idx"])
            if orig is None:
                continue
            full_prompt = build_correction_prompt(orig["problem"], prefix)
            random_prompts.append(full_prompt)
            random_meta.append({
                "problem_idx": pv["problem_idx"],
                "gt_answer": pv["gt_answer"],
                "pivot_depth": (rand_seg + 0.5) / max(T, 1),
                "level": pv.get("level", "?"),
                "subject": pv.get("subject", "?"),
            })

        # First-divergence baseline (oracle pivot = where correct/incorrect first differ)
        diverge_prompts = []
        diverge_meta = []
        if diff_data:
            # Build a map from (problem_idx, rollout_id) -> diff_char_pos
            diff_map = {(d["problem_idx"], d["rollout_id"]): d["diff_char_pos"]
                        for d in diff_data}
            for pv in pivot_data:
                key = (pv["problem_idx"], pv["rollout_id"])
                if key not in diff_map:
                    continue
                # Use the first-divergence position (snapped to nearest segment boundary)
                div_pos = diff_map[key]
                # Find nearest segment boundary (snap to segment index)
                best_seg_idx = 0
                best_seg_end = pv["segments"][0][1]
                for seg_i, (s, e) in enumerate(pv["segments"]):
                    if e <= div_pos:
                        best_seg_idx = seg_i
                        best_seg_end = e
                T = pv["n_segments"]
                prefix = pv["text"][:best_seg_end]
                if not prefix.strip():
                    continue
                orig = goldilocks_map.get(pv["problem_idx"])
                if orig is None:
                    continue
                full_prompt = build_correction_prompt(orig["problem"], prefix)
                diverge_prompts.append(full_prompt)
                diverge_meta.append({
                    "problem_idx": pv["problem_idx"],
                    "gt_answer": pv["gt_answer"],
                    "pivot_depth": (best_seg_idx + 0.5) / max(T, 1),
                    "level": pv.get("level", "?"),
                    "subject": pv.get("subject", "?"),
                })

        # Root resample baseline (= Best-of-N from scratch)
        root_prompts = []
        root_meta = []
        n_root_per_problem = max(1, len(corrections) // max(len(goldilocks), 1))
        for g in goldilocks:
            msgs = [{"role": "user", "content": g["problem"]}]
            chat_prefix = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            for _ in range(min(n_root_per_problem, args.n_branches)):
                root_prompts.append(chat_prefix)
                root_meta.append({
                    "problem_idx": g["problem_idx"],
                    "gt_answer": g["gt_answer"],
                    "level": str(g.get("level", "?")),
                    "subject": g.get("subject", "?"),
                })

        print(f"  Random baseline prompts: {len(random_prompts)}")
        print(f"  First-divergence baseline prompts: {len(diverge_prompts)}")
        print(f"  Root resample (BoN) prompts: {len(root_prompts)}")

        # Generate baselines — FAIR: same n_branches per position as DEEP-GRPO
        baseline_results = {}
        for baseline_name, bl_prompts, bl_meta, use_branches in [
            ("random_position", random_prompts, random_meta, True),
            ("first_divergence", diverge_prompts, diverge_meta, True),
            ("root_resample", root_prompts, root_meta, True),
        ]:
            if not bl_prompts:
                baseline_results[baseline_name] = {"yield_rate": 0, "n": 0}
                continue
            t_bl = time.time()
            sp = branch_gen_sp if use_branches else single_sp
            bl_outputs = llm.generate(bl_prompts, sp)
            n_bl_correct = 0
            n_bl_total = 0
            bl_tokens = 0
            bl_per_level = defaultdict(lambda: {"correct": 0, "total": 0})
            bl_per_subject = defaultdict(lambda: {"correct": 0, "total": 0})
            bl_per_pivot = []  # per-pivot success for CI computation
            bl_yield_by_depth = {f"{lo:.1f}-{hi:.1f}": {"correct": 0, "total": 0}
                                 for lo, hi in [(i/10, (i+1)/10) for i in range(10)]}
            for meta, out in zip(bl_meta, bl_outputs):
                pivot_correct = 0
                for branch_out in out.outputs:
                    text = branch_out.text
                    ans = extract_clean_answer(text)
                    correct = is_equiv(ans, meta["gt_answer"])
                    n_tok = len(branch_out.token_ids) if hasattr(branch_out, 'token_ids') else len(text.split())
                    bl_tokens += n_tok
                    if correct:
                        n_bl_correct += 1
                        pivot_correct += 1
                    n_bl_total += 1
                    bl_per_level[meta.get("level", "?")]["total"] += 1
                    bl_per_subject[meta.get("subject", "?")]["total"] += 1
                    if correct:
                        bl_per_level[meta.get("level", "?")]["correct"] += 1
                        bl_per_subject[meta.get("subject", "?")]["correct"] += 1
                    # Depth bucket for baseline
                    d = meta.get("pivot_depth", 0)
                    for lo, hi in [(i/10, (i+1)/10) for i in range(10)]:
                        if lo <= d < hi or (hi == 1.0 and d == 1.0):
                            bucket = f"{lo:.1f}-{hi:.1f}"
                            bl_yield_by_depth[bucket]["total"] += 1
                            if correct:
                                bl_yield_by_depth[bucket]["correct"] += 1
                            break
                bl_per_pivot.append(pivot_correct / max(len(out.outputs), 1))
            bl_yield = n_bl_correct / max(n_bl_total, 1)
            bl_mean_depth = float(np.mean([m.get("pivot_depth", 0) for m in bl_meta])) if bl_meta else 0
            baseline_results[baseline_name] = {
                "yield_rate": bl_yield,
                "n_correct": n_bl_correct,
                "n_total": n_bl_total,
                "n_prompts": len(bl_prompts),
                "time_s": time.time() - t_bl,
                "yield_per_1k_tokens": n_bl_correct / max(bl_tokens, 1) * 1000,
                "effective_yield": bl_yield * bl_mean_depth,
                "mean_depth": bl_mean_depth,
                "total_tokens": bl_tokens,
                "yield_by_depth": {k: {
                    "yield": v["correct"] / max(v["total"], 1),
                    "correct": v["correct"],
                    "total": v["total"],
                } for k, v in bl_yield_by_depth.items()},
                "per_level": {str(k): v for k, v in sorted(bl_per_level.items())},
                "per_subject": {str(k): v for k, v in sorted(bl_per_subject.items())},
                "per_pivot_yields": bl_per_pivot,  # for bootstrap CI
            }
            print(f"  {baseline_name}: {bl_yield:.3f} "
                  f"({n_bl_correct}/{n_bl_total})")

    # Save Final Summary
    t_total = time.time() - t_start

    # Per-level analysis
    level_stats = defaultdict(lambda: {"n_gold": 0, "n_correct_corr": 0, "n_total_corr": 0})
    level_map = {g["problem_idx"]: g.get("level", "?") for g in goldilocks}
    for c in corrections:
        lev = level_map.get(c["problem_idx"], "?")
        level_stats[lev]["n_total_corr"] += 1
        if c["correction_correct"]:
            level_stats[lev]["n_correct_corr"] += 1
    for g in goldilocks:
        level_stats[g.get("level", "?")]["n_gold"] += 1

    # Diversity: pairwise edit distance among correct corrections per problem
    diversity_scores = []
    for pid in problem_yields:
        correct_texts = [c["correction_text"] for c in corrections
                         if c["problem_idx"] == pid and c["correction_correct"]]
        if len(correct_texts) >= 2:
            # Simple word-level Jaccard as diversity proxy
            pairs = []
            for i in range(len(correct_texts)):
                for j in range(i + 1, len(correct_texts)):
                    words_i = set(correct_texts[i].split())
                    words_j = set(correct_texts[j].split())
                    if words_i | words_j:
                        jaccard = len(words_i & words_j) / len(words_i | words_j)
                        pairs.append(1 - jaccard)  # diversity = 1 - similarity
                    else:
                        pairs.append(0)
            diversity_scores.append(float(np.mean(pairs)))

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "model": args.model,
            "dataset": args.dataset,
            "n_rollouts": args.n_rollouts,
            "n_branches": args.n_branches,
            "gamma": args.gamma,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "enable_thinking": enable_thinking,
        },
        "pilot": pilot_results,
        "step1": {
            "n_problems": len(rollout_data),
            "mean_accuracy": float(np.mean(accs)),
            "per_level_accuracy": {str(k): v for k, v in sorted(level_acc.items())},
        },
        "step2": {
            "n_goldilocks": len(goldilocks),
            "goldilocks_rate": len(goldilocks) / max(len(rollout_data), 1),
        },
        "step3": {
            "n_pivots": len(pivot_data),
            "mean_pivot_depth": float(np.mean([p["pivot_depth"] for p in pivot_data])) if pivot_data else 0,
            "p_phi_params": pivot_selector.get_params(),
        },
        "step4": divergence_stats,
        "step6": {
            "yield_rate": yield_rate,
            "n_correct": n_correct_corrections if corrections else 0,
            "n_total": len(corrections),
            "mean_per_problem_yield": float(np.mean(yields_per_problem)) if corrections else 0,
            "problems_with_any_correct": sum(1 for y in yields_per_problem if y > 0) if corrections else 0,
        },
        "instrumentation": {
            "depth_histogram": depth_histogram,
            "yield_by_depth": {k: {
                "yield": v["correct"] / max(v["total"], 1),
                "correct": v["correct"],
                "total": v["total"],
            } for k, v in yield_by_depth.items()},
            "yield_per_1k_tokens": yield_per_token * 1000,
            "total_generated_tokens": total_generated_tokens,
            "effective_yield": effective_yield,
            "mean_pivot_depth": mean_pivot_depth,
        },
        "baselines": baseline_results,
        "diversity": {
            "mean_pairwise_diversity": float(np.mean(diversity_scores)) if diversity_scores else None,
            "n_problems_measured": len(diversity_scores),
        },
        "per_level": {str(k): v for k, v in sorted(level_stats.items())},
        "kill_criteria": {
            "yield_lt_5pct": yield_rate < 0.05,
            "pivot_eq_random": (
                bool(abs(yield_rate - baseline_results.get("random_position", {}).get("yield_rate", 0)) < 0.02)
                if baseline_results.get("random_position") else None
            ),
            "no_diversity": (
                bool(np.mean(diversity_scores) < 0.1) if diversity_scores else None
            ),
        },
        "runtime_s": t_total,
    }

    # Print kill criteria
    print("[Eval] Kill criteria check")
    kc = summary["kill_criteria"]
    print(f"  yield < 5%: {kc['yield_lt_5pct']} (yield={yield_rate:.3f})")
    print(f"  pivot ≈ random: {kc['pivot_eq_random']}")
    print(f"  no diversity: {kc['no_diversity']}")

    # Overall status
    killed = kc["yield_lt_5pct"]
    if kc["pivot_eq_random"]:
        killed = True
    status = "KILLED" if killed else "ALIVE"
    summary["status"] = status
    print(f"\n  Pipeline status: {status}")
    print(f"  Total runtime: {t_total:.1f}s ({t_total / 60:.1f} min)")

    # Save
    with open(output_dir / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "pivots.jsonl", "w") as f:
        for p in pivot_data:
            # Don't save full text in pivots (already in rollouts)
            save_p = {k: v for k, v in p.items() if k != "text"}
            f.write(json.dumps(save_p) + "\n")
    if diff_data:
        with open(output_dir / "divergence_analysis.json", "w") as f:
            json.dump(diff_data, f, indent=2)

    print(f"\n  Output files in {output_dir}/:")
    for p in sorted(output_dir.glob("*")):
        size = p.stat().st_size / 1024
        print(f"    {p.name} ({size:.1f} KB)")

    return summary


# CLI

def main():
    parser = argparse.ArgumentParser(description="ExIt Pipeline: DEEP-GRPO + Correction Generation")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--dataset", default="HuggingFaceH4/MATH-500")
    parser.add_argument("--n-rollouts", type=int, default=8)
    parser.add_argument("--n-branches", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="/mnt/local/results")
    parser.add_argument("--skip-pilot", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true",
                        help="Force thinking mode (only used with --skip-pilot)")
    parser.add_argument("--force", action="store_true",
                        help="Force regeneration even if rollouts exist")
    parser.add_argument("--skip-baselines", action="store_true",
                        help="Skip baseline generation (random, divergence, root-resample)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick smoke test: 5 problems, 4 rollouts, 4 branches")
    args = parser.parse_args()

    # Override for smoke test
    if args.smoke_test:
        args.n_rollouts = 4
        args.n_branches = 4
        args.skip_pilot = True
        print("[SMOKE TEST] Running with 5 problems, 4 rollouts, 4 branches")

    summary = run_pipeline(args)
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
