#!/usr/bin/env python3
"""SFT Ablation — Does correction data have unique training value?

3-way comparison:
  Model A: ~636 clean traces from goldilocks train problems
  Model B: ~636 clean + ~160 correction traces (20% mix, single-stage)
  Model C: ~2000 clean traces (volume control — root resample + all-correct)

Usage:
  # Step 1: Prepare data (CPU, local)
  python sft_ablation.py prep \
    --rollouts-file data/rollouts.jsonl \
    --corrections-file data/corrections_max.jsonl \
    --output-dir data/sft_data

  # Step 2: Train (GPU)
  python sft_ablation.py train \
    --data-dir data/sft_data \
    --model-variant A \
    --output-dir /path/to/models

  # Step 3: Evaluate (GPU)
  python sft_ablation.py eval \
    --data-dir data/sft_data \
    --model-dir /path/to/models/model_A \
    --output-dir /path/to/eval
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

SEED = 17
CORRECTION_RATIO = 0.20  # Target: 20% corrections in Model B mix
MAX_SEQ_LENGTH = 4096
N_RECOVERY_ROLLOUTS = 15  # Maximize prefixes per problem for statistical power

# Answer Extraction

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


# Step 1: Data Preparation

def load_rollouts(path: str) -> list[dict]:
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_corrections(path: str) -> list[dict]:
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def create_problem_split(
    rollouts: list[dict], seed: int = SEED
) -> tuple[list[int], list[int], list[int]]:
    """Stratified train/val/test split (106/15/20) from goldilocks problems."""
    rng = random.Random(seed)

    by_level: dict[int, list[int]] = defaultdict(list)
    for r in rollouts:
        nc = r["n_correct"]
        if 1 <= nc <= 7:
            by_level[r["level"]].append(r["problem_idx"])

    for level in by_level:
        by_level[level].sort()

    train_idxs, val_idxs, test_idxs = [], [], []
    total_gold = sum(len(v) for v in by_level.values())

    for level in sorted(by_level.keys()):
        problems = by_level[level][:]
        rng.shuffle(problems)
        n = len(problems)
        n_test = max(1, round(20 * n / total_gold))
        n_val = max(1, round(15 * n / total_gold))
        n_train = n - n_test - n_val
        if n_train < 1:
            n_train = n
            n_val = 0
            n_test = 0

        test_idxs.extend(problems[:n_test])
        val_idxs.extend(problems[n_test : n_test + n_val])
        train_idxs.extend(problems[n_test + n_val :])

    print(f"Split: {len(train_idxs)} train / {len(val_idxs)} val / {len(test_idxs)} test")
    print(f"  Levels in test: {sorted(set(next(r['level'] for r in rollouts if r['problem_idx'] == idx) for idx in test_idxs))}")
    return sorted(train_idxs), sorted(val_idxs), sorted(test_idxs)


def format_chat(problem: str, solution: str) -> str:
    """Format as Qwen3 ChatML (non-thinking mode)."""
    return f"<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n{solution}<|im_end|>"


def build_clean_traces(
    rollouts: list[dict], problem_idxs: set[int]
) -> list[dict]:
    traces = []
    for r in rollouts:
        if r["problem_idx"] not in problem_idxs:
            continue
        for rollout in r["rollouts"]:
            if rollout["is_correct"]:
                traces.append({
                    "problem_idx": r["problem_idx"],
                    "problem": r["problem"],
                    "gt_answer": r["gt_answer"],
                    "level": r["level"],
                    "subject": r["subject"],
                    "text": format_chat(r["problem"], rollout["text"]),
                    "source": "clean",
                    "rollout_id": rollout["rollout_id"],
                })
    return traces


def build_correction_traces(
    rollouts: list[dict],
    corrections: list[dict],
    problem_idxs: set[int],
    max_corrections: int | None = None,
    seed: int = SEED,
) -> list[dict]:
    """Build correction traces for SFT.

    Each correction trace = problem + incorrect prefix + "Wait" + correct continuation.
    Quality filters:
    - correction_correct must be True
    - Must be from a train problem
    - Prefer diverse problems (spread corrections across problems)
    """
    rng = random.Random(seed)

    # Index rollouts by problem_idx for prefix lookup
    rollout_index: dict[int, dict[int, str]] = {}
    for r in rollouts:
        rollout_index[r["problem_idx"]] = {
            roll["rollout_id"]: roll["text"] for roll in r["rollouts"]
        }

    # Problem text lookup
    problem_text: dict[int, dict] = {}
    for r in rollouts:
        problem_text[r["problem_idx"]] = {
            "problem": r["problem"],
            "gt_answer": r["gt_answer"],
            "level": r["level"],
            "subject": r["subject"],
        }

    # Collect all valid corrections from train problems
    valid = []
    for c in corrections:
        if c["problem_idx"] not in problem_idxs:
            continue
        if not c["correction_correct"]:
            continue
        # Get the incorrect rollout's prefix
        rid = c["rollout_id"]
        pidx = c["problem_idx"]
        if pidx not in rollout_index or rid not in rollout_index[pidx]:
            continue
        prefix = rollout_index[pidx][rid][: c["prefix_len"]]
        if len(prefix.strip()) < 10:
            continue  # Too short prefix — likely degenerate

        valid.append({
            "problem_idx": pidx,
            "rollout_id": rid,
            "prefix": prefix,
            "correction_text": c["correction_text"],
            "div_depth": c["div_depth"],
            "gt_answer": c["gt_answer"],
        })

    print(f"  Valid corrections from train problems: {len(valid)}")

    # If we need to subsample, spread across problems for diversity
    if max_corrections and len(valid) > max_corrections:
        # Group by problem, take at most ceil(max/n_problems) per problem
        by_prob: dict[int, list] = defaultdict(list)
        for v in valid:
            by_prob[v["problem_idx"]].append(v)

        # Shuffle within each problem
        for pidx in by_prob:
            rng.shuffle(by_prob[pidx])

        # Round-robin selection
        selected = []
        prob_ids = sorted(by_prob.keys())
        rng.shuffle(prob_ids)
        idx_per_prob = {p: 0 for p in prob_ids}
        while len(selected) < max_corrections:
            added = False
            for p in prob_ids:
                if idx_per_prob[p] < len(by_prob[p]):
                    selected.append(by_prob[p][idx_per_prob[p]])
                    idx_per_prob[p] += 1
                    added = True
                    if len(selected) >= max_corrections:
                        break
            if not added:
                break
        valid = selected

    # Format as SFT traces
    traces = []
    for v in valid:
        pinfo = problem_text[v["problem_idx"]]
        transition = "\n\nWait, let me reconsider this step.\n\n"
        solution = v["prefix"] + transition + v["correction_text"]
        traces.append({
            "problem_idx": v["problem_idx"],
            "problem": pinfo["problem"],
            "gt_answer": pinfo["gt_answer"],
            "level": pinfo["level"],
            "subject": pinfo["subject"],
            "text": format_chat(pinfo["problem"], solution),
            "source": "correction",
            "rollout_id": v["rollout_id"],
            "div_depth": v["div_depth"],
        })
    return traces


def build_recovery_eval_data(
    rollouts: list[dict],
    corrections: list[dict],
    problem_idxs: set[int],
    max_per_problem: int = N_RECOVERY_ROLLOUTS,
) -> list[dict]:
    """Build forced-prefix recovery evaluation data.

    For each held-out problem, collect up to max_per_problem incorrect rollouts.
    For each, find the divergence point and create a prefix for forced-prefix eval.
    """
    # Use correction data to get divergence points for held-out problems
    # Group corrections by problem
    by_prob: dict[int, list] = defaultdict(list)
    for c in corrections:
        if c["problem_idx"] not in problem_idxs:
            continue
        # Use both correct and incorrect corrections — we just need the prefix
        by_prob[c["problem_idx"]].append(c)

    # Also need rollout texts
    rollout_index: dict[int, dict] = {}
    problem_text: dict[int, dict] = {}
    for r in rollouts:
        rollout_index[r["problem_idx"]] = {
            roll["rollout_id"]: roll for roll in r["rollouts"]
        }
        problem_text[r["problem_idx"]] = {
            "problem": r["problem"],
            "gt_answer": r["gt_answer"],
            "level": r["level"],
            "subject": r["subject"],
        }

    eval_data = []
    for pidx in sorted(problem_idxs):
        if pidx not in by_prob:
            continue
        # Deduplicate by rollout_id, prefer corrections with explicit divergence
        seen_rollouts = set()
        problem_evals = []
        for c in by_prob[pidx]:
            rid = c["rollout_id"]
            if rid in seen_rollouts:
                continue
            seen_rollouts.add(rid)
            if pidx not in rollout_index or rid not in rollout_index[pidx]:
                continue
            prefix = rollout_index[pidx][rid]["text"][: c["prefix_len"]]
            if len(prefix.strip()) < 10:
                continue
            pinfo = problem_text[pidx]
            problem_evals.append({
                "problem_idx": pidx,
                "problem": pinfo["problem"],
                "gt_answer": pinfo["gt_answer"],
                "level": pinfo["level"],
                "subject": pinfo["subject"],
                "rollout_id": rid,
                "prefix": prefix,
                "div_depth": c["div_depth"],
                "prefix_len": c["prefix_len"],
            })
            if len(problem_evals) >= max_per_problem:
                break
        eval_data.extend(problem_evals)

    return eval_data


def cmd_prep(args):
    """Prepare all data for SFT ablation."""
    print("SFT Ablation — Data Preparation")

    rollouts = load_rollouts(args.rollouts_file)
    corrections = load_corrections(args.corrections_file)
    print(f"Loaded {len(rollouts)} problems, {len(corrections)} corrections")

    # Create split
    train_idxs, val_idxs, test_idxs = create_problem_split(rollouts)
    train_set = set(train_idxs)
    val_set = set(val_idxs)
    test_set = set(test_idxs)

    # Save split
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    split_info = {
        "seed": SEED,
        "train": train_idxs,
        "val": val_idxs,
        "test": test_idxs,
    }
    with open(out / "split.json", "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"\nSaved split to {out / 'split.json'}")

    # === Model A: Clean traces from goldilocks train problems ===
    print("\n--- Model A: Clean baseline ---")
    clean_train = build_clean_traces(rollouts, train_set)
    clean_val = build_clean_traces(rollouts, val_set)
    print(f"  Train: {len(clean_train)} clean traces")
    print(f"  Val: {len(clean_val)} clean traces")

    save_jsonl(clean_train, out / "model_a_train.jsonl")
    save_jsonl(clean_val, out / "model_a_val.jsonl")

    # === Model B: Clean + corrections (20% mix) ===
    print("\n--- Model B: Clean + corrections ---")
    n_clean = len(clean_train)
    # Target: corrections = 20% of total → n_corr = n_clean * 0.20 / 0.80
    n_target_corrections = int(n_clean * CORRECTION_RATIO / (1 - CORRECTION_RATIO))
    print(f"  Target corrections: {n_target_corrections} (for {CORRECTION_RATIO:.0%} mix)")

    correction_train = build_correction_traces(
        rollouts, corrections, train_set,
        max_corrections=n_target_corrections,
    )
    print(f"  Actual corrections: {len(correction_train)}")
    actual_ratio = len(correction_train) / (n_clean + len(correction_train))
    print(f"  Actual mix: {actual_ratio:.1%} corrections, {1-actual_ratio:.1%} clean")

    # Model B train = clean + corrections
    model_b_train = clean_train + correction_train
    random.Random(SEED).shuffle(model_b_train)

    save_jsonl(model_b_train, out / "model_b_train.jsonl")
    save_jsonl(clean_val, out / "model_b_val.jsonl")  # Val is always clean-only

    # === Model C: Large clean control ===
    print("\n--- Model C: Volume control ---")
    # Oversample goldilocks train clean traces to ~2000
    # MUST use same problem distribution as A/B to avoid confound
    clean_c_train = build_clean_traces(rollouts, train_set)
    base_n = len(clean_c_train)
    rng = random.Random(SEED)
    while len(clean_c_train) < 2000:
        # Oversample with shuffle
        extra = build_clean_traces(rollouts, train_set)
        rng.shuffle(extra)
        clean_c_train.extend(extra)
    clean_c_train = clean_c_train[:2000]
    rng.shuffle(clean_c_train)
    print(f"  Train: {len(clean_c_train)} clean traces (oversampled from {base_n} unique, {len(train_set)} problems)")

    save_jsonl(clean_c_train, out / "model_c_train.jsonl")
    save_jsonl(clean_val, out / "model_c_val.jsonl")

    # === Evaluation data ===
    print("\n--- Evaluation data ---")
    # Recovery eval: forced-prefix data for val and test
    recovery_val = build_recovery_eval_data(rollouts, corrections, val_set)
    recovery_test = build_recovery_eval_data(rollouts, corrections, test_set)
    print(f"  Recovery val: {len(recovery_val)} prefixes from {len(val_set)} problems")
    print(f"  Recovery test: {len(recovery_test)} prefixes from {len(test_set)} problems")

    save_jsonl(recovery_val, out / "recovery_val.jsonl")
    save_jsonl(recovery_test, out / "recovery_test.jsonl")

    # From-scratch eval: just problems (no prefixes) — for BOTH val and test
    for split_name, split_idxs in [("val", val_set), ("test", test_set)]:
        scratch_problems = []
        for r in rollouts:
            if r["problem_idx"] in split_idxs:
                scratch_problems.append({
                    "problem_idx": r["problem_idx"],
                    "problem": r["problem"],
                    "gt_answer": r["gt_answer"],
                    "level": r["level"],
                    "subject": r["subject"],
                })
        save_jsonl(scratch_problems, out / f"scratch_{split_name}.jsonl")
        print(f"  Scratch {split_name}: {len(scratch_problems)} problems")

    # Summary
    summary = {
        "split": {"train": len(train_idxs), "val": len(val_idxs), "test": len(test_idxs)},
        "model_a": {"train": len(clean_train), "val": len(clean_val)},
        "model_b": {
            "train": len(model_b_train),
            "clean": n_clean,
            "corrections": len(correction_train),
            "correction_ratio": actual_ratio,
            "val": len(clean_val),
        },
        "model_c": {"train": len(clean_c_train), "val": len(clean_val)},
        "eval": {
            "recovery_val": len(recovery_val),
            "recovery_test": len(recovery_test),
            "scratch_test": len(scratch_problems),
        },
    }
    with open(out / "prep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print("Data prep complete!")
    print(json.dumps(summary, indent=2))
    return summary


def save_jsonl(data: list[dict], path: Path):
    with open(path, "w") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


# Step 2: Training (Unsloth + TRL)

def cmd_train(args):
    """Train one model variant (A, B, or C)."""
    print(f"SFT Ablation — Training Model {args.model_variant}")

    variant = args.model_variant.upper()
    assert variant in ("A", "B", "C"), f"Invalid variant: {variant}"

    data_dir = Path(args.data_dir)
    train_file = data_dir / f"model_{variant.lower()}_train.jsonl"
    val_file = data_dir / f"model_{variant.lower()}_val.jsonl"

    if not train_file.exists():
        print(f"ERROR: {train_file} not found. Run 'prep' first.")
        sys.exit(1)

    # Load data
    train_data = [json.loads(l) for l in open(train_file)]
    val_data = [json.loads(l) for l in open(val_file)]
    print(f"Train: {len(train_data)} examples, Val: {len(val_data)} examples")

    # Count sources
    n_clean = sum(1 for d in train_data if d.get("source") == "clean")
    n_corr = sum(1 for d in train_data if d.get("source") == "correction")
    print(f"  Clean: {n_clean}, Corrections: {n_corr}")

    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer
    from datasets import Dataset

    # Verify tokenizer tokens
    print("\nLoading Qwen3-8B with QLoRA...")
    # Note: Unsloth handles pad/eos tokens, but we verify after load
    model, tokenizer = FastLanguageModel.from_pretrained(
        "Qwen/Qwen3-8B",
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,  # auto-detect
    )

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

    # Verify tokenizer configuration
    print(f"  eos_token: {tokenizer.eos_token!r} (id={tokenizer.eos_token_id})")
    print(f"  pad_token: {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})")
    # Verify ChatML tokens exist in vocabulary
    for token in ["<|im_start|>", "<|im_end|>"]:
        tid = tokenizer.convert_tokens_to_ids(token)
        assert tid != tokenizer.unk_token_id, f"{token} not in vocabulary!"
    print("  ChatML tokens verified in vocabulary")

    # Prepare datasets
    train_ds = Dataset.from_list([{"text": d["text"]} for d in train_data])
    val_ds = Dataset.from_list([{"text": d["text"]} for d in val_data])

    # Output directory
    output_dir = Path(args.output_dir) / f"model_{variant.lower()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training config
    # Model C has ~4x more data (oversampled), needs fewer epochs
    # A/B need more epochs due to small dataset (465 examples)
    n_epochs = 3 if variant == "C" else 5
    lr = 1e-4  # Conservative for small dataset

    config = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,  # effective batch = 8
        num_train_epochs=n_epochs,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        max_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        save_total_limit=3,
        weight_decay=0.01,
        max_grad_norm=1.0,
        seed=SEED,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=config,
    )

    print(f"\nStarting training: {config.num_train_epochs} epochs, "
          f"lr={config.learning_rate}, batch={config.per_device_train_batch_size}×"
          f"{config.gradient_accumulation_steps}")

    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    print(f"\nTraining complete in {elapsed/60:.1f} min")

    # Save final model
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    print(f"Model saved to {output_dir / 'final'}")

    # Save training log
    log = {
        "variant": variant,
        "n_train": len(train_data),
        "n_clean": n_clean,
        "n_corrections": n_corr,
        "training_time_s": elapsed,
        "final_train_loss": trainer.state.log_history[-1].get("train_loss"),
        "best_eval_loss": trainer.state.best_metric,
    }
    with open(output_dir / "train_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(json.dumps(log, indent=2))


# Step 3: Evaluation

def cmd_eval(args):
    """Evaluate a trained model on recovery@1, pass@1, over-triggering."""
    print(f"SFT Ablation — Evaluation")

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which eval set to use
    eval_set = args.eval_set  # "val" or "test"

    # Load eval data
    recovery_file = data_dir / f"recovery_{eval_set}.jsonl"
    scratch_file = data_dir / f"scratch_{eval_set}.jsonl"

    recovery_data = [json.loads(l) for l in open(recovery_file)]
    print(f"Recovery data: {len(recovery_data)} prefixes")

    has_scratch = scratch_file.exists()
    scratch_data = [json.loads(l) for l in open(scratch_file)] if has_scratch else []
    print(f"Scratch data: {len(scratch_data)} problems")

    # Load model
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("Falling back to transformers for inference")
        from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nLoading model from {model_dir / 'final'}...")

    if args.use_vllm:
        results = _eval_vllm(model_dir, recovery_data, scratch_data, output_dir, eval_set)
    else:
        results = _eval_hf(model_dir, recovery_data, scratch_data, output_dir, eval_set)

    # Save results
    with open(output_dir / f"eval_{eval_set}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir / f'eval_{eval_set}.json'}")
    print(json.dumps(results, indent=2))
    return results


def _eval_vllm(model_dir, recovery_data, scratch_data, output_dir, eval_set):
    """Evaluate using vLLM for faster inference."""
    from vllm import LLM, SamplingParams

    # Merge LoRA with base for vLLM
    # First check if merged model exists
    merged_dir = model_dir / "merged"
    if not merged_dir.exists():
        print("Merging LoRA weights for vLLM...")
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            str(model_dir / "final"),
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=False,  # Need full precision for merge
        )
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
        print(f"Merged model saved to {merged_dir}")
        # Free GPU memory before vLLM init
        del model, tokenizer
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()
        print(f"GPU memory freed after merge")

    llm = LLM(
        str(merged_dir),
        max_model_len=8192,
        gpu_memory_utilization=0.70,
        enable_prefix_caching=True,
    )
    sampling = SamplingParams(
        temperature=0,
        top_p=1.0,
        max_tokens=2048,
    )

    results = {}

    # 1. Forced-prefix recovery@1
    if recovery_data:
        print("\n--- Forced-prefix recovery@1 ---")
        prompts = []
        for d in recovery_data:
            prompt = f"<|im_start|>user\n{d['problem']}<|im_end|>\n<|im_start|>assistant\n{d['prefix']}"
            prompts.append(prompt)

        outputs = llm.generate(prompts, sampling)

        correct = 0
        per_problem = defaultdict(list)
        for d, out in zip(recovery_data, outputs):
            text = out.outputs[0].text
            ans = extract_clean_answer(text)
            hit = answers_match(ans, d["gt_answer"])
            correct += int(hit)
            per_problem[d["problem_idx"]].append(int(hit))

        raw_recovery = correct / len(recovery_data) if recovery_data else 0
        per_prob_recovery = np.mean([np.mean(v) for v in per_problem.values()])

        results["recovery"] = {
            "raw": raw_recovery,
            "per_problem": float(per_prob_recovery),
            "n_total": len(recovery_data),
            "n_correct": correct,
            "n_problems": len(per_problem),
        }
        print(f"  Raw: {raw_recovery:.3f} ({correct}/{len(recovery_data)})")
        print(f"  Per-problem: {per_prob_recovery:.3f}")

    # 2. From-scratch pass@1
    if scratch_data:
        print("\n--- From-scratch pass@1 ---")
        prompts = [f"<|im_start|>user\n{d['problem']}<|im_end|>\n<|im_start|>assistant\n" for d in scratch_data]
        outputs = llm.generate(prompts, sampling)

        correct = 0
        generations = []
        for d, out in zip(scratch_data, outputs):
            text = out.outputs[0].text
            ans = extract_clean_answer(text)
            hit = answers_match(ans, d["gt_answer"])
            correct += int(hit)
            generations.append({
                "problem_idx": d["problem_idx"],
                "text": text,
                "answer": ans,
                "gt_answer": d["gt_answer"],
                "correct": hit,
            })

        pass_at_1 = correct / len(scratch_data) if scratch_data else 0
        results["pass_at_1"] = {
            "accuracy": pass_at_1,
            "n_total": len(scratch_data),
            "n_correct": correct,
        }
        print(f"  Accuracy: {pass_at_1:.3f} ({correct}/{len(scratch_data)})")

        # 3. Over-triggering rate
        print("\n--- Over-triggering rate ---")
        # Check for correction-style language outside <think> tags
        trigger_phrases = ["wait, let me reconsider", "wait, let me re-evaluate",
                           "wait, that's not right", "wait, i made"]
        n_triggered = 0
        for g in generations:
            # Strip <think>...</think> content before checking
            text = re.sub(r"<think>.*?</think>", "", g["text"][:1024], flags=re.DOTALL)
            first_portion = text[:512].lower()
            if any(phrase in first_portion for phrase in trigger_phrases):
                n_triggered += 1

        trigger_rate = n_triggered / len(generations) if generations else 0
        results["over_triggering"] = {
            "rate": trigger_rate,
            "n_triggered": n_triggered,
            "n_total": len(generations),
        }
        print(f"  Rate: {trigger_rate:.3f} ({n_triggered}/{len(generations)})")

        # 4. Average generation length
        avg_len = np.mean([len(g["text"]) for g in generations])
        results["avg_gen_length"] = float(avg_len)
        print(f"  Avg generation length: {avg_len:.0f} chars")

        # Save generations for inspection
        save_jsonl(generations, output_dir / f"generations_{eval_set}.jsonl")

    return results


def _eval_hf(model_dir, recovery_data, scratch_data, output_dir, eval_set):
    """Evaluate using HuggingFace generate (slower, no vLLM needed)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir / "final"))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir / "final"),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    results = {}

    def generate_one(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # 1. Recovery
    if recovery_data:
        print("\n--- Forced-prefix recovery@1 ---")
        correct = 0
        per_problem = defaultdict(list)
        for i, d in enumerate(recovery_data):
            prompt = f"<|im_start|>user\n{d['problem']}<|im_end|>\n<|im_start|>assistant\n{d['prefix']}"
            text = generate_one(prompt)
            ans = extract_clean_answer(text)
            hit = answers_match(ans, d["gt_answer"])
            correct += int(hit)
            per_problem[d["problem_idx"]].append(int(hit))
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(recovery_data)}] running accuracy: {correct/(i+1):.3f}")

        raw_recovery = correct / len(recovery_data) if recovery_data else 0
        per_prob_recovery = np.mean([np.mean(v) for v in per_problem.values()])
        results["recovery"] = {
            "raw": raw_recovery,
            "per_problem": float(per_prob_recovery),
            "n_total": len(recovery_data),
            "n_correct": correct,
            "n_problems": len(per_problem),
        }
        print(f"  Raw: {raw_recovery:.3f}, Per-problem: {per_prob_recovery:.3f}")

    # 2. From-scratch
    if scratch_data:
        print("\n--- From-scratch pass@1 ---")
        correct = 0
        generations = []
        for i, d in enumerate(scratch_data):
            prompt = f"<|im_start|>user\n{d['problem']}<|im_end|>\n<|im_start|>assistant\n"
            text = generate_one(prompt)
            ans = extract_clean_answer(text)
            hit = answers_match(ans, d["gt_answer"])
            correct += int(hit)
            generations.append({"problem_idx": d["problem_idx"], "text": text,
                                "answer": ans, "gt_answer": d["gt_answer"], "correct": hit})

        pass_at_1 = correct / len(scratch_data) if scratch_data else 0
        results["pass_at_1"] = {"accuracy": pass_at_1, "n_total": len(scratch_data), "n_correct": correct}
        print(f"  Accuracy: {pass_at_1:.3f}")

        # Over-triggering (match vLLM eval logic)
        trigger_phrases = ["wait, let me reconsider", "wait, let me re-evaluate",
                           "wait, that's not right", "wait, i made"]
        n_triggered = 0
        for g in generations:
            text = re.sub(r"<think>.*?</think>", "", g["text"][:1024], flags=re.DOTALL)
            if any(p in text[:512].lower() for p in trigger_phrases):
                n_triggered += 1
        results["over_triggering"] = {"rate": n_triggered / len(generations), "n_triggered": n_triggered,
                                       "n_total": len(generations)}
        results["avg_gen_length"] = float(np.mean([len(g["text"]) for g in generations]))

        save_jsonl(generations, output_dir / f"generations_{eval_set}.jsonl")

    return results


# Step 4: Compare all models

def cmd_compare(args):
    """Compare evaluation results across all 3 models. Apply kill criteria."""
    print("SFT Ablation — Model Comparison")

    eval_dir = Path(args.eval_dir)
    eval_set = args.eval_set

    results = {}
    for variant in ["base", "a", "b", "c"]:
        path = eval_dir / f"model_{variant}" / f"eval_{eval_set}.json"
        if path.exists():
            with open(path) as f:
                results[variant.upper()] = json.load(f)
            print(f"Loaded Model {variant.upper()} results")
        else:
            print(f"WARNING: {path} not found")

    if len(results) < 2:
        print("Need at least 2 models to compare. Exiting.")
        sys.exit(1)

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"{'Metric':<30} {'Base':>10} {'Model A':>10} {'Model B':>10} {'Model C':>10}")
    print(f"{'-'*70}")

    for metric, key_path in [
        ("Recovery@1 (raw)", ("recovery", "raw")),
        ("Recovery@1 (per-prob)", ("recovery", "per_problem")),
        ("Pass@1", ("pass_at_1", "accuracy")),
        ("Over-triggering rate", ("over_triggering", "rate")),
        ("Avg gen length", ("avg_gen_length",)),
    ]:
        vals = []
        for v in ["BASE", "A", "B", "C"]:
            if v in results:
                obj = results[v]
                for k in key_path:
                    obj = obj.get(k, {}) if isinstance(obj, dict) else None
                vals.append(f"{obj:.3f}" if isinstance(obj, (int, float)) else "N/A")
            else:
                vals.append("N/A")
        print(f"{metric:<30} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} {vals[3]:>10}")

    # Kill criteria
    print(f"\n{'='*60}")
    print("Kill Criteria Check")
    print(f"{'='*60}")

    kills = []
    if "A" in results and "B" in results:
        a_rec = results["A"].get("recovery", {}).get("per_problem", 0)
        b_rec = results["B"].get("recovery", {}).get("per_problem", 0)
        a_pass = results["A"].get("pass_at_1", {}).get("accuracy", 0)
        b_pass = results["B"].get("pass_at_1", {}).get("accuracy", 0)
        b_trig = results["B"].get("over_triggering", {}).get("rate", 0)

        delta_rec = b_rec - a_rec
        delta_pass = b_pass - a_pass

        print(f"B recovery - A recovery = {delta_rec:+.3f} (need ≥+0.05)")
        print(f"B pass@1 - A pass@1 = {delta_pass:+.3f} (need ≥-0.02)")
        print(f"B over-triggering = {b_trig:.3f} (need <0.15)")

        if delta_rec < 0.05:
            kills.append("KILL: B recovery < A + 5pp")
        if delta_pass < -0.02:
            kills.append("KILL: B pass@1 < A - 2pp (over-triggering)")
        if b_trig > 0.15:
            kills.append("KILL: B over-triggering >15%")

    if "B" in results and "C" in results:
        b_rec = results["B"].get("recovery", {}).get("per_problem", 0)
        c_rec = results["C"].get("recovery", {}).get("per_problem", 0)
        b_pass = results["B"].get("pass_at_1", {}).get("accuracy", 0)
        c_pass = results["C"].get("pass_at_1", {}).get("accuracy", 0)

        if c_rec >= b_rec and c_pass >= b_pass:
            kills.append("KILL: C ≥ B on both metrics (corrections = just more data)")

    if kills:
        print("\n⚠️  KILL CRITERIA FIRED:")
        for k in kills:
            print(f"  {k}")
    else:
        print("\n✓ All kill criteria PASSED — corrections have unique value")

    # Save comparison
    comparison = {
        "eval_set": eval_set,
        "results": results,
        "kills": kills,
        "passed": len(kills) == 0,
    }
    with open(eval_dir / f"comparison_{eval_set}.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nComparison saved to {eval_dir / f'comparison_{eval_set}.json'}")


# Step 3b: Base Model Evaluation (no SFT)

def cmd_eval_base(args):
    """Evaluate unmodified Qwen3-8B as a baseline."""
    print("SFT Ablation — Base Model Evaluation")

    from vllm import LLM, SamplingParams

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) / "model_base"
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_set = args.eval_set

    recovery_data = [json.loads(l) for l in open(data_dir / f"recovery_{eval_set}.jsonl")]
    scratch_file = data_dir / f"scratch_{eval_set}.jsonl"
    scratch_data = [json.loads(l) for l in open(scratch_file)] if scratch_file.exists() else []

    print(f"Recovery: {len(recovery_data)}, Scratch: {len(scratch_data)}")

    llm = LLM("Qwen/Qwen3-8B", max_model_len=8192, gpu_memory_utilization=0.70)
    sampling = SamplingParams(temperature=0, top_p=1.0, max_tokens=2048)

    results = {}

    # Recovery
    if recovery_data:
        prompts = [f"<|im_start|>user\n{d['problem']}<|im_end|>\n<|im_start|>assistant\n{d['prefix']}"
                   for d in recovery_data]
        outputs = llm.generate(prompts, sampling)
        correct = 0
        per_problem = defaultdict(list)
        for d, out in zip(recovery_data, outputs):
            text = out.outputs[0].text
            ans = extract_clean_answer(text)
            hit = answers_match(ans, d["gt_answer"])
            correct += int(hit)
            per_problem[d["problem_idx"]].append(int(hit))
        results["recovery"] = {
            "raw": correct / len(recovery_data),
            "per_problem": float(np.mean([np.mean(v) for v in per_problem.values()])),
            "n_total": len(recovery_data), "n_correct": correct,
        }
        print(f"  Recovery: {results['recovery']['per_problem']:.3f}")

    # From-scratch
    if scratch_data:
        prompts = [f"<|im_start|>user\n{d['problem']}<|im_end|>\n<|im_start|>assistant\n"
                   for d in scratch_data]
        outputs = llm.generate(prompts, sampling)
        correct = 0
        generations = []
        for d, out in zip(scratch_data, outputs):
            text = out.outputs[0].text
            ans = extract_clean_answer(text)
            hit = answers_match(ans, d["gt_answer"])
            correct += int(hit)
            generations.append({"problem_idx": d["problem_idx"], "text": text,
                                "answer": ans, "gt_answer": d["gt_answer"], "correct": hit})
        results["pass_at_1"] = {"accuracy": correct / len(scratch_data),
                                 "n_total": len(scratch_data), "n_correct": correct}
        print(f"  Pass@1: {results['pass_at_1']['accuracy']:.3f}")

        # Over-triggering baseline
        trigger_phrases = ["wait, let me reconsider", "wait, let me re-evaluate",
                           "wait, that's not right", "wait, i made"]
        n_triggered = 0
        for g in generations:
            text = re.sub(r"<think>.*?</think>", "", g["text"][:1024], flags=re.DOTALL)
            if any(p in text[:512].lower() for p in trigger_phrases):
                n_triggered += 1
        results["over_triggering"] = {"rate": n_triggered / len(generations),
                                       "n_triggered": n_triggered, "n_total": len(generations)}
        results["avg_gen_length"] = float(np.mean([len(g["text"]) for g in generations]))

    with open(output_dir / f"eval_{eval_set}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBase model results saved to {output_dir / f'eval_{eval_set}.json'}")
    print(json.dumps(results, indent=2))


# CLI

def main():
    parser = argparse.ArgumentParser(description="SFT Ablation")
    sub = parser.add_subparsers(dest="command", required=True)

    # Prep
    p_prep = sub.add_parser("prep", help="Prepare SFT data")
    p_prep.add_argument("--rollouts-file", required=True)
    p_prep.add_argument("--corrections-file", required=True)
    p_prep.add_argument("--output-dir", required=True)

    # Train
    p_train = sub.add_parser("train", help="Train one model variant")
    p_train.add_argument("--data-dir", required=True)
    p_train.add_argument("--model-variant", required=True, choices=["A", "B", "C", "a", "b", "c"])
    p_train.add_argument("--output-dir", required=True)

    # Eval
    p_eval = sub.add_parser("eval", help="Evaluate a trained model")
    p_eval.add_argument("--data-dir", required=True)
    p_eval.add_argument("--model-dir", required=True)
    p_eval.add_argument("--output-dir", required=True)
    p_eval.add_argument("--eval-set", default="val", choices=["val", "test"])
    p_eval.add_argument("--use-vllm", action="store_true")

    # Eval-base (unmodified Qwen3-8B baseline)
    p_base = sub.add_parser("eval-base", help="Evaluate base Qwen3-8B (no SFT)")
    p_base.add_argument("--data-dir", required=True)
    p_base.add_argument("--output-dir", required=True)
    p_base.add_argument("--eval-set", default="test", choices=["val", "test"])

    # Compare
    p_cmp = sub.add_parser("compare", help="Compare all models + kill criteria")
    p_cmp.add_argument("--eval-dir", required=True)
    p_cmp.add_argument("--eval-set", default="test", choices=["val", "test"])

    args = parser.parse_args()
    if args.command == "prep":
        cmd_prep(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "eval-base":
        cmd_eval_base(args)
    elif args.command == "compare":
        cmd_compare(args)


if __name__ == "__main__":
    main()
