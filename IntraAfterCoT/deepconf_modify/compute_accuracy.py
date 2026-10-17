#!/usr/bin/env python3
"""
Sample valid traces from DeepConf-style PKL files and compute repeated accuracy.

Definition used here:
1) valid trace: `extracted_answer` exists and is non-empty
2) per repeat:
   - sample `sample_size` traces for each question
   - compute question accuracy = mean(correctness of sampled traces)
   - compute repeat accuracy = macro average over questions
3) repeat the above `num_repeats` times

conda activate deepconf
cd /path/to/this_repo/code/IntraAfterCoT/deepconf_modify

base
python compute_accuracy.py --inputs runs/offline_pkls/instruct_aime24 --sample_with_replacement

========== Sampled Accuracy Summary ==========
PKLs found: 30
Usable questions: 30
Bad files: 0
Unusable files: 0
sample_size: 5
num_repeats: 5
sample_with_replacement: True
---------------------------------------------
Repeat 1: accuracy=0.093333 (questions=30)
Repeat 2: accuracy=0.040000 (questions=30)
Repeat 3: accuracy=0.080000 (questions=30)
Repeat 4: accuracy=0.106667 (questions=30)
Repeat 5: accuracy=0.106667 (questions=30)
---------------------------------------------
Mean accuracy over repeats: 0.085333
Std  accuracy over repeats: 0.024730
=============================================





inst
python compute_accuracy.py --inputs runs/offline_pkls/sft_aime24 --sample_with_replacement
========== Sampled Accuracy Summary ==========
PKLs found: 30
Usable questions: 30
Bad files: 0
Unusable files: 0
sample_size: 5
num_repeats: 5
sample_with_replacement: True
---------------------------------------------
Repeat 1: accuracy=0.140000 (questions=30)
Repeat 2: accuracy=0.140000 (questions=30)
Repeat 3: accuracy=0.113333 (questions=30)
Repeat 4: accuracy=0.113333 (questions=30)
Repeat 5: accuracy=0.146667 (questions=30)
---------------------------------------------
Mean accuracy over repeats: 0.130667
Std  accuracy over repeats: 0.014360
=============================================



sft
python compute_accuracy.py --inputs runs/offline_pkls/sft_aime24 --sample_with_replacement

========== Sampled Accuracy Summary ==========
PKLs found: 30
Usable questions: 30
Bad files: 0
Unusable files: 0
sample_size: 5
num_repeats: 5
sample_with_replacement: True
---------------------------------------------
Repeat 1: accuracy=0.173333 (questions=30)
Repeat 2: accuracy=0.166667 (questions=30)
Repeat 3: accuracy=0.200000 (questions=30)
Repeat 4: accuracy=0.146667 (questions=30)
Repeat 5: accuracy=0.140000 (questions=30)
---------------------------------------------
Mean accuracy over repeats: 0.165333
Std  accuracy over repeats: 0.021250
=============================================



rl
python compute_accuracy.py --inputs runs/offline_pkls/rl_aime24 --sample_with_replacement
========== Sampled Accuracy Summary ==========
PKLs found: 30
Usable questions: 30
Bad files: 0
Unusable files: 0
sample_size: 5
num_repeats: 5
sample_with_replacement: True
---------------------------------------------
Repeat 1: accuracy=0.100000 (questions=30)
Repeat 2: accuracy=0.140000 (questions=30)
Repeat 3: accuracy=0.126667 (questions=30)
Repeat 4: accuracy=0.093333 (questions=30)
Repeat 5: accuracy=0.120000 (questions=30)
---------------------------------------------
Mean accuracy over repeats: 0.116000
Std  accuracy over repeats: 0.017179
=============================================



--sample_size：每题采样条数（默认 5）
--num_repeats：重复次数（默认 5）
--seed：随机种子（默认 42）
--sample_with_replacement：当某题 valid trace 少于 sample_size 时允许有放回采样（默认不开启，默认会跳过该题）
--json_output result.json：保存结果到 JSON


"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import random
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import List, Optional


try:
    from dynasor.core.evaluator import math_equal as _math_equal
except Exception:
    _math_equal = None


@dataclass
class QuestionSamples:
    path: str
    qid: str
    labels: List[int]


def expand_inputs(inputs: List[str]) -> List[str]:
    paths: List[str] = []
    for inp in inputs:
        if os.path.isdir(inp):
            paths.extend(sorted(glob.glob(os.path.join(inp, "*.pkl"))))
        elif any(ch in inp for ch in ["*", "?", "["]):
            paths.extend(sorted(glob.glob(inp)))
        else:
            paths.append(inp)

    out: List[str] = []
    seen = set()
    for p in paths:
        ap = os.path.abspath(p)
        if os.path.isfile(ap) and ap not in seen:
            out.append(ap)
            seen.add(ap)
    return out


def quick_parse(text: str) -> str:
    out = str(text)
    while "\\text{" in out:
        st = out.find("\\text{")
        if st < 0:
            break
        ed = out.find("}", st)
        if ed < 0:
            break
        out = out[:st] + out[st + 6 : ed] + out[ed + 1 :]
    return out


def equal_func(answer: str, ground_truth: str) -> bool:
    a = quick_parse(answer).strip()
    g = str(ground_truth).strip()
    if len(a) == 1 and a.isalpha() and len(g) == 1 and g.isalpha():
        return a.lower() == g.lower()
    if _math_equal is not None:
        try:
            return bool(_math_equal(a, g))
        except Exception:
            pass
    return a == g


def _trace_answer(trace: dict) -> Optional[str]:
    ans = trace.get("extracted_answer")
    if ans is None:
        return None
    ans = str(ans).strip()
    if not ans:
        return None
    return ans


def build_question_samples(path: str) -> Optional[QuestionSamples]:
    with open(path, "rb") as f:
        record = pickle.load(f)

    traces = record.get("all_traces", [])
    if not isinstance(traces, list) or not traces:
        return None

    gt = str(record.get("ground_truth", "")).strip()
    qid = str(record.get("qid", os.path.basename(path)))

    labels: List[int] = []
    for tr in traces:
        if not isinstance(tr, dict):
            continue
        ans = _trace_answer(tr)
        if ans is None:
            continue

        if tr.get("is_correct", None) is not None:
            labels.append(1 if bool(tr["is_correct"]) else 0)
            continue

        if gt:
            labels.append(1 if equal_func(ans, gt) else 0)

    if not labels:
        return None

    return QuestionSamples(path=path, qid=qid, labels=labels)


def sample_labels(
    labels: List[int],
    sample_size: int,
    rng: random.Random,
    sample_with_replacement: bool,
) -> Optional[List[int]]:
    n = len(labels)
    if n == 0:
        return None

    if n >= sample_size:
        return rng.sample(labels, sample_size)

    if sample_with_replacement:
        return [labels[rng.randrange(n)] for _ in range(sample_size)]

    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "From DeepConf PKL folder: sample valid traces per question, "
            "compute repeated macro accuracy."
        )
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="PKL folder / glob / files",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=5,
        help="How many valid traces to sample per question (default: 5)",
    )
    parser.add_argument(
        "--num_repeats",
        type=int,
        default=5,
        help="How many repeated runs (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42). Repeat i uses seed+i.",
    )
    parser.add_argument(
        "--sample_with_replacement",
        action="store_true",
        help="If a question has <sample_size valid traces, sample with replacement.",
    )
    parser.add_argument(
        "--json_output",
        default=None,
        help="Optional path to save JSON summary.",
    )
    args = parser.parse_args()

    if args.sample_size <= 0:
        raise SystemExit("--sample_size must be > 0")
    if args.num_repeats <= 0:
        raise SystemExit("--num_repeats must be > 0")

    pkl_paths = expand_inputs(args.inputs)
    if not pkl_paths:
        raise SystemExit("No PKL files found from --inputs.")

    questions: List[QuestionSamples] = []
    bad_files = 0
    unusable_files = 0
    for p in pkl_paths:
        try:
            item = build_question_samples(p)
        except Exception as e:
            print(f"[SKIP] failed to read {p} ({type(e).__name__}: {e})")
            bad_files += 1
            continue
        if item is None:
            unusable_files += 1
            continue
        questions.append(item)

    if not questions:
        raise SystemExit("No usable questions (no evaluable valid traces).")

    insufficient = sum(1 for q in questions if len(q.labels) < args.sample_size)
    if insufficient > 0 and not args.sample_with_replacement:
        kept = [q for q in questions if len(q.labels) >= args.sample_size]
        if not kept:
            raise SystemExit(
                "No question has enough valid traces for sampling. "
                "Try --sample_with_replacement."
            )
        questions = kept

    repeat_accs: List[float] = []
    repeat_used_questions: List[int] = []

    for i in range(args.num_repeats):
        rng = random.Random(args.seed + i)
        per_question_accs: List[float] = []
        for q in questions:
            sampled = sample_labels(
                labels=q.labels,
                sample_size=args.sample_size,
                rng=rng,
                sample_with_replacement=args.sample_with_replacement,
            )
            if sampled is None:
                continue
            per_question_accs.append(sum(sampled) / len(sampled))

        if not per_question_accs:
            raise SystemExit(f"Repeat {i + 1} has no usable questions.")

        repeat_used_questions.append(len(per_question_accs))
        repeat_accs.append(mean(per_question_accs))

    overall_mean = mean(repeat_accs)
    overall_std = pstdev(repeat_accs) if len(repeat_accs) > 1 else 0.0

    print("\n========== Sampled Accuracy Summary ==========")
    print(f"PKLs found: {len(pkl_paths)}")
    print(f"Usable questions: {len(questions)}")
    print(f"Bad files: {bad_files}")
    print(f"Unusable files: {unusable_files}")
    print(f"sample_size: {args.sample_size}")
    print(f"num_repeats: {args.num_repeats}")
    print(f"sample_with_replacement: {bool(args.sample_with_replacement)}")
    print("---------------------------------------------")
    for i, (acc, used_q) in enumerate(zip(repeat_accs, repeat_used_questions), start=1):
        print(f"Repeat {i}: accuracy={acc:.6f} (questions={used_q})")
    print("---------------------------------------------")
    print(f"Mean accuracy over repeats: {overall_mean:.6f}")
    print(f"Std  accuracy over repeats: {overall_std:.6f}")
    print("=============================================")

    if args.json_output:
        out = {
            "meta": {
                "num_pkls_found": len(pkl_paths),
                "num_usable_questions": len(questions),
                "num_bad_files": bad_files,
                "num_unusable_files": unusable_files,
                "sample_size": int(args.sample_size),
                "num_repeats": int(args.num_repeats),
                "seed": int(args.seed),
                "sample_with_replacement": bool(args.sample_with_replacement),
            },
            "repeat_results": [
                {
                    "repeat_id": i + 1,
                    "accuracy": float(acc),
                    "num_questions_used": int(used_q),
                }
                for i, (acc, used_q) in enumerate(zip(repeat_accs, repeat_used_questions))
            ],
            "summary": {
                "mean_accuracy": float(overall_mean),
                "std_accuracy": float(overall_std),
            },
        }

        out_path = os.path.abspath(args.json_output)
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Wrote JSON summary to: {out_path}")


if __name__ == "__main__":
    main()
