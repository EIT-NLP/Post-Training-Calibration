#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone Pass@K (paper-style unbiased estimator) over a directory/glob of pkls.
Computes: Pass@1,2,4,8,16,32,64,128,256 (macro average over questions).

Paper estimator per question:
  pass@k = 1 - C(n-c, k) / C(n, k)
where n = #samples(traces), c = #correct samples.

This script supports two labeling modes:
1) Prefer trace["is_correct"] if present.
2) Else fallback to compare trace["extracted_answer"] vs data["ground_truth"] using equal_func.

Expected pkl schema (one question per pkl):
  data = {
    "qid": ...,
    "ground_truth": "...",          # optional if is_correct exists
    "all_traces": [
        {"extracted_answer": "...", "is_correct": bool, ...},
        ...
    ]
  }
"""

import argparse
import glob
import os
import pickle
from collections import Counter
from typing import List, Optional, Tuple

import numpy as np

# Optional: better math equivalence if available
try:
    from dynasor.core.evaluator import math_equal as _math_equal
except Exception:
    _math_equal = None


PASS_KS = [1, 2, 4, 8, 16, 32, 64, 128, 256]


# -----------------------------
# Utilities
# -----------------------------
def expand_inputs(inputs: List[str]) -> List[str]:
    paths: List[str] = []
    for inp in inputs:
        if os.path.isdir(inp):
            paths.extend(sorted(glob.glob(os.path.join(inp, "*.pkl"))))
        elif any(ch in inp for ch in ["*", "?", "["]):
            paths.extend(sorted(glob.glob(inp)))
        else:
            paths.append(inp)

    # dedup while preserving order
    seen = set()
    out: List[str] = []
    for p in paths:
        ap = os.path.abspath(p)
        if ap not in seen and os.path.isfile(ap):
            seen.add(ap)
            out.append(ap)
    return out


def quick_parse(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    while "\\text{" in text:
        start = text.find("\\text{")
        if start == -1:
            break
        end = text.find("}", start)
        if end == -1:
            break
        content = text[start + 6 : end]
        text = text[:start] + content + text[end + 1 :]
    return text


def equal_func(answer: str, ground_truth: str) -> bool:
    a = quick_parse(answer).strip()
    g = str(ground_truth).strip()

    # single-letter multiple choice normalization
    if len(a) == 1 and a.isalpha() and len(g) == 1 and g.isalpha():
        return a.lower() == g.lower()

    if _math_equal is not None:
        try:
            return bool(_math_equal(a, g))
        except Exception:
            pass

    return a == g


# -----------------------------
# Pass@k estimator (stable product form)
# pass@k = 1 - C(n-c, k) / C(n, k)
# -----------------------------
def pass_at_k(n: int, c: int, k: int) -> float:
    if k <= 0:
        return 0.0
    if c <= 0:
        return 0.0
    if k >= n:
        return 1.0 if c > 0 else 0.0
    if (n - c) < k:
        return 1.0
    # stable product form used in many codebases:
    # 1 - Π_{i=n-c+1..n} (1 - k/i)
    arr = np.arange(n - c + 1, n + 1, dtype=np.float64)
    return float(1.0 - np.prod(1.0 - (k / arr)))


# -----------------------------
# Extract labels y for a question
# -----------------------------
def extract_trace_labels(
    data: dict,
    include_empty_as_incorrect: bool,
) -> Optional[np.ndarray]:
    traces = data.get("all_traces", [])
    if not isinstance(traces, list) or len(traces) == 0:
        return None

    gt = str(data.get("ground_truth", "")).strip()
    has_any_is_correct = any(isinstance(tr, dict) and ("is_correct" in tr) for tr in traces)

    # need either ground truth or per-trace is_correct
    if gt == "" and not has_any_is_correct:
        return None

    y_list: List[float] = []
    for tr in traces:
        if not isinstance(tr, dict):
            continue

        ans = tr.get("extracted_answer", None)

        # empty answer handling
        if ans is None or str(ans).strip() == "":
            if not include_empty_as_incorrect:
                continue
            # treat as incorrect unless explicit is_correct exists
            if tr.get("is_correct", None) is not None:
                y_list.append(1.0 if bool(tr["is_correct"]) else 0.0)
            else:
                y_list.append(0.0)
            continue

        # label source priority: trace["is_correct"] if present else compare with gt
        if tr.get("is_correct", None) is not None:
            y_list.append(1.0 if bool(tr["is_correct"]) else 0.0)
        else:
            if gt == "":
                continue
            y_list.append(1.0 if equal_func(str(ans).strip(), gt) else 0.0)

    if len(y_list) == 0:
        return None
    return np.asarray(y_list, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser(
        description="Compute Pass@K (paper-style unbiased estimator) for per-question PKLs."
    )
    ap.add_argument("inputs", nargs="+", help="Directory / glob / pkl files")
    ap.add_argument("--include_empty_as_incorrect", action="store_true")
    ap.add_argument(
        "--require_full_n",
        action="store_true",
        help="Paper-style: require each question has n >= max(K). If not, skip that question.",
    )
    args = ap.parse_args()

    ks = PASS_KS
    max_k = max(ks)

    pkl_paths = expand_inputs(args.inputs)
    if len(pkl_paths) == 0:
        raise SystemExit("No pkl files found from inputs.")

    used = 0
    skipped = 0
    ns: List[int] = []
    per_k_vals = {k: [] for k in ks}

    for path in pkl_paths:
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"[SKIP] bad/empty pickle: {path} ({type(e).__name__})")
            skipped += 1
            continue
        except Exception as e:
            print(f"[SKIP] failed to read: {path} ({type(e).__name__}: {e})")
            skipped += 1
            continue

        y = extract_trace_labels(data, include_empty_as_incorrect=args.include_empty_as_incorrect)
        if y is None:
            skipped += 1
            continue

        n = int(y.size)
        c = int(np.sum(y))

        if args.require_full_n and n < max_k:
            skipped += 1
            continue

        used += 1
        ns.append(n)

        for k in ks:
            if (not args.require_full_n) and (n < k):
                # cannot compute pass@k for this question
                continue
            per_k_vals[k].append(pass_at_k(n, c, k))

    if used == 0:
        raise SystemExit("No usable questions (all skipped).")

    print("\n================ Pass@K Summary ================")
    print(f"Found pkls: {len(pkl_paths)} | Used questions: {used} | Skipped: {skipped}")
    print(f"include_empty_as_incorrect: {bool(args.include_empty_as_incorrect)}")
    print(f"require_full_n (n >= {max_k}): {bool(args.require_full_n)}")
    print(f"Traces per used question: mean={float(np.mean(ns)):.2f} min={int(np.min(ns))} max={int(np.max(ns))}")

    print("\n--- Pass@k (macro average over questions) ---")
    for k in ks:
        vals = per_k_vals[k]
        if len(vals) == 0:
            print(f"pass@{k}: NaN (no questions had n >= {k})")
        else:
            print(f"pass@{k}: {float(np.mean(vals)):.6f}")

    print("===============================================")


if __name__ == "__main__":
    main()

'''
Example:
python compute_metrics_new.py runs/offline_pkls/sft_aime24 --include_empty_as_incorrect
'''
