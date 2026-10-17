#!/usr/bin/env python3
"""
Compute Pass@K from DeepConf-style per-question PKL files.

Default K list:
  1, 4, 8, 16, 32, 64, 128, 256

Usage examples:
  python compute_passk_from_pkls.py --inputs outputs
  python compute_passk_from_pkls.py --inputs "outputs/*.pkl"
  python compute_passk_from_pkls.py --inputs outputs --ks 1,2,4,8,16,32,64,128,256
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pickle
from typing import Dict, List, Optional


try:
    from dynasor.core.evaluator import math_equal as _math_equal
except Exception:
    _math_equal = None


DEFAULT_KS = [1, 4, 8, 16, 32, 64, 128, 256]


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


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased estimator used in pass@k literature:
      pass@k = 1 - C(n-c, k) / C(n, k)
    """
    if k <= 0:
        return 0.0
    if c <= 0:
        return 0.0
    if k >= n:
        return 1.0 if c > 0 else 0.0
    if (n - c) < k:
        return 1.0

    # Stable product form:
    # C(n-c,k)/C(n,k) = Π_{i=0..k-1} (n-c-i)/(n-i)
    prod = 1.0
    for i in range(k):
        prod *= (n - c - i) / (n - i)
    return 1.0 - prod


def extract_trace_labels(record: Dict, include_empty_as_incorrect: bool) -> Optional[List[int]]:
    traces = record.get("all_traces", [])
    if not isinstance(traces, list) or len(traces) == 0:
        return None

    ground_truth = str(record.get("ground_truth", "")).strip()
    has_any_is_correct = any(isinstance(tr, dict) and ("is_correct" in tr) for tr in traces)
    if not ground_truth and not has_any_is_correct:
        return None

    labels: List[int] = []
    for tr in traces:
        if not isinstance(tr, dict):
            continue

        ans = tr.get("extracted_answer")
        if ans is None or str(ans).strip() == "":
            if include_empty_as_incorrect:
                if tr.get("is_correct", None) is not None:
                    labels.append(1 if bool(tr["is_correct"]) else 0)
                else:
                    labels.append(0)
            continue

        if tr.get("is_correct", None) is not None:
            labels.append(1 if bool(tr["is_correct"]) else 0)
            continue

        if ground_truth:
            labels.append(1 if equal_func(str(ans), ground_truth) else 0)

    if not labels:
        return None
    return labels


def parse_ks(ks_str: str) -> List[int]:
    if not ks_str.strip():
        return DEFAULT_KS
    ks = []
    for x in ks_str.split(","):
        x = x.strip()
        if not x:
            continue
        v = int(x)
        if v <= 0:
            raise ValueError(f"Invalid K: {v}")
        ks.append(v)
    ks = sorted(set(ks))
    if not ks:
        raise ValueError("No valid Ks parsed.")
    return ks


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Pass@K from PKL files.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Dirs/globs/files for PKLs")
    parser.add_argument(
        "--ks",
        default="1,4,8,16,32,64,128,256",
        help="Comma-separated K list, e.g. 1,4,8,16,32,64,128,256",
    )
    parser.add_argument("--include_empty_as_incorrect", action="store_true")
    parser.add_argument(
        "--require_full_n",
        action="store_true",
        help="If set, only use questions where n >= max(K).",
    )
    parser.add_argument(
        "--json_output",
        default=None,
        help="Optional path to write a simple JSON summary.",
    )
    args = parser.parse_args()

    ks = parse_ks(args.ks)
    max_k = max(ks)
    paths = expand_inputs(args.inputs)
    if not paths:
        raise SystemExit("No PKL files found.")

    per_k_values: Dict[int, List[float]] = {k: [] for k in ks}
    used = 0
    skipped = 0
    ns: List[int] = []

    for p in paths:
        try:
            with open(p, "rb") as f:
                rec = pickle.load(f)
        except Exception as e:
            print(f"[SKIP] {p} ({type(e).__name__}: {e})")
            skipped += 1
            continue

        labels = extract_trace_labels(rec, include_empty_as_incorrect=args.include_empty_as_incorrect)
        if labels is None:
            skipped += 1
            continue

        n = len(labels)
        c = sum(labels)
        if args.require_full_n and n < max_k:
            skipped += 1
            continue

        used += 1
        ns.append(n)
        for k in ks:
            if (not args.require_full_n) and n < k:
                continue
            per_k_values[k].append(pass_at_k(n, c, k))

    if used == 0:
        raise SystemExit("No usable questions after filtering.")

    print("\n================ Pass@K Summary ================")
    print(f"PKLs found: {len(paths)} | Used questions: {used} | Skipped: {skipped}")
    print(f"Ks: {ks}")
    print(f"include_empty_as_incorrect: {bool(args.include_empty_as_incorrect)}")
    print(f"require_full_n (n >= {max_k}): {bool(args.require_full_n)}")
    print(f"Traces/question: mean={sum(ns)/len(ns):.2f} min={min(ns)} max={max(ns)}")
    print("\n--- Macro Pass@K (average over questions) ---")
    passk_summary: Dict[str, Optional[float]] = {}
    for k in ks:
        vals = per_k_values[k]
        if not vals:
            print(f"Pass@{k}: NaN (no question had n >= {k})")
            passk_summary[f"pass@{k}"] = None
        else:
            score = sum(vals) / len(vals)
            print(f"Pass@{k}: {score:.6f}")
            passk_summary[f"pass@{k}"] = float(score)
    print("===============================================")

    if args.json_output:
        per_k_details: Dict[str, Dict[str, Optional[float]]] = {}
        for k in ks:
            vals = per_k_values[k]
            key = f"pass@{k}"
            if not vals:
                per_k_details[key] = {
                    "macro_pass_at_k": None,
                    "numerator_sum_pass_estimator": 0.0,
                    "denominator_question_count": 0.0,
                    "formula": "macro_pass@k = sum(per-question pass@k estimator) / num_questions_with_n_ge_k",
                }
                continue

            per_k_details[key] = {
                "macro_pass_at_k": float(sum(vals) / len(vals)),
                "numerator_sum_pass_estimator": float(sum(vals)),
                "denominator_question_count": float(len(vals)),
                "formula": "macro_pass@k = sum(per-question pass@k estimator) / num_questions_with_n_ge_k",
            }

        out = {
            "meta": {
                "num_pkls_found": len(paths),
                "num_questions_used": used,
                "num_questions_skipped": skipped,
                "ks": ks,
                "include_empty_as_incorrect": bool(args.include_empty_as_incorrect),
                "require_full_n": bool(args.require_full_n),
                "trace_count_mean": float(sum(ns) / len(ns)),
                "trace_count_min": int(min(ns)),
                "trace_count_max": int(max(ns)),
                "per_k_details": per_k_details,
            },
            "results": passk_summary,
        }
        out_path = os.path.abspath(args.json_output)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"[INFO] Wrote JSON summary: {out_path}")


if __name__ == "__main__":
    main()
