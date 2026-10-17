#!/usr/bin/env python3
"""
Post-hoc DeepConf analysis from offline SFT traces.

This script reads `deepthink_offline_qid*.pkl` files and computes:

1) DeepConf-online style sweep (post-hoc simulation)
   - warmup_traces=32 (configurable)
   - no consensus stop
   - adaptive group length per question from warmup traces: round(mean_trace_len / divisor)
   - sweep confidence thresholds derived from warmup traces

2) DeepConf-offline confidence subsets over repeated subsamples
   - majority voting
   - most-confidence
   - top 5% confidence
   - top 10% confidence
   - trace mean confidence (new)

Outputs:
- a full JSON (`--output_json`)
- a concise dataset-level JSON (`--dataset_output_json`)


python examples/analyze_sft_deepconf.py \
  --results_dir runs/offline_pkls/sft_aime24 \
  --dataset examples/aime_2024.jsonl \
  --online_warmup_traces 32 \
  --online_total_budget 288 \
  --adaptive_divisor 10 \
  --online_sweep_points 50 \
  --online_resamples 5 \
  --offline_sample_size 256 \
  --offline_resamples 5 \
  --output_json runs/posthoc/sft_aime24_deepconf_full.json \
  --dataset_output_json runs/posthoc/sft_aime24_deepconf_dataset_view.json \
  --keep_per_case


"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

try:
    from dynasor.core.evaluator import math_equal as _math_equal
except Exception:
    _math_equal = None


OFFLINE_PATTERN = re.compile(
    r"^deepthink_offline_qid(?P<qid>\d+)_rid(?P<rid>.+)_(?P<ts>\d{8}_\d{6})\.pkl$"
)


def quick_parse(text: str) -> str:
    if text is None:
        return ""
    out = str(text)
    while "\\text{" in out:
        start = out.find("\\text{")
        if start < 0:
            break
        end = out.find("}", start)
        if end < 0:
            break
        content = out[start + 6 : end]
        out = out[:start] + content + out[end + 1 :]
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


def parse_filename(path: Path) -> Optional[Tuple[int, str, str]]:
    m = OFFLINE_PATTERN.match(path.name)
    if not m:
        return None
    return int(m.group("qid")), m.group("rid"), m.group("ts")


def load_ground_truth_map(dataset_path: Optional[Path]) -> Dict[int, str]:
    if dataset_path is None:
        return {}
    gt_map: Dict[int, str] = {}
    with dataset_path.open("r", encoding="utf-8") as f:
        for qid, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            gt_map[qid] = str(record.get("answer", "")).strip()
    return gt_map


def pick_latest_files(
    results_dir: Path,
    rid_filter: Optional[str],
    max_qid: Optional[int],
) -> List[Tuple[int, str, Path]]:
    by_qid_rid: Dict[Tuple[int, str], Tuple[str, Path]] = {}

    for pkl_path in sorted(results_dir.glob("deepthink_offline_qid*_rid*.pkl")):
        parsed = parse_filename(pkl_path)
        if parsed is None:
            continue
        qid, rid, ts = parsed
        if max_qid is not None and qid > max_qid:
            continue
        if rid_filter and rid != rid_filter:
            continue
        key = (qid, rid)
        current = by_qid_rid.get(key)
        if current is None or ts > current[0]:
            by_qid_rid[key] = (ts, pkl_path)

    return [
        (qid, rid, item[1])
        for (qid, rid), item in sorted(by_qid_rid.items(), key=lambda x: (x[0][0], x[0][1]))
    ]


def token_count(trace: Dict[str, Any]) -> int:
    if isinstance(trace.get("num_tokens"), int):
        return max(0, int(trace["num_tokens"]))
    token_ids = trace.get("token_ids")
    if isinstance(token_ids, list):
        return len(token_ids)
    confs = trace.get("confs")
    if isinstance(confs, list):
        return len(confs)
    return 0


def adaptive_group_size(traces: List[Dict[str, Any]], divisor: float) -> int:
    lengths = [token_count(t) for t in traces if token_count(t) > 0]
    if not lengths:
        return 1
    mean_len = float(mean(lengths))
    group = int(round(mean_len / divisor))
    return max(1, group)


def min_group_conf(trace: Dict[str, Any], group_size: int) -> float:
    confs = trace.get("confs") or []
    if not confs:
        return float("-inf")
    if len(confs) < group_size:
        return float(sum(confs) / len(confs))

    window_sum = sum(confs[:group_size])
    best = window_sum / group_size
    for idx in range(group_size, len(confs)):
        window_sum += confs[idx] - confs[idx - group_size]
        mean_val = window_sum / group_size
        if mean_val < best:
            best = mean_val
    return float(best)

def tail_fraction_trace_conf(trace: Dict[str, Any], tail_ratio: float = 0.1) -> float:
    confs = trace.get("confs") or []
    if not confs:
        return float("-inf")

    tail_len = max(1, int(math.ceil(len(confs) * tail_ratio)))
    tail_confs = confs[-tail_len:]
    return float(sum(tail_confs) / len(tail_confs))


def mean_trace_conf(trace: Dict[str, Any]) -> float:
    confs = trace.get("confs") or []
    if not confs:
        return float("-inf")
    return float(sum(confs) / len(confs))


def stop_token_count(
    trace: Dict[str, Any],
    threshold: float,
    group_size: int,
) -> int:
    confs = trace.get("confs") or []
    full_tokens = token_count(trace)
    if not confs or full_tokens <= 0:
        return full_tokens

    if len(confs) < group_size:
        return full_tokens

    window_sum = sum(confs[:group_size])
    if window_sum / group_size < threshold:
        return min(full_tokens, group_size)

    for idx in range(group_size, len(confs)):
        window_sum += confs[idx] - confs[idx - group_size]
        if window_sum / group_size < threshold:
            return min(full_tokens, idx + 1)

    return full_tokens


def simple_majority_vote(answers: List[str]) -> Optional[str]:
    if not answers:
        return None
    counts: Dict[str, int] = defaultdict(int)
    for answer in answers:
        counts[str(answer)] += 1
    return max(counts.items(), key=lambda x: x[1])[0]


def weighted_majority_vote(scored_answers: List[Tuple[float, str]]) -> Optional[str]:
    """
    Weighted majority vote by summing confidence scores for each answer.

    Each item is (weight, answer), where higher weight means higher confidence.
    Non-finite or negative weights are clamped to 0.0.

    Tie-break:
    1) larger total weight
    2) larger vote count
    3) lexicographically smaller answer
    """
    if not scored_answers:
        return None

    total_weight: Dict[str, float] = defaultdict(float)
    vote_count: Dict[str, int] = defaultdict(int)

    for weight, answer in scored_answers:
        ans = str(answer)
        try:
            w = float(weight)
        except Exception:
            w = 0.0

        if not math.isfinite(w) or w < 0:
            w = 0.0

        total_weight[ans] += w
        vote_count[ans] += 1

    if not total_weight:
        return None

    return sorted(
        total_weight.keys(),
        key=lambda ans: (-total_weight[ans], -vote_count[ans], ans)
    )[0]


def evaluate_answer(pred: Optional[str], gt: str) -> bool:
    if pred is None:
        return False
    try:
        return equal_func(str(pred), gt)
    except Exception:
        return str(pred).strip() == str(gt).strip()


def trace_answer(trace: Dict[str, Any]) -> Optional[str]:
    ans = trace.get("extracted_answer")
    if ans is None:
        return None
    ans_str = str(ans).strip()
    return ans_str if ans_str else None


def trace_is_correct(trace: Dict[str, Any], gt: str) -> bool:
    ans = trace_answer(trace)
    return evaluate_answer(ans, gt)


def safe_ratio(num: int, den: int) -> float:
    return float(num / den) if den > 0 else 0.0


def quantile_thresholds(values: List[float], points: int) -> List[float]:
    uniq = sorted(set(float(v) for v in values if math.isfinite(float(v))))
    if not uniq:
        return []
    if points <= 1:
        mid = len(uniq) // 2
        if len(uniq) % 2 == 1:
            return [float(uniq[mid])]
        return [float((uniq[mid - 1] + uniq[mid]) / 2.0)]

    def quantile_linear(sorted_values: List[float], q: float) -> float:
        if q <= 0.0:
            return float(sorted_values[0])
        if q >= 1.0:
            return float(sorted_values[-1])
        pos = q * (len(sorted_values) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return float(sorted_values[lo])
        frac = pos - lo
        return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)

    quantiles = [i / (points - 1) for i in range(points)]
    return [quantile_linear(uniq, q) for q in quantiles]


def run_online_sweep_for_question(
    traces: List[Dict[str, Any]],
    ground_truth: str,
    qid: int,
    rng: random.Random,
    warmup_traces: int,
    total_budget: int,
    sweep_points: int,
    adaptive_divisor: float,
    resamples: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if len(traces) < max(2, warmup_traces + 1):
        return rows

    budget = min(total_budget, len(traces))

    for sample_id in range(resamples):
        picked_idx = rng.sample(range(len(traces)), budget)
        picked = [traces[i] for i in picked_idx]
        rng.shuffle(picked)

        warm = picked[:warmup_traces]
        final = picked[warmup_traces:]
        if not warm or not final:
            continue

        group_size = adaptive_group_size(warm, adaptive_divisor)
        warm_scores = [min_group_conf(t, group_size) for t in warm]
        thresholds = quantile_thresholds(warm_scores, sweep_points)
        if not thresholds:
            continue

        warm_scores_map = [min_group_conf(t, group_size) for t in warm]
        final_scores_map = [min_group_conf(t, group_size) for t in final]

        full_tokens = sum(token_count(t) for t in picked)
        warm_tokens = sum(token_count(t) for t in warm)

        # no-early-stop baseline: all sampled paths are counted
        # path without extracted_answer is treated as incorrect
        no_early_stop_path_count = len(picked)
        no_early_stop_correct_path_count = sum(
            1 for trace in picked if trace_is_correct(trace, ground_truth)
        )

        for sweep_idx, threshold in enumerate(thresholds):
            voting_answers: List[str] = []

            surviving_path_count = 0
            surviving_correct_path_count = 0
            surviving_valid_answer_count = 0
            surviving_valid_correct_path_count = 0

            # warm traces: not early-stopped, but still filtered by threshold
            for trace, score in zip(warm, warm_scores_map):
                if score < threshold:
                    continue

                is_correct = trace_is_correct(trace, ground_truth)
                surviving_path_count += 1
                if is_correct:
                    surviving_correct_path_count += 1

                ans = trace_answer(trace)
                if ans is not None:
                    surviving_valid_answer_count += 1
                    if is_correct:
                        surviving_valid_correct_path_count += 1
                    voting_answers.append(ans)

            final_tokens = 0
            stopped_final = 0

            for trace, score in zip(final, final_scores_map):
                final_tokens += stop_token_count(trace, threshold, group_size)

                if score < threshold:
                    stopped_final += 1
                    continue

                is_correct = trace_is_correct(trace, ground_truth)
                surviving_path_count += 1
                if is_correct:
                    surviving_correct_path_count += 1

                ans = trace_answer(trace)
                if ans is not None:
                    surviving_valid_answer_count += 1
                    if is_correct:
                        surviving_valid_correct_path_count += 1
                    voting_answers.append(ans)

            pred = simple_majority_vote(voting_answers)
            majority_vote_is_correct = evaluate_answer(pred, ground_truth)

            total_tokens = warm_tokens + final_tokens
            token_ratio = (total_tokens / full_tokens) if full_tokens > 0 else 0.0

            rows.append(
                {
                    "qid": qid,
                    "sample_id": sample_id,
                    "sweep_index": sweep_idx,
                    "threshold": float(threshold),
                    "group_size": int(group_size),
                    "warmup_used": len(warm),
                    "final_used": len(final),
                    "final_stopped": int(stopped_final),
                    "total_tokens": int(total_tokens),
                    "full_tokens": int(full_tokens),
                    "token_ratio": float(token_ratio),

                    # old majority-vote outputs kept for comparison
                    "has_prediction": bool(pred is not None),
                    "is_correct": bool(majority_vote_is_correct),
                    "majority_vote_has_prediction": bool(pred is not None),
                    "majority_vote_is_correct": bool(majority_vote_is_correct),

                    # new path-level outputs
                    "surviving_path_count": int(surviving_path_count),
                    "surviving_correct_path_count": int(surviving_correct_path_count),
                    "surviving_path_accuracy": safe_ratio(
                        surviving_correct_path_count, surviving_path_count
                    ),
                    "surviving_valid_answer_count": int(surviving_valid_answer_count),
                    "surviving_valid_correct_path_count": int(surviving_valid_correct_path_count),
                    "valid_answer_accuracy": safe_ratio(
                        surviving_valid_correct_path_count, surviving_valid_answer_count
                    ),

                    "no_early_stop_path_count": int(no_early_stop_path_count),
                    "no_early_stop_correct_path_count": int(no_early_stop_correct_path_count),
                    "no_early_stop_path_accuracy": safe_ratio(
                        no_early_stop_correct_path_count, no_early_stop_path_count
                    ),

                    "surviving_path_keep_rate": safe_ratio(
                        surviving_path_count, no_early_stop_path_count
                    ),
                }
            )

    return rows


def run_offline_confidence_for_question(
    traces: List[Dict[str, Any]],
    ground_truth: str,
    qid: int,
    rng: random.Random,
    sample_size: int,
    adaptive_divisor: float,
    resamples: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if len(traces) < 2:
        return rows

    n = min(sample_size, len(traces))
    methods = [
        "majority_voting",
        "most_confidence",
        "top5_confidence_valid",
        "top10_confidence_valid",
        "top5_confidence_base256",
        "top10_confidence_base256",
    ]

    for sample_id in range(resamples):
        picked_idx = rng.sample(range(len(traces)), n)
        picked = [traces[i] for i in picked_idx]

        group_size = adaptive_group_size(picked, adaptive_divisor)

        # ---------- trace-level statistics ----------
        sample_size_used = len(picked)  # usually 256

        # base-count version: denominator = sample_size_used
        # trace without extracted_answer is counted as incorrect
        base_correct_trace_count = sum(
            1 for trace in picked if trace_is_correct(trace, ground_truth)
        )
        trace_accuracy_base_count = safe_ratio(
            base_correct_trace_count, sample_size_used
        )

        # valid-only version: denominator = number of traces with extracted_answer
        valid_mean: List[Tuple[float, str]] = []
        valid_answer_count = 0
        valid_correct_trace_count = 0

        # all-ranked version for base256 top-k selection
        all_ranked: List[Tuple[float, Optional[str]]] = []

        for trace in picked:
            score = mean_trace_conf(trace)
            # score = tail_fraction_trace_conf(trace, tail_ratio=0.1)
            
            ans = trace_answer(trace)

            all_ranked.append((score, ans))

            if ans is None:
                continue

            valid_mean.append((score, ans))
            valid_answer_count += 1

            if evaluate_answer(ans, ground_truth):
                valid_correct_trace_count += 1

        trace_accuracy_valid_only = safe_ratio(
            valid_correct_trace_count, valid_answer_count
        )
        valid_answer_ratio = safe_ratio(valid_answer_count, sample_size_used)
        # -------------------------------------------

        if not valid_mean:
            for method in methods:
                rows.append(
                    {
                        "qid": qid,
                        "sample_id": sample_id,
                        "group_size": group_size,
                        "method": method,
                        "has_prediction": False,
                        "is_correct": False,

                        # trace-level summary
                        "sample_size_used": int(sample_size_used),
                        "valid_answer_count": int(valid_answer_count),
                        "valid_correct_trace_count": int(valid_correct_trace_count),
                        "trace_accuracy_valid_only": float(trace_accuracy_valid_only),
                        "base_correct_trace_count": int(base_correct_trace_count),
                        "trace_accuracy_base_count": float(trace_accuracy_base_count),
                        "valid_answer_ratio": float(valid_answer_ratio),
                    }
                )
            continue

        # sort by confidence descending
        valid_mean.sort(key=lambda x: x[0], reverse=True)
        all_ranked.sort(key=lambda x: x[0], reverse=True)

        def vote_top_valid(
            sorted_pairs: List[Tuple[float, str]],
            frac: float,
        ) -> Optional[str]:
            k = max(1, int(math.ceil(len(sorted_pairs) * frac)))
            chosen = sorted_pairs[:k]
            scored_answers = [(score, ans) for score, ans in chosen]
            return weighted_majority_vote(scored_answers)

        def vote_top_base(
            sorted_pairs_all: List[Tuple[float, Optional[str]]],
            frac: float,
            base_n: int,
        ) -> Optional[str]:
            k = max(1, int(math.ceil(base_n * frac)))
            chosen = sorted_pairs_all[:k]
            scored_answers = [
                (score, ans) for score, ans in chosen if ans is not None
            ]
            return weighted_majority_vote(scored_answers)

        pred_most = valid_mean[0][1]
        pred_majority = simple_majority_vote([ans for _, ans in valid_mean])

        pred_top5_valid = vote_top_valid(valid_mean, 0.05)
        pred_top10_valid = vote_top_valid(valid_mean, 0.10)

        pred_top5_base256 = vote_top_base(all_ranked, 0.05, sample_size_used)
        pred_top10_base256 = vote_top_base(all_ranked, 0.10, sample_size_used)

        preds = {
            "majority_voting": pred_majority,
            "most_confidence": pred_most,
            "top5_confidence_valid": pred_top5_valid,
            "top10_confidence_valid": pred_top10_valid,
            "top5_confidence_base256": pred_top5_base256,
            "top10_confidence_base256": pred_top10_base256,
        }

        for method in methods:
            pred = preds[method]
            rows.append(
                {
                    "qid": qid,
                    "sample_id": sample_id,
                    "group_size": group_size,
                    "method": method,
                    "has_prediction": pred is not None,
                    "is_correct": evaluate_answer(pred, ground_truth),

                    # trace-level summary
                    "sample_size_used": int(sample_size_used),
                    "valid_answer_count": int(valid_answer_count),
                    "valid_correct_trace_count": int(valid_correct_trace_count),
                    "trace_accuracy_valid_only": float(trace_accuracy_valid_only),
                    "base_correct_trace_count": int(base_correct_trace_count),
                    "trace_accuracy_base_count": float(trace_accuracy_base_count),
                    "valid_answer_ratio": float(valid_answer_ratio),
                }
            )

    return rows


def aggregate_online(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_sweep: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sweep[int(row["sweep_index"])].append(row)

    summary: List[Dict[str, Any]] = []
    for sweep_idx in sorted(by_sweep.keys()):
        group = by_sweep[sweep_idx]

        surviving_path_count = sum(int(g["surviving_path_count"]) for g in group)
        surviving_correct_path_count = sum(int(g["surviving_correct_path_count"]) for g in group)
        surviving_valid_answer_count = sum(int(g["surviving_valid_answer_count"]) for g in group)
        surviving_valid_correct_path_count = sum(
            int(g["surviving_valid_correct_path_count"]) for g in group
        )

        no_early_stop_path_count = sum(int(g["no_early_stop_path_count"]) for g in group)
        no_early_stop_correct_path_count = sum(
            int(g["no_early_stop_correct_path_count"]) for g in group
        )
        total_tokens_sum = sum(int(g["total_tokens"]) for g in group)
        full_tokens_sum = sum(int(g["full_tokens"]) for g in group)

        surviving_path_accuracy = safe_ratio(
            surviving_correct_path_count, surviving_path_count
        )
        valid_answer_accuracy = safe_ratio(
            surviving_valid_correct_path_count, surviving_valid_answer_count
        )
        no_early_stop_path_accuracy = safe_ratio(
            no_early_stop_correct_path_count, no_early_stop_path_count
        )

        majority_vote_correct_count = sum(
            1 for g in group if bool(g["majority_vote_is_correct"])
        )
        majority_vote_prediction_count = sum(
            1 for g in group if bool(g["majority_vote_has_prediction"])
        )
        majority_vote_accuracy = safe_ratio(majority_vote_correct_count, len(group))
        majority_vote_answer_rate = safe_ratio(majority_vote_prediction_count, len(group))

        summary.append(
            {
                "sweep_index": sweep_idx,
                "cases": len(group),
                "mean_threshold": float(mean([g["threshold"] for g in group])),
                "mean_group_size": float(mean([g["group_size"] for g in group])),
                "mean_tokens": float(mean([g["total_tokens"] for g in group])),
                "mean_token_ratio": float(mean([g["token_ratio"] for g in group])),
                "total_tokens_sum": int(total_tokens_sum),
                "full_tokens_sum": int(full_tokens_sum),
                "global_token_ratio_by_tokens": safe_ratio(total_tokens_sum, full_tokens_sum),

                # new primary online metric
                "accuracy": float(surviving_path_accuracy),
                "surviving_path_accuracy": float(surviving_path_accuracy),
                "valid_answer_accuracy": float(valid_answer_accuracy),
                "no_early_stop_path_accuracy": float(no_early_stop_path_accuracy),

                "surviving_path_count": int(surviving_path_count),
                "surviving_correct_path_count": int(surviving_correct_path_count),
                "surviving_valid_answer_count": int(surviving_valid_answer_count),
                "surviving_valid_correct_path_count": int(surviving_valid_correct_path_count),
                "no_early_stop_path_count": int(no_early_stop_path_count),
                "no_early_stop_correct_path_count": int(no_early_stop_correct_path_count),
                "surviving_path_keep_rate": safe_ratio(
                    surviving_path_count, no_early_stop_path_count
                ),

                # old majority-vote metrics retained for comparison
                "majority_vote_accuracy": float(majority_vote_accuracy),
                "answer_rate": float(majority_vote_answer_rate),
                "majority_vote_answer_rate": float(majority_vote_answer_rate),
                "majority_vote_correct_count": int(majority_vote_correct_count),
                "majority_vote_prediction_count": int(majority_vote_prediction_count),
            }
        )
    return summary


def aggregate_offline(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_method: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[str(row["method"])].append(row)

    summary: Dict[str, Dict[str, float]] = {}
    for method, group in by_method.items():
        cases_total = len(group)
        correct_prediction_count = sum(1 for g in group if bool(g.get("is_correct", False)))
        has_prediction_count = sum(1 for g in group if bool(g.get("has_prediction", False)))

        valid_answer_count_total = sum(int(g.get("valid_answer_count", 0)) for g in group)
        valid_correct_trace_count_total = sum(
            int(g.get("valid_correct_trace_count", 0)) for g in group
        )

        sample_size_used_total = sum(int(g.get("sample_size_used", 0)) for g in group)
        base_correct_trace_count_total = sum(
            int(g.get("base_correct_trace_count", 0)) for g in group
        )

        summary[method] = {
            # existing prediction-level metrics
            "cases": int(cases_total),
            "accuracy": float(safe_ratio(correct_prediction_count, cases_total)),
            "answer_rate": float(safe_ratio(has_prediction_count, cases_total)),
            "mean_group_size": float(mean([g["group_size"] for g in group])),
            "correct_prediction_count": int(correct_prediction_count),
            "has_prediction_count": int(has_prediction_count),

            # new trace-level metrics
            "trace_accuracy_valid_only": float(
                safe_ratio(valid_correct_trace_count_total, valid_answer_count_total)
            ),
            "trace_accuracy_base_count": float(
                safe_ratio(base_correct_trace_count_total, sample_size_used_total)
            ),
            "valid_correct_trace_count_total": int(valid_correct_trace_count_total),
            "valid_answer_count_total": int(valid_answer_count_total),
            "base_correct_trace_count_total": int(base_correct_trace_count_total),
            "sample_size_used_total": int(sample_size_used_total),
            "mean_valid_answer_ratio": float(mean([g.get("valid_answer_ratio", 0.0) for g in group])),
        }
    return summary


def build_online_dataset_token_vs_accuracy(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_sweep: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sweep[int(row["sweep_index"])].append(row)

    points: List[Dict[str, Any]] = []
    for sweep_idx in sorted(by_sweep.keys()):
        group = by_sweep[sweep_idx]

        surviving_path_count = sum(int(g["surviving_path_count"]) for g in group)
        surviving_correct_path_count = sum(int(g["surviving_correct_path_count"]) for g in group)
        surviving_valid_answer_count = sum(int(g["surviving_valid_answer_count"]) for g in group)
        surviving_valid_correct_path_count = sum(
            int(g["surviving_valid_correct_path_count"]) for g in group
        )

        no_early_stop_path_count = sum(int(g["no_early_stop_path_count"]) for g in group)
        no_early_stop_correct_path_count = sum(
            int(g["no_early_stop_correct_path_count"]) for g in group
        )
        total_tokens_sum = sum(int(g["total_tokens"]) for g in group)
        full_tokens_sum = sum(int(g["full_tokens"]) for g in group)

        surviving_path_accuracy = safe_ratio(
            surviving_correct_path_count, surviving_path_count
        )
        valid_answer_accuracy = safe_ratio(
            surviving_valid_correct_path_count, surviving_valid_answer_count
        )
        no_early_stop_path_accuracy = safe_ratio(
            no_early_stop_correct_path_count, no_early_stop_path_count
        )
        majority_vote_correct_count = sum(
            1 for g in group if bool(g["majority_vote_is_correct"])
        )
        majority_vote_prediction_count = sum(
            1 for g in group if bool(g["majority_vote_has_prediction"])
        )
        majority_vote_accuracy = safe_ratio(majority_vote_correct_count, len(group))

        points.append(
            {
                "sweep_index": sweep_idx,
                "threshold": float(mean([g["threshold"] for g in group])),
                "token_usage": float(mean([g["total_tokens"] for g in group])),
                "token_ratio": float(mean([g["token_ratio"] for g in group])),
                "total_tokens_sum": int(total_tokens_sum),
                "full_tokens_sum": int(full_tokens_sum),
                "global_token_ratio_by_tokens": safe_ratio(total_tokens_sum, full_tokens_sum),

                # new primary metric
                "accuracy": float(surviving_path_accuracy),
                "surviving_path_accuracy": float(surviving_path_accuracy),
                "valid_answer_accuracy": float(valid_answer_accuracy),
                "no_early_stop_path_accuracy": float(no_early_stop_path_accuracy),

                "surviving_path_count": int(surviving_path_count),
                "surviving_correct_path_count": int(surviving_correct_path_count),
                "surviving_valid_answer_count": int(surviving_valid_answer_count),
                "surviving_valid_correct_path_count": int(surviving_valid_correct_path_count),
                "no_early_stop_path_count": int(no_early_stop_path_count),
                "no_early_stop_correct_path_count": int(no_early_stop_correct_path_count),
                "surviving_path_keep_rate": safe_ratio(
                    surviving_path_count, no_early_stop_path_count
                ),

                # old metric for comparison
                "majority_vote_accuracy": float(majority_vote_accuracy),
                "majority_vote_correct_count": int(majority_vote_correct_count),
                "majority_vote_prediction_count": int(majority_vote_prediction_count),

                "num_rows": len(group),
            }
        )
    return points


def build_offline_qid_correctness(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    target_methods = {
        "majority_voting",
        "most_confidence",
        "top5_confidence_valid",
        "top10_confidence_valid",
        "top5_confidence_base256",
        "top10_confidence_base256",
    }
    by_method_qid: Dict[str, Dict[int, List[bool]]] = defaultdict(lambda: defaultdict(list))

    for row in rows:
        method = str(row["method"])
        if method not in target_methods:
            continue
        qid = int(row["qid"])
        by_method_qid[method][qid].append(bool(row["is_correct"]))

    result: Dict[str, List[Dict[str, Any]]] = {}
    for method in sorted(target_methods):
        qid_rows: List[Dict[str, Any]] = []
        for qid in sorted(by_method_qid[method].keys()):
            vals = by_method_qid[method][qid]
            correct_rate = float(sum(1 for v in vals if v) / len(vals))
            is_correct_majority = correct_rate >= 0.5
            qid_rows.append(
                {
                    "qid": qid,
                    "is_correct": bool(is_correct_majority),
                    "correct_rate": correct_rate,
                    "num_resamples": len(vals),
                }
            )
        result[method] = qid_rows
    return result


def build_dataset_view_json(
    config: Dict[str, Any],
    summary: Dict[str, Any],
    online_rows: List[Dict[str, Any]],
    offline_rows: List[Dict[str, Any]],
    offline_summary: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    return {
        "config": {
            "rid": config["rid"],
            "max_qid": config["max_qid"],
            "seed": config["seed"],
            "online_warmup_traces": config["online_warmup_traces"],
            "online_total_budget": config["online_total_budget"],
            "online_sweep_points": config["online_sweep_points"],
            "online_resamples": config["online_resamples"],
            "offline_sample_size": config["offline_sample_size"],
            "offline_resamples": config["offline_resamples"],
            "adaptive_divisor": config["adaptive_divisor"],
            "consensus_stop_used": config["consensus_stop_used"],
        },
        "summary": summary,
        "online_dataset_token_vs_accuracy": build_online_dataset_token_vs_accuracy(online_rows),
        "offline_dataset_metrics": {
            method: {
                "accuracy": stats["accuracy"],
                "answer_rate": stats["answer_rate"],
                "cases": stats["cases"],
                "correct_prediction_count": stats["correct_prediction_count"],
                "has_prediction_count": stats["has_prediction_count"],

                "trace_accuracy_valid_only": stats["trace_accuracy_valid_only"],
                "trace_accuracy_base_count": stats["trace_accuracy_base_count"],
                "valid_correct_trace_count_total": stats["valid_correct_trace_count_total"],
                "valid_answer_count_total": stats["valid_answer_count_total"],
                "base_correct_trace_count_total": stats["base_correct_trace_count_total"],
                "sample_size_used_total": stats["sample_size_used_total"],
                "mean_valid_answer_ratio": stats["mean_valid_answer_ratio"],
            }
            for method, stats in offline_summary.items()
        },
        "offline_qid_correctness": build_offline_qid_correctness(offline_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze DeepConf from SFT offline traces.")
    parser.add_argument("--results_dir", required=True, help="Directory with deepthink_offline_qid*.pkl files")
    parser.add_argument("--dataset", default=None, help="Optional JSONL dataset for fallback ground truth")
    parser.add_argument("--rid", default=None, help="Use only this run id")
    parser.add_argument("--max_qid", type=int, default=None, help="Optional qid upper bound")
    parser.add_argument("--output_json", default="deepconf_sft_analysis.json", help="Where to save full analysis JSON")
    parser.add_argument(
        "--dataset_output_json",
        default="deepconf_sft_dataset_view.json",
        help="Where to save concise dataset-level JSON",
    )

    parser.add_argument("--online_warmup_traces", type=int, default=32)
    parser.add_argument("--online_total_budget", type=int, default=288)
    parser.add_argument("--online_sweep_points", type=int, default=50)
    parser.add_argument("--online_resamples", type=int, default=5)

    parser.add_argument("--offline_sample_size", type=int, default=256)
    parser.add_argument("--offline_resamples", type=int, default=5)

    parser.add_argument("--adaptive_divisor", type=float, default=10.0, help="group_size = round(mean_len / adaptive_divisor)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep_per_case", action="store_true", help="Store per-case rows in full output JSON")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"results_dir not found: {results_dir}")

    dataset_map = load_ground_truth_map(Path(args.dataset) if args.dataset else None)
    analysis_files = pick_latest_files(results_dir, args.rid, args.max_qid)
    if not analysis_files:
        raise FileNotFoundError(f"No matching deepthink_offline_qid*.pkl found in: {results_dir}")

    unique_qids = {qid for qid, _rid, _path in analysis_files}
    print(f"Found {len(analysis_files)} qid/run files across {len(unique_qids)} qids to analyze.")

    rng = random.Random(args.seed)
    online_rows: List[Dict[str, Any]] = []
    offline_rows: List[Dict[str, Any]] = []
    skipped_qids: List[str] = []

    for qid, rid, pkl_path in analysis_files:
        with pkl_path.open("rb") as f:
            data = pickle.load(f)

        traces = data.get("all_traces") or []
        if not traces:
            skipped_qids.append(f"{qid}:{rid}")
            continue

        gt = str(data.get("ground_truth", "")).strip()
        if not gt:
            gt = dataset_map.get(qid, "")
        if not gt:
            skipped_qids.append(f"{qid}:{rid}")
            continue

        online_rows.extend(
            run_online_sweep_for_question(
                traces=traces,
                ground_truth=gt,
                qid=qid,
                rng=rng,
                warmup_traces=args.online_warmup_traces,
                total_budget=args.online_total_budget,
                sweep_points=args.online_sweep_points,
                adaptive_divisor=args.adaptive_divisor,
                resamples=args.online_resamples,
            )
        )

        offline_rows.extend(
            run_offline_confidence_for_question(
                traces=traces,
                ground_truth=gt,
                qid=qid,
                rng=rng,
                sample_size=args.offline_sample_size,
                adaptive_divisor=args.adaptive_divisor,
                resamples=args.offline_resamples,
            )
        )

    online_summary = aggregate_online(online_rows)
    offline_summary = aggregate_offline(offline_rows)

    config = {
        "results_dir": str(results_dir),
        "dataset": args.dataset,
        "rid": args.rid,
        "max_qid": args.max_qid,
        "seed": args.seed,
        "online_warmup_traces": args.online_warmup_traces,
        "online_total_budget": args.online_total_budget,
        "online_sweep_points": args.online_sweep_points,
        "online_resamples": args.online_resamples,
        "offline_sample_size": args.offline_sample_size,
        "offline_resamples": args.offline_resamples,
        "adaptive_divisor": args.adaptive_divisor,
        "consensus_stop_used": False,
    }

    summary = {
        "num_qids_loaded": len(unique_qids),
        "num_qid_run_files_loaded": len(analysis_files),
        "num_qids_skipped": len(skipped_qids),
        "skipped_qids": skipped_qids,
        "online_rows": len(online_rows),
        "offline_rows": len(offline_rows),
    }

    full_output = {
        "config": config,
        "summary": summary,
        "online_sweep": {
            "aggregate": online_summary,
        },
        "offline_confidence": {
            "aggregate": offline_summary,
        },
    }

    if args.keep_per_case:
        full_output["online_sweep"]["per_case"] = online_rows
        full_output["offline_confidence"]["per_case"] = offline_rows

    dataset_view = build_dataset_view_json(
        config=config,
        summary=summary,
        online_rows=online_rows,
        offline_rows=offline_rows,
        offline_summary=offline_summary,
    )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    dataset_output_path = Path(args.dataset_output_json)
    dataset_output_path.parent.mkdir(parents=True, exist_ok=True)
    with dataset_output_path.open("w", encoding="utf-8") as f:
        json.dump(dataset_view, f, indent=2)

    print(f"Saved full analysis JSON: {output_path}")
    print(f"Saved dataset-level JSON: {dataset_output_path}")

    if online_summary:
        best_acc = max(online_summary, key=lambda x: x["accuracy"])
        print(
            "Online best surviving-path accuracy sweep index="
            f"{best_acc['sweep_index']} acc={best_acc['accuracy']:.4f} "
            f"token_ratio={best_acc['mean_token_ratio']:.4f} "
            f"no_early_stop_path_accuracy={best_acc['no_early_stop_path_accuracy']:.4f} "
            f"majority_vote_accuracy={best_acc['majority_vote_accuracy']:.4f}"
        )

    if offline_summary:
        for method in [
            "majority_voting",
            "most_confidence",
            "top5_confidence_valid",
            "top10_confidence_valid",
            "top5_confidence_base256",
            "top10_confidence_base256",
        ]:
            stats = offline_summary.get(method)
            if stats:
                print(
                    f"Offline {method}: "
                    f"accuracy={stats['accuracy']:.4f} "
                    f"trace_acc_valid={stats['trace_accuracy_valid_only']:.4f} "
                    f"trace_acc_base256={stats['trace_accuracy_base_count']:.4f}"
                )


if __name__ == "__main__":
    main()
