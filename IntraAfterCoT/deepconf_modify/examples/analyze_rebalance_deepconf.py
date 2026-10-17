#!/usr/bin/env python3
"""
Post-hoc ReBalance-style DeepConf analysis from offline traces.

This script keeps the original DeepConf postprocess contract, but replaces the
one-dimensional confidence gate with a two-dimensional signal:

  confidence: local group confidence, where higher means more certain
  variance:   recent variance of local group confidence

The online/Intra simulation stops a trace only when it enters a low-confidence
and high-variance region. This is a conservative proxy for the ReBalance
"overthinking" signature and avoids stopping high-confidence, low-variance
segments that may simply be concise.

The offline/Post simulation adds balanced voting methods that reward confident
traces while penalizing two risks:

  overthinking:   low local confidence + high confidence variance
  underthinking:  high confidence + low variance + unusually short trace
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional, Sequence, Tuple

import analyze_sft_deepconf as base


def quantile(values: Sequence[float], q: float) -> float:
    clean = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not clean:
        return 0.0
    if q <= 0.0:
        return float(clean[0])
    if q >= 1.0:
        return float(clean[-1])
    pos = q * (len(clean) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(clean[lo])
    frac = pos - lo
    return float(clean[lo] * (1.0 - frac) + clean[hi] * frac)


def minmax(values: Sequence[float], default: float = 0.5) -> List[float]:
    clean = [float(v) if math.isfinite(float(v)) else 0.0 for v in values]
    if not clean:
        return []
    lo = min(clean)
    hi = max(clean)
    if hi <= lo:
        return [default for _ in clean]
    return [(v - lo) / (hi - lo) for v in clean]


def rolling_group_means(
    confs: Sequence[float],
    group_size: int,
    stride_divisor: float,
) -> List[Tuple[int, float]]:
    if not confs:
        return []

    n = len(confs)
    group = max(1, int(group_size))
    stride = max(1, int(round(group / max(stride_divisor, 1.0))))

    if n <= group:
        return [(n, float(sum(confs) / n))]

    out: List[Tuple[int, float]] = []
    window_sum = sum(confs[:group])
    out.append((group, float(window_sum / group)))

    # Sample overlapping windows to approximate step-level states without
    # materializing one window per token for very long traces.
    end = group + stride
    while end <= n:
        start = end - group
        window_sum = sum(confs[start:end])
        out.append((end, float(window_sum / group)))
        end += stride

    if out[-1][0] != n:
        tail = confs[max(0, n - group):n]
        out.append((n, float(sum(tail) / len(tail))))

    return out


def confidence_variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = sum(values) / len(values)
    return float(sum((v - avg) ** 2 for v in values) / len(values))


def local_state_series(
    trace: Dict[str, Any],
    group_size: int,
    variance_window: int,
    stride_divisor: float,
) -> List[Dict[str, float]]:
    confs = trace.get("confs") or []
    groups = rolling_group_means(confs, group_size, stride_divisor)
    if not groups:
        return []

    states: List[Dict[str, float]] = []
    recent: List[float] = []
    var_window = max(1, int(variance_window))
    for end_token, conf in groups:
        recent.append(conf)
        if len(recent) > var_window:
            recent.pop(0)
        states.append(
            {
                "end_token": float(end_token),
                "confidence": float(conf),
                "variance": confidence_variance(recent),
            }
        )
    return states


def trace_state_features(
    trace: Dict[str, Any],
    group_size: int,
    variance_window: int,
    stride_divisor: float,
) -> Dict[str, float]:
    states = local_state_series(trace, group_size, variance_window, stride_divisor)
    confs = trace.get("confs") or []

    if confs:
        mean_conf = float(sum(confs) / len(confs))
        tail_len = max(1, int(math.ceil(len(confs) * 0.10)))
        tail_conf = float(sum(confs[-tail_len:]) / tail_len)
    else:
        mean_conf = float("-inf")
        tail_conf = float("-inf")

    if not states:
        return {
            "mean_conf": mean_conf,
            "tail_conf": tail_conf,
            "min_state_conf": float("-inf"),
            "max_state_var": 0.0,
            "mean_state_var": 0.0,
            "num_states": 0.0,
            "num_tokens": float(base.token_count(trace)),
        }

    state_confs = [s["confidence"] for s in states]
    state_vars = [s["variance"] for s in states]
    return {
        "mean_conf": mean_conf,
        "tail_conf": tail_conf,
        "min_state_conf": float(min(state_confs)),
        "max_state_var": float(max(state_vars)),
        "mean_state_var": float(sum(state_vars) / len(state_vars)),
        "num_states": float(len(states)),
        "num_tokens": float(base.token_count(trace)),
    }


def collect_state_values(
    traces: Sequence[Dict[str, Any]],
    group_size: int,
    variance_window: int,
    stride_divisor: float,
) -> Tuple[List[float], List[float]]:
    conf_values: List[float] = []
    var_values: List[float] = []
    for trace in traces:
        for state in local_state_series(trace, group_size, variance_window, stride_divisor):
            conf_values.append(float(state["confidence"]))
            var_values.append(float(state["variance"]))
    return conf_values, var_values


def trace_hits_region(
    trace: Dict[str, Any],
    group_size: int,
    variance_window: int,
    stride_divisor: float,
    conf_threshold: float,
    var_threshold: float,
) -> bool:
    for state in local_state_series(trace, group_size, variance_window, stride_divisor):
        if state["confidence"] <= conf_threshold and state["variance"] >= var_threshold:
            return True
    return False


def rebalance_stop_token_count(
    trace: Dict[str, Any],
    group_size: int,
    variance_window: int,
    stride_divisor: float,
    conf_threshold: float,
    var_threshold: float,
) -> int:
    full_tokens = base.token_count(trace)
    if full_tokens <= 0:
        return 0

    for state in local_state_series(trace, group_size, variance_window, stride_divisor):
        if state["confidence"] <= conf_threshold and state["variance"] >= var_threshold:
            return min(full_tokens, max(1, int(state["end_token"])))
    return full_tokens


def rebalance_scores_for_traces(
    traces: Sequence[Dict[str, Any]],
    group_size: int,
    variance_window: int,
    stride_divisor: float,
    over_penalty: float,
    under_penalty: float,
) -> List[Dict[str, float]]:
    features = [
        trace_state_features(trace, group_size, variance_window, stride_divisor)
        for trace in traces
    ]
    if not features:
        return []

    tail_norm = minmax([f["tail_conf"] for f in features])
    mean_norm = minmax([f["mean_conf"] for f in features])
    min_conf_norm = minmax([f["min_state_conf"] for f in features])
    max_var_norm = minmax([f["max_state_var"] for f in features], default=0.0)

    lengths = [max(0.0, f["num_tokens"]) for f in features]
    med_len = max(1.0, float(median(lengths))) if lengths else 1.0

    raw_scores: List[float] = []
    out: List[Dict[str, float]] = []
    for idx, f in enumerate(features):

        '''
        base_conf = (
            1.0 * mean_norm[idx]
        )
        '''
        
        base_conf = (
            0.50 * tail_norm[idx]
            + 0.30 * min_conf_norm[idx]
            + 0.20 * mean_norm[idx]
        )
        
        over_risk = (1.0 - min_conf_norm[idx]) * max_var_norm[idx]
        shortness = max(0.0, (med_len - lengths[idx]) / med_len)
        under_risk = mean_norm[idx] * (1.0 - max_var_norm[idx]) * shortness
        raw_score = base_conf - over_penalty * over_risk - under_penalty * under_risk
        raw_scores.append(raw_score)
        out.append(
            {
                **f,
                "base_conf_norm": float(base_conf),
                "overthink_risk": float(over_risk),
                "underthink_risk": float(under_risk),
                "rebalance_raw_score": float(raw_score),
            }
        )

    # Keep weights positive for weighted voting while preserving ranking.
    min_score = min(raw_scores)
    for item in out:
        item["rebalance_weight"] = float(item["rebalance_raw_score"] - min_score + 1e-6)
    return out


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
    variance_window: int,
    q_min: float,
    q_max: float,
    stride_divisor: float,
    over_penalty: float,
    under_penalty: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if len(traces) < max(2, warmup_traces + 1):
        return rows

    budget = min(total_budget, len(traces))
    if sweep_points <= 1:
        q_values = [(q_min + q_max) / 2.0]
    else:
        q_values = [
            q_min + (q_max - q_min) * i / (sweep_points - 1)
            for i in range(sweep_points)
        ]

    for sample_id in range(resamples):
        picked_idx = rng.sample(range(len(traces)), budget)
        picked = [traces[i] for i in picked_idx]
        rng.shuffle(picked)

        warm = picked[:warmup_traces]
        final = picked[warmup_traces:]
        if not warm or not final:
            continue

        group_size = base.adaptive_group_size(warm, adaptive_divisor)
        warm_conf_values, warm_var_values = collect_state_values(
            warm, group_size, variance_window, stride_divisor
        )
        if not warm_conf_values:
            continue

        warm_scores = rebalance_scores_for_traces(
            warm,
            group_size,
            variance_window,
            stride_divisor,
            over_penalty,
            under_penalty,
        )
        final_scores = rebalance_scores_for_traces(
            final,
            group_size,
            variance_window,
            stride_divisor,
            over_penalty,
            under_penalty,
        )

        full_tokens = sum(base.token_count(t) for t in picked)
        warm_tokens = sum(base.token_count(t) for t in warm)

        no_early_stop_path_count = len(picked)
        no_early_stop_correct_path_count = sum(
            1 for trace in picked if base.trace_is_correct(trace, ground_truth)
        )

        for sweep_idx, q_low in enumerate(q_values):
            q_high = 1.0 - q_low
            conf_threshold = quantile(warm_conf_values, q_low)
            var_threshold = quantile(warm_var_values, q_high)

            voting_answers: List[str] = []
            balanced_scored_answers: List[Tuple[float, str]] = []

            surviving_path_count = 0
            surviving_correct_path_count = 0
            surviving_valid_answer_count = 0
            surviving_valid_correct_path_count = 0

            for trace, score_info in zip(warm, warm_scores):
                if trace_hits_region(
                    trace,
                    group_size,
                    variance_window,
                    stride_divisor,
                    conf_threshold,
                    var_threshold,
                ):
                    continue

                is_correct = base.trace_is_correct(trace, ground_truth)
                surviving_path_count += 1
                if is_correct:
                    surviving_correct_path_count += 1

                ans = base.trace_answer(trace)
                if ans is not None:
                    surviving_valid_answer_count += 1
                    if is_correct:
                        surviving_valid_correct_path_count += 1
                    voting_answers.append(ans)
                    balanced_scored_answers.append((score_info["rebalance_weight"], ans))

            final_tokens = 0
            stopped_final = 0

            for trace, score_info in zip(final, final_scores):
                stopped_tokens = rebalance_stop_token_count(
                    trace,
                    group_size,
                    variance_window,
                    stride_divisor,
                    conf_threshold,
                    var_threshold,
                )
                final_tokens += stopped_tokens

                overthinking_hit = stopped_tokens < base.token_count(trace)
                if overthinking_hit:
                    stopped_final += 1
                    continue

                is_correct = base.trace_is_correct(trace, ground_truth)
                surviving_path_count += 1
                if is_correct:
                    surviving_correct_path_count += 1

                ans = base.trace_answer(trace)
                if ans is not None:
                    surviving_valid_answer_count += 1
                    if is_correct:
                        surviving_valid_correct_path_count += 1
                    voting_answers.append(ans)
                    balanced_scored_answers.append((score_info["rebalance_weight"], ans))

            majority_pred = base.simple_majority_vote(voting_answers)
            balanced_pred = base.weighted_majority_vote(balanced_scored_answers)
            majority_vote_is_correct = base.evaluate_answer(majority_pred, ground_truth)
            balanced_vote_is_correct = base.evaluate_answer(balanced_pred, ground_truth)

            total_tokens = warm_tokens + final_tokens
            token_ratio = (total_tokens / full_tokens) if full_tokens > 0 else 0.0

            rows.append(
                {
                    "qid": qid,
                    "sample_id": sample_id,
                    "sweep_index": sweep_idx,
                    "threshold": float(conf_threshold),
                    "confidence_threshold": float(conf_threshold),
                    "variance_threshold": float(var_threshold),
                    "q_low": float(q_low),
                    "q_high": float(q_high),
                    "group_size": int(group_size),
                    "variance_window": int(variance_window),
                    "warmup_used": len(warm),
                    "final_used": len(final),
                    "final_stopped": int(stopped_final),
                    "total_tokens": int(total_tokens),
                    "full_tokens": int(full_tokens),
                    "token_ratio": float(token_ratio),

                    "has_prediction": bool(balanced_pred is not None),
                    "is_correct": bool(balanced_vote_is_correct),
                    "majority_vote_has_prediction": bool(majority_pred is not None),
                    "majority_vote_is_correct": bool(majority_vote_is_correct),
                    "balanced_vote_has_prediction": bool(balanced_pred is not None),
                    "balanced_vote_is_correct": bool(balanced_vote_is_correct),

                    "surviving_path_count": int(surviving_path_count),
                    "surviving_correct_path_count": int(surviving_correct_path_count),
                    "surviving_path_accuracy": base.safe_ratio(
                        surviving_correct_path_count, surviving_path_count
                    ),
                    "surviving_valid_answer_count": int(surviving_valid_answer_count),
                    "surviving_valid_correct_path_count": int(surviving_valid_correct_path_count),
                    "valid_answer_accuracy": base.safe_ratio(
                        surviving_valid_correct_path_count, surviving_valid_answer_count
                    ),

                    "no_early_stop_path_count": int(no_early_stop_path_count),
                    "no_early_stop_correct_path_count": int(no_early_stop_correct_path_count),
                    "no_early_stop_path_accuracy": base.safe_ratio(
                        no_early_stop_correct_path_count, no_early_stop_path_count
                    ),
                    "surviving_path_keep_rate": base.safe_ratio(
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
    variance_window: int,
    stride_divisor: float,
    over_penalty: float,
    under_penalty: float,
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
        "rebalance_weighted",
        "rebalance_most",
        "top5_rebalance_valid",
        "top10_rebalance_valid",
        "top5_rebalance_base256",
        "top10_rebalance_base256",
    ]

    for sample_id in range(resamples):
        picked_idx = rng.sample(range(len(traces)), n)
        picked = [traces[i] for i in picked_idx]
        group_size = base.adaptive_group_size(picked, adaptive_divisor)
        rebalance_infos = rebalance_scores_for_traces(
            picked,
            group_size,
            variance_window,
            stride_divisor,
            over_penalty,
            under_penalty,
        )

        sample_size_used = len(picked)
        base_correct_trace_count = sum(
            1 for trace in picked if base.trace_is_correct(trace, ground_truth)
        )
        trace_accuracy_base_count = base.safe_ratio(
            base_correct_trace_count, sample_size_used
        )

        valid_tail: List[Tuple[float, str]] = []
        valid_rebalance: List[Tuple[float, str]] = []
        valid_answer_count = 0
        valid_correct_trace_count = 0
        all_tail_ranked: List[Tuple[float, Optional[str]]] = []
        all_rebalance_ranked: List[Tuple[float, Optional[str]]] = []

        for trace, info in zip(picked, rebalance_infos):
            tail_score = base.tail_fraction_trace_conf(trace, tail_ratio=0.1)
            rebalance_score = info["rebalance_weight"]
            ans = base.trace_answer(trace)

            all_tail_ranked.append((tail_score, ans))
            all_rebalance_ranked.append((rebalance_score, ans))

            if ans is None:
                continue

            valid_tail.append((tail_score, ans))
            valid_rebalance.append((rebalance_score, ans))
            valid_answer_count += 1
            if base.evaluate_answer(ans, ground_truth):
                valid_correct_trace_count += 1

        trace_accuracy_valid_only = base.safe_ratio(
            valid_correct_trace_count, valid_answer_count
        )
        valid_answer_ratio = base.safe_ratio(valid_answer_count, sample_size_used)

        def add_empty_rows() -> None:
            for method in methods:
                rows.append(
                    {
                        "qid": qid,
                        "sample_id": sample_id,
                        "group_size": group_size,
                        "variance_window": variance_window,
                        "method": method,
                        "has_prediction": False,
                        "is_correct": False,
                        "sample_size_used": int(sample_size_used),
                        "valid_answer_count": int(valid_answer_count),
                        "valid_correct_trace_count": int(valid_correct_trace_count),
                        "trace_accuracy_valid_only": float(trace_accuracy_valid_only),
                        "base_correct_trace_count": int(base_correct_trace_count),
                        "trace_accuracy_base_count": float(trace_accuracy_base_count),
                        "valid_answer_ratio": float(valid_answer_ratio),
                    }
                )

        if not valid_tail:
            add_empty_rows()
            continue

        valid_tail.sort(key=lambda x: x[0], reverse=True)
        valid_rebalance.sort(key=lambda x: x[0], reverse=True)
        all_tail_ranked.sort(key=lambda x: x[0], reverse=True)
        all_rebalance_ranked.sort(key=lambda x: x[0], reverse=True)

        def vote_top_valid(sorted_pairs: List[Tuple[float, str]], frac: float) -> Optional[str]:
            k = max(1, int(math.ceil(len(sorted_pairs) * frac)))
            return base.weighted_majority_vote(sorted_pairs[:k])

        def vote_top_base(
            sorted_pairs_all: List[Tuple[float, Optional[str]]],
            frac: float,
            base_n: int,
        ) -> Optional[str]:
            k = max(1, int(math.ceil(base_n * frac)))
            scored_answers = [
                (score, ans) for score, ans in sorted_pairs_all[:k] if ans is not None
            ]
            return base.weighted_majority_vote(scored_answers)

        pred_majority = base.simple_majority_vote([ans for _, ans in valid_tail])
        preds = {
            "majority_voting": pred_majority,
            "most_confidence": valid_tail[0][1],
            "top5_confidence_valid": vote_top_valid(valid_tail, 0.05),
            "top10_confidence_valid": vote_top_valid(valid_tail, 0.10),
            "top5_confidence_base256": vote_top_base(all_tail_ranked, 0.05, sample_size_used),
            "top10_confidence_base256": vote_top_base(all_tail_ranked, 0.10, sample_size_used),
            "rebalance_weighted": base.weighted_majority_vote(valid_rebalance),
            "rebalance_most": valid_rebalance[0][1],
            "top5_rebalance_valid": vote_top_valid(valid_rebalance, 0.05),
            "top10_rebalance_valid": vote_top_valid(valid_rebalance, 0.10),
            "top5_rebalance_base256": vote_top_base(all_rebalance_ranked, 0.05, sample_size_used),
            "top10_rebalance_base256": vote_top_base(all_rebalance_ranked, 0.10, sample_size_used),
        }

        for method in methods:
            pred = preds[method]
            rows.append(
                {
                    "qid": qid,
                    "sample_id": sample_id,
                    "group_size": group_size,
                    "variance_window": variance_window,
                    "method": method,
                    "has_prediction": pred is not None,
                    "is_correct": base.evaluate_answer(pred, ground_truth),
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
    summary = base.aggregate_online(rows)
    by_sweep: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sweep[int(row["sweep_index"])].append(row)

    for item in summary:
        group = by_sweep[int(item["sweep_index"])]
        balanced_correct = sum(1 for g in group if bool(g.get("balanced_vote_is_correct")))
        balanced_pred = sum(1 for g in group if bool(g.get("balanced_vote_has_prediction")))
        item["balanced_vote_accuracy"] = base.safe_ratio(balanced_correct, len(group))
        item["balanced_vote_answer_rate"] = base.safe_ratio(balanced_pred, len(group))
        item["mean_confidence_threshold"] = item.get("mean_threshold")
        item["mean_variance_threshold"] = float(mean([g["variance_threshold"] for g in group]))
        item["mean_q_low"] = float(mean([g["q_low"] for g in group]))
        item["mean_q_high"] = float(mean([g["q_high"] for g in group]))
    return summary


def build_online_dataset_token_vs_accuracy(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    points = base.build_online_dataset_token_vs_accuracy(rows)
    by_sweep: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sweep[int(row["sweep_index"])].append(row)

    for item in points:
        group = by_sweep[int(item["sweep_index"])]
        balanced_correct = sum(1 for g in group if bool(g.get("balanced_vote_is_correct")))
        balanced_pred = sum(1 for g in group if bool(g.get("balanced_vote_has_prediction")))
        item["balanced_vote_accuracy"] = base.safe_ratio(balanced_correct, len(group))
        item["balanced_vote_answer_rate"] = base.safe_ratio(balanced_pred, len(group))
        item["variance_threshold"] = float(mean([g["variance_threshold"] for g in group]))
        item["q_low"] = float(mean([g["q_low"] for g in group]))
        item["q_high"] = float(mean([g["q_high"] for g in group]))
    return points


def build_offline_qid_correctness(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by_method_qid: Dict[str, Dict[int, List[bool]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        method = str(row["method"])
        qid = int(row["qid"])
        by_method_qid[method][qid].append(bool(row["is_correct"]))

    result: Dict[str, List[Dict[str, Any]]] = {}
    for method in sorted(by_method_qid.keys()):
        qid_rows: List[Dict[str, Any]] = []
        for qid in sorted(by_method_qid[method].keys()):
            vals = by_method_qid[method][qid]
            correct_rate = float(sum(1 for v in vals if v) / len(vals))
            qid_rows.append(
                {
                    "qid": qid,
                    "is_correct": bool(correct_rate >= 0.5),
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
        "config": config,
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
    parser = argparse.ArgumentParser(
        description="Analyze DeepConf with ReBalance-style confidence/variance signals."
    )
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--rid", default=None)
    parser.add_argument("--max_qid", type=int, default=None)
    parser.add_argument("--output_json", default="rebalance_deepconf_full.json")
    parser.add_argument("--dataset_output_json", default="rebalance_deepconf_dataset_view.json")

    parser.add_argument("--online_warmup_traces", type=int, default=32)
    parser.add_argument("--online_total_budget", type=int, default=288)
    parser.add_argument("--online_sweep_points", type=int, default=50)
    parser.add_argument("--online_resamples", type=int, default=5)
    parser.add_argument("--offline_sample_size", type=int, default=256)
    parser.add_argument("--offline_resamples", type=int, default=5)
    parser.add_argument("--adaptive_divisor", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep_per_case", action="store_true")

    parser.add_argument("--rebalance_variance_window", type=int, default=4)
    parser.add_argument("--rebalance_q_min", type=float, default=0.05)
    parser.add_argument("--rebalance_q_max", type=float, default=0.50)
    parser.add_argument("--rebalance_stride_divisor", type=float, default=4.0)
    parser.add_argument("--rebalance_over_penalty", type=float, default=0.35)
    parser.add_argument("--rebalance_under_penalty", type=float, default=0.20)

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"results_dir not found: {results_dir}")

    dataset_map = base.load_ground_truth_map(Path(args.dataset) if args.dataset else None)
    analysis_files = base.pick_latest_files(results_dir, args.rid, args.max_qid)
    if not analysis_files:
        raise FileNotFoundError(f"No matching deepthink_offline_qid*.pkl found in: {results_dir}")

    unique_qids = {qid for qid, _rid, _path in analysis_files}
    print(
        f"Found {len(analysis_files)} qid/run files across {len(unique_qids)} qids "
        "to analyze with ReBalance-style 2D signal."
    )

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
                variance_window=args.rebalance_variance_window,
                q_min=args.rebalance_q_min,
                q_max=args.rebalance_q_max,
                stride_divisor=args.rebalance_stride_divisor,
                over_penalty=args.rebalance_over_penalty,
                under_penalty=args.rebalance_under_penalty,
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
                variance_window=args.rebalance_variance_window,
                stride_divisor=args.rebalance_stride_divisor,
                over_penalty=args.rebalance_over_penalty,
                under_penalty=args.rebalance_under_penalty,
            )
        )

    online_summary = aggregate_online(online_rows)
    offline_summary = base.aggregate_offline(offline_rows)

    config = {
        "analysis": "rebalance_2d",
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
        "rebalance_variance_window": args.rebalance_variance_window,
        "rebalance_q_min": args.rebalance_q_min,
        "rebalance_q_max": args.rebalance_q_max,
        "rebalance_stride_divisor": args.rebalance_stride_divisor,
        "rebalance_over_penalty": args.rebalance_over_penalty,
        "rebalance_under_penalty": args.rebalance_under_penalty,
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
    output_path.write_text(json.dumps(full_output, indent=2), encoding="utf-8")

    dataset_output_path = Path(args.dataset_output_json)
    dataset_output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_output_path.write_text(json.dumps(dataset_view, indent=2), encoding="utf-8")

    print(f"Saved full ReBalance analysis JSON: {output_path}")
    print(f"Saved dataset-level ReBalance JSON: {dataset_output_path}")

    if online_summary:
        best_acc = max(online_summary, key=lambda x: x["accuracy"])
        print(
            "ReBalance-online best surviving-path accuracy sweep index="
            f"{best_acc['sweep_index']} acc={best_acc['accuracy']:.4f} "
            f"token_ratio={best_acc['mean_token_ratio']:.4f} "
            f"keep_rate={best_acc['surviving_path_keep_rate']:.4f} "
            f"balanced_vote_accuracy={best_acc.get('balanced_vote_accuracy', 0.0):.4f}"
        )

    if offline_summary:
        for method in [
            "majority_voting",
            "top10_confidence_base256",
            "rebalance_weighted",
            "rebalance_most",
            "top5_rebalance_base256",
            "top10_rebalance_base256",
        ]:
            stats = offline_summary.get(method)
            if stats:
                print(
                    f"ReBalance-offline {method}: "
                    f"accuracy={stats['accuracy']:.4f} "
                    f"trace_acc_valid={stats['trace_accuracy_valid_only']:.4f} "
                    f"trace_acc_base={stats['trace_accuracy_base_count']:.4f}"
                )


if __name__ == "__main__":
    main()
