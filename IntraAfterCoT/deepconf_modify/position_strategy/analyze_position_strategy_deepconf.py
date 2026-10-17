#!/usr/bin/env python3
"""
Position-aware DeepConf analysis from offline traces.

This wraps the existing examples/analyze_sft_deepconf.py helpers but changes
the two confidence decisions described in the paper and rebuttal:

1) Post-CoT trace score:
   - Qwen SFT/RL:  mean confidence over [0.6, 1.0]
   - Qwen OPD:     mean confidence over [0.0, 0.6]
   - Llama SFT/RL: mean confidence over [0.55, 1.0]
   - Llama OPD:    mean confidence over [0.0, 0.55]

2) Intra-CoT early stop active region:
   - Qwen SFT:  [0.0, 1.0]
   - Qwen RL:   [0.2, 1.0]
   - Qwen OPD:  [0.0, 0.6]
   - Llama SFT: [0.0, 1.0]
   - Llama RL:  [0.25, 1.0]
   - Llama OPD: [0.0, 0.55]
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
DEEPCONF_DIR = SCRIPT_DIR.parent
EXAMPLES_DIR = DEEPCONF_DIR / "examples"
sys.path.insert(0, str(EXAMPLES_DIR))

import analyze_sft_deepconf as base  # noqa: E402


@dataclass(frozen=True)
class PositionStrategy:
    name: str
    post_start: float
    post_end: float
    intra_start: float
    intra_end: float
    description: str


DEFAULT_STRATEGIES: Dict[str, PositionStrategy] = {
    "sft": PositionStrategy(
        name="sft",
        post_start=0.60,
        post_end=1.00,
        intra_start=0.00,
        intra_end=1.00,
        description="Post-CoT [0.6, 1.0]; Intra-CoT early stop over the full trace.",
    ),
    "rl": PositionStrategy(
        name="rl",
        post_start=0.60,
        post_end=1.00,
        intra_start=0.20,
        intra_end=1.00,
        description="Post-CoT [0.6, 1.0]; Intra-CoT skips the first 20%.",
    ),
    "opd": PositionStrategy(
        name="opd",
        post_start=0.00,
        post_end=0.60,
        intra_start=0.00,
        intra_end=0.60,
        description="Post-CoT [0.0, 0.6]; Intra-CoT only stops in the first 60%.",
    ),
    "qwen_sft": PositionStrategy(
        name="qwen_sft",
        post_start=0.60,
        post_end=1.00,
        intra_start=0.00,
        intra_end=1.00,
        description="Qwen SFT PosConf intervals selected on the disjoint development split.",
    ),
    "qwen_rl": PositionStrategy(
        name="qwen_rl",
        post_start=0.60,
        post_end=1.00,
        intra_start=0.20,
        intra_end=1.00,
        description="Qwen RL PosConf intervals selected on the disjoint development split.",
    ),
    "qwen_opd": PositionStrategy(
        name="qwen_opd",
        post_start=0.00,
        post_end=0.60,
        intra_start=0.00,
        intra_end=0.60,
        description="Qwen OPD PosConf intervals selected on the disjoint development split.",
    ),
    "llama_sft": PositionStrategy(
        name="llama_sft",
        post_start=0.55,
        post_end=1.00,
        intra_start=0.00,
        intra_end=1.00,
        description="Llama SFT PosConf intervals selected on the disjoint development split.",
    ),
    "llama_rl": PositionStrategy(
        name="llama_rl",
        post_start=0.55,
        post_end=1.00,
        intra_start=0.25,
        intra_end=1.00,
        description="Llama RL PosConf intervals selected on the disjoint development split.",
    ),
    "llama_opd": PositionStrategy(
        name="llama_opd",
        post_start=0.00,
        post_end=0.55,
        intra_start=0.00,
        intra_end=0.55,
        description="Llama OPD PosConf intervals selected on the disjoint development split.",
    ),
}


def clamp_fraction(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def make_strategy(args: argparse.Namespace) -> PositionStrategy:
    base_strategy = DEFAULT_STRATEGIES[args.strategy_model]
    post_start = base_strategy.post_start if args.post_start_fraction is None else args.post_start_fraction
    post_end = base_strategy.post_end if args.post_end_fraction is None else args.post_end_fraction
    intra_start = base_strategy.intra_start if args.intra_start_fraction is None else args.intra_start_fraction
    intra_end = base_strategy.intra_end if args.intra_end_fraction is None else args.intra_end_fraction

    post_start = clamp_fraction(post_start)
    post_end = clamp_fraction(post_end)
    intra_start = clamp_fraction(intra_start)
    intra_end = clamp_fraction(intra_end)
    if post_end <= post_start:
        raise ValueError(f"Invalid Post-CoT window: [{post_start}, {post_end}]")
    if intra_end <= intra_start:
        raise ValueError(f"Invalid Intra-CoT active range: [{intra_start}, {intra_end}]")

    return PositionStrategy(
        name=base_strategy.name,
        post_start=post_start,
        post_end=post_end,
        intra_start=intra_start,
        intra_end=intra_end,
        description=(
            f"Post-CoT score window [{post_start:.2f}, {post_end:.2f}]; "
            f"Intra-CoT early-stop active range [{intra_start:.2f}, {intra_end:.2f}]."
        ),
    )


def confidence_window(confs: List[float], start_fraction: float, end_fraction: float) -> List[float]:
    if not confs:
        return []
    n = len(confs)
    start = int(math.floor(n * clamp_fraction(start_fraction)))
    end = int(math.ceil(n * clamp_fraction(end_fraction)))
    start = min(max(start, 0), n - 1)
    end = min(max(end, start + 1), n)
    return confs[start:end]


def trace_position_conf(trace: Dict[str, Any], start_fraction: float, end_fraction: float) -> float:
    confs = trace.get("confs") or []
    window = confidence_window(confs, start_fraction, end_fraction)
    if not window:
        return float("-inf")
    return float(sum(window) / len(window))


def expected_length_from_warmup(traces: List[Dict[str, Any]]) -> float:
    lengths = [base.token_count(t) for t in traces if base.token_count(t) > 0]
    if not lengths:
        return 1.0
    return max(1.0, float(median(lengths)))


def position_min_group_conf(
    trace: Dict[str, Any],
    group_size: int,
    active_start: float,
    active_end: float,
    expected_length: Optional[float] = None,
) -> float:
    confs = trace.get("confs") or []
    if not confs:
        return float("-inf")

    n = len(confs)
    group_size = max(1, int(group_size))
    active_start = clamp_fraction(active_start)
    active_end = clamp_fraction(active_end)
    denom = max(1.0, float(expected_length or n))

    if n < group_size:
        rel_end = min(n / denom, 1.0)
        if active_start <= rel_end <= active_end:
            return float(sum(confs) / n)
        return float("inf")

    best: Optional[float] = None
    window_sum = sum(confs[:group_size])
    for end_idx in range(group_size, n + 1):
        if end_idx > group_size:
            window_sum += confs[end_idx - 1] - confs[end_idx - group_size - 1]

        rel_end = min(end_idx / denom, 1.0)
        if rel_end < active_start:
            continue
        if rel_end > active_end:
            break

        mean_val = window_sum / group_size
        best = mean_val if best is None else min(best, mean_val)

    if best is None:
        return float("inf")
    return float(best)


def position_stop_token_count(
    trace: Dict[str, Any],
    threshold: float,
    group_size: int,
    active_start: float,
    active_end: float,
    expected_length: Optional[float] = None,
) -> int:
    confs = trace.get("confs") or []
    full_tokens = base.token_count(trace)
    if not confs or full_tokens <= 0:
        return full_tokens

    n = len(confs)
    group_size = max(1, int(group_size))
    active_start = clamp_fraction(active_start)
    active_end = clamp_fraction(active_end)
    denom = max(1.0, float(expected_length or n))

    if n < group_size:
        rel_end = min(n / denom, 1.0)
        mean_val = sum(confs) / n
        if active_start <= rel_end <= active_end and mean_val < threshold:
            return min(full_tokens, n)
        return full_tokens

    window_sum = sum(confs[:group_size])
    for end_idx in range(group_size, n + 1):
        if end_idx > group_size:
            window_sum += confs[end_idx - 1] - confs[end_idx - group_size - 1]

        rel_end = min(end_idx / denom, 1.0)
        if rel_end < active_start:
            continue
        if rel_end > active_end:
            break

        if window_sum / group_size < threshold:
            return min(full_tokens, end_idx)

    return full_tokens


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
    strategy: PositionStrategy,
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

        group_size = base.adaptive_group_size(warm, adaptive_divisor)
        expected_length = expected_length_from_warmup(warm)
        warm_scores = [
            position_min_group_conf(
                t,
                group_size,
                strategy.intra_start,
                strategy.intra_end,
                expected_length=expected_length,
            )
            for t in warm
        ]
        thresholds = base.quantile_thresholds(warm_scores, sweep_points)
        if not thresholds:
            continue

        warm_scores_map = [
            position_min_group_conf(
                t,
                group_size,
                strategy.intra_start,
                strategy.intra_end,
                expected_length=expected_length,
            )
            for t in warm
        ]
        final_scores_map = [
            position_min_group_conf(
                t,
                group_size,
                strategy.intra_start,
                strategy.intra_end,
                expected_length=expected_length,
            )
            for t in final
        ]

        full_tokens = sum(base.token_count(t) for t in picked)
        warm_tokens = sum(base.token_count(t) for t in warm)
        no_early_stop_path_count = len(picked)
        no_early_stop_correct_path_count = sum(
            1 for trace in picked if base.trace_is_correct(trace, ground_truth)
        )

        for sweep_idx, threshold in enumerate(thresholds):
            voting_answers: List[str] = []
            surviving_path_count = 0
            surviving_correct_path_count = 0
            surviving_valid_answer_count = 0
            surviving_valid_correct_path_count = 0

            for trace, score in zip(warm, warm_scores_map):
                if score < threshold:
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

            final_tokens = 0
            stopped_final = 0

            for trace, score in zip(final, final_scores_map):
                final_tokens += position_stop_token_count(
                    trace=trace,
                    threshold=threshold,
                    group_size=group_size,
                    active_start=strategy.intra_start,
                    active_end=strategy.intra_end,
                    expected_length=expected_length,
                )

                if score < threshold:
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

            pred = base.simple_majority_vote(voting_answers)
            majority_vote_is_correct = base.evaluate_answer(pred, ground_truth)
            total_tokens = warm_tokens + final_tokens
            token_ratio = (total_tokens / full_tokens) if full_tokens > 0 else 0.0

            rows.append(
                {
                    "qid": qid,
                    "sample_id": sample_id,
                    "sweep_index": sweep_idx,
                    "threshold": float(threshold),
                    "group_size": int(group_size),
                    "expected_length": float(expected_length),
                    "warmup_used": len(warm),
                    "final_used": len(final),
                    "final_stopped": int(stopped_final),
                    "total_tokens": int(total_tokens),
                    "full_tokens": int(full_tokens),
                    "token_ratio": float(token_ratio),
                    "has_prediction": bool(pred is not None),
                    "is_correct": bool(majority_vote_is_correct),
                    "majority_vote_has_prediction": bool(pred is not None),
                    "majority_vote_is_correct": bool(majority_vote_is_correct),
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
                    "intra_active_start_fraction": float(strategy.intra_start),
                    "intra_active_end_fraction": float(strategy.intra_end),
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
    strategy: PositionStrategy,
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
        group_size = base.adaptive_group_size(picked, adaptive_divisor)

        sample_size_used = len(picked)
        base_correct_trace_count = sum(
            1 for trace in picked if base.trace_is_correct(trace, ground_truth)
        )
        trace_accuracy_base_count = base.safe_ratio(
            base_correct_trace_count, sample_size_used
        )

        valid_scored: List[Tuple[float, str]] = []
        valid_answer_count = 0
        valid_correct_trace_count = 0
        all_ranked: List[Tuple[float, Optional[str]]] = []

        for trace in picked:
            score = trace_position_conf(trace, strategy.post_start, strategy.post_end)
            ans = base.trace_answer(trace)
            all_ranked.append((score, ans))

            if ans is None:
                continue

            valid_scored.append((score, ans))
            valid_answer_count += 1
            if base.evaluate_answer(ans, ground_truth):
                valid_correct_trace_count += 1

        trace_accuracy_valid_only = base.safe_ratio(
            valid_correct_trace_count, valid_answer_count
        )
        valid_answer_ratio = base.safe_ratio(valid_answer_count, sample_size_used)

        def append_row(method: str, pred: Optional[str]) -> None:
            rows.append(
                {
                    "qid": qid,
                    "sample_id": sample_id,
                    "group_size": group_size,
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
                    "post_score_start_fraction": float(strategy.post_start),
                    "post_score_end_fraction": float(strategy.post_end),
                }
            )

        if not valid_scored:
            for method in methods:
                append_row(method, None)
            continue

        valid_scored.sort(key=lambda x: x[0], reverse=True)
        all_ranked.sort(key=lambda x: x[0], reverse=True)

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

        preds = {
            "majority_voting": base.simple_majority_vote([ans for _, ans in valid_scored]),
            "most_confidence": valid_scored[0][1],
            "top5_confidence_valid": vote_top_valid(valid_scored, 0.05),
            "top10_confidence_valid": vote_top_valid(valid_scored, 0.10),
            "top5_confidence_base256": vote_top_base(all_ranked, 0.05, sample_size_used),
            "top10_confidence_base256": vote_top_base(all_ranked, 0.10, sample_size_used),
        }

        for method in methods:
            append_row(method, preds[method])

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Position-aware DeepConf analysis from offline traces."
    )
    parser.add_argument("--results_dir", required=True, help="Directory with deepthink_offline_qid*.pkl files")
    parser.add_argument("--dataset", default=None, help="Optional JSONL dataset for fallback ground truth")
    parser.add_argument("--rid", default=None, help="Use only this run id")
    parser.add_argument("--max_qid", type=int, default=None, help="Optional qid upper bound")
    parser.add_argument("--output_json", default="position_strategy_deepconf_full.json")
    parser.add_argument("--dataset_output_json", default="position_strategy_deepconf_dataset_view.json")
    parser.add_argument("--strategy_model", required=True, choices=sorted(DEFAULT_STRATEGIES.keys()))

    parser.add_argument("--online_warmup_traces", type=int, default=32)
    parser.add_argument("--online_total_budget", type=int, default=288)
    parser.add_argument("--online_sweep_points", type=int, default=50)
    parser.add_argument("--online_resamples", type=int, default=5)
    parser.add_argument("--offline_sample_size", type=int, default=256)
    parser.add_argument("--offline_resamples", type=int, default=5)
    parser.add_argument("--adaptive_divisor", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep_per_case", action="store_true")

    parser.add_argument("--post_start_fraction", type=float, default=None)
    parser.add_argument("--post_end_fraction", type=float, default=None)
    parser.add_argument("--intra_start_fraction", type=float, default=None)
    parser.add_argument("--intra_end_fraction", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    strategy = make_strategy(args)

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"results_dir not found: {results_dir}")

    dataset_map = base.load_ground_truth_map(Path(args.dataset) if args.dataset else None)
    analysis_files = base.pick_latest_files(results_dir, args.rid, args.max_qid)
    if not analysis_files:
        raise FileNotFoundError(f"No matching deepthink_offline_qid*.pkl found in: {results_dir}")

    unique_qids = {qid for qid, _rid, _path in analysis_files}
    print(f"Found {len(analysis_files)} qid/run files across {len(unique_qids)} qids to analyze.")
    print(f"Strategy model: {strategy.name}")
    print(f"Post-CoT score window: [{strategy.post_start}, {strategy.post_end}]")
    print(f"Intra-CoT active range: [{strategy.intra_start}, {strategy.intra_end}]")

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
                strategy=strategy,
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
                strategy=strategy,
            )
        )

    online_summary = base.aggregate_online(online_rows)
    offline_summary = base.aggregate_offline(offline_rows)

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
        "strategy_model": strategy.name,
        "strategy_description": strategy.description,
        "post_score_start_fraction": strategy.post_start,
        "post_score_end_fraction": strategy.post_end,
        "intra_active_start_fraction": strategy.intra_start,
        "intra_active_end_fraction": strategy.intra_end,
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
        "online_sweep": {"aggregate": online_summary},
        "offline_confidence": {"aggregate": offline_summary},
    }
    if args.keep_per_case:
        full_output["online_sweep"]["per_case"] = online_rows
        full_output["offline_confidence"]["per_case"] = offline_rows

    dataset_view = base.build_dataset_view_json(
        config=config,
        summary=summary,
        online_rows=online_rows,
        offline_rows=offline_rows,
        offline_summary=offline_summary,
    )
    dataset_view["config"].update(
        {
            "strategy_model": strategy.name,
            "strategy_description": strategy.description,
            "post_score_start_fraction": strategy.post_start,
            "post_score_end_fraction": strategy.post_end,
            "intra_active_start_fraction": strategy.intra_start,
            "intra_active_end_fraction": strategy.intra_end,
        }
    )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(full_output, indent=2), encoding="utf-8")

    dataset_output_path = Path(args.dataset_output_json)
    dataset_output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_output_path.write_text(json.dumps(dataset_view, indent=2), encoding="utf-8")

    print(f"Saved full analysis JSON: {output_path}")
    print(f"Saved dataset-level JSON: {dataset_output_path}")

    if online_summary:
        best_acc = max(online_summary, key=lambda x: x["accuracy"])
        print(
            "Online best surviving-path accuracy "
            f"sweep={best_acc['sweep_index']} "
            f"acc={best_acc['accuracy']:.4f} "
            f"token_ratio={best_acc['mean_token_ratio']:.4f} "
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
