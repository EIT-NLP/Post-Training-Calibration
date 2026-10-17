#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Read DeepConf full JSON and export dataset-level / per-case CSV tables.

Supports the newer analyze_sft_deepconf.py schema:
- online aggregate:
    accuracy (= surviving_path_accuracy),
    surviving_path_accuracy,
    no_early_stop_path_accuracy,
    surviving_path_count,
    no_early_stop_path_count,
    surviving_path_keep_rate,
    majority_vote_accuracy,
    answer_rate / majority_vote_answer_rate
- offline aggregate:
    majority_voting,
    most_confidence,
    top5_confidence_valid,
    top10_confidence_valid,
    top5_confidence_base256,
    top10_confidence_base256
  with:
    accuracy,
    answer_rate,
    trace_accuracy_valid_only,
    trace_accuracy_base_count,
    mean_valid_answer_ratio

If --keep_per_case was used when generating the JSON, this script also exports:
- online_per_case.csv
- offline_per_case.csv

Usage:
python read_deepconf_json.py \
  --input_json deepconf_instruct.json \
  --output_dir csv_view \
  --print_tables
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


ONLINE_AGG_COLUMNS = [
    "sweep_index",
    "cases",
    "mean_threshold",
    "mean_group_size",
    "mean_tokens",
    "mean_token_ratio",
    "accuracy",
    "surviving_path_accuracy",
    "no_early_stop_path_accuracy",
    "surviving_path_count",
    "no_early_stop_path_count",
    "surviving_path_keep_rate",
    "majority_vote_accuracy",
    "answer_rate",
    "majority_vote_answer_rate",
]

OFFLINE_AGG_COLUMNS = [
    "method",
    "cases",
    "accuracy",
    "answer_rate",
    "mean_group_size",
    "trace_accuracy_valid_only",
    "trace_accuracy_base_count",
    "mean_valid_answer_ratio",
    "valid_answer_count_total",
    "sample_size_used_total",
]

ONLINE_PER_CASE_COLUMNS = [
    "qid",
    "sample_id",
    "sweep_index",
    "threshold",
    "group_size",
    "warmup_used",
    "final_used",
    "final_stopped",
    "total_tokens",
    "full_tokens",
    "token_ratio",
    "has_prediction",
    "is_correct",
    "majority_vote_has_prediction",
    "majority_vote_is_correct",
    "surviving_path_count",
    "surviving_correct_path_count",
    "surviving_path_accuracy",
    "no_early_stop_path_count",
    "no_early_stop_correct_path_count",
    "no_early_stop_path_accuracy",
    "surviving_path_keep_rate",
]

OFFLINE_PER_CASE_COLUMNS = [
    "qid",
    "sample_id",
    "group_size",
    "method",
    "has_prediction",
    "is_correct",
    "sample_size_used",
    "valid_answer_count",
    "valid_correct_trace_count",
    "trace_accuracy_valid_only",
    "base_correct_trace_count",
    "trace_accuracy_base_count",
    "valid_answer_ratio",
]


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_nested(data: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key, default)
    return cur


def build_online_agg_rows(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = get_nested(data, "online_sweep", "aggregate", default=[])
    if not isinstance(rows, list):
        raise ValueError("Invalid format: online_sweep.aggregate should be a list.")

    clean_rows: List[Dict[str, Any]] = []
    for row in rows:
        clean_rows.append(
            {
                "sweep_index": row.get("sweep_index"),
                "cases": row.get("cases"),
                "mean_threshold": row.get("mean_threshold"),
                "mean_group_size": row.get("mean_group_size"),
                "mean_tokens": row.get("mean_tokens"),
                "mean_token_ratio": row.get("mean_token_ratio"),
                "accuracy": row.get("accuracy"),
                "surviving_path_accuracy": row.get("surviving_path_accuracy", row.get("accuracy")),
                "no_early_stop_path_accuracy": row.get("no_early_stop_path_accuracy"),
                "surviving_path_count": row.get("surviving_path_count"),
                "no_early_stop_path_count": row.get("no_early_stop_path_count"),
                "surviving_path_keep_rate": row.get("surviving_path_keep_rate"),
                "majority_vote_accuracy": row.get("majority_vote_accuracy"),
                "answer_rate": row.get("answer_rate"),
                "majority_vote_answer_rate": row.get(
                    "majority_vote_answer_rate", row.get("answer_rate")
                ),
            }
        )
    return clean_rows


def build_offline_agg_rows(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    agg = get_nested(data, "offline_confidence", "aggregate", default={})
    if not isinstance(agg, dict):
        raise ValueError("Invalid format: offline_confidence.aggregate should be a dict.")

    rows: List[Dict[str, Any]] = []

    preferred_order = [
        "majority_voting",
        "most_confidence",
        "top5_confidence_valid",
        "top10_confidence_valid",
        "top5_confidence_base256",
        "top10_confidence_base256",
        # backward compatibility
        "top5_confidence",
        "top10_confidence",
    ]

    seen = set()
    for method in preferred_order:
        if method in agg and isinstance(agg[method], dict):
            stats = agg[method]
            rows.append(
                {
                    "method": method,
                    "cases": stats.get("cases"),
                    "accuracy": stats.get("accuracy"),
                    "answer_rate": stats.get("answer_rate"),
                    "mean_group_size": stats.get("mean_group_size"),
                    "trace_accuracy_valid_only": stats.get("trace_accuracy_valid_only"),
                    "trace_accuracy_base_count": stats.get("trace_accuracy_base_count"),
                    "mean_valid_answer_ratio": stats.get("mean_valid_answer_ratio"),
                    "valid_answer_count_total": stats.get("valid_answer_count_total"),
                    "sample_size_used_total": stats.get("sample_size_used_total"),
                }
            )
            seen.add(method)

    for method, stats in agg.items():
        if method in seen or not isinstance(stats, dict):
            continue
        rows.append(
            {
                "method": method,
                "cases": stats.get("cases"),
                "accuracy": stats.get("accuracy"),
                "answer_rate": stats.get("answer_rate"),
                "mean_group_size": stats.get("mean_group_size"),
                "trace_accuracy_valid_only": stats.get("trace_accuracy_valid_only"),
                "trace_accuracy_base_count": stats.get("trace_accuracy_base_count"),
                "mean_valid_answer_ratio": stats.get("mean_valid_answer_ratio"),
                "valid_answer_count_total": stats.get("valid_answer_count_total"),
                "sample_size_used_total": stats.get("sample_size_used_total"),
            }
        )

    return rows


def build_online_per_case_rows(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = get_nested(data, "online_sweep", "per_case", default=[])
    if not isinstance(rows, list):
        return []

    clean_rows: List[Dict[str, Any]] = []
    for row in rows:
        clean_rows.append(
            {
                "qid": row.get("qid"),
                "sample_id": row.get("sample_id"),
                "sweep_index": row.get("sweep_index"),
                "threshold": row.get("threshold"),
                "group_size": row.get("group_size"),
                "warmup_used": row.get("warmup_used"),
                "final_used": row.get("final_used"),
                "final_stopped": row.get("final_stopped"),
                "total_tokens": row.get("total_tokens"),
                "full_tokens": row.get("full_tokens"),
                "token_ratio": row.get("token_ratio"),
                "has_prediction": row.get("has_prediction"),
                "is_correct": row.get("is_correct"),
                "majority_vote_has_prediction": row.get("majority_vote_has_prediction"),
                "majority_vote_is_correct": row.get("majority_vote_is_correct"),
                "surviving_path_count": row.get("surviving_path_count"),
                "surviving_correct_path_count": row.get("surviving_correct_path_count"),
                "surviving_path_accuracy": row.get("surviving_path_accuracy"),
                "no_early_stop_path_count": row.get("no_early_stop_path_count"),
                "no_early_stop_correct_path_count": row.get("no_early_stop_correct_path_count"),
                "no_early_stop_path_accuracy": row.get("no_early_stop_path_accuracy"),
                "surviving_path_keep_rate": row.get("surviving_path_keep_rate"),
            }
        )
    return clean_rows


def build_offline_per_case_rows(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = get_nested(data, "offline_confidence", "per_case", default=[])
    if not isinstance(rows, list):
        return []

    clean_rows: List[Dict[str, Any]] = []
    for row in rows:
        clean_rows.append(
            {
                "qid": row.get("qid"),
                "sample_id": row.get("sample_id"),
                "group_size": row.get("group_size"),
                "method": row.get("method"),
                "has_prediction": row.get("has_prediction"),
                "is_correct": row.get("is_correct"),
                "sample_size_used": row.get("sample_size_used"),
                "valid_answer_count": row.get("valid_answer_count"),
                "valid_correct_trace_count": row.get("valid_correct_trace_count"),
                "trace_accuracy_valid_only": row.get("trace_accuracy_valid_only"),
                "base_correct_trace_count": row.get("base_correct_trace_count"),
                "trace_accuracy_base_count": row.get("trace_accuracy_base_count"),
                "valid_answer_ratio": row.get("valid_answer_ratio"),
            }
        )
    return clean_rows


def write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_value(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.6f}"
    if v is None:
        return ""
    return str(v)


def build_text_table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    if not rows:
        return "(empty)"

    widths = {}
    for col in columns:
        max_width = len(col)
        for row in rows:
            max_width = max(max_width, len(format_value(row.get(col))))
        widths[col] = max_width

    def render_line(values: List[str]) -> str:
        return " | ".join(val.ljust(widths[col]) for val, col in zip(values, columns))

    header = render_line(columns)
    sep = "-+-".join("-" * widths[col] for col in columns)
    body = [render_line([format_value(row.get(col)) for col in columns]) for row in rows]
    return "\n".join([header, sep] + body)


def summarize_best_online(
    online_rows: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any] | None, Dict[str, Any] | None]:
    if not online_rows:
        return None, None

    best_acc = max(
        online_rows,
        key=lambda x: (
            float("-inf") if x.get("accuracy") is None else float(x["accuracy"]),
            float("inf") if x.get("mean_token_ratio") is None else -float(x["mean_token_ratio"]),
        ),
    )

    best_eff = min(
        online_rows,
        key=lambda x: (
            float("inf") if x.get("mean_token_ratio") is None else float(x["mean_token_ratio"]),
            float("-inf") if x.get("accuracy") is None else -float(x["accuracy"]),
        ),
    )

    return best_acc, best_eff


def summarize_best_offline(
    offline_rows: List[Dict[str, Any]],
) -> Dict[str, Any] | None:
    if not offline_rows:
        return None
    return max(
        offline_rows,
        key=lambda x: (
            float("-inf") if x.get("accuracy") is None else float(x["accuracy"]),
            float("-inf")
            if x.get("trace_accuracy_base_count") is None
            else float(x["trace_accuracy_base_count"]),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read deepconf_instruct.json and export dataset-level / per-case CSV tables."
    )
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Path to deepconf_instruct.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save CSVs. Default: <input_json_dir>/csv_view",
    )
    parser.add_argument(
        "--print_tables",
        action="store_true",
        help="Print aggregate tables in terminal.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_json)
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")

    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent / "csv_view"

    data = read_json(input_path)

    online_agg_rows = build_online_agg_rows(data)
    offline_agg_rows = build_offline_agg_rows(data)

    online_agg_csv = output_dir / "online_dataset.csv"
    offline_agg_csv = output_dir / "offline_dataset.csv"

    write_csv(online_agg_csv, online_agg_rows, ONLINE_AGG_COLUMNS)
    write_csv(offline_agg_csv, offline_agg_rows, OFFLINE_AGG_COLUMNS)

    print(f"Input JSON: {input_path}")
    print(f"Saved online aggregate CSV: {online_agg_csv}")
    print(f"Saved offline aggregate CSV: {offline_agg_csv}")

    print(f"Online aggregate rows: {len(online_agg_rows)}")
    print(f"Offline aggregate rows: {len(offline_agg_rows)}")

    best_acc, best_eff = summarize_best_online(online_agg_rows)
    if best_acc is not None:
        print(
            "\n[Online best surviving-path accuracy point] "
            f"sweep_index={best_acc.get('sweep_index')} "
            f"accuracy={format_value(best_acc.get('accuracy'))} "
            f"mean_token_ratio={format_value(best_acc.get('mean_token_ratio'))} "
            f"mean_threshold={format_value(best_acc.get('mean_threshold'))} "
            f"no_early_stop_path_accuracy={format_value(best_acc.get('no_early_stop_path_accuracy'))} "
            f"majority_vote_accuracy={format_value(best_acc.get('majority_vote_accuracy'))}"
        )
    if best_eff is not None:
        print(
            "[Online lowest token-ratio point] "
            f"sweep_index={best_eff.get('sweep_index')} "
            f"accuracy={format_value(best_eff.get('accuracy'))} "
            f"mean_token_ratio={format_value(best_eff.get('mean_token_ratio'))} "
            f"mean_threshold={format_value(best_eff.get('mean_threshold'))}"
        )

    best_offline = summarize_best_offline(offline_agg_rows)
    if best_offline is not None:
        print(
            "\n[Offline best method] "
            f"method={best_offline.get('method')} "
            f"accuracy={format_value(best_offline.get('accuracy'))} "
            f"trace_accuracy_valid_only={format_value(best_offline.get('trace_accuracy_valid_only'))} "
            f"trace_accuracy_base_count={format_value(best_offline.get('trace_accuracy_base_count'))}"
        )

    if args.print_tables:
        print("\n=== Online aggregate table ===")
        print(build_text_table(online_agg_rows, ONLINE_AGG_COLUMNS))
        print("\n=== Offline aggregate table ===")
        print(build_text_table(offline_agg_rows, OFFLINE_AGG_COLUMNS))


if __name__ == "__main__":
    main()