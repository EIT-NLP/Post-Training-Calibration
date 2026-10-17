#!/usr/bin/env python3
"""Export CSV/Markdown summary tables from a position_strategy output directory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

'''
python position_strategy/export_position_strategy_tables.py \
  position_strategy/runs/position_strategy_batch_<timestamp>/opd_math500
'''

ONLINE_COLUMNS = [
    "sweep_index",
    "mean_threshold",
    "mean_token_ratio",
    "global_token_ratio_by_tokens",
    "accuracy",
    "valid_answer_accuracy",
    "surviving_path_keep_rate",
    "majority_vote_accuracy",
]

OFFLINE_COLUMNS = [
    "method",
    "accuracy",
    "answer_rate",
    "trace_accuracy_valid_only",
    "trace_accuracy_base_count",
]

OFFLINE_METHOD_ORDER = [
    "majority_voting",
    "most_confidence",
    "top5_confidence_base256",
    "top10_confidence_base256",
    "top5_confidence_valid",
    "top10_confidence_valid",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export online/offline summary tables from position_strategy results."
    )
    parser.add_argument(
        "input_path",
        help=(
            "Position strategy output root, reports directory, "
            "or reports/strategy_deepconf_dataset_view.json."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional output root. Defaults to the input output root.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_layout(input_path: Path, output_dir: Optional[str]) -> Tuple[Path, Path]:
    input_path = input_path.expanduser().resolve()

    if input_path.is_file():
        dataset_json = input_path
        reports_dir = dataset_json.parent
        output_root = Path(output_dir).expanduser().resolve() if output_dir else reports_dir.parent
    elif input_path.name == "reports":
        reports_dir = input_path
        dataset_json = reports_dir / "strategy_deepconf_dataset_view.json"
        output_root = Path(output_dir).expanduser().resolve() if output_dir else reports_dir.parent
    else:
        output_root = Path(output_dir).expanduser().resolve() if output_dir else input_path
        reports_dir = input_path / "reports"
        dataset_json = reports_dir / "strategy_deepconf_dataset_view.json"

    if not dataset_json.exists():
        raise FileNotFoundError(f"Missing strategy dataset JSON: {dataset_json}")

    return dataset_json, output_root


def write_csv(path: Path, columns: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col) for col in columns})


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_markdown(path: Path, columns: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_cell(row.get(col)) for col in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_online_rows(dataset_view: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_rows = dataset_view.get("online_dataset_token_vs_accuracy", [])
    if not isinstance(raw_rows, list):
        raise ValueError("Invalid online_dataset_token_vs_accuracy in strategy JSON.")

    rows: List[Dict[str, Any]] = []
    for row in raw_rows:
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "sweep_index": row.get("sweep_index"),
                "mean_threshold": row.get("mean_threshold", row.get("threshold")),
                "mean_token_ratio": row.get("mean_token_ratio", row.get("token_ratio")),
                "global_token_ratio_by_tokens": row.get("global_token_ratio_by_tokens"),
                "accuracy": row.get("accuracy"),
                "valid_answer_accuracy": row.get("valid_answer_accuracy"),
                "surviving_path_keep_rate": row.get("surviving_path_keep_rate"),
                "majority_vote_accuracy": row.get("majority_vote_accuracy"),
            }
        )
    return rows


def build_offline_rows(dataset_view: Dict[str, Any]) -> List[Dict[str, Any]]:
    offline = dataset_view.get("offline_dataset_metrics", {})
    if not isinstance(offline, dict):
        raise ValueError("Invalid offline_dataset_metrics in strategy JSON.")

    rows: List[Dict[str, Any]] = []
    seen = set()
    for method in OFFLINE_METHOD_ORDER:
        stats = offline.get(method)
        if not isinstance(stats, dict):
            continue
        seen.add(method)
        rows.append(
            {
                "method": method,
                "accuracy": stats.get("accuracy"),
                "answer_rate": stats.get("answer_rate"),
                "trace_accuracy_valid_only": stats.get("trace_accuracy_valid_only"),
                "trace_accuracy_base_count": stats.get("trace_accuracy_base_count"),
            }
        )

    for method in sorted(set(offline.keys()) - seen):
        stats = offline.get(method)
        if not isinstance(stats, dict):
            continue
        rows.append(
            {
                "method": method,
                "accuracy": stats.get("accuracy"),
                "answer_rate": stats.get("answer_rate"),
                "trace_accuracy_valid_only": stats.get("trace_accuracy_valid_only"),
                "trace_accuracy_base_count": stats.get("trace_accuracy_base_count"),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    dataset_json, output_root = resolve_layout(Path(args.input_path), args.output_dir)
    dataset_view = read_json(dataset_json)

    reports_dir = output_root / "reports"
    online_rows = build_online_rows(dataset_view)
    offline_rows = build_offline_rows(dataset_view)

    online_csv = reports_dir / "online_summary_table.csv"
    online_md = reports_dir / "online_summary_table.md"
    offline_csv = reports_dir / "offline_summary_table.csv"
    offline_md = reports_dir / "offline_summary_table.md"

    write_csv(online_csv, ONLINE_COLUMNS, online_rows)
    write_markdown(online_md, ONLINE_COLUMNS, online_rows)
    write_csv(offline_csv, OFFLINE_COLUMNS, offline_rows)
    write_markdown(offline_md, OFFLINE_COLUMNS, offline_rows)

    print(f"[INFO] Wrote: {online_csv}")
    print(f"[INFO] Wrote: {online_md}")
    print(f"[INFO] Wrote: {offline_csv}")
    print(f"[INFO] Wrote: {offline_md}")


if __name__ == "__main__":
    main()
