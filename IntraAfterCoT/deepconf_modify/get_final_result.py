#!/usr/bin/env python3
"""
Export chart/table artifacts from a postprocess_single_* directory.

Inputs:
- a postprocess_single_* directory, or
- a metrics_summary.json path

Outputs:
- figures/passk_curve.png
- figures/precot_confidence_1d.png
- figures/precot_tsne_confidence_2d.png
- reports/passk_curve.csv
- reports/online_summary_table.csv
- reports/online_summary_table.md
- reports/offline_summary_table.csv
- reports/offline_summary_table.md
- reports/precot_confidence_bins.csv
- reports/precot_confidence_metrics.json
- reports/precot_confidence_metrics.md

Important behavior change in this version:
- Pre-CoT ECE is computed against qid-level mean `trace_accuracy_base_count`
  aggregated from `reports/deepconf_full.json`.
- `trace_accuracy_base_count` uses denominator = sampled trace count, so traces
  without answers are counted as incorrect.
- If your upstream postprocess used `OFFLINE_RESAMPLES=100`, then this ECE is
  based on 100 repeated samples per qid. The script reads the actual repeat
  count from `deepconf_full.json` and records it in the summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


PASSK_COLUMNS = ["k", "pass_at_k"]
ONLINE_COLUMNS = [
    "sweep_index",
    "mean_threshold",
    "mean_token_ratio",
    "accuracy",
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
]
PRECOT_BINS_COLUMNS = [
    "normalization_method",
    "bin_index",
    "bin_lower",
    "bin_upper",
    "count",
    "fraction",
    "mean_confidence",
    "empirical_accuracy",
    "abs_gap",
]
PRECOT_LABEL_COLORS = {
    1: "#3fa7ff",  # correct
    0: "#ff5f5f",  # wrong
}
PRECOT_LABEL_NAMES = {
    1: "correct",
    0: "wrong",
}
PRECOT_ECE_TARGET_METRIC = "trace_accuracy_base_count"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export pass@k chart and DeepConf summary tables from a postprocess directory. "
            "Pre-CoT ECE uses qid-level mean trace_accuracy_base_count."
        )
    )
    parser.add_argument(
        "input_path",
        help="Path to a postprocess_single_* directory or to reports/metrics_summary.json",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional output root directory. Defaults to the postprocess root.",
    )
    parser.add_argument(
        "--expect_offline_resamples",
        type=int,
        default=100,
        help="Expected upstream offline resample count for the ECE target summary (default: 100).",
    )
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(data).__name__}.")
    return data


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def resolve_layout(input_path: Path, output_dir: str | None) -> Tuple[Path, Path, Path]:
    if input_path.is_file():
        if input_path.name != "metrics_summary.json":
            raise ValueError("When input_path is a file, it must be metrics_summary.json.")
        reports_dir = input_path.parent
        postprocess_dir = reports_dir.parent
    else:
        postprocess_dir = input_path
        reports_dir = postprocess_dir / "reports"

    if not postprocess_dir.exists():
        raise FileNotFoundError(f"Path not found: {postprocess_dir}")
    if not reports_dir.exists():
        raise FileNotFoundError(f"Reports directory not found: {reports_dir}")

    metrics_path = reports_dir / "metrics_summary.json"
    output_root = Path(output_dir) if output_dir else postprocess_dir
    return postprocess_dir, metrics_path, output_root


def synthesize_metrics_from_reports(reports_dir: Path) -> Dict[str, Any]:
    passk_path = reports_dir / "passk.json"
    deepconf_path = reports_dir / "deepconf_dataset_view.json"
    if not passk_path.exists() or not deepconf_path.exists():
        missing = [str(p) for p in [passk_path, deepconf_path] if not p.exists()]
        raise FileNotFoundError(
            "metrics_summary.json not found, and fallback reports are missing: "
            + ", ".join(missing)
        )

    passk = read_json(passk_path)
    deepconf_raw = read_json(deepconf_path)

    offline_summary = []
    raw_offline = deepconf_raw.get("offline_dataset_metrics", {})
    for method in OFFLINE_METHOD_ORDER:
        stats = raw_offline.get(method)
        if not isinstance(stats, dict):
            continue
        offline_summary.append(
            {
                "method": method,
                "accuracy": stats.get("accuracy"),
                "answer_rate": stats.get("answer_rate"),
                "trace_accuracy_valid_only": stats.get("trace_accuracy_valid_only"),
                "trace_accuracy_base_count": stats.get("trace_accuracy_base_count"),
            }
        )

    return {
        "passk": passk,
        "deepconf": {
            "summary": deepconf_raw.get("summary", {}),
            "online_summary": deepconf_raw.get("online_dataset_token_vs_accuracy", []),
            "offline_summary": offline_summary,
        },
    }


def load_metrics(metrics_path: Path) -> Dict[str, Any]:
    if metrics_path.exists():
        return read_json(metrics_path)
    return synthesize_metrics_from_reports(metrics_path.parent)


def parse_passk_key(key: str) -> int:
    prefix = "pass@"
    if not isinstance(key, str) or not key.startswith(prefix):
        raise ValueError(f"Invalid pass@k key: {key!r}")
    return int(key[len(prefix) :])


def build_passk_rows(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    passk = metrics.get("passk")
    if not isinstance(passk, dict):
        raise ValueError("Missing 'passk' section in metrics data.")

    meta = passk.get("meta", {})
    results = passk.get("results", {})
    if not isinstance(results, dict):
        raise ValueError("Invalid 'passk.results' in metrics data.")

    ks = meta.get("ks")
    if not isinstance(ks, list) or not ks:
        ks = sorted(parse_passk_key(key) for key in results.keys())

    rows: List[Dict[str, Any]] = []
    for k in ks:
        value = results.get(f"pass@{k}")
        rows.append({"k": int(k), "pass_at_k": value})
    return rows


def build_online_rows(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    deepconf = metrics.get("deepconf")
    if not isinstance(deepconf, dict):
        raise ValueError("Missing 'deepconf' section in metrics data.")

    rows = deepconf.get("online_summary", [])
    if not isinstance(rows, list):
        raise ValueError("Invalid 'deepconf.online_summary' in metrics data.")

    clean_rows: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        clean_rows.append(
            {
                "sweep_index": row.get("sweep_index"),
                "mean_threshold": row.get("mean_threshold", row.get("threshold")),
                "mean_token_ratio": row.get("mean_token_ratio", row.get("token_ratio")),
                "accuracy": row.get("accuracy"),
                "surviving_path_keep_rate": row.get("surviving_path_keep_rate"),
                "majority_vote_accuracy": row.get("majority_vote_accuracy"),
            }
        )

    clean_rows.sort(key=lambda row: (row["sweep_index"] is None, row["sweep_index"]))
    return clean_rows


def build_offline_rows(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    deepconf = metrics.get("deepconf")
    if not isinstance(deepconf, dict):
        raise ValueError("Missing 'deepconf' section in metrics data.")

    raw_rows = deepconf.get("offline_summary", [])
    if not isinstance(raw_rows, list):
        raise ValueError("Invalid 'deepconf.offline_summary' in metrics data.")

    by_method: Dict[str, Dict[str, Any]] = {}
    for row in raw_rows:
        if isinstance(row, dict) and isinstance(row.get("method"), str):
            by_method[row["method"]] = row

    rows: List[Dict[str, Any]] = []
    for method in OFFLINE_METHOD_ORDER:
        row = by_method.get(method)
        if row is None:
            continue
        rows.append(
            {
                "method": method,
                "accuracy": row.get("accuracy"),
                "answer_rate": row.get("answer_rate"),
                "trace_accuracy_valid_only": row.get("trace_accuracy_valid_only"),
                "trace_accuracy_base_count": row.get("trace_accuracy_base_count"),
            }
        )

    if not rows:
        raise ValueError(
            "No offline rows found for target methods: "
            + ", ".join(OFFLINE_METHOD_ORDER)
        )

    return rows


def load_precot_points(points_path: Path) -> List[Dict[str, Any]]:
    if not points_path.exists():
        return []

    raw_rows = read_csv_rows(points_path)
    clean_rows: List[Dict[str, Any]] = []
    for row in raw_rows:
        confidence = safe_float(row.get("first_token_deepconf_confidence"))
        entropy = safe_float(row.get("first_token_entropy"))
        tsne_x = safe_float(row.get("tsne_x"))
        tsne_y = safe_float(row.get("tsne_y"))
        qid = safe_int(row.get("qid"))
        label_id = safe_int(row.get("label_id"))
        label_name = str(row.get("label_name", "")).strip().lower()

        if confidence is None or tsne_x is None or tsne_y is None or qid is None:
            continue

        if label_name == "correct":
            is_correct = 1
        elif label_name == "wrong":
            is_correct = 0
        elif label_id is not None:
            is_correct = 1 if label_id == 0 else 0
        else:
            continue

        clean_rows.append(
            {
                "qid": qid,
                "label_name": PRECOT_LABEL_NAMES[is_correct],
                "is_correct": is_correct,
                "tsne_x": tsne_x,
                "tsne_y": tsne_y,
                "first_token_entropy": entropy,
                "first_token_deepconf_confidence": confidence,
            }
        )

    if not clean_rows:
        raise ValueError(f"No usable Pre-CoT points found in: {points_path}")
    return clean_rows


def load_qid_trace_accuracy_base_targets(
    deepconf_full_path: Path,
) -> Tuple[Dict[int, float], Dict[str, Any]]:
    if not deepconf_full_path.exists():
        raise FileNotFoundError(
            "Pre-CoT trace-accuracy ECE requires deepconf_full.json, but it was not found: "
            f"{deepconf_full_path}"
        )

    data = read_json(deepconf_full_path)
    offline_conf = data.get("offline_confidence", {})
    if not isinstance(offline_conf, dict):
        raise ValueError(f"Invalid offline_confidence section in {deepconf_full_path}")

    per_case = offline_conf.get("per_case", [])
    if not isinstance(per_case, list):
        raise ValueError(f"Invalid offline_confidence.per_case in {deepconf_full_path}")

    by_qid_sample: Dict[Tuple[int, int], float] = {}
    duplicate_inconsistencies = 0

    for row in per_case:
        if not isinstance(row, dict):
            continue
        qid = safe_int(row.get("qid"))
        sample_id = safe_int(row.get("sample_id"))
        metric_value = safe_float(row.get(PRECOT_ECE_TARGET_METRIC))
        if qid is None or sample_id is None or metric_value is None:
            continue

        key = (qid, sample_id)
        prev = by_qid_sample.get(key)
        if prev is not None and abs(prev - metric_value) > 1e-12:
            duplicate_inconsistencies += 1
        by_qid_sample.setdefault(key, metric_value)

    by_qid: Dict[int, List[float]] = defaultdict(list)
    for (qid, _sample_id), metric_value in by_qid_sample.items():
        by_qid[qid].append(metric_value)

    if not by_qid:
        raise ValueError(
            f"No per-qid {PRECOT_ECE_TARGET_METRIC} targets found in {deepconf_full_path}"
        )

    qid_to_target: Dict[int, float] = {}
    repeat_counts: List[int] = []
    for qid, values in by_qid.items():
        if not values:
            continue
        qid_to_target[qid] = float(sum(values) / len(values))
        repeat_counts.append(len(values))

    config = data.get("config", {})
    offline_resamples = safe_int(config.get("offline_resamples")) if isinstance(config, dict) else None

    meta = {
        "ece_target_metric": PRECOT_ECE_TARGET_METRIC,
        "offline_resamples_config": offline_resamples,
        "qid_count_with_target": len(qid_to_target),
        "repeat_count_min": min(repeat_counts) if repeat_counts else None,
        "repeat_count_max": max(repeat_counts) if repeat_counts else None,
        "repeat_count_mean": mean_or_none([float(v) for v in repeat_counts]),
        "duplicate_qid_sample_inconsistencies": duplicate_inconsistencies,
        "source_json": str(deepconf_full_path),
    }
    return qid_to_target, meta


def attach_trace_accuracy_targets(
    points: Sequence[Dict[str, Any]],
    qid_to_target: Dict[int, float],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    missing_qids = 0
    for row in points:
        qid = safe_int(row.get("qid"))
        if qid is None or qid not in qid_to_target:
            missing_qids += 1
            continue
        new_row = dict(row)
        new_row[PRECOT_ECE_TARGET_METRIC] = float(qid_to_target[qid])
        rows.append(new_row)

    if not rows:
        raise ValueError(
            f"No Pre-CoT points matched qid-level {PRECOT_ECE_TARGET_METRIC} targets."
        )
    if missing_qids:
        print(
            f"[WARN] Dropped {missing_qids} Pre-CoT point rows without "
            f"{PRECOT_ECE_TARGET_METRIC} targets."
        )
    return rows


def minmax_normalize(values: Sequence[float]) -> Tuple[List[float], float | None, float | None]:
    if not values:
        return [], None, None

    min_value = min(values)
    max_value = max(values)
    if max_value == min_value:
        return [0.5 for _ in values], float(min_value), float(max_value)

    return [
        float((value - min_value) / (max_value - min_value))
        for value in values
    ], float(min_value), float(max_value)


def clamp_to_unit_interval(values: Sequence[float]) -> Tuple[List[float], Dict[str, Any]]:
    below_zero = sum(1 for value in values if value < 0.0)
    above_one = sum(1 for value in values if value > 1.0)
    clipped = [float(min(max(value, 0.0), 1.0)) for value in values]
    return clipped, {
        "method": "raw",
        "clipped_to_unit_interval": bool(below_zero or above_one),
        "num_below_zero": int(below_zero),
        "num_above_one": int(above_one),
    }


def stable_sigmoid(value: float) -> float:
    if value >= 0:
        exp_neg = math.exp(-value)
        return float(1.0 / (1.0 + exp_neg))
    exp_pos = math.exp(value)
    return float(exp_pos / (1.0 + exp_pos))


def platt_normalize(
    scores: Sequence[float],
    labels: Sequence[int],
    max_iter: int = 100,
    tol: float = 1e-7,
    l2: float = 1e-6,
) -> Tuple[List[float], Dict[str, Any]]:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length for Platt scaling.")
    if not scores:
        return [], {"method": "platt", "status": "empty"}

    n = len(scores)
    positives = int(sum(int(v) for v in labels))
    negatives = int(n - positives)

    score_mean = float(sum(scores) / n)
    score_std = float(math.sqrt(sum((value - score_mean) ** 2 for value in scores) / n))
    if score_std <= 0.0:
        score_std = 1.0
    normalized_scores = [float((value - score_mean) / score_std) for value in scores]

    prior = float((positives + 1.0) / (n + 2.0))
    prior = min(max(prior, 1e-6), 1.0 - 1e-6)

    if positives == 0 or negatives == 0:
        probs = [prior for _ in scores]
        return probs, {
            "method": "platt",
            "status": "degenerate_labels",
            "A": 0.0,
            "B": float(math.log(prior / (1.0 - prior))),
            "score_mean": score_mean,
            "score_std": score_std,
            "positive_count": positives,
            "negative_count": negatives,
        }

    A = 0.0
    B = float(math.log(prior / (1.0 - prior)))
    status = "max_iter_reached"

    for _ in range(max_iter):
        g_a = l2 * A
        g_b = l2 * B
        h_aa = l2
        h_ab = 0.0
        h_bb = l2

        for x_value, y_value in zip(normalized_scores, labels):
            probability = stable_sigmoid(A * x_value + B)
            probability = min(max(probability, 1e-12), 1.0 - 1e-12)
            diff = probability - int(y_value)
            weight = probability * (1.0 - probability)

            g_a += diff * x_value
            g_b += diff
            h_aa += weight * x_value * x_value
            h_ab += weight * x_value
            h_bb += weight

        det = h_aa * h_bb - h_ab * h_ab
        if abs(det) <= 1e-12:
            status = "singular_hessian"
            break

        delta_a = (h_bb * g_a - h_ab * g_b) / det
        delta_b = (-h_ab * g_a + h_aa * g_b) / det
        A -= delta_a
        B -= delta_b

        if max(abs(delta_a), abs(delta_b)) < tol:
            status = "converged"
            break

    probabilities = [
        float(min(max(stable_sigmoid(A * x_value + B), 1e-6), 1.0 - 1e-6))
        for x_value in normalized_scores
    ]
    return probabilities, {
        "method": "platt",
        "status": status,
        "A": float(A),
        "B": float(B),
        "score_mean": score_mean,
        "score_std": score_std,
        "positive_count": positives,
        "negative_count": negatives,
    }


def compute_auroc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length for AUROC.")
    if not scores:
        return None

    positives = int(sum(labels))
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None

    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    rank = 1
    positive_rank_sum = 0.0
    idx = 0
    while idx < len(pairs):
        jdx = idx
        while jdx < len(pairs) and pairs[jdx][0] == pairs[idx][0]:
            jdx += 1
        avg_rank = (rank + rank + (jdx - idx) - 1) / 2.0
        positive_rank_sum += avg_rank * sum(label for _, label in pairs[idx:jdx])
        rank += (jdx - idx)
        idx = jdx

    return float(
        (positive_rank_sum - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def build_ece_bins(
    confidences: Sequence[float],
    targets: Sequence[float],
    num_bins: int = 10,
) -> Tuple[float | None, List[Dict[str, Any]]]:
    if len(confidences) != len(targets):
        raise ValueError("confidences and targets must have the same length for ECE.")
    if not confidences:
        return None, []

    total = len(confidences)
    rows: List[Dict[str, Any]] = []
    ece = 0.0

    for bin_index in range(num_bins):
        lower = bin_index / num_bins
        upper = (bin_index + 1) / num_bins

        in_bin: List[int] = []
        for idx, value in enumerate(confidences):
            if bin_index == num_bins - 1:
                if lower <= value <= upper:
                    in_bin.append(idx)
            else:
                if lower <= value < upper:
                    in_bin.append(idx)

        count = len(in_bin)
        fraction = count / total
        if count == 0:
            mean_conf = None
            empirical_acc = None
            abs_gap = None
        else:
            mean_conf = sum(confidences[idx] for idx in in_bin) / count
            empirical_acc = sum(targets[idx] for idx in in_bin) / count
            abs_gap = abs(empirical_acc - mean_conf)
            ece += fraction * abs_gap

        rows.append(
            {
                "bin_index": bin_index,
                "bin_lower": lower,
                "bin_upper": upper,
                "count": count,
                "fraction": fraction,
                "mean_confidence": mean_conf,
                "empirical_accuracy": empirical_acc,
                "abs_gap": abs_gap,
            }
        )

    return float(ece), rows


def build_precot_summary(
    points: Sequence[Dict[str, Any]],
    target_meta: Dict[str, Any],
    num_bins: int = 10,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    confidences = [float(row["first_token_deepconf_confidence"]) for row in points]
    entropies = [
        float(row["first_token_entropy"])
        for row in points
        if row.get("first_token_entropy") is not None
    ]
    labels = [int(row["is_correct"]) for row in points]
    trace_accuracy_targets = [float(row[PRECOT_ECE_TARGET_METRIC]) for row in points]

    raw_confidences, raw_meta = clamp_to_unit_interval(confidences)
    platt_confidences, platt_meta = platt_normalize(confidences, labels)
    minmax_confidences, min_conf, max_conf = minmax_normalize(confidences)

    auroc = compute_auroc(confidences, labels)
    ece_raw, raw_rows = build_ece_bins(raw_confidences, trace_accuracy_targets, num_bins=num_bins)
    ece_platt, platt_rows = build_ece_bins(platt_confidences, trace_accuracy_targets, num_bins=num_bins)
    ece_minmax, minmax_rows = build_ece_bins(minmax_confidences, trace_accuracy_targets, num_bins=num_bins)

    ece_rows: List[Dict[str, Any]] = []
    for method, rows in [
        ("raw", raw_rows),
        ("platt", platt_rows),
        ("minmax", minmax_rows),
    ]:
        for row in rows:
            row_out = dict(row)
            row_out["normalization_method"] = method
            ece_rows.append(row_out)

    correct_confidences = [
        float(row["first_token_deepconf_confidence"])
        for row in points
        if int(row["is_correct"]) == 1
    ]
    wrong_confidences = [
        float(row["first_token_deepconf_confidence"])
        for row in points
        if int(row["is_correct"]) == 0
    ]
    correct_entropies = [
        float(row["first_token_entropy"])
        for row in points
        if int(row["is_correct"]) == 1 and row.get("first_token_entropy") is not None
    ]
    wrong_entropies = [
        float(row["first_token_entropy"])
        for row in points
        if int(row["is_correct"]) == 0 and row.get("first_token_entropy") is not None
    ]
    correct_target_values = [
        float(row[PRECOT_ECE_TARGET_METRIC])
        for row in points
        if int(row["is_correct"]) == 1
    ]
    wrong_target_values = [
        float(row[PRECOT_ECE_TARGET_METRIC])
        for row in points
        if int(row["is_correct"]) == 0
    ]

    summary = {
        "num_points": len(points),
        "num_correct": int(sum(labels)),
        "num_wrong": int(len(points) - sum(labels)),
        "confidence_field": "first_token_deepconf_confidence",
        "positive_label": "correct",
        "confidence_mean_overall": mean_or_none(confidences),
        "confidence_mean_correct": mean_or_none(correct_confidences),
        "confidence_mean_wrong": mean_or_none(wrong_confidences),
        "entropy_mean_overall": mean_or_none(entropies),
        "entropy_mean_correct": mean_or_none(correct_entropies),
        "entropy_mean_wrong": mean_or_none(wrong_entropies),
        "ece_target_metric": PRECOT_ECE_TARGET_METRIC,
        "ece_target_mean_overall": mean_or_none(trace_accuracy_targets),
        "ece_target_mean_correct": mean_or_none(correct_target_values),
        "ece_target_mean_wrong": mean_or_none(wrong_target_values),
        "auroc_confidence_correct_vs_wrong": auroc,
        "ece_raw_confidence": ece_raw,
        "ece_platt_confidence": ece_platt,
        "ece_minmax_confidence": ece_minmax,
        "ece_num_bins": num_bins,
        "ece_confidence_normalization": {
            "raw": raw_meta,
            "platt": platt_meta,
            "minmax": {
                "method": "minmax",
                "min_value": min_conf,
                "max_value": max_conf,
            },
        },
        "trace_accuracy_target_meta": target_meta,
    }
    return summary, ece_rows


def write_csv(path: Path, columns: Sequence[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def format_markdown_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_markdown_table(path: Path, columns: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, separator]
    for row in rows:
        values = [format_markdown_cell(row.get(column)) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_precot_summary_markdown(path: Path, summary: Dict[str, Any]) -> None:
    target_meta = summary.get("trace_accuracy_target_meta", {})
    lines = [
        "# Pre-CoT Confidence Summary",
        "",
        f"- num_points: {summary.get('num_points')}",
        f"- num_correct: {summary.get('num_correct')}",
        f"- num_wrong: {summary.get('num_wrong')}",
        f"- confidence_mean_overall: {format_markdown_cell(summary.get('confidence_mean_overall'))}",
        f"- confidence_mean_correct: {format_markdown_cell(summary.get('confidence_mean_correct'))}",
        f"- confidence_mean_wrong: {format_markdown_cell(summary.get('confidence_mean_wrong'))}",
        f"- entropy_mean_overall: {format_markdown_cell(summary.get('entropy_mean_overall'))}",
        f"- entropy_mean_correct: {format_markdown_cell(summary.get('entropy_mean_correct'))}",
        f"- entropy_mean_wrong: {format_markdown_cell(summary.get('entropy_mean_wrong'))}",
        f"- ece_target_metric: {summary.get('ece_target_metric')}",
        f"- ece_target_mean_overall: {format_markdown_cell(summary.get('ece_target_mean_overall'))}",
        f"- ece_target_mean_correct: {format_markdown_cell(summary.get('ece_target_mean_correct'))}",
        f"- ece_target_mean_wrong: {format_markdown_cell(summary.get('ece_target_mean_wrong'))}",
        f"- auroc_confidence_correct_vs_wrong: {format_markdown_cell(summary.get('auroc_confidence_correct_vs_wrong'))}",
        f"- ece_raw_confidence: {format_markdown_cell(summary.get('ece_raw_confidence'))}",
        f"- ece_platt_confidence: {format_markdown_cell(summary.get('ece_platt_confidence'))}",
        f"- ece_minmax_confidence: {format_markdown_cell(summary.get('ece_minmax_confidence'))}",
        f"- ece_num_bins: {summary.get('ece_num_bins')}",
        f"- offline_resamples_config: {target_meta.get('offline_resamples_config')}",
        f"- repeat_count_min: {target_meta.get('repeat_count_min')}",
        f"- repeat_count_max: {target_meta.get('repeat_count_max')}",
        f"- repeat_count_mean: {format_markdown_cell(target_meta.get('repeat_count_mean'))}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_passk(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required to render the pass@k plot. "
            "Please install it in the current environment."
        ) from exc

    ks = [int(row["k"]) for row in rows if row.get("pass_at_k") is not None]
    values = [float(row["pass_at_k"]) for row in rows if row.get("pass_at_k") is not None]
    if not ks:
        raise ValueError("No valid pass@k values found for plotting.")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, values, marker="o", linewidth=2.0, color="#1f77b4")
    ax.set_title("Pass@K")
    ax.set_xlabel("K")
    ax.set_ylabel("Score")
    ax.set_xticks(ks)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_ylim(0.0, min(1.0, max(values) + 0.05))

    for x_value, y_value in zip(ks, values):
        ax.annotate(
            f"{y_value:.3f}",
            (x_value, y_value),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required to render plots. "
            "Please install matplotlib (and its dependencies) in the current environment."
        ) from exc
    return plt


def plot_precot_confidence_1d(path: Path, points: Sequence[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    plt = import_matplotlib()

    correct_scores = [
        float(row["first_token_deepconf_confidence"])
        for row in points
        if int(row["is_correct"]) == 1
    ]
    wrong_scores = [
        float(row["first_token_deepconf_confidence"])
        for row in points
        if int(row["is_correct"]) == 0
    ]
    if not correct_scores and not wrong_scores:
        raise ValueError("No Pre-CoT confidence values found for 1D plotting.")

    bin_count = max(8, min(24, int(math.sqrt(len(points))) or 8))
    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    if wrong_scores:
        ax.hist(
            wrong_scores,
            bins=bin_count,
            alpha=0.6,
            color=PRECOT_LABEL_COLORS[0],
            label=f"wrong (n={len(wrong_scores)})",
        )
        ax.axvline(
            sum(wrong_scores) / len(wrong_scores),
            color=PRECOT_LABEL_COLORS[0],
            linestyle="--",
            linewidth=1.5,
        )
    if correct_scores:
        ax.hist(
            correct_scores,
            bins=bin_count,
            alpha=0.6,
            color=PRECOT_LABEL_COLORS[1],
            label=f"correct (n={len(correct_scores)})",
        )
        ax.axvline(
            sum(correct_scores) / len(correct_scores),
            color=PRECOT_LABEL_COLORS[1],
            linestyle="--",
            linewidth=1.5,
        )

    title = "Pre-CoT Confidence Distribution"
    auroc = summary.get("auroc_confidence_correct_vs_wrong")
    ece_raw = summary.get("ece_raw_confidence")
    ece_platt = summary.get("ece_platt_confidence")
    ece_minmax = summary.get("ece_minmax_confidence")
    if auroc is not None or ece_raw is not None or ece_platt is not None or ece_minmax is not None:
        subtitle_parts = []
        if auroc is not None:
            subtitle_parts.append(f"AUROC={auroc:.4f}")
        if ece_raw is not None:
            subtitle_parts.append(f"ECE(raw)={ece_raw:.4f}")
        if ece_platt is not None:
            subtitle_parts.append(f"ECE(platt)={ece_platt:.4f}")
        if ece_minmax is not None:
            subtitle_parts.append(f"ECE(min-max)={ece_minmax:.4f}")
        title += " | " + " | ".join(subtitle_parts)

    ax.set_title(title)
    ax.set_xlabel("first_token_deepconf_confidence")
    ax.set_ylabel("Count")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_precot_tsne_2d(path: Path, points: Sequence[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    plt = import_matplotlib()

    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    confidences = [float(row["first_token_deepconf_confidence"]) for row in points]
    scatter_handle = None

    for label_value in [1, 0]:
        subset = [row for row in points if int(row["is_correct"]) == label_value]
        if not subset:
            continue

        xs = [float(row["tsne_x"]) for row in subset]
        ys = [float(row["tsne_y"]) for row in subset]
        cs = [float(row["first_token_deepconf_confidence"]) for row in subset]
        marker = "o" if label_value == 1 else "X"

        scatter_handle = ax.scatter(
            xs,
            ys,
            c=cs,
            cmap="viridis",
            vmin=min(confidences),
            vmax=max(confidences),
            marker=marker,
            s=64 if label_value == 1 else 76,
            alpha=0.9,
            edgecolors="white",
            linewidths=0.35,
            label=f"{PRECOT_LABEL_NAMES[label_value]} (n={len(subset)})",
        )

    ax.set_title("Pre-CoT t-SNE (color = confidence, marker = correctness)")
    ax.set_xlabel("t-SNE x")
    ax.set_ylabel("t-SNE y")
    ax.grid(True, linestyle="--", alpha=0.2)
    ax.legend()

    if scatter_handle is not None:
        colorbar = fig.colorbar(scatter_handle, ax=ax)
        colorbar.set_label("first_token_deepconf_confidence")

    auroc = summary.get("auroc_confidence_correct_vs_wrong")
    ece_raw = summary.get("ece_raw_confidence")
    ece_platt = summary.get("ece_platt_confidence")
    ece_minmax = summary.get("ece_minmax_confidence")
    metrics_text_parts = []
    if auroc is not None:
        metrics_text_parts.append(f"AUROC={auroc:.4f}")
    if ece_raw is not None:
        metrics_text_parts.append(f"ECE(raw)={ece_raw:.4f}")
    if ece_platt is not None:
        metrics_text_parts.append(f"ECE(platt)={ece_platt:.4f}")
    if ece_minmax is not None:
        metrics_text_parts.append(f"ECE(min-max)={ece_minmax:.4f}")
    if metrics_text_parts:
        ax.text(
            0.02,
            0.98,
            "\n".join(metrics_text_parts),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path).expanduser().resolve()
    postprocess_dir, metrics_path, output_root = resolve_layout(input_path, args.output_dir)
    metrics = load_metrics(metrics_path)

    passk_rows = build_passk_rows(metrics)
    online_rows = build_online_rows(metrics)
    offline_rows = build_offline_rows(metrics)

    reports_dir = output_root / "reports"
    figures_dir = output_root / "figures"
    precot_points_path = postprocess_dir / "figures" / "figure1_points.csv"
    deepconf_full_path = postprocess_dir / "reports" / "deepconf_full.json"

    passk_csv_path = reports_dir / "passk_curve.csv"
    passk_png_path = figures_dir / "passk_curve.png"
    online_csv_path = reports_dir / "online_summary_table.csv"
    online_md_path = reports_dir / "online_summary_table.md"
    offline_csv_path = reports_dir / "offline_summary_table.csv"
    offline_md_path = reports_dir / "offline_summary_table.md"
    precot_bins_csv_path = reports_dir / "precot_confidence_bins.csv"
    precot_metrics_json_path = reports_dir / "precot_confidence_metrics.json"
    precot_metrics_md_path = reports_dir / "precot_confidence_metrics.md"
    precot_1d_png_path = figures_dir / "precot_confidence_1d.png"
    precot_tsne_png_path = figures_dir / "precot_tsne_confidence_2d.png"

    write_csv(passk_csv_path, PASSK_COLUMNS, passk_rows)
    plot_passk(passk_png_path, passk_rows)
    write_csv(online_csv_path, ONLINE_COLUMNS, online_rows)
    write_markdown_table(online_md_path, ONLINE_COLUMNS, online_rows)
    write_csv(offline_csv_path, OFFLINE_COLUMNS, offline_rows)
    write_markdown_table(offline_md_path, OFFLINE_COLUMNS, offline_rows)

    precot_points = load_precot_points(precot_points_path)
    if precot_points:
        qid_to_target, target_meta = load_qid_trace_accuracy_base_targets(deepconf_full_path)
        precot_points = attach_trace_accuracy_targets(precot_points, qid_to_target)

        expected = args.expect_offline_resamples
        observed = target_meta.get("offline_resamples_config")
        if expected is not None and observed is not None and observed != expected:
            print(
                f"[WARN] deepconf_full.json reports offline_resamples={observed}, "
                f"not the expected {expected}."
            )
        target_meta["expected_offline_resamples"] = expected

        precot_summary, precot_bin_rows = build_precot_summary(
            precot_points,
            target_meta=target_meta,
            num_bins=10,
        )
        write_csv(precot_bins_csv_path, PRECOT_BINS_COLUMNS, precot_bin_rows)
        write_json(precot_metrics_json_path, precot_summary)
        write_precot_summary_markdown(precot_metrics_md_path, precot_summary)
        plot_precot_confidence_1d(precot_1d_png_path, precot_points, precot_summary)
        plot_precot_tsne_2d(precot_tsne_png_path, precot_points, precot_summary)
    else:
        print(f"[WARN] Pre-CoT points CSV not found: {precot_points_path}")

    print(f"[INFO] Input root: {postprocess_dir}")
    print(f"[INFO] Metrics source: {metrics_path if metrics_path.exists() else metrics_path.parent}")
    print(f"[INFO] Pass@K plot: {passk_png_path}")
    print(f"[INFO] Pass@K CSV: {passk_csv_path}")
    print(f"[INFO] Online table CSV: {online_csv_path}")
    print(f"[INFO] Online table MD: {online_md_path}")
    print(f"[INFO] Offline table CSV: {offline_csv_path}")
    print(f"[INFO] Offline table MD: {offline_md_path}")
    if precot_points:
        print(f"[INFO] Pre-CoT bins CSV: {precot_bins_csv_path}")
        print(f"[INFO] Pre-CoT metrics JSON: {precot_metrics_json_path}")
        print(f"[INFO] Pre-CoT metrics MD: {precot_metrics_md_path}")
        print(f"[INFO] Pre-CoT 1D plot: {precot_1d_png_path}")
        print(f"[INFO] Pre-CoT t-SNE plot: {precot_tsne_png_path}")


if __name__ == "__main__":
    main()