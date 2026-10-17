#!/usr/bin/env python3
"""Compute Pre-CoT AUROC and Prediction-Rejection Ratio.

Two modes are supported:
1) Existing-points mode:
   python calc_precot_auroc_prr.py /path/to/figure1_points.csv

2) Recompute mode, no shell wrapper required:
   python calc_precot_auroc_prr.py \
     --model /path/to/model \
     --data /path/to/data.jsonl \
     --results_dir /path/to/pkls \
     --output_dir /path/to/out \
     --device cuda:0 \
     --recompute_logits



"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


DEFAULT_SCORES = [
    ("first_token_deepconf_confidence", "low", "deepconf_confidence"),
    ("first_token_entropy", "low", "entropy"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute AUROC and Prediction-Rejection Ratio (PRR) for Pre-CoT signals. "
            "Pass points_csv to reuse existing points, or pass --model/--data/--results_dir "
            "to recompute first-token logits before scoring."
        )
    )
    parser.add_argument(
        "points_csv",
        nargs="?",
        default=None,
        help="Optional path to existing figure1_points.csv / points CSV",
    )
    parser.add_argument("--output_dir", default=None, help="Output root for recompute mode")
    parser.add_argument("--output_json", default=None, help="Optional JSON summary path")
    parser.add_argument("--output_csv", default=None, help="Optional CSV summary path")
    parser.add_argument("--output_md", default=None, help="Optional Markdown summary path")

    parser.add_argument("--model", default=None, help="HF model name/path for recompute mode")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer name/path, defaults to --model")
    parser.add_argument("--data", default=None, help="JSONL dataset path for recompute mode")
    parser.add_argument("--results_dir", default=None, help="Directory with deepthink_offline_qid*.pkl")
    parser.add_argument("--device", default="auto", help="Device for recompute mode, e.g. cuda:0")
    parser.add_argument("--cache", default=None, help="Optional logits cache path")
    parser.add_argument("--points_output", default=None, help="Optional generated points CSV path")
    parser.add_argument("--figure_output", default=None, help="Optional generated figure PNG path")
    parser.add_argument("--metrics_output", default=None, help="Optional generated group metrics CSV path")
    parser.add_argument("--recompute_logits", action="store_true", help="Ignore cache and recompute logits")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--confidence_top_k", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--pca_dim", type=int, default=50)
    parser.add_argument("--tsne_perplexity", type=int, default=10)
    parser.add_argument("--tsne_iters", type=int, default=2000)
    parser.add_argument("--system_prompt", default=None)

    parser.add_argument(
        "--score",
        action="append",
        default=None,
        help=(
            "Score spec as column:direction[:name], where direction is high or low. "
            "high means larger score predicts correct; low means smaller score predicts correct. "
            "Can be passed multiple times. Defaults to deepconf confidence and entropy."
        ),
    )
    return parser.parse_args()


def require_recompute_args(args: argparse.Namespace) -> None:
    missing = [
        name
        for name in ["model", "data", "results_dir"]
        if not getattr(args, name)
    ]
    if missing:
        raise ValueError(
            "points_csv was not provided, so recompute mode requires: "
            + ", ".join(f"--{name}" for name in missing)
        )


def resolve_generated_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path, Path]:
    output_root = Path(args.output_dir) if args.output_dir else Path("precot_auroc_prr_run")
    points_path = Path(args.points_output) if args.points_output else output_root / "figures" / "figure1_points.csv"
    figure_path = Path(args.figure_output) if args.figure_output else output_root / "figures" / "figure1_first_token_logits.png"
    metrics_path = Path(args.metrics_output) if args.metrics_output else output_root / "figures" / "figure1_metrics.csv"
    cache_path = Path(args.cache) if args.cache else output_root / "figures" / "first_token_logits_cache.npz"
    return points_path, figure_path, metrics_path, cache_path


def default_metric_paths(args: argparse.Namespace, points_path: Path) -> Tuple[Path, Path, Path]:
    if args.output_dir:
        reports_dir = Path(args.output_dir) / "reports"
        output_json = Path(args.output_json) if args.output_json else reports_dir / "precot_auroc_prr.json"
        output_csv = Path(args.output_csv) if args.output_csv else reports_dir / "precot_auroc_prr.csv"
        output_md = Path(args.output_md) if args.output_md else reports_dir / "precot_auroc_prr.md"
    else:
        output_json = Path(args.output_json) if args.output_json else points_path.with_name("precot_auroc_prr.json")
        output_csv = Path(args.output_csv) if args.output_csv else points_path.with_name("precot_auroc_prr.csv")
        output_md = Path(args.output_md) if args.output_md else points_path.with_name("precot_auroc_prr.md")
    return output_json, output_csv, output_md


def generate_points_from_model(args: argparse.Namespace) -> Path:
    require_recompute_args(args)

    import numpy as np

    precot_dir = Path(__file__).resolve().parent
    if str(precot_dir) not in sys.path:
        sys.path.insert(0, str(precot_dir))
    import figure1_first_token_logits as ftl

    points_path, figure_path, metrics_path, cache_path = resolve_generated_paths(args)
    data_path = Path(args.data)
    results_dir = Path(args.results_dir)
    tokenizer_name = args.tokenizer if args.tokenizer else args.model
    device = args.device
    if device == "auto":
        device = "cuda" if ftl.torch.cuda.is_available() else "cpu"
    system_prompt = (
        ftl.DEFAULT_SYSTEM_PROMPT
        if args.system_prompt is None
        else args.system_prompt
    )

    print(f"[INFO] Loading dataset: {data_path}")
    problems = ftl.load_problems(data_path)
    print(f"[INFO] Loaded {len(problems)} problems.")

    print(f"[INFO] Collecting labels from: {results_dir}")
    labels, keep_idx = ftl.collect_labels(results_dir, problems)
    if labels.size == 0:
        raise RuntimeError("No valid correctness labels were found in result PKLs.")
    print(
        f"[INFO] Labeled problems: {labels.size} | "
        f"correct={(labels == 0).sum()} | wrong={(labels == 1).sum()}"
    )

    if cache_path.exists() and not args.recompute_logits:
        cache = np.load(cache_path)
        reps = cache["reps"]
        cached_n = reps.shape[0]
        if cached_n != len(problems):
            raise RuntimeError(
                f"Cache rows ({cached_n}) do not match dataset size ({len(problems)}). "
                "Use --recompute_logits or a different --cache path."
            )
        print(f"[INFO] Loaded logits cache: {cache_path}")
    else:
        print(f"[INFO] Loading model: {args.model} on {device}")
        print(f"[INFO] Loading tokenizer: {tokenizer_name}")
        tokenizer = ftl.AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        model = ftl.AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=(ftl.torch.float16 if "cuda" in device else ftl.torch.float32),
            device_map=device,
            trust_remote_code=True,
        )
        model.eval()

        prompts = [
            ftl.build_prompt(tokenizer, problem["question"], system_prompt)
            for problem in problems
        ]
        print("[INFO] Extracting first-token logits representations...")
        reps = ftl.extract_first_token_representations(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            temperature=args.temperature,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, reps=reps)
        print(f"[INFO] Saved logits cache: {cache_path}")

    reps = reps[keep_idx]
    if reps.shape[0] < 3:
        raise RuntimeError("Need at least 3 labeled samples for PCA/t-SNE visualization.")

    entropies = ftl.compute_first_token_entropy(reps)
    deepconf_confidences = ftl.compute_deepconf_confidence_topk(
        reps,
        args.confidence_top_k,
    )

    n_pca = min(args.pca_dim, reps.shape[0] - 1, reps.shape[1])
    print(f"[INFO] PCA: {reps.shape[1]} -> {n_pca}")
    pca = ftl.PCA(n_components=n_pca, random_state=42)
    reps_pca = pca.fit_transform(reps)

    perplexity = min(args.tsne_perplexity, max(1, reps.shape[0] - 1))
    print(f"[INFO] t-SNE perplexity: {perplexity}")
    tsne = ftl.TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=args.tsne_iters,
        random_state=42,
        init="pca",
        learning_rate="auto",
    )
    coords = tsne.fit_transform(reps_pca)

    ftl.save_points_data(
        coords=coords,
        labels=labels,
        keep_idx=keep_idx,
        problems=problems,
        entropies=entropies,
        deepconf_confidences=deepconf_confidences,
        output=points_path,
    )
    ftl.save_metrics_summary(
        labels=labels,
        entropies=entropies,
        deepconf_confidences=deepconf_confidences,
        output=metrics_path,
    )
    ftl.plot_figure(coords, labels, figure_path)

    print(f"[INFO] Generated points CSV: {points_path}")
    print(f"[INFO] Generated group metrics CSV: {metrics_path}")
    print(f"[INFO] Generated figure: {figure_path}")
    return points_path


def parse_score_specs(raw_specs: Sequence[str] | None) -> List[Tuple[str, str, str]]:
    if not raw_specs:
        return list(DEFAULT_SCORES)

    specs: List[Tuple[str, str, str]] = []
    for raw in raw_specs:
        parts = raw.split(":")
        if len(parts) not in {2, 3}:
            raise ValueError(f"Invalid --score spec: {raw!r}")
        column = parts[0].strip()
        direction = parts[1].strip().lower()
        if direction not in {"high", "low"}:
            raise ValueError(f"Score direction must be high or low in spec: {raw!r}")
        name = parts[2].strip() if len(parts) == 3 else column
        if not column or not name:
            raise ValueError(f"Invalid --score spec: {raw!r}")
        specs.append((column, direction, name))
    return specs


def read_points(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def label_to_correct(row: Dict[str, str]) -> int:
    label_name = str(row.get("label_name", "")).strip().lower()
    if label_name == "correct":
        return 1
    if label_name == "wrong":
        return 0

    label_id = row.get("label_id")
    if label_id is None or label_id == "":
        raise ValueError("Missing label_name/label_id column.")
    # figure1_first_token_logits.py writes 0=correct, 1=wrong.
    return 1 if int(label_id) == 0 else 0


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def rank_auc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length.")
    n = len(scores)
    positives = sum(1 for y in labels if int(y) == 1)
    negatives = n - positives
    if positives == 0 or negatives == 0:
        return None

    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    rank = 1
    positive_rank_sum = 0.0
    idx = 0
    while idx < n:
        jdx = idx + 1
        while jdx < n and pairs[jdx][0] == pairs[idx][0]:
            jdx += 1
        avg_rank = (rank + rank + (jdx - idx) - 1) / 2.0
        positive_rank_sum += avg_rank * sum(int(label) for _, label in pairs[idx:jdx])
        rank += jdx - idx
        idx = jdx

    return float(
        (positive_rank_sum - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def accuracy_rejection_curve(
    scores: Sequence[float],
    labels: Sequence[int],
) -> List[Dict[str, float]]:
    order = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    sorted_labels = [int(labels[idx]) for idx in order]
    n = len(sorted_labels)
    total_correct = sum(sorted_labels)

    rows: List[Dict[str, float]] = []
    kept_correct = total_correct
    kept_count = n
    rows.append(
        {
            "rejection_fraction": 0.0,
            "coverage": 1.0,
            "accuracy": kept_correct / kept_count if kept_count else 0.0,
        }
    )

    # Reject one low-confidence sample at a time. Accuracy is only defined while
    # at least one sample remains, so the curve ends at coverage=1/N.
    for rejected in range(1, n):
        removed_label = sorted_labels[n - rejected]
        kept_correct -= removed_label
        kept_count -= 1
        rows.append(
            {
                "rejection_fraction": rejected / n,
                "coverage": kept_count / n,
                "accuracy": kept_correct / kept_count if kept_count else 0.0,
            }
        )
    return rows


def curve_auc(curve: Sequence[Dict[str, float]]) -> float:
    if len(curve) < 2:
        return 0.0
    auc = 0.0
    for left, right in zip(curve[:-1], curve[1:]):
        x0 = float(left["rejection_fraction"])
        x1 = float(right["rejection_fraction"])
        y0 = float(left["accuracy"])
        y1 = float(right["accuracy"])
        auc += (x1 - x0) * (y0 + y1) / 2.0
    return float(auc)


def ranked_accuracy_area(scores: Sequence[float], labels: Sequence[int]) -> float:
    """
    Paper PRR area: average accuracy among the top-r retained examples for
    r = 1..N after ranking by confidence.
    """
    order = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    kept_correct = 0
    acc_sum = 0.0
    for rank, idx in enumerate(order, start=1):
        kept_correct += int(labels[idx])
        acc_sum += kept_correct / rank
    return float(acc_sum / len(order))


def compute_prr(scores: Sequence[float], labels: Sequence[int]) -> Tuple[float | None, Dict[str, float]]:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length.")
    if len(scores) < 2:
        return None, {}

    model_auc = ranked_accuracy_area(scores, labels)
    random_accuracy = sum(int(y) for y in labels) / len(labels)
    random_auc = random_accuracy

    oracle_scores = [float(label) for label in labels]
    oracle_auc = ranked_accuracy_area(oracle_scores, labels)

    denom = oracle_auc - random_auc
    prr = None if abs(denom) <= 1e-12 else float((model_auc - random_auc) / denom)
    return prr, {
        "accuracy_rejection_auc": float(model_auc),
        "random_auc": float(random_auc),
        "oracle_auc": float(oracle_auc),
        "base_accuracy": float(random_accuracy),
    }


def build_summary(
    rows: Sequence[Dict[str, str]],
    score_specs: Sequence[Tuple[str, str, str]],
) -> List[Dict[str, Any]]:
    labels_all = [label_to_correct(row) for row in rows]
    out: List[Dict[str, Any]] = []

    for column, direction, name in score_specs:
        scores: List[float] = []
        labels: List[int] = []
        dropped = 0
        for row, label in zip(rows, labels_all):
            raw_score = safe_float(row.get(column))
            if raw_score is None:
                dropped += 1
                continue
            score_for_correct = raw_score if direction == "high" else -raw_score
            scores.append(float(score_for_correct))
            labels.append(int(label))

        if not scores:
            raise ValueError(f"No usable scores found for column {column!r}")

        auroc = rank_auc(scores, labels)
        prr, prr_meta = compute_prr(scores, labels)
        out.append(
            {
                "score_name": name,
                "score_column": column,
                "direction": direction,
                "num_points": len(scores),
                "num_dropped": dropped,
                "num_correct": int(sum(labels)),
                "num_wrong": int(len(labels) - sum(labels)),
                "base_accuracy": prr_meta.get("base_accuracy"),
                "auroc_correct": auroc,
                "prr": prr,
                "accuracy_rejection_auc": prr_meta.get("accuracy_rejection_auc"),
                "random_auc": prr_meta.get("random_auc"),
                "oracle_auc": prr_meta.get("oracle_auc"),
            }
        )
    return out


def write_json(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "definition": {
            "positive_label": "correct",
            "auroc": "AUROC for predicting correct traces/problems.",
            "prr": (
                "(ranked retained-accuracy area - random area) / "
                "(oracle area - random area), using the paper's average over "
                "top-r retained samples."
            ),
        },
        "metrics": list(rows),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "score_name",
        "score_column",
        "direction",
        "num_points",
        "num_dropped",
        "num_correct",
        "num_wrong",
        "base_accuracy",
        "auroc_correct",
        "prr",
        "accuracy_rejection_auc",
        "random_auc",
        "oracle_auc",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_markdown(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "score_name",
        "direction",
        "num_points",
        "base_accuracy",
        "auroc_correct",
        "prr",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_cell(row.get(col)) for col in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(rows: Iterable[Dict[str, Any]]) -> None:
    print("============== Pre-CoT AUROC / PRR ==============")
    for row in rows:
        print(
            "{name}: n={n} base_acc={base:.4f} auroc={auroc:.4f} prr={prr:.4f}".format(
                name=row["score_name"],
                n=int(row["num_points"]),
                base=float(row["base_accuracy"]),
                auroc=float(row["auroc_correct"]) if row["auroc_correct"] is not None else float("nan"),
                prr=float(row["prr"]) if row["prr"] is not None else float("nan"),
            )
        )
    print("=================================================")


def main() -> None:
    args = parse_args()
    if args.points_csv:
        points_path = Path(args.points_csv)
    else:
        points_path = generate_points_from_model(args)

    rows = read_points(points_path)
    score_specs = parse_score_specs(args.score)
    summary = build_summary(rows, score_specs)

    output_json, output_csv, output_md = default_metric_paths(args, points_path)

    write_json(output_json, summary)
    write_csv(output_csv, summary)
    write_markdown(output_md, summary)
    print_summary(summary)
    print(f"[INFO] JSON: {output_json}")
    print(f"[INFO] CSV:  {output_csv}")
    print(f"[INFO] MD:   {output_md}")


if __name__ == "__main__":
    main()
