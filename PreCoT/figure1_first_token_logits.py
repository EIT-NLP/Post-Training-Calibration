#!/usr/bin/env python3
"""
Figure-1 style visualization using:
1) question-level correctness labels (correct/wrong) from deepconf result PKLs
2) first-token logits before generation from an LLM forward pass

Typical usage:
python figure1_first_token_logits.py \
  --model /path/to/model \
  --data aime2025.jsonl \
  --results_dir /path/to/offline_results \
  --output figure1_logits.png



"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import AutoModelForCausalLM, AutoTokenizer
import csv

try:
    from dynasor.core.evaluator import math_equal as _math_equal
except Exception:
    _math_equal = None


DEFAULT_SYSTEM_PROMPT = (
    ""
)

FILENAME_RE = re.compile(
    r"^deepthink_offline_qid(?P<qid>\d+)_rid(?P<rid>.+)_(?P<ts>\d{8}_\d{6})\.pkl$"
)
DEFAULT_CONFIDENCE_TOP_K = 20


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

def save_points_data(
    coords: np.ndarray,
    labels: np.ndarray,
    keep_idx: np.ndarray,
    problems: List[dict],
    entropies: np.ndarray,
    deepconf_confidences: np.ndarray,
    output: Path,
    ) -> None:
    label_map = {
        0: "correct",
        1: "wrong",
    }

    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "row_id",
            "qid",
            "label_id",
            "label_name",
            "tsne_x",
            "tsne_y",
            "first_token_entropy",
            "first_token_deepconf_confidence",
        ])

        for row_id, (qid, coord, label, entropy, deepconf_conf) in enumerate(
            zip(keep_idx, coords, labels, entropies, deepconf_confidences)
        ):
            qid = int(qid)
            label = int(label)
            x, y = float(coord[0]), float(coord[1])
            entropy = float(entropy)
            deepconf_conf = float(deepconf_conf)

            problem = problems[qid]
            question = str(problem.get("question", ""))
            ground_truth = str(problem.get("answer", "")).strip()

            writer.writerow([
                row_id,
                qid,
                label,
                label_map[label],
                x,
                y,
                entropy,
                deepconf_conf
            ])


def save_metrics_summary(
    labels: np.ndarray,
    entropies: np.ndarray,
    deepconf_confidences: np.ndarray,
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    def stats(vals: np.ndarray) -> Tuple[float, float]:
        if vals.size == 0:
            return float("nan"), float("nan")
        return float(np.mean(vals)), float(np.std(vals))

    rows = []
    groups = [
        ("overall", np.ones_like(labels, dtype=bool)),
        ("correct", labels == 0),
        ("wrong", labels == 1),
    ]
    for name, mask in groups:
        ent_mean, ent_std = stats(entropies[mask])
        dconf_mean, dconf_std = stats(deepconf_confidences[mask])
        rows.append(
            {
                "group": name,
                "count": int(mask.sum()),
                "entropy_mean": ent_mean,
                "entropy_std": ent_std,
                "deepconf_confidence_mean": dconf_mean,
                "deepconf_confidence_std": dconf_std,
            }
        )

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "group",
                "count",
                "entropy_mean",
                "entropy_std",
                "deepconf_confidence_mean",
                "deepconf_confidence_std",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["group"],
                    row["count"],
                    row["entropy_mean"],
                    row["entropy_std"],
                    row["deepconf_confidence_mean"],
                    row["deepconf_confidence_std"],
                ]
            )

def load_problems(path: Path) -> List[dict]:
    problems: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            problems.append(json.loads(line))
    return problems


def parse_result_name(filename: str) -> Optional[Tuple[int, str]]:
    match = FILENAME_RE.match(filename)
    if not match:
        return None
    qid = int(match.group("qid"))
    ts = match.group("ts")
    return qid, ts


def majority_vote(answers: List[str]) -> Optional[str]:
    if not answers:
        return None
    counts: Dict[str, int] = {}
    for ans in answers:
        key = str(ans)
        counts[key] = counts.get(key, 0) + 1
    return max(counts.items(), key=lambda x: x[1])[0]


def get_correctness_label(record: dict, ground_truth: str) -> Optional[bool]:
    evaluation = record.get("evaluation") or {}
    if "majority" in evaluation and isinstance(evaluation["majority"], dict):
        is_correct = evaluation["majority"].get("is_correct")
        if isinstance(is_correct, bool):
            return is_correct

    for method_eval in evaluation.values():
        if isinstance(method_eval, dict):
            is_correct = method_eval.get("is_correct")
            if isinstance(is_correct, bool):
                return is_correct

    voting_results = record.get("voting_results") or {}
    majority = voting_results.get("majority") if isinstance(voting_results, dict) else None
    if isinstance(majority, dict):
        answer = majority.get("answer")
        if answer is not None and str(answer).strip() and ground_truth:
            return equal_func(str(answer), ground_truth)

    traces = record.get("all_traces") or []
    answers = []
    for trace in traces:
        if not isinstance(trace, dict):
            continue
        ans = trace.get("extracted_answer")
        if ans is None:
            continue
        ans = str(ans).strip()
        if ans:
            answers.append(ans)

    pred = majority_vote(answers)
    if pred is None or not ground_truth:
        return None
    return equal_func(pred, ground_truth)


def collect_labels(results_dir: Path, problems: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
    runs_by_qid: Dict[int, List[Path]] = {}
    for pkl_path in results_dir.glob("deepthink_offline_qid*_rid*.pkl"):
        parsed = parse_result_name(pkl_path.name)
        if parsed is None:
            continue
        qid, _ts = parsed
        runs_by_qid.setdefault(qid, []).append(pkl_path)

    keep_idx: List[int] = []
    labels: List[int] = []
    for qid, pkl_paths in sorted(runs_by_qid.items(), key=lambda x: x[0]):
        if qid < 0 or qid >= len(problems):
            continue
        gt = str(problems[qid].get("answer", "")).strip()
        run_labels: List[bool] = []
        for pkl_path in sorted(pkl_paths):
            with pkl_path.open("rb") as f:
                record = pickle.load(f)
            label = get_correctness_label(record, gt)
            if label is not None:
                run_labels.append(bool(label))
        if not run_labels:
            continue
        label = sum(run_labels) >= (len(run_labels) / 2)
        keep_idx.append(qid)
        labels.append(0 if label else 1)  # 0=correct, 1=wrong

    return np.array(labels, dtype=np.int64), np.array(keep_idx, dtype=np.int64)


def build_prompt(tokenizer, question: str, system_prompt: str) -> str:
    user_msg = (
    f"{question}\n"
    r"Please reason step by step, and put your final answer within \boxed{}."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # Fallback for tokenizers without chat template support.
        return user_msg


@torch.inference_mode()
def extract_first_token_representations(
    model,
    tokenizer,
    prompts: List[str],
    temperature: float,
    device: str,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    reps: List[np.ndarray] = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        outputs = model(**enc)
        seq_lens = enc["attention_mask"].sum(dim=1)
        for b in range(len(batch)):
            last_idx = int(seq_lens[b].item()) - 1
            logits = outputs.logits[b, last_idx, :].float()
            rep = torch.log_softmax(logits / temperature, dim=-1)
            reps.append(rep.cpu().numpy())
        print(f"Processed {min(i + len(batch), len(prompts))}/{len(prompts)}", end="\r", flush=True)
    print()
    return np.stack(reps, axis=0)


def compute_first_token_entropy(log_probs: np.ndarray) -> np.ndarray:
    """
    log_probs: shape (N, vocab), already log_softmax.
    returns:
      entropies: shape (N,), -sum(p * log p)
    """
    probs = np.exp(log_probs)
    entropies = -np.sum(probs * log_probs, axis=1)
    return entropies


def compute_deepconf_confidence_topk(
    log_probs: np.ndarray,
    top_k: int = DEFAULT_CONFIDENCE_TOP_K,
) -> np.ndarray:
    """
    Paper Pre-CoT confidence:
      C_pre(x) = -1/k * sum_{v in top-k next-token set} log p(v | P_x)
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    k = min(int(top_k), log_probs.shape[1])
    top_log_probs = np.partition(log_probs, -k, axis=1)[:, -k:]
    return -np.mean(top_log_probs, axis=1)


def compute_deepconf_confidence_whole_vocab(log_probs: np.ndarray) -> np.ndarray:
    """
    Deprecated compatibility helper. The paper uses top-k=20 instead.
    """
    return -np.mean(log_probs, axis=1)


def plot_figure(coords: np.ndarray, labels: np.ndarray, output: Path) -> None:
    palette = {
        0: {"name": "correct", "color": "#3fa7ff"},
        1: {"name": "wrong", "color": "#ff5f5f"},
    }

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_facecolor("#0f1220")
    fig.patch.set_facecolor("#0f1220")

    for cls in [0, 1]:
        mask = labels == cls
        points = coords[mask]
        if points.shape[0] == 0:
            continue
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=55,
            c=palette[cls]["color"],
            alpha=0.85,
            edgecolors="white",
            linewidths=0.35,
            label=palette[cls]["name"],
            zorder=3,
        )
        if points.shape[0] >= 3:
            try:
                kde = gaussian_kde(points.T, bw_method="scott")
                xmin, xmax = coords[:, 0].min() - 4, coords[:, 0].max() + 4
                ymin, ymax = coords[:, 1].min() - 4, coords[:, 1].max() + 4
                xx, yy = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
                zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                ax.contour(
                    xx,
                    yy,
                    zz,
                    levels=5,
                    colors=palette[cls]["color"],
                    alpha=0.55,
                    linewidths=0.9,
                    zorder=2,
                )
            except np.linalg.LinAlgError:
                pass

    ax.legend(
        handles=[
            mpatches.Patch(color=palette[0]["color"], label="correct"),
            mpatches.Patch(color=palette[1]["color"], label="wrong"),
        ],
        facecolor="#1b1f33",
        edgecolor="#444",
        framealpha=0.35,
        labelcolor="white",
    )
    ax.set_title("First-token Logits Embedding (Correct vs Wrong)", color="white", pad=12)
    ax.set_xlabel("t-SNE dim 1", color="#c4c9df")
    ax.set_ylabel("t-SNE dim 2", color="#c4c9df")
    ax.tick_params(colors="#a7adc6")
    for spine in ax.spines.values():
        spine.set_edgecolor("#445")

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Figure-1 style visualization with first-token logits and correct/wrong labels."
    )
    parser.add_argument("--model", required=True, help="HF model name or local path")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="HF tokenizer name or local path (defaults to --model)",
    )
    parser.add_argument("--data", default=str(Path(__file__).parent / "aime2025.jsonl"), help="JSONL dataset path")
    parser.add_argument("--results_dir", required=True, help="Directory containing deepthink_offline_qid*.pkl")
    parser.add_argument("--output", default="figure1_first_token_logits.png", help="Output figure path")
    parser.add_argument("--cache", default="first_token_logits_cache.npz", help="Cache path for logits representations")
    parser.add_argument("--recompute_logits", action="store_true", help="Ignore cache and recompute logits")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature before log_softmax")
    parser.add_argument("--confidence_top_k", type=int, default=DEFAULT_CONFIDENCE_TOP_K, help="Top-k next tokens used by paper Pre-CoT confidence")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--pca_dim", type=int, default=50)
    parser.add_argument("--tsne_perplexity", type=int, default=10)
    parser.add_argument("--tsne_iters", type=int, default=2000)
    parser.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--points_output", default="figure1_first_token_logits_points.csv", help="Output CSV file containing point coordinates and labels")
    parser.add_argument("--metrics_output", default="figure1_first_token_logits_metrics.csv", help="Output CSV file for confidence/entropy summary")
    args = parser.parse_args()

    data_path = Path(args.data)
    results_dir = Path(args.results_dir)
    cache_path = Path(args.cache)
    output_path = Path(args.output)
    points_output_path = Path(args.points_output)
    metrics_output_path = Path(args.metrics_output)
    tokenizer_name = args.tokenizer if args.tokenizer else args.model

    print(f"Loading dataset: {data_path}")
    problems = load_problems(data_path)
    print(f"Loaded {len(problems)} problems.")

    print(f"Collecting labels from: {results_dir}")
    labels, keep_idx = collect_labels(results_dir, problems)
    if labels.size == 0:
        raise RuntimeError("No valid correctness labels were found in result PKLs.")
    print(
        f"Labeled problems: {labels.size} | "
        f"correct={(labels == 0).sum()} | wrong={(labels == 1).sum()}"
    )

    if cache_path.exists() and not args.recompute_logits:
        cache = np.load(cache_path)
        reps = cache["reps"]
        cached_n = reps.shape[0]
        if cached_n != len(problems):
            raise RuntimeError(
                f"Cache rows ({cached_n}) do not match dataset size ({len(problems)}). "
                "Delete cache or use --recompute_logits."
            )
        print(f"Loaded logits cache: {cache_path}")
    else:
        print(f"Loading model: {args.model} on {args.device}")
        print(f"Loading tokenizer: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=(torch.float16 if "cuda" in args.device else torch.float32),
            device_map=args.device,
            trust_remote_code=True,
        )
        model.eval()

        prompts = [build_prompt(tokenizer, p["question"], args.system_prompt) for p in problems]
        print("Extracting first-token logits representations...")
        reps = extract_first_token_representations(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            temperature=args.temperature,
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, reps=reps)
        print(f"Saved logits cache: {cache_path}")

    reps = reps[keep_idx]
    if reps.shape[0] < 3:
        raise RuntimeError("Need at least 3 labeled samples for PCA/t-SNE visualization.")

    entropies = compute_first_token_entropy(reps)
    deepconf_confidences = compute_deepconf_confidence_topk(reps, args.confidence_top_k)

    n_pca = min(args.pca_dim, reps.shape[0] - 1, reps.shape[1])
    print(f"PCA: {reps.shape[1]} -> {n_pca}")
    pca = PCA(n_components=n_pca, random_state=42)
    reps_pca = pca.fit_transform(reps)

    perplexity = min(args.tsne_perplexity, max(1, reps.shape[0] - 1))
    print(f"t-SNE perplexity: {perplexity}")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=args.tsne_iters,
        random_state=42,
        init="pca",
        learning_rate="auto",
    )
    coords = tsne.fit_transform(reps_pca)
    save_points_data(
        coords, labels, keep_idx, problems,
        entropies, deepconf_confidences, points_output_path
    )
    save_metrics_summary(labels, entropies, deepconf_confidences, metrics_output_path)
    
    plot_figure(coords, labels, output_path)


if __name__ == "__main__":
    main()
