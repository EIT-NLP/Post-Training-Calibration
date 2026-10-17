#!/usr/bin/env python3
"""
Entropy baseline for Pre-CoT analysis.

This script uses the paper prompt and repeated-run labels, then visualizes
first-token entropy distributions for correct vs. wrong problems.

Typical usage:
python figure1_first_token_entropy.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --data ../IntraAfterCoT/deepconf_modify/examples/aime_2024.jsonl \
  --results_dir runs/offline_pkls/instruct_aime24 \
  --output runs/precot/entropy.png
"""

import argparse
import csv
import json
import math
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from scipy.stats import gaussian_kde
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from dynasor.core.evaluator import math_equal as _math_equal
except Exception:
    _math_equal = None


DEFAULT_SYSTEM_PROMPT = ""

FILENAME_RE = re.compile(
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
        return user_msg



def infer_input_device(model, requested_device: str, use_device_map: bool) -> str:
    if use_device_map and hasattr(model, "hf_device_map"):
        device_values = []
        for value in model.hf_device_map.values():
            if value in {"cpu", "disk", None}:
                continue
            device_values.append(value)
        if device_values:
            first = device_values[0]
            if isinstance(first, int):
                return f"cuda:{first}"
            return str(first)
    return requested_device


@torch.inference_mode()
def extract_first_token_entropy(
    model,
    tokenizer,
    prompts: List[str],
    temperature: float,
    input_device: str,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    entropies: List[float] = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(input_device) for k, v in enc.items()}
        outputs = model(**enc)
        seq_lens = enc["attention_mask"].sum(dim=1)
        for b in range(len(batch)):
            last_idx = int(seq_lens[b].item()) - 1
            logits = outputs.logits[b, last_idx, :].float()
            log_probs = torch.log_softmax(logits / temperature, dim=-1)
            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum().item()
            entropies.append(float(entropy))
        print(f"Processed {min(i + len(batch), len(prompts))}/{len(prompts)}", end="\r", flush=True)
    print()
    return np.array(entropies, dtype=np.float32)



def save_points_data(
    entropies: np.ndarray,
    labels: np.ndarray,
    keep_idx: np.ndarray,
    problems: List[dict],
    output: Path,
) -> None:
    label_map = {0: "correct", 1: "wrong"}
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "row_id",
                "qid",
                "label_id",
                "label_name",
                "first_token_entropy",
                "ground_truth",
                "question",
            ]
        )
        for row_id, (qid, entropy, label) in enumerate(zip(keep_idx, entropies, labels)):
            qid = int(qid)
            label = int(label)
            problem = problems[qid]
            question = str(problem.get("question", ""))
            ground_truth = str(problem.get("answer", "")).strip()
            writer.writerow(
                [
                    row_id,
                    qid,
                    label,
                    label_map[label],
                    float(entropy),
                    ground_truth,
                    question,
                ]
            )



def _safe_kde(values: np.ndarray):
    if values.size < 2:
        return None
    if np.allclose(values, values[0]):
        return None
    try:
        return gaussian_kde(values)
    except Exception:
        return None



def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    x_var = x.var(ddof=1)
    y_var = y.var(ddof=1)
    pooled = ((x.size - 1) * x_var + (y.size - 1) * y_var) / (x.size + y.size - 2)
    if pooled <= 0:
        return float("nan")
    return float((x.mean() - y.mean()) / math.sqrt(pooled))



def plot_figure(entropies: np.ndarray, labels: np.ndarray, output: Path, title: str) -> None:
    palette = {
        0: {"name": "correct", "color": "#3fa7ff"},
        1: {"name": "wrong", "color": "#ff5f5f"},
    }

    correct = entropies[labels == 0]
    wrong = entropies[labels == 1]
    all_vals = entropies
    xmin = float(all_vals.min())
    xmax = float(all_vals.max())
    if math.isclose(xmin, xmax):
        xmin -= 1.0
        xmax += 1.0
    margin = 0.08 * (xmax - xmin)
    xmin -= margin
    xmax += margin

    fig = plt.figure(figsize=(10, 7), facecolor="#0f1220")
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.3], hspace=0.08)
    ax_top = fig.add_subplot(gs[0])
    ax_bottom = fig.add_subplot(gs[1], sharex=ax_top)

    for ax in [ax_top, ax_bottom]:
        ax.set_facecolor("#0f1220")
        ax.tick_params(colors="#a7adc6")
        for spine in ax.spines.values():
            spine.set_edgecolor("#445")

    bins = min(30, max(10, int(round(np.sqrt(len(entropies))))))
    ax_top.hist(
        correct,
        bins=bins,
        density=True,
        alpha=0.35,
        color=palette[0]["color"],
        edgecolor="white",
        linewidth=0.25,
        label="correct",
    )
    ax_top.hist(
        wrong,
        bins=bins,
        density=True,
        alpha=0.35,
        color=palette[1]["color"],
        edgecolor="white",
        linewidth=0.25,
        label="wrong",
    )

    xs = np.linspace(xmin, xmax, 400)
    for cls, vals in [(0, correct), (1, wrong)]:
        kde = _safe_kde(vals)
        if kde is not None:
            ys = kde(xs)
            ax_top.plot(xs, ys, color=palette[cls]["color"], linewidth=2.0)
            ax_top.fill_between(xs, 0, ys, color=palette[cls]["color"], alpha=0.10)
        if vals.size > 0:
            mean_val = float(vals.mean())
            ax_top.axvline(mean_val, color=palette[cls]["color"], linestyle="--", linewidth=1.5, alpha=0.9)

    rng = np.random.default_rng(42)
    correct_y = rng.normal(loc=1.0, scale=0.04, size=correct.size)
    wrong_y = rng.normal(loc=0.0, scale=0.04, size=wrong.size)

    ax_bottom.scatter(
        correct,
        correct_y,
        s=34,
        c=palette[0]["color"],
        alpha=0.85,
        edgecolors="white",
        linewidths=0.3,
        zorder=3,
    )
    ax_bottom.scatter(
        wrong,
        wrong_y,
        s=34,
        c=palette[1]["color"],
        alpha=0.85,
        edgecolors="white",
        linewidths=0.3,
        zorder=3,
    )

    ax_bottom.set_yticks([0.0, 1.0])
    ax_bottom.set_yticklabels(["wrong", "correct"], color="#c4c9df")
    ax_bottom.set_xlabel("First-token entropy", color="#c4c9df")
    ax_top.set_ylabel("Density", color="#c4c9df")
    ax_bottom.set_ylabel("Label", color="#c4c9df")
    ax_top.set_xlim(xmin, xmax)

    d_val = _cohens_d(correct, wrong)
    subtitle = (
        f"correct n={correct.size}, mean={correct.mean():.3f} | "
        f"wrong n={wrong.size}, mean={wrong.mean():.3f}"
    )
    if not math.isnan(d_val):
        subtitle += f" | Cohen's d={d_val:.3f}"

    ax_top.set_title(title, color="white", pad=12)
    ax_top.text(
        0.01,
        0.98,
        subtitle,
        transform=ax_top.transAxes,
        va="top",
        ha="left",
        color="#d8dcf0",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#1b1f33", edgecolor="#444", alpha=0.5),
    )

    ax_top.legend(
        handles=[
            mpatches.Patch(color=palette[0]["color"], label="correct"),
            mpatches.Patch(color=palette[1]["color"], label="wrong"),
        ],
        facecolor="#1b1f33",
        edgecolor="#444",
        framealpha=0.35,
        labelcolor="white",
        loc="upper right",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {output}")



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate entropy-based visualization with first-token entropy and correct/wrong labels."
    )
    parser.add_argument("--model", required=True, help="HF model name or local path")
    parser.add_argument("--data", default=str(Path(__file__).parent / "aime2025.jsonl"), help="JSONL dataset path")
    parser.add_argument("--results_dir", required=True, help="Directory containing deepthink_offline_qid*.pkl")
    parser.add_argument("--output", default="figure1_first_token_entropy.png", help="Output figure path")
    parser.add_argument("--cache", default="first_token_entropy_cache.npz", help="Cache path for entropy values")
    parser.add_argument("--recompute_entropy", action="store_true", help="Ignore cache and recompute entropy")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature before log_softmax")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument(
        "--device_map",
        default=None,
        help='Optional HuggingFace device_map, e.g. "auto" for model sharding across multiple GPUs.',
    )
    parser.add_argument(
        "--points_output",
        default="figure1_first_token_entropy_points.csv",
        help="Output CSV file containing entropy values and labels",
    )
    parser.add_argument(
        "--title",
        default="First-token Entropy Distribution (Correct vs Wrong)",
        help="Figure title",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    results_dir = Path(args.results_dir)
    cache_path = Path(args.cache)
    output_path = Path(args.output)
    points_output_path = Path(args.points_output)

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

    if cache_path.exists() and not args.recompute_entropy:
        cache = np.load(cache_path)
        entropies = cache["entropies"]
        cached_n = entropies.shape[0]
        if cached_n != len(problems):
            raise RuntimeError(
                f"Cache rows ({cached_n}) do not match dataset size ({len(problems)}). "
                "Delete cache or use --recompute_entropy."
            )
        print(f"Loaded entropy cache: {cache_path}")
    else:
        print(f"Loading model: {args.model} | device={args.device} | device_map={args.device_map}")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        torch_dtype = torch.float16 if args.device.startswith("cuda") else torch.float32

        load_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
        }

        use_device_map = args.device_map is not None and args.device != "cpu"
        if use_device_map:
            load_kwargs["device_map"] = args.device_map
            load_kwargs["low_cpu_mem_usage"] = True

        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
        if not use_device_map:
            model.to(args.device)
        model.eval()

        input_device = infer_input_device(model, args.device, use_device_map)
        print(f"Input tensors will be placed on: {input_device}")
        if use_device_map and hasattr(model, "hf_device_map"):
            print(f"Resolved hf_device_map: {model.hf_device_map}")

        prompts = [build_prompt(tokenizer, p["question"], args.system_prompt) for p in problems]
        print("Extracting first-token entropy...")
        entropies = extract_first_token_entropy(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            temperature=args.temperature,
            input_device=input_device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, entropies=entropies)
        print(f"Saved entropy cache: {cache_path}")

    entropies = entropies[keep_idx]
    if entropies.shape[0] < 3:
        raise RuntimeError("Need at least 3 labeled samples for visualization.")

    save_points_data(entropies, labels, keep_idx, problems, points_output_path)
    plot_figure(entropies, labels, output_path, args.title)


if __name__ == "__main__":
    main()
