#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Single-model position-aware postprocess (offline PKLs only).
#
# Required inputs:
#   PKL_DIR: directory containing deepthink_offline_qid*_rid*.pkl
#   MODEL_FAMILY: one of sft, rl, opd, qwen_sft, qwen_rl, qwen_opd,
#                 llama_sft, llama_rl, llama_opd
#
# Outputs:
#   - passk.json
#   - strategy_deepconf_full.json
#   - strategy_deepconf_dataset_view.json
#   - strategy_metrics_summary.json
#   - strategy_metrics_summary.txt
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEEPCONF_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ANALYZER="${ANALYZER:-${SCRIPT_DIR}/analyze_position_strategy_deepconf.py}"
TABLE_EXPORTER="${TABLE_EXPORTER:-${SCRIPT_DIR}/export_position_strategy_tables.py}"

PKL_DIR="${PKL_DIR:-}"
MODEL_FAMILY="${MODEL_FAMILY:-}"
MODEL="${MODEL:-}"

ENV_NAME="${ENV_NAME:-deepconf}"
DATASET="${DATASET:-${DEEPCONF_DIR}/examples/aime_2024_convert.jsonl}"
RID="${RID:-}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/runs/position_strategy_single_${MODEL_FAMILY}_${RUN_TAG}}"
REPORT_DIR="${REPORT_DIR:-${OUT_ROOT}/reports}"

PASSK_LIST="${PASSK_LIST:-8,16,32,64,128,256}"
ONLINE_WARMUP_TRACES="${ONLINE_WARMUP_TRACES:-32}"
ONLINE_TOTAL_BUDGET="${ONLINE_TOTAL_BUDGET:-288}"
ONLINE_SWEEP_POINTS="${ONLINE_SWEEP_POINTS:-50}"
ONLINE_RESAMPLES="${ONLINE_RESAMPLES:-5}"
OFFLINE_SAMPLE_SIZE="${OFFLINE_SAMPLE_SIZE:-256}"
OFFLINE_RESAMPLES="${OFFLINE_RESAMPLES:-5}"
ADAPTIVE_DIVISOR="${ADAPTIVE_DIVISOR:-10}"
SEED="${SEED:-42}"

# Optional strategy overrides. Fractions are relative token positions in [0, 1].
# Example for tail 30% Post-CoT: POST_START_FRACTION=0.70 POST_END_FRACTION=1.00
POST_START_FRACTION="${POST_START_FRACTION:-}"
POST_END_FRACTION="${POST_END_FRACTION:-}"
INTRA_START_FRACTION="${INTRA_START_FRACTION:-}"
INTRA_END_FRACTION="${INTRA_END_FRACTION:-}"

if [[ -z "${PKL_DIR}" ]]; then
  echo "[ERROR] PKL_DIR is required."
  exit 1
fi

if [[ -z "${MODEL_FAMILY}" ]]; then
  echo "[ERROR] MODEL_FAMILY is required. Use one of: sft, rl, opd, qwen_sft, qwen_rl, qwen_opd, llama_sft, llama_rl, llama_opd"
  exit 1
fi

case "${MODEL_FAMILY}" in
  sft|rl|opd|qwen_sft|qwen_rl|qwen_opd|llama_sft|llama_rl|llama_opd) ;;
  *)
    echo "[ERROR] MODEL_FAMILY must be one of: sft, rl, opd, qwen_sft, qwen_rl, qwen_opd, llama_sft, llama_rl, llama_opd. Got: ${MODEL_FAMILY}"
    exit 1
    ;;
esac

if [[ ! -d "${PKL_DIR}" ]]; then
  echo "[ERROR] PKL_DIR does not exist: ${PKL_DIR}"
  exit 1
fi

if [[ ! -f "${ANALYZER}" ]]; then
  echo "[ERROR] Missing analyzer: ${ANALYZER}"
  exit 1
fi

if [[ ! -f "${TABLE_EXPORTER}" ]]; then
  echo "[ERROR] Missing table exporter: ${TABLE_EXPORTER}"
  exit 1
fi

shopt -s nullglob
PKL_FILES=("${PKL_DIR}"/*.pkl)
shopt -u nullglob
if [[ ${#PKL_FILES[@]} -eq 0 ]]; then
  echo "[ERROR] No .pkl files found in: ${PKL_DIR}"
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found."
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

mkdir -p "${REPORT_DIR}"

PASSK_JSON="${REPORT_DIR}/passk.json"
DEEP_FULL_JSON="${REPORT_DIR}/strategy_deepconf_full.json"
DEEP_DATASET_JSON="${REPORT_DIR}/strategy_deepconf_dataset_view.json"
METRICS_JSON="${REPORT_DIR}/strategy_metrics_summary.json"
METRICS_TXT="${REPORT_DIR}/strategy_metrics_summary.txt"

echo "[INFO] PKL_DIR=${PKL_DIR}"
echo "[INFO] MODEL_FAMILY=${MODEL_FAMILY}"
echo "[INFO] MODEL=${MODEL}"
echo "[INFO] DATASET=${DATASET}"
if [[ -n "${RID}" ]]; then
  echo "[INFO] RID=${RID}"
else
  echo "[INFO] RID=<all>"
fi
echo "[INFO] REPORT_DIR=${REPORT_DIR}"

echo "[STEP] Computing Pass@K JSON..."
python "${DEEPCONF_DIR}/compute_passk_from_pkls.py" \
  --inputs "${PKL_DIR}" \
  --ks "${PASSK_LIST}" \
  --include_empty_as_incorrect \
  --json_output "${PASSK_JSON}"

echo "[STEP] Running position-aware DeepConf analysis..."
ANALYZE_RID_ARGS=()
if [[ -n "${RID}" ]]; then
  ANALYZE_RID_ARGS=(--rid "${RID}")
fi

STRATEGY_OVERRIDE_ARGS=()
if [[ -n "${POST_START_FRACTION}" ]]; then
  STRATEGY_OVERRIDE_ARGS+=(--post_start_fraction "${POST_START_FRACTION}")
fi
if [[ -n "${POST_END_FRACTION}" ]]; then
  STRATEGY_OVERRIDE_ARGS+=(--post_end_fraction "${POST_END_FRACTION}")
fi
if [[ -n "${INTRA_START_FRACTION}" ]]; then
  STRATEGY_OVERRIDE_ARGS+=(--intra_start_fraction "${INTRA_START_FRACTION}")
fi
if [[ -n "${INTRA_END_FRACTION}" ]]; then
  STRATEGY_OVERRIDE_ARGS+=(--intra_end_fraction "${INTRA_END_FRACTION}")
fi

python "${ANALYZER}" \
  --results_dir "${PKL_DIR}" \
  --dataset "${DATASET}" \
  --strategy_model "${MODEL_FAMILY}" \
  "${ANALYZE_RID_ARGS[@]}" \
  "${STRATEGY_OVERRIDE_ARGS[@]}" \
  --online_warmup_traces "${ONLINE_WARMUP_TRACES}" \
  --online_total_budget "${ONLINE_TOTAL_BUDGET}" \
  --online_sweep_points "${ONLINE_SWEEP_POINTS}" \
  --online_resamples "${ONLINE_RESAMPLES}" \
  --offline_sample_size "${OFFLINE_SAMPLE_SIZE}" \
  --offline_resamples "${OFFLINE_RESAMPLES}" \
  --adaptive_divisor "${ADAPTIVE_DIVISOR}" \
  --seed "${SEED}" \
  --output_json "${DEEP_FULL_JSON}" \
  --dataset_output_json "${DEEP_DATASET_JSON}" \
  --keep_per_case

echo "[STEP] Building strategy metrics summary..."
python - <<PY
import json
from pathlib import Path

def fmt(v, ndigits=4):
    try:
        return f"{float(v):.{ndigits}f}"
    except Exception:
        return "NaN"

def fmt_int(v):
    try:
        return str(int(v))
    except Exception:
        return "0"

passk_path = Path(r"${PASSK_JSON}")
deep_path = Path(r"${DEEP_DATASET_JSON}")
metrics_json_path = Path(r"${METRICS_JSON}")
metrics_txt_path = Path(r"${METRICS_TXT}")

passk = json.loads(passk_path.read_text(encoding="utf-8"))
deep = json.loads(deep_path.read_text(encoding="utf-8"))

passk_meta = passk.get("meta", {})
passk_results = passk.get("results", {})
config = deep.get("config", {})
online_points = deep.get("online_dataset_token_vs_accuracy", []) or []
offline = deep.get("offline_dataset_metrics", {}) or {}

best_online = max(online_points, key=lambda x: float(x.get("accuracy", -1.0))) if online_points else None

summary = {
    "inputs": {
        "pkl_dir": r"${PKL_DIR}",
        "model": r"${MODEL}",
        "model_family": r"${MODEL_FAMILY}",
        "dataset": r"${DATASET}",
        "rid": r"${RID}",
    },
    "strategy": {
        "post_score_start_fraction": config.get("post_score_start_fraction"),
        "post_score_end_fraction": config.get("post_score_end_fraction"),
        "intra_active_start_fraction": config.get("intra_active_start_fraction"),
        "intra_active_end_fraction": config.get("intra_active_end_fraction"),
        "description": config.get("strategy_description"),
    },
    "passk": {
        "meta": passk_meta,
        "results": passk_results,
    },
    "deepconf": {
        "summary": deep.get("summary", {}),
        "best_online_point": best_online,
        "offline_summary": offline,
    },
}
metrics_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

def passk_sort_key(key):
    try:
        return int(str(key).split("@", 1)[1])
    except Exception:
        return 10**9

lines = []
lines.append("============== Position-Aware Strategy Summary ==============")
lines.append(f"PKL_DIR:      {r'${PKL_DIR}'}")
lines.append(f"MODEL:        {r'${MODEL}'}")
lines.append(f"MODEL_FAMILY: {r'${MODEL_FAMILY}'}")
lines.append(f"DATASET:      {r'${DATASET}'}")
lines.append(f"RID:          {r'${RID}'}")
lines.append("")
lines.append("[Strategy]")
lines.append(f"Post-CoT score window: [{config.get('post_score_start_fraction')}, {config.get('post_score_end_fraction')}]")
lines.append(f"Intra-CoT active range: [{config.get('intra_active_start_fraction')}, {config.get('intra_active_end_fraction')}]")
lines.append(str(config.get("strategy_description", "")))
lines.append("")
lines.append("[Pass@K]")
lines.append(
    "PKLs found: {found} | used: {used} | skipped: {skipped}".format(
        found=fmt_int(passk_meta.get("num_pkls_found")),
        used=fmt_int(passk_meta.get("num_questions_used")),
        skipped=fmt_int(passk_meta.get("num_questions_skipped")),
    )
)
lines.append(" ".join(
    f"{key}={fmt(passk_results.get(key))}"
    for key in sorted(passk_results.keys(), key=passk_sort_key)
) or "No pass@k values found.")
lines.append("")
lines.append("[Intra-CoT]")
if best_online:
    lines.append(
        "best_sweep={sweep} acc={acc} token_ratio={tok} keep_rate={keep} majority_vote_acc={mv}".format(
            sweep=fmt_int(best_online.get("sweep_index")),
            acc=fmt(best_online.get("accuracy")),
            tok=fmt(best_online.get("token_ratio")),
            keep=fmt(best_online.get("surviving_path_keep_rate")),
            mv=fmt(best_online.get("majority_vote_accuracy")),
        )
    )
else:
    lines.append("No online sweep rows found.")
lines.append("")
lines.append("[Post-CoT]")
for method in [
    "majority_voting",
    "most_confidence",
    "top5_confidence_base256",
    "top10_confidence_base256",
]:
    stats = offline.get(method)
    if not stats:
        continue
    lines.append(
        "{method}: acc={acc} answer_rate={ar} trace_base={tb}".format(
            method=method,
            acc=fmt(stats.get("accuracy")),
            ar=fmt(stats.get("answer_rate")),
            tb=fmt(stats.get("trace_accuracy_base_count")),
        )
    )
lines.append("")
lines.append(f"metrics_json: {metrics_json_path}")
lines.append(f"metrics_txt:  {metrics_txt_path}")
lines.append(f"deepconf_dataset_json: {deep_path}")
lines.append("=============================================================")

text = "\n".join(lines) + "\n"
metrics_txt_path.write_text(text, encoding="utf-8")
print(text, end="")
PY

echo "[STEP] Exporting online/offline summary tables..."
python "${TABLE_EXPORTER}" "${OUT_ROOT}"

echo
echo "[DONE] Position-aware single-model postprocess finished."
echo "[INFO] Pass@K JSON: ${PASSK_JSON}"
echo "[INFO] Strategy DeepConf full JSON: ${DEEP_FULL_JSON}"
echo "[INFO] Strategy DeepConf dataset JSON: ${DEEP_DATASET_JSON}"
echo "[INFO] Online summary table CSV: ${REPORT_DIR}/online_summary_table.csv"
echo "[INFO] Offline summary table CSV: ${REPORT_DIR}/offline_summary_table.csv"
echo "[INFO] Strategy metrics summary JSON: ${METRICS_JSON}"
echo "[INFO] Strategy metrics summary TXT: ${METRICS_TXT}"
