#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Batch runner for position-aware Post-CoT/Intra-CoT strategies.
#
# Strategy table implemented by analyze_position_strategy_deepconf.py:
#   Qwen SFT/RL/OPD use sft/rl/opd by default.
#   Llama replication can use SFT_STRATEGY_MODEL=llama_sft,
#   RL_STRATEGY_MODEL=llama_rl, and OPD_STRATEGY_MODEL=llama_opd.
#
# Examples:
#   bash postprocess_all_position_strategy.sh
#
#   PARALLEL=1 MAX_JOBS=4 bash postprocess_all_position_strategy.sh
#
#   PKL_ROOT=/path/to/offline_pkls BASE_OUTPUT_ROOT=runs/position_strategy_results \
#     bash postprocess_all_position_strategy.sh
#
#   SFT_MODEL=/path/to/qwen_sft RL_MODEL=/path/to/qwen_rl OPD_MODEL=/path/to/qwen_opd \
#     PKL_ROOT=/path/to/offline_pkls bash postprocess_all_position_strategy.sh
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SINGLE_SCRIPT="${SINGLE_SCRIPT:-${SCRIPT_DIR}/postprocess_strategy_single_model.sh}"

ENV_NAME="${ENV_NAME:-deepconf}"
BATCH_TAG="${BATCH_TAG:-$(date +%Y%m%d_%H%M%S)}"
BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-${SCRIPT_DIR}/runs/position_strategy_batch_${BATCH_TAG}}"
LOG_DIR="${LOG_DIR:-${BASE_OUTPUT_ROOT}/logs}"

PARALLEL="${PARALLEL:-0}"
MAX_JOBS="${MAX_JOBS:-1}"

PASSK_LIST="${PASSK_LIST:-8,16,32,64,128,256}"
ONLINE_WARMUP_TRACES="${ONLINE_WARMUP_TRACES:-32}"
ONLINE_TOTAL_BUDGET="${ONLINE_TOTAL_BUDGET:-288}"
ONLINE_SWEEP_POINTS="${ONLINE_SWEEP_POINTS:-50}"
ONLINE_RESAMPLES="${ONLINE_RESAMPLES:-5}"
OFFLINE_SAMPLE_SIZE="${OFFLINE_SAMPLE_SIZE:-256}"
OFFLINE_RESAMPLES="${OFFLINE_RESAMPLES:-5}"
ADAPTIVE_DIVISOR="${ADAPTIVE_DIVISOR:-10}"
SEED="${SEED:-42}"

# Optional per-model-family strategy overrides. Leave empty to use the analyzer
# defaults. For tail 30% Post-CoT, use start=0.70 and end=1.00.
SFT_POST_START_FRACTION="${SFT_POST_START_FRACTION:-}"
SFT_POST_END_FRACTION="${SFT_POST_END_FRACTION:-}"
SFT_INTRA_START_FRACTION="${SFT_INTRA_START_FRACTION:-}"
SFT_INTRA_END_FRACTION="${SFT_INTRA_END_FRACTION:-}"

RL_POST_START_FRACTION="${RL_POST_START_FRACTION:-}"
RL_POST_END_FRACTION="${RL_POST_END_FRACTION:-}"
RL_INTRA_START_FRACTION="${RL_INTRA_START_FRACTION:-}"
RL_INTRA_END_FRACTION="${RL_INTRA_END_FRACTION:-}"

OPD_POST_START_FRACTION="${OPD_POST_START_FRACTION:-}"
OPD_POST_END_FRACTION="${OPD_POST_END_FRACTION:-}"
OPD_INTRA_START_FRACTION="${OPD_INTRA_START_FRACTION:-}"
OPD_INTRA_END_FRACTION="${OPD_INTRA_END_FRACTION:-}"

EXAMPLES_DIR="${EXAMPLES_DIR:-${SCRIPT_DIR}/../examples}"
PKL_ROOT="${PKL_ROOT:-${SCRIPT_DIR}/runs/offline_pkls}"

OPD_MODEL="${OPD_MODEL:-Qwen-OPD}"
RL_MODEL="${RL_MODEL:-Qwen-RL}"
SFT_MODEL="${SFT_MODEL:-Qwen-SFT}"
SFT_STRATEGY_MODEL="${SFT_STRATEGY_MODEL:-sft}"
RL_STRATEGY_MODEL="${RL_STRATEGY_MODEL:-rl}"
OPD_STRATEGY_MODEL="${OPD_STRATEGY_MODEL:-opd}"

AIME24_DATASET="${AIME24_DATASET:-${EXAMPLES_DIR}/aime_2024.jsonl}"
AIME25_DATASET="${AIME25_DATASET:-${EXAMPLES_DIR}/aime_2025.jsonl}"
AMC23_DATASET="${AMC23_DATASET:-${EXAMPLES_DIR}/amc23_convert.jsonl}"
MATH500_DATASET="${MATH500_DATASET:-${EXAMPLES_DIR}/math500_convert.jsonl}"

SFT_AIME24_PKL_DIR="${SFT_AIME24_PKL_DIR:-${PKL_ROOT}/sft_aime24}"
SFT_AIME25_PKL_DIR="${SFT_AIME25_PKL_DIR:-${PKL_ROOT}/sft_aime25}"
SFT_AMC23_PKL_DIR="${SFT_AMC23_PKL_DIR:-${PKL_ROOT}/sft_amc23}"
SFT_MATH500_PKL_DIR="${SFT_MATH500_PKL_DIR:-${PKL_ROOT}/sft_math500}"

RL_AIME24_PKL_DIR="${RL_AIME24_PKL_DIR:-${PKL_ROOT}/rl_aime24}"
RL_AIME25_PKL_DIR="${RL_AIME25_PKL_DIR:-${PKL_ROOT}/rl_aime25}"
RL_AMC23_PKL_DIR="${RL_AMC23_PKL_DIR:-${PKL_ROOT}/rl_amc23}"
RL_MATH500_PKL_DIR="${RL_MATH500_PKL_DIR:-${PKL_ROOT}/rl_math500}"

OPD_AIME24_PKL_DIR="${OPD_AIME24_PKL_DIR:-${PKL_ROOT}/opd_aime24}"
OPD_AIME25_PKL_DIR="${OPD_AIME25_PKL_DIR:-${PKL_ROOT}/opd_aime25}"
OPD_AMC23_PKL_DIR="${OPD_AMC23_PKL_DIR:-${PKL_ROOT}/opd_amc23}"
OPD_MATH500_PKL_DIR="${OPD_MATH500_PKL_DIR:-${PKL_ROOT}/opd_math500}"


JOBS=(
  "sft_aime24|${SFT_STRATEGY_MODEL}|${SFT_MODEL}|${SFT_AIME24_PKL_DIR}|${AIME24_DATASET}"
  "sft_aime25|${SFT_STRATEGY_MODEL}|${SFT_MODEL}|${SFT_AIME25_PKL_DIR}|${AIME25_DATASET}"
  "sft_amc23|${SFT_STRATEGY_MODEL}|${SFT_MODEL}|${SFT_AMC23_PKL_DIR}|${AMC23_DATASET}"
  "sft_math500|${SFT_STRATEGY_MODEL}|${SFT_MODEL}|${SFT_MATH500_PKL_DIR}|${MATH500_DATASET}"
  "rl_aime24|${RL_STRATEGY_MODEL}|${RL_MODEL}|${RL_AIME24_PKL_DIR}|${AIME24_DATASET}"
  "rl_aime25|${RL_STRATEGY_MODEL}|${RL_MODEL}|${RL_AIME25_PKL_DIR}|${AIME25_DATASET}"
  "rl_amc23|${RL_STRATEGY_MODEL}|${RL_MODEL}|${RL_AMC23_PKL_DIR}|${AMC23_DATASET}"
  "rl_math500|${RL_STRATEGY_MODEL}|${RL_MODEL}|${RL_MATH500_PKL_DIR}|${MATH500_DATASET}"
  "opd_aime24|${OPD_STRATEGY_MODEL}|${OPD_MODEL}|${OPD_AIME24_PKL_DIR}|${AIME24_DATASET}"
  "opd_aime25|${OPD_STRATEGY_MODEL}|${OPD_MODEL}|${OPD_AIME25_PKL_DIR}|${AIME25_DATASET}"
  "opd_amc23|${OPD_STRATEGY_MODEL}|${OPD_MODEL}|${OPD_AMC23_PKL_DIR}|${AMC23_DATASET}"
  "opd_math500|${OPD_STRATEGY_MODEL}|${OPD_MODEL}|${OPD_MATH500_PKL_DIR}|${MATH500_DATASET}"
)

if [[ ! -f "${SINGLE_SCRIPT}" ]]; then
  echo "[ERROR] Missing SINGLE_SCRIPT: ${SINGLE_SCRIPT}"
  exit 1
fi

if ! [[ "${MAX_JOBS}" =~ ^[0-9]+$ ]] || (( MAX_JOBS < 1 )); then
  echo "[ERROR] MAX_JOBS must be a positive integer. Got: ${MAX_JOBS}"
  exit 1
fi

mkdir -p "${BASE_OUTPUT_ROOT}" "${LOG_DIR}"

family_setting() {
  local family="$1"
  local suffix="$2"

  case "${family}" in
    sft|qwen_sft|llama_sft)
      case "${suffix}" in
        POST_START_FRACTION) printf '%s\n' "${SFT_POST_START_FRACTION}" ;;
        POST_END_FRACTION) printf '%s\n' "${SFT_POST_END_FRACTION}" ;;
        INTRA_START_FRACTION) printf '%s\n' "${SFT_INTRA_START_FRACTION}" ;;
        INTRA_END_FRACTION) printf '%s\n' "${SFT_INTRA_END_FRACTION}" ;;
      esac
      ;;
    rl|qwen_rl|llama_rl)
      case "${suffix}" in
        POST_START_FRACTION) printf '%s\n' "${RL_POST_START_FRACTION}" ;;
        POST_END_FRACTION) printf '%s\n' "${RL_POST_END_FRACTION}" ;;
        INTRA_START_FRACTION) printf '%s\n' "${RL_INTRA_START_FRACTION}" ;;
        INTRA_END_FRACTION) printf '%s\n' "${RL_INTRA_END_FRACTION}" ;;
      esac
      ;;
    opd|qwen_opd|llama_opd)
      case "${suffix}" in
        POST_START_FRACTION) printf '%s\n' "${OPD_POST_START_FRACTION}" ;;
        POST_END_FRACTION) printf '%s\n' "${OPD_POST_END_FRACTION}" ;;
        INTRA_START_FRACTION) printf '%s\n' "${OPD_INTRA_START_FRACTION}" ;;
        INTRA_END_FRACTION) printf '%s\n' "${OPD_INTRA_END_FRACTION}" ;;
      esac
      ;;
  esac
}

run_one_job() {
  local name="$1"
  local family="$2"
  local model="$3"
  local pkl_dir="$4"
  local dataset="$5"
  local out_root="${BASE_OUTPUT_ROOT}/${name}"
  local post_start
  local post_end
  local intra_start
  local intra_end

  post_start="$(family_setting "${family}" POST_START_FRACTION)"
  post_end="$(family_setting "${family}" POST_END_FRACTION)"
  intra_start="$(family_setting "${family}" INTRA_START_FRACTION)"
  intra_end="$(family_setting "${family}" INTRA_END_FRACTION)"

  echo "[JOB:${name}] MODEL_FAMILY=${family}"
  echo "[JOB:${name}] MODEL=${model}"
  echo "[JOB:${name}] PKL_DIR=${pkl_dir}"
  echo "[JOB:${name}] DATASET=${dataset}"
  echo "[JOB:${name}] OUT_ROOT=${out_root}"
  if [[ -n "${post_start}" || -n "${post_end}" ]]; then
    echo "[JOB:${name}] POST_COT_WINDOW=[${post_start:-default}, ${post_end:-default}]"
  fi
  if [[ -n "${intra_start}" || -n "${intra_end}" ]]; then
    echo "[JOB:${name}] INTRA_COT_RANGE=[${intra_start:-default}, ${intra_end:-default}]"
  fi

  ENV_NAME="${ENV_NAME}" \
  PKL_DIR="${pkl_dir}" \
  MODEL_FAMILY="${family}" \
  MODEL="${model}" \
  DATASET="${dataset}" \
  POST_START_FRACTION="${post_start}" \
  POST_END_FRACTION="${post_end}" \
  INTRA_START_FRACTION="${intra_start}" \
  INTRA_END_FRACTION="${intra_end}" \
  RUN_TAG="${name}" \
  OUT_ROOT="${out_root}" \
  PASSK_LIST="${PASSK_LIST}" \
  ONLINE_WARMUP_TRACES="${ONLINE_WARMUP_TRACES}" \
  ONLINE_TOTAL_BUDGET="${ONLINE_TOTAL_BUDGET}" \
  ONLINE_SWEEP_POINTS="${ONLINE_SWEEP_POINTS}" \
  ONLINE_RESAMPLES="${ONLINE_RESAMPLES}" \
  OFFLINE_SAMPLE_SIZE="${OFFLINE_SAMPLE_SIZE}" \
  OFFLINE_RESAMPLES="${OFFLINE_RESAMPLES}" \
  ADAPTIVE_DIVISOR="${ADAPTIVE_DIVISOR}" \
  SEED="${SEED}" \
  bash "${SINGLE_SCRIPT}"

  echo "[JOB:${name}] Done."
}

echo "[INFO] Starting position-aware batch postprocess."
echo "[INFO] SINGLE_SCRIPT=${SINGLE_SCRIPT}"
echo "[INFO] BASE_OUTPUT_ROOT=${BASE_OUTPUT_ROOT}"
echo "[INFO] LOG_DIR=${LOG_DIR}"
echo "[INFO] PARALLEL=${PARALLEL}"
echo "[INFO] MAX_JOBS=${MAX_JOBS}"
echo

if [[ "${PARALLEL}" == "1" ]]; then
  running=0
  failed=0

  for i in "${!JOBS[@]}"; do
    IFS='|' read -r name family model pkl_dir dataset <<< "${JOBS[$i]}"
    log_path="${LOG_DIR}/${name}.log"

    while (( running >= MAX_JOBS )); do
      if wait -n; then
        :
      else
        failed=1
      fi
      running=$((running - 1))
    done

    echo "[INFO] Launching ${name}; log=${log_path}"
    run_one_job "${name}" "${family}" "${model}" "${pkl_dir}" "${dataset}" >"${log_path}" 2>&1 &
    running=$((running + 1))
  done

  while (( running > 0 )); do
    if wait -n; then
      :
    else
      failed=1
    fi
    running=$((running - 1))
  done

  if (( failed != 0 )); then
    echo "[ERROR] One or more strategy jobs failed. Check logs in ${LOG_DIR}."
    exit 1
  fi
else
  for i in "${!JOBS[@]}"; do
    IFS='|' read -r name family model pkl_dir dataset <<< "${JOBS[$i]}"
    log_path="${LOG_DIR}/${name}.log"

    echo "[INFO] Running ${name}; log=${log_path}"
    run_one_job "${name}" "${family}" "${model}" "${pkl_dir}" "${dataset}" 2>&1 | tee "${log_path}"
    echo
  done
fi

echo "[DONE] Position-aware batch postprocess finished."
echo "[INFO] Output root: ${BASE_OUTPUT_ROOT}"
echo "[INFO] Logs: ${LOG_DIR}"
