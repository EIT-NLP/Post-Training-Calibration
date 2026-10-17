#!/usr/bin/env bash
set -euo pipefail

# Rebuttal Llama SFT launcher. This is a thin wrapper around train_sft.sh so the
# base script remains the single source of installation and data preparation.
# If the gated HuggingFace checkpoint is not directly downloadable in your
# environment, set MS_MODEL_ID to a permitted mirror or set MS_MODEL_DIR to a
# local checkpoint directory before running this wrapper.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MS_MODEL_ID="${MS_MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"
export MS_MODEL_CACHE_NAME="${MS_MODEL_CACHE_NAME:-Llama-3.1-8B-Instruct}"
export LF_TEMPLATE="${LF_TEMPLATE:-llama3}"
export SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-saves/Llama-3.1-8B-Instruct/full/train_deepscaler_simplerl_r1_qwen32b_correct}"
export SWANLAB_RUN_NAME="${SWANLAB_RUN_NAME:-sft_llama31_8b_r1_qwen32b_correct}"

exec "${SCRIPT_DIR}/train_sft.sh"
