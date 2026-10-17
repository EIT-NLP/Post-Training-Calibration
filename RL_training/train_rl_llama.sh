#!/usr/bin/env bash
set -euo pipefail

# Rebuttal Llama RL launcher. All heavy lifting stays in train_rl.sh; this file
# only switches the backbone and output naming.
# If the gated HuggingFace checkpoint is not directly downloadable in your
# environment, set MS_MODEL_REPO to a permitted mirror or pre-populate the local
# model directory used by train_rl.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MS_MODEL_REPO="${MS_MODEL_REPO:-meta-llama/Llama-3.1-8B-Instruct}"
export MODEL_NAME="${MODEL_NAME:-Llama-3.1-8B-Instruct}"
export SWANLAB_RUN_NAME="${SWANLAB_RUN_NAME:-rl_llama31_8b}"

exec "${SCRIPT_DIR}/train_rl.sh"
