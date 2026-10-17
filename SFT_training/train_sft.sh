#!/usr/bin/env bash
set -euo pipefail

# 让 conda activate 在脚本里可用
source "$(conda info --base)/etc/profile.d/conda.sh"

# 1) 从 base 环境开始
conda activate base

# 3) 创建 finetune 环境
conda create -y -n finetune python=3.11



# 4) 激活新环境
conda activate finetune




detect_cuda_version() {
  if command -v nvcc >/dev/null 2>&1; then
    nvcc --version | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p' | tail -n1
  elif [ -f /usr/local/cuda/version.txt ]; then
    sed -n 's/.*CUDA Version \([0-9]\+\.[0-9]\+\).*/\1/p' /usr/local/cuda/version.txt | head -n1
  else
    echo ""
  fi
}

CUDA_VERSION="$(detect_cuda_version)"
if [ -z "${CUDA_VERSION}" ]; then
  echo "[ERROR] Cannot detect CUDA toolkit version from nvcc or /usr/local/cuda/version.txt"
  exit 1
fi

case "${CUDA_VERSION}" in
  11.8) TORCH_CUDA_TAG="cu118" ;;
  12.4) TORCH_CUDA_TAG="cu124" ;;
  12.6) TORCH_CUDA_TAG="cu126" ;;
  *)
    echo "[ERROR] Unsupported CUDA version for this script: ${CUDA_VERSION}"
    echo "[ERROR] Please check https://pytorch.org/get-started/locally/ and choose the correct wheel index."
    exit 1
    ;;
esac

echo "[INFO] Detected CUDA toolkit: ${CUDA_VERSION}"
echo "[INFO] Installing PyTorch for: ${TORCH_CUDA_TAG}"


python -m pip install -U pip setuptools wheel ninja
python -m pip install \
    torch==2.6.0 \
    torchvision==0.21.0 \
    torchaudio==2.6.0 \
    -f "https://mirrors.aliyun.com/pytorch-wheels/${TORCH_CUDA_TAG}"


# 5) 安装代码到当前目录
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LlamaFactory

pip install -U pip
pip install -e .
pip install -r requirements/metrics.txt
pip install -r requirements/deepspeed.txt
pip install swanlab
pip install 'swanlab[dashboard]'

echo "Done."


source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate finetune

LLAMAFACTORY_DIR="$(pwd)"


MS_DATASET_ID="${MS_DATASET_ID:-sft_deepscaler_simplerl}"
MS_TARGET_JSON="${MS_TARGET_JSON:-deepscaler_simplerl_r1_qwen32b_correct.json}"
MS_TARGET_INFO="${MS_TARGET_INFO:-dataset_info.json}"

LF_DATASET_NAME="${LF_DATASET_NAME:-deepscaler_simplerl_r1_qwen32b_correct}"
LF_TEMPLATE="${LF_TEMPLATE:-qwen}"

MS_CACHE_DIR="${MS_CACHE_DIR:-${LLAMAFACTORY_DIR}/.ms_cache/sft_deepscaler_simplerl}"
LF_DATA_DIR="${LF_DATA_DIR:-${LLAMAFACTORY_DIR}/data}"
LF_DATA_JSON="${LF_DATA_DIR}/${MS_TARGET_JSON}"
LF_DATA_INFO="${LF_DATA_DIR}/dataset_info.json"

echo "[INFO] conda env: $(conda info --envs | sed -n '/\*/p')"
echo "[INFO] python: $(which python)"
echo "[INFO] llamafactory-cli: $(which llamafactory-cli || true)"
echo "[INFO] LlamaFactory dir: ${LLAMAFACTORY_DIR}"

cd "${LLAMAFACTORY_DIR}"

llamafactory-cli version
python -m pip install -U modelscope


export MS_DATASET_ID
export MS_TARGET_JSON
export MS_TARGET_INFO
export MS_CACHE_DIR
export LF_DATA_DIR
export LF_DATA_JSON
export LF_DATA_INFO

python - <<'PY'
import os
import shutil
from pathlib import Path
from modelscope import dataset_snapshot_download

ms_dataset_id = os.environ["MS_DATASET_ID"]
ms_target_json = os.environ["MS_TARGET_JSON"]
ms_target_info = os.environ["MS_TARGET_INFO"]
ms_cache_dir = Path(os.environ["MS_CACHE_DIR"])
lf_data_dir = Path(os.environ["LF_DATA_DIR"])
lf_data_json = Path(os.environ["LF_DATA_JSON"])
lf_data_info = Path(os.environ["LF_DATA_INFO"])

lf_data_dir.mkdir(parents=True, exist_ok=True)
ms_cache_dir.mkdir(parents=True, exist_ok=True)

print(f"[PY] downloading from ModelScope: {ms_dataset_id}")
dataset_snapshot_download(
    dataset_id=ms_dataset_id,
    local_dir=str(ms_cache_dir),
    allow_file_pattern=[ms_target_info, ms_target_json],
)

json_matches = list(ms_cache_dir.rglob(ms_target_json))
info_matches = list(ms_cache_dir.rglob(ms_target_info))

if not json_matches:
    raise FileNotFoundError(
        f"Cannot find {ms_target_json} under downloaded dataset cache: {ms_cache_dir}"
    )
if not info_matches:
    raise FileNotFoundError(
        f"Cannot find {ms_target_info} under downloaded dataset cache: {ms_cache_dir}"
    )

src_json = json_matches[0]
src_info = info_matches[0]

print(f"[PY] found dataset json: {src_json}")
print(f"[PY] found dataset info: {src_info}")

# 可选：先备份原文件
if lf_data_info.exists():
    backup = lf_data_info.with_suffix(".json.bak")
    shutil.copy2(lf_data_info, backup)
    print(f"[PY] backup old dataset_info.json -> {backup}")

# 1) 覆盖 LlamaFactory/data/dataset_info.json
shutil.copy2(src_info, lf_data_info)
print(f"[PY] copied dataset_info.json -> {lf_data_info}")

# 2) 复制目标数据文件到 LlamaFactory/data/
shutil.copy2(src_json, lf_data_json)
print(f"[PY] copied dataset json -> {lf_data_json}")
PY

MS_MODEL_ID="${MS_MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
MS_MODEL_CACHE_NAME="${MS_MODEL_CACHE_NAME:-Qwen2.5-7B-Instruct}"
MS_MODEL_DIR="${MS_MODEL_DIR:-${LLAMAFACTORY_DIR}/.ms_cache/models/${MS_MODEL_CACHE_NAME}}"
export MS_MODEL_ID MS_MODEL_DIR

python - <<'PY'
import os
from modelscope import snapshot_download

ms_model_id = os.environ["MS_MODEL_ID"]
ms_model_dir = os.environ["MS_MODEL_DIR"]

local_model_dir = snapshot_download(
    model_id=ms_model_id,
    local_dir=ms_model_dir,
)
print(f"[PY] model downloaded to: {local_model_dir}")
PY




# Paper SFT setting: full-parameter Qwen2.5-7B-Instruct on correct-only
# DeepSeek-R1-Distill-Qwen-32B long-CoT targets, cutoff length 16,384.
llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path "${MS_MODEL_DIR}" \
    --preprocessing_num_workers 16 \
    --finetuning_type full \
    --template "${LF_TEMPLATE}" \
    --flash_attn auto \
    --dataset_dir "${LF_DATA_DIR}" \
    --dataset "${LF_DATASET_NAME}" \
    --cutoff_len 16384 \
    --learning_rate "${SFT_LEARNING_RATE:-1e-06}" \
    --num_train_epochs "${SFT_NUM_EPOCHS:-2.0}" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 10 \
    --save_steps 500 \
    --warmup_ratio 0.1 \
    --packing False \
    --enable_thinking False \
    --report_to "${REPORT_TO:-none}" \
    --use_swanlab "${USE_SWANLAB:-False}" \
    --output_dir "${SFT_OUTPUT_DIR:-saves/Qwen2.5-7B-Instruct/full/train_deepscaler_simplerl_r1_qwen32b_correct}" \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --swanlab_project "${SWANLAB_PROJECT:-Calibration}" \
    --swanlab_run_name "${SWANLAB_RUN_NAME:-sft_qwen25_7b_r1_qwen32b_correct}" \
    --swanlab_mode "${SWANLAB_MODE:-local}" \
    --deepspeed examples/deepspeed/ds_z3_offload_config.json


