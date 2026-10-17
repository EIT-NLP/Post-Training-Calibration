#!/usr/bin/env bash
set -euo pipefail

# 让 conda activate 在脚本里可用
source "$(conda info --base)/etc/profile.d/conda.sh"

# 1) 从 base 环境开始
conda activate base

# 2) 创建 / 激活 conda 环境
ENV_NAME="verl"
PYTHON_VERSION="3.10"

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "[INFO] Conda environment already exists: $ENV_NAME"
else
  echo "[INFO] Creating conda environment: $ENV_NAME (python=$PYTHON_VERSION)"
  conda create -y -n "$ENV_NAME" "python=${PYTHON_VERSION}"
fi

conda activate "$ENV_NAME"

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
  12.1) TORCH_CUDA_TAG="cu121" ;;
  12.4) TORCH_CUDA_TAG="cu124" ;;
  *)
    echo "[ERROR] Unsupported CUDA version for this script: ${CUDA_VERSION}"
    echo "[ERROR] Please check https://pytorch.org/get-started/locally/ and choose the correct wheel index."
    exit 1
    ;;
esac

echo "[INFO] Detected CUDA toolkit: ${CUDA_VERSION}"
echo "[INFO] Target PyTorch CUDA tag: ${TORCH_CUDA_TAG}"

python -m pip install -U pip setuptools wheel ninja packaging

python -m pip install torch==2.4.0 -f "https://mirrors.aliyun.com/pytorch-wheels/${TORCH_CUDA_TAG}"

# =========================================================
# 统一目录：以“当前脚本所在目录”为根目录，任何人放哪都能直接运行
# =========================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

REPO_DIR="$PROJECT_ROOT/simpleRL-reason"
MODEL_ROOT="$PROJECT_ROOT/models"
DATA_ROOT="$PROJECT_ROOT/data"
CHECKPOINT_ROOT="$PROJECT_ROOT/checkpoints"
LOG_ROOT="$PROJECT_ROOT/logs"

mkdir -p "$MODEL_ROOT" "$DATA_ROOT" "$CHECKPOINT_ROOT" "$LOG_ROOT"

echo "[INFO] PROJECT_ROOT=$PROJECT_ROOT"
echo "[INFO] REPO_DIR=$REPO_DIR"
echo "[INFO] MODEL_ROOT=$MODEL_ROOT"
echo "[INFO] DATA_ROOT=$DATA_ROOT"
echo "[INFO] CHECKPOINT_ROOT=$CHECKPOINT_ROOT"
echo "[INFO] LOG_ROOT=$LOG_ROOT"

# 3) 安装代码到当前目录
if [[ ! -d "$REPO_DIR/.git" ]]; then
  echo "[INFO] Cloning simpleRL-reason ..."
  git clone https://github.com/hkust-nlp/simpleRL-reason.git "$REPO_DIR"
else
  echo "[INFO] Repo already exists, skip clone: $REPO_DIR"
fi

cd "$REPO_DIR"

# 避免 Ray 把 repo 内的大文件一起上传
cat > "$REPO_DIR/.rayignore" <<'EOF'
.git/
Qwen2.5-7B-Instruct/
rl_data/
checkpoints/
logs/
examples/simplelr_math_eval/data/
__pycache__/
*.pyc
*.parquet
*.safetensors
*.pt
*.bin
EOF

install_flash_attn() {
  local PYBIN
  PYBIN="$(command -v python3 || command -v python || true)"

  if [ -z "$PYBIN" ]; then
    echo "[ERROR] Neither python3 nor python was found in PATH."
    return 1
  fi

  echo "[INFO] Using Python: $PYBIN"

  echo "[INFO] Installing build dependencies first..."
  "$PYBIN" -m pip install -U pip setuptools wheel packaging ninja

  FLASH_ATTN_CXX11_ABI="$("$PYBIN" - <<'PY'
import sys
try:
    import torch
except Exception as e:
    print(f"[ERROR] Cannot import torch: {e}", file=sys.stderr)
    sys.exit(1)

try:
    abi = torch.compiled_with_cxx11_abi()
except Exception:
    abi = getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", None)

if abi is None:
    print("[ERROR] Cannot determine torch CXX11 ABI", file=sys.stderr)
    sys.exit(1)

print("TRUE" if abi else "FALSE")
PY
)" || return 1

  FLASH_ATTN_CUDA_MAJOR="$("$PYBIN" - <<'PY'
import sys
try:
    import torch
except Exception as e:
    print(f"[ERROR] Cannot import torch: {e}", file=sys.stderr)
    sys.exit(1)

cuda_ver = getattr(torch.version, "cuda", None)
if not cuda_ver:
    print("[ERROR] torch.version.cuda is empty; this torch build may be CPU-only", file=sys.stderr)
    sys.exit(1)

print(str(cuda_ver).split(".")[0])
PY
)" || return 1

  PYTAG="$("$PYBIN" - <<'PY'
import sys
print(f"cp{sys.version_info.major}{sys.version_info.minor}")
PY
)" || return 1

  echo "[INFO] torch CXX11 ABI: ${FLASH_ATTN_CXX11_ABI}"
  echo "[INFO] torch CUDA major: ${FLASH_ATTN_CUDA_MAJOR}"
  echo "[INFO] python tag: ${PYTAG}"

  case "${FLASH_ATTN_CUDA_MAJOR}_${FLASH_ATTN_CXX11_ABI}_${PYTAG}" in
    12_TRUE_cp310)
      FLASH_ATTN_WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.4cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
      ;;
    12_FALSE_cp310)
      FLASH_ATTN_WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
      ;;
    11_TRUE_cp310)
      FLASH_ATTN_WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2+cu11torch2.4cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
      ;;
    11_FALSE_cp310)
      FLASH_ATTN_WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2+cu11torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
      ;;
    *)
      echo "[ERROR] Unsupported torch CUDA/ABI/Python combination: CUDA=${FLASH_ATTN_CUDA_MAJOR}, ABI=${FLASH_ATTN_CXX11_ABI}, PY=${PYTAG}"
      return 1
      ;;
  esac

  echo "[INFO] Installing flash-attn wheel: ${FLASH_ATTN_WHEEL_URL}"
  "$PYBIN" -m pip install "${FLASH_ATTN_WHEEL_URL}"
}

install_flash_attn

pip install -e .
pip install swanlab
pip install -U modelscope
pip install -U "click==8.2.1" --force-reinstall

WORK_DIR="$REPO_DIR"

# Keep experiment tracking opt-in for public release. Export SWANLAB_API_KEY
# before running this script if cloud logging is desired.
if [[ -n "${SWANLAB_API_KEY:-}" ]]; then
  export SWANLAB_API_KEY
fi
export VLLM_ATTENTION_BACKEND=XFORMERS

# ModelScope 仓库
MS_DATASET_REPO="${MS_DATASET_REPO:-deepscalar_simplerl}"
MS_MODEL_REPO="${MS_MODEL_REPO:-qwen/Qwen2.5-7B-Instruct}"
MODEL_NAME="${MODEL_NAME:-Qwen2.5-7B-Instruct}"
DATASET_NAME="${DATASET_NAME:-rl_data}"

# 训练参数
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
# 16384
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
ROLLOUT_N="${ROLLOUT_N:-8}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.0001}"
ENTROPY_COEFFIENT="${ENTROPY_COEFFIENT:-0.001}"
ROLLOUT_GPU_MEMORY_UTIL="${ROLLOUT_GPU_MEMORY_UTIL:-0.75}"
ROLLOUT_TP="${ROLLOUT_TP:-2}"
SAVE_FREQ="${SAVE_FREQ:-500}"
RAY_NUM_GPUS="${RAY_NUM_GPUS:-8}"

# 自动生成路径：全部放在 simpleRL-reason 的同级目录
HDFS_DATA_PATH="$DATA_ROOT"
HDFS_MODEL_PATH="$MODEL_ROOT"
HDFS_CHECKPOINT_PATH="$CHECKPOINT_ROOT"
HDFS_LOG_PATH="$LOG_ROOT"
SWANLAB_LOG_DIR="$HDFS_LOG_PATH/swanlab"

mkdir -p \
  "$HDFS_DATA_PATH/$DATASET_NAME" \
  "$HDFS_MODEL_PATH/$MODEL_NAME" \
  "$HDFS_CHECKPOINT_PATH" \
  "$HDFS_LOG_PATH" \
  "$SWANLAB_LOG_DIR"

echo "[INFO] WORK_DIR=$WORK_DIR"
echo "[INFO] HDFS_DATA_PATH=$HDFS_DATA_PATH"
echo "[INFO] HDFS_MODEL_PATH=$HDFS_MODEL_PATH"
echo "[INFO] HDFS_CHECKPOINT_PATH=$HDFS_CHECKPOINT_PATH"
echo "[INFO] HDFS_LOG_PATH=$HDFS_LOG_PATH"
echo "[INFO] SWANLAB_LOG_DIR=$SWANLAB_LOG_DIR"

# 若历史上误把数据/模型下载进 repo，则自动迁移到同级目录
if [[ -d "$WORK_DIR/$MODEL_NAME" && ! -d "$HDFS_MODEL_PATH/$MODEL_NAME" ]]; then
  echo "[INFO] Migrating legacy model dir from repo to sibling directory..."
  mv "$WORK_DIR/$MODEL_NAME" "$HDFS_MODEL_PATH/$MODEL_NAME"
fi

if [[ -d "$WORK_DIR/$DATASET_NAME" && ! -d "$HDFS_DATA_PATH/$DATASET_NAME" ]]; then
  echo "[INFO] Migrating legacy dataset dir from repo to sibling directory..."
  mv "$WORK_DIR/$DATASET_NAME" "$HDFS_DATA_PATH/$DATASET_NAME"
fi

# 检查/安装 modelscope
if ! python - <<'PY' >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("modelscope") else 1)
PY
then
  echo "[INFO] modelscope 未安装，正在安装..."
  python -m pip install -U modelscope
fi

# 1) 下载数据集中的 train/test/grpo_rl_train.sh
echo "[INFO] 正在下载数据集文件..."
python - <<PY
from modelscope import snapshot_download

snapshot_download(
    repo_id="${MS_DATASET_REPO}",
    repo_type="dataset",
    allow_patterns=["train.parquet", "test.parquet", "grpo_rl_train.sh"],
    local_dir="${HDFS_DATA_PATH}/${DATASET_NAME}",
)
PY

test -f "$HDFS_DATA_PATH/$DATASET_NAME/train.parquet"
test -f "$HDFS_DATA_PATH/$DATASET_NAME/test.parquet"
test -f "$HDFS_DATA_PATH/$DATASET_NAME/grpo_rl_train.sh"

# 2) 下载模型到同级目录 ./models/Qwen2.5-1.5B-Instruct
if [[ ! -f "$HDFS_MODEL_PATH/$MODEL_NAME/config.json" ]]; then
  echo "[INFO] 正在下载模型 ${MS_MODEL_REPO} ..."
  python - <<PY
from modelscope import snapshot_download

snapshot_download(
    repo_id="${MS_MODEL_REPO}",
    local_dir="${HDFS_MODEL_PATH}/${MODEL_NAME}",
)
PY
else
  echo "[INFO] 检测到本地模型目录已存在，跳过模型下载: $HDFS_MODEL_PATH/$MODEL_NAME"
fi

if [[ ! -f "$HDFS_MODEL_PATH/$MODEL_NAME/config.json" ]]; then
  echo "[ERROR] 模型下载后未找到 config.json: $HDFS_MODEL_PATH/$MODEL_NAME" >&2
  exit 1
fi

# 3) 复制训练脚本到 repo 当前目录
cp -f "$HDFS_DATA_PATH/$DATASET_NAME/grpo_rl_train.sh" "$WORK_DIR/grpo_rl_train.sh"
chmod +x "$WORK_DIR/grpo_rl_train.sh"

# 4) patch grpo_rl_train.sh 中的固定路径
export PATCH_HDFS_DATA_PATH="$HDFS_DATA_PATH"
export PATCH_HDFS_MODEL_PATH="$HDFS_MODEL_PATH"
export PATCH_HDFS_CHECKPOINT_PATH="$HDFS_CHECKPOINT_PATH"
export PATCH_HDFS_LOG_PATH="$HDFS_LOG_PATH"
export PATCH_SWANLAB_LOG_DIR="$SWANLAB_LOG_DIR"

python - "$WORK_DIR/grpo_rl_train.sh" <<'PY'
import os
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")

defaults = {
    "HDFS_DATA_PATH": os.environ["PATCH_HDFS_DATA_PATH"],
    "HDFS_MODEL_PATH": os.environ["PATCH_HDFS_MODEL_PATH"],
    "HDFS_CHECKPOINT_PATH": os.environ["PATCH_HDFS_CHECKPOINT_PATH"],
    "HDFS_LOG_PATH": os.environ["PATCH_HDFS_LOG_PATH"],
    "SWANLAB_LOG_DIR": os.environ["PATCH_SWANLAB_LOG_DIR"],
}

def esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')

for name, default in defaults.items():
    default = esc(default)
    pattern = rf'^export {name}=.*$'
    replacement = f'export {name}="${{{name}:-{default}}}"'
    text, count = re.subn(pattern, replacement, text, count=1, flags=re.M)
    if count != 1:
        raise SystemExit(f"[ERROR] Failed to patch {name}")


path.write_text(text, encoding="utf-8")
PY

# 5) 启动 Ray
if ray status >/dev/null 2>&1; then
  echo "[INFO] 检测到已有 Ray 集群，跳过 ray start"
else
  echo "[INFO] 正在启动 Ray head ..."
  ray start --head --node-ip-address 0.0.0.0 --num-gpus "$RAY_NUM_GPUS"
fi

# 6) 导出路径变量
export HDFS_DATA_PATH
export HDFS_MODEL_PATH
export HDFS_CHECKPOINT_PATH
export HDFS_LOG_PATH
export SWANLAB_LOG_DIR

# 7) 启动训练
echo "[INFO] 开始训练 ..."
bash "$WORK_DIR/grpo_rl_train.sh" \
  --model_name "$MODEL_NAME" \
  --dataset_name "$DATASET_NAME" \
  --total_epochs "$TOTAL_EPOCHS" \
  --max_response_length "$MAX_RESPONSE_LENGTH" \
  --train_batch_size "$TRAIN_BATCH_SIZE" \
  --rollout_n "$ROLLOUT_N" \
  --kl_loss_coef "$KL_LOSS_COEF" \
  --entropy_coeffient "$ENTROPY_COEFFIENT" \
  --rollout_gpu_memory_util "$ROLLOUT_GPU_MEMORY_UTIL" \
  --rollout_tp "$ROLLOUT_TP" \
  --save_freq "$SAVE_FREQ"
