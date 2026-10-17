#!/usr/bin/env bash
set -euo pipefail

# =========================
# Config
# =========================
ENV_NAME="${ENV_NAME:-deepconf}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
REPO_URL="${REPO_URL:-}"
REPO_DIR="${REPO_DIR:-$PWD}"

# =========================
# Basic checks
# =========================
if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found. Please install Miniconda/Anaconda first."
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "[ERROR] git not found. Please install git first."
  exit 1
fi

# Make 'conda activate' available in shell scripts
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "[INFO] Repo dir: $REPO_DIR"
echo "[INFO] Env name: $ENV_NAME"
echo "[INFO] Python version: $PYTHON_VERSION"

# =========================
# Clone or update repo
# =========================
if [ ! -d "$REPO_DIR/.git" ]; then
  if [ -z "$REPO_URL" ]; then
    echo "[ERROR] REPO_URL is required when REPO_DIR is not an existing git checkout."
    echo "[ERROR] Or run this script from the extracted anonymous review repository."
    exit 1
  fi
  echo "[INFO] Cloning DeepConf..."
  git clone "$REPO_URL" "$REPO_DIR"
else
  echo "[INFO] Using existing DeepConf checkout: $REPO_DIR"
  if [ -n "$REPO_URL" ]; then
    git -C "$REPO_DIR" pull --ff-only
  fi
fi

# =========================
# Create / activate conda env
# =========================
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[INFO] Creating conda env: $ENV_NAME"
  conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
else
  echo "[INFO] Conda env $ENV_NAME already exists."
fi

conda activate "$ENV_NAME"

# =========================
# Upgrade installers
# =========================
python -m pip install --upgrade pip setuptools wheel uv

# =========================
# Install dependencies
# =========================
# DeepConf requirements.txt pins vllm==0.10.2
# and installs Dynasor from GitHub.
echo "[INFO] Installing vLLM..."
uv pip install "vllm==0.10.2" --torch-backend=auto

echo "[INFO] Installing Dynasor..."
uv pip install "git+https://github.com/hao-ai-lab/Dynasor.git"

# =========================
# Install DeepConf itself
# =========================
echo "[INFO] Installing DeepConf from local repo..."
cd "$REPO_DIR"
uv pip install -e .

# =========================
# Force compatible transformers version
# =========================

python -m pip uninstall -y transformers || true
python -m pip install "transformers<5"
python -m pip install "scikit-learn"
python -m pip install "accelerate"
python -m pip install "matplotlib"
python -m pip install "numpy"



# =========================
# Smoke test
# =========================
echo "[INFO] Running smoke test..."
python - <<'PY'
import importlib

mods = ["deepconf", "vllm"]
for m in mods:
    importlib.import_module(m)

print("Smoke test passed: deepconf and vllm imported successfully.")
PY

echo
echo "[DONE] Installation finished."
echo "Activate with: conda activate $ENV_NAME"
echo "Repo path: $REPO_DIR"
