#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/or-rl}"
VENV_DIR="${VENV_DIR:-/workspace/or-rl-venv}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/workspace/.cache/huggingface}"
RESET_VENV="${RESET_VENV:-false}"
CAUSAL_CONV1D_WHL="${CAUSAL_CONV1D_WHL:-https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.4/causal_conv1d-1.5.4+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-true}"

if [[ ! -d "$REPO_DIR" ]]; then
  echo "Repository directory not found: $REPO_DIR" >&2
  echo "Copy or clone the repo to /workspace/or-rl, or set REPO_DIR=/path/to/repo." >&2
  exit 1
fi

cd "$REPO_DIR"

echo "Checking base PyTorch/CUDA install from the RunPod template..."
python - <<'PY'
import sys
try:
    import torch
except Exception as exc:
    raise SystemExit(
        "PyTorch is not installed in the base image. Use a RunPod PyTorch CUDA template "
        "instead of installing torch from this project setup script."
    ) from exc

print("base_python", sys.executable)
print("base_torch", torch.__version__)
print("base_cuda_available", torch.cuda.is_available())
print("base_cuda_device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is not available from the base PyTorch install. Pick a GPU pod/template with "
        "a working CUDA PyTorch stack before running training."
    )
PY

if [[ "$RESET_VENV" == "true" ]]; then
  rm -rf "$VENV_DIR"
fi

python -m venv --system-site-packages "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip wheel setuptools

TORCH_CONSTRAINT_FILE="$(mktemp /tmp/or-rl-torch-constraints.XXXXXX.txt)"
python - <<'PY' > "$TORCH_CONSTRAINT_FILE"
import importlib.util
import torch

packages = {"torch": torch.__version__}
for name in ("torchvision", "torchaudio"):
    try:
        spec = importlib.util.find_spec(name)
    except Exception:
        spec = None
    if spec:
        try:
            mod = __import__(name)
        except Exception:
            continue
        packages[name] = getattr(mod, "__version__", "")

for name, version in packages.items():
    if version:
        print(f"{name}=={version}")
PY

echo "Constraining pip to the template's PyTorch stack:"
cat "$TORCH_CONSTRAINT_FILE"

python -m pip install \
  --upgrade \
  --upgrade-strategy only-if-needed \
  --constraint "$TORCH_CONSTRAINT_FILE" \
  -r requirements-runpod.txt

python -m pip install --no-deps "$CAUSAL_CONV1D_WHL"
if [[ "$INSTALL_FLASH_ATTN" == "true" ]]; then
  python -m pip install flash-attn --no-build-isolation
fi

mkdir -p "$HF_CACHE_DIR" outputs
export HF_HOME="$HF_CACHE_DIR"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR"
export HF_HUB_CACHE="$HF_CACHE_DIR/hub"

cat > .runpod_env <<EOF
export HF_HOME="$HF_HOME"
export TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE"
export HF_HUB_CACHE="$HF_HUB_CACHE"
export ATTN_IMPLEMENTATION="\${ATTN_IMPLEMENTATION:-sdpa}"
source "$VENV_DIR/bin/activate"
EOF

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
if not torch.cuda.is_available():
    raise SystemExit("CUDA became unavailable after dependency installation.")

import transformers, trl, peft, ortools, bitsandbytes, fla, causal_conv1d
print("transformers", transformers.__version__)
print("trl", trl.__version__)
print("peft", peft.__version__)
print("ortools", ortools.__version__)
print("bitsandbytes", bitsandbytes.__version__)
print("flash_linear_attention", getattr(fla, "__version__", "installed"))
print("causal_conv1d", getattr(causal_conv1d, "__version__", "installed"))
try:
    import flash_attn
except Exception as exc:
    print("flash_attn", f"unavailable: {exc}")
else:
    print("flash_attn", getattr(flash_attn, "__version__", "installed"))

try:
    from trl import GRPOTrainer
    print("GRPOTrainer", "available")
except Exception as exc:
    raise SystemExit(f"TRL GRPOTrainer import failed: {exc}") from exc

from transformers import AutoConfig, AutoModelForCausalLM
config = AutoConfig.from_pretrained("Qwen/Qwen3.5-2B", trust_remote_code=True)
try:
    mapped_cls = AutoModelForCausalLM._model_mapping[type(config)]
except Exception as exc:
    raise SystemExit(f"Qwen/Qwen3.5-2B is not supported by AutoModelForCausalLM: {exc}") from exc
print("Qwen/Qwen3.5-2B AutoModelForCausalLM", mapped_cls.__name__)
PY

if ! python -m pip check; then
  echo "pip check reported dependency conflicts. Continuing because RunPod base images often include unrelated system packages." >&2
fi

echo
echo "RunPod setup complete."
echo "Activate this environment with:"
echo "  cd $REPO_DIR && source .runpod_env"
