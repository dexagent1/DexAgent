#!/usr/bin/env bash
# One-time setup for DexAgent Stage 4 (robot data generation).
# Run inside a fresh Python 3.11 environment:
#
#   conda create -y -n dexagent-stage4 python=3.11
#   conda activate dexagent-stage4
#   bash setup.sh
#
# Steps: (1) clone VOID at its pinned commit, (2) install Python packages, (3) download model weights.
# A VOID checkout elsewhere: point VOID_DIR at it (run_dexagent.py passes models.void_dir).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

python - <<'PY'
import sys
assert sys.version_info[:2] == (3, 11), f"Python 3.11 is required (bpy wheel), found {sys.version.split()[0]}"
PY

# The exact revisions Stage 4 was tested with (override at your own risk)
VOID_COMMIT="${VOID_COMMIT:-e3914f8f551dd4b880661991fd6b28cd1699a97a}"
SAM3_COMMIT="${SAM3_COMMIT:-2345a4ad109ac29c569da749c91d84f10dc08c40}"

# 1. VOID (video object and interaction deletion), Apache-2.0, pinned
mkdir -p third_party
if [ ! -d third_party/void-model ]; then
  git clone https://github.com/netflix/void-model.git third_party/void-model
fi
if [ "$(git -C third_party/void-model rev-parse HEAD)" != "$VOID_COMMIT" ]; then
  git -C third_party/void-model fetch origin
  git -C third_party/void-model checkout --detach "$VOID_COMMIT"
fi

# 2. Python packages: VOID's pins first, then SAM 3 (pinned) and the Stage 4 extras (MuJoCo, Blender)
pip install -r third_party/void-model/requirements.txt
pip install "git+https://github.com/facebookresearch/sam3.git@${SAM3_COMMIT}"
pip install -r requirements.txt

# 3. Model weights (cached under $HF_HOME, default ~/.cache/huggingface)
if [ "${SKIP_WEIGHTS:-0}" != "1" ]; then
  python - <<'PY'
from huggingface_hub import hf_hub_download, snapshot_download
print("VOID pass-1 checkpoint ...")
hf_hub_download("netflix/void-model", "void_pass1.safetensors")
print("CogVideoX-Fun base model (VOID backbone) ...")
snapshot_download("alibaba-pai/CogVideoX-Fun-V1.5-5b-InP")
print("SAM 3.1 checkpoint (gated: request access on the model page and run `hf auth login` first) ...")
try:
    from sam3.model_builder import download_ckpt_from_hf
    download_ckpt_from_hf(version="sam3.1")
except Exception as e:
    print(f"\n[setup] SAM 3.1 weights not downloaded: {type(e).__name__}: {e}\n"
          "        Request access at https://huggingface.co/facebook/sam3.1, run `hf auth login`, then re-run setup.sh.\n"
          "        (Stage 4 also runs without SAM 3 if you provide your own erase masks, see README.)")
PY
fi
echo "Stage 4 setup complete."
