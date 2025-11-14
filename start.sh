#!/usr/bin/env bash
set -euo pipefail

cd /workspace

# -----------------------------
# WebUI args
# -----------------------------
export COMMANDLINE_ARGS="${WEBUI_ARGS:-"--api --listen --xformers --enable-insecure-extension-access --port 3000"}"
export LAUNCH_BROWSER=0
echo "[start] COMMANDLINE_ARGS=${COMMANDLINE_ARGS}"

# -----------------------------
# Ensure repos exist
# -----------------------------
if [[ ! -d /workspace/stable-diffusion-webui ]]; then
  echo "[start] Cloning AUTOMATIC1111 repo…"
  git clone --depth 1 https://github.com/AUTOMATIC1111/stable-diffusion-webui /workspace/stable-diffusion-webui
fi

# Deforum (safety net; Dockerfile clones it already)
DEF_EXT="/workspace/stable-diffusion-webui/extensions/sd-webui-deforum"
if [[ ! -d "$DEF_EXT" ]]; then
  echo "[start] Cloning Deforum extension…"
  git clone --depth 1 https://github.com/deforum-art/sd-webui-deforum "$DEF_EXT" || true
fi

# ControlNet (safety net; Dockerfile clones it already)
CN_EXT="/workspace/stable-diffusion-webui/extensions/sd-webui-controlnet"
if [[ ! -d "$CN_EXT" ]]; then
  echo "[start] Cloning ControlNet extension…"
  git clone --depth 1 https://github.com/Mikubill/sd-webui-controlnet "$CN_EXT" || true
fi

# If Deforum exposes deforum_api.py, make it visible to A1111
if [[ -f "$DEF_EXT/scripts/deforum_api.py" ]]; then
  cp -f "$DEF_EXT/scripts/deforum_api.py" /workspace/stable-diffusion-webui/scripts/ || true
fi

# -----------------------------
# Launch A1111 using system Python so it uses its OWN venv
# (prewarmed in Docker build). Do NOT activate /workspace/venv here.
# -----------------------------
pushd /workspace/stable-diffusion-webui >/dev/null
echo "[start] Launching A1111…"
# Pass a hint to prefer binary wheels
export PIP_PREFER_BINARY=1
/usr/bin/python3 launch.py ${COMMANDLINE_ARGS} &
popd >/dev/null

# -----------------------------
# Wait for API
# -----------------------------
echo "[start] Waiting for A1111 API..."
url="http://127.0.0.1:3000/sdapi/v1/sd-models"

for i in {1..360}; do # up to ~6 minutes
    if curl -s --head --fail "$url" > /dev/null; then
        echo "[start] A1111 API is ready"
        break
    fi
    sleep 1
    if [ $i -eq 360 ]; then
        echo "[start] ERROR: A1111 API did not start in time" >&2
        exit 1
    fi
done

# -----------------------------
# Optional: list routes to verify deforum endpoints quickly
# -----------------------------
( curl -s http://127.0.0.1:3000/openapi.json | jq -r 'try .paths | keys[] catch empty' | sed 's/^/[route] /' ) || true

# -----------------------------
# NOW activate the worker venv just for the RunPod handler
# -----------------------------
if [[ ! -d /workspace/venv ]]; then
  echo "[start] Creating worker venv…"
  python3 -m venv /workspace/venv
fi
# shellcheck disable=SC1091
source /workspace/venv/bin/activate
pip install --no-cache-dir -q runpod==1.7.7 requests imageio imageio-ffmpeg pillow || true

echo "[start] Starting RunPod worker…"
exec python /workspace/rp_handler.py