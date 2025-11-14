#!/usr/bin/env bash
set -euo pipefail

cd /workspace

# -----------------------------
# Environment / passthrough
# -----------------------------
export COMMANDLINE_ARGS="${WEBUI_ARGS:-"--api --listen --xformers --enable-insecure-extension-access --port 3000"}"
export LAUNCH_BROWSER=0
export PYTHONUNBUFFERED=1

echo "[start] COMMANDLINE_ARGS=${COMMANDLINE_ARGS}"
echo "[start] A1111_PORT(for CLI)=${A1111_PORT:-3001}"
echo "[start] CKPT_PATH=${CKPT_PATH:-<unset>}"

# -----------------------------
# Python venv for the worker
# -----------------------------
if [[ ! -d /workspace/venv ]]; then
  echo "[start] Creating worker venv…"
  python3 -m venv /workspace/venv
fi
# shellcheck disable=SC1091
source /workspace/venv/bin/activate

# -----------------------------
# Ensure A1111 repo exists
# -----------------------------
if [[ ! -d /workspace/stable-diffusion-webui ]]; then
  echo "[start] Cloning AUTOMATIC1111 repo…"
  git clone --depth 1 https://github.com/AUTOMATIC1111/stable-diffusion-webui /workspace/stable-diffusion-webui
fi

# -----------------------------
# Log available checkpoints (helps diagnose downloads vs baked models)
# -----------------------------
echo "[start] Listing baked SD models:"
ls -lh /workspace/stable-diffusion-webui/models/Stable-diffusion || true

# -----------------------------
# Start a background A1111 (optional, dev convenience) on :3000
# -----------------------------
pushd /workspace/stable-diffusion-webui >/dev/null
echo "[start] Launching A1111 (background, optional)…"
# If it fails, we don't want to kill the worker; ignore errors.
( python launch.py ${COMMANDLINE_ARGS} || true ) &
popd >/dev/null

# -----------------------------
# Wait briefly (but don't block the worker)
# -----------------------------
python - <<'PY'
import time, requests
url = "http://127.0.0.1:3000/sdapi/v1/sd-models"
for i in range(30):
    try:
        requests.get(url, timeout=2)
        print("[start] A1111 API on :3000 responded.")
        break
    except Exception:
        time.sleep(1)
else:
    print("[start] A1111 API on :3000 not ready (continuing anyway).")
PY

# -----------------------------
# Start RunPod worker (BLOCKS). Use -u for unbuffered logs.
# -----------------------------
echo "[start] Starting RunPod worker…"
exec python -u /workspace/rp_handler.py