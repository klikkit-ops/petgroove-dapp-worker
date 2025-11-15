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
A1111_DIR="/workspace/stable-diffusion-webui"
if [[ ! -d "$A1111_DIR" ]]; then
  echo "[start] Cloning AUTOMATIC1111 repo…"
  git clone --depth 1 https://github.com/AUTOMATIC1111/stable-diffusion-webui "$A1111_DIR"
fi

# -----------------------------
# Ensure Deforum + ControlNet extensions exist
# -----------------------------
DEF_EXT="$A1111_DIR/extensions/sd-webui-deforum"
if [[ ! -d "$DEF_EXT" ]]; then
  echo "[start] Deforum extension not found, cloning…"
  git clone --depth 1 https://github.com/deforum-art/sd-webui-deforum "$DEF_EXT" || true
fi

CN_EXT="$A1111_DIR/extensions/sd-webui-controlnet"
if [[ ! -d "$CN_EXT" ]]; then
  echo "[start] ControlNet extension not found, cloning…"
  git clone --depth 1 https://github.com/Mikubill/sd-webui-controlnet "$CN_EXT" || true
fi
mkdir -p "$CN_EXT/models"

# -----------------------------
# Checkpoint & ControlNet model presence (clear, log-first)
# -----------------------------
echo "[check] CKPT_PATH=${CKPT_PATH:-<unset>}"
if [[ -n "${CKPT_PATH:-}" ]]; then
  if [[ -f "$CKPT_PATH" ]]; then
    echo "[check] ✅ Found checkpoint at CKPT_PATH"
  else
    echo "[check] ❌ CKPT_PATH points to a missing file"
  fi
fi

echo "[check] Stable-diffusion models in $A1111_DIR/models/Stable-diffusion:"
ls -lh "$A1111_DIR/models/Stable-diffusion" || true

echo "[check] ControlNet models dir: $CN_EXT/models"
ls -lh "$CN_EXT/models" || true

# -----------------------------
# Optional: copy deforum_api.py to /scripts if present
# -----------------------------
if [[ -f "$DEF_EXT/scripts/deforum_api.py" ]]; then
  echo "[start] Found deforum_api.py; copying to /scripts so A1111 can import it…"
  cp -f "$DEF_EXT/scripts/deforum_api.py" "$A1111_DIR/scripts/" || true
else
  echo "[start] No deforum_api.py present (OK; we can still drive via launch payload)."
fi

# -----------------------------
# Deforum schema introspection (prints exact cn_* keys used by THIS build)
# -----------------------------
echo "[schema] Grepping Deforum for cn_* key names…"
grep -RhoE "cn_[A-Za-z0-9_]+" "$DEF_EXT"/scripts | sort -u | sed 's/^/[deforum.cn] /' || true

echo "[schema] Inspecting Deforum helper modules (best effort)…"
python - <<'PY'
import importlib.util, json, traceback, re, os
root = "/workspace/stable-diffusion-webui/extensions/sd-webui-deforum/scripts/deforum_helpers"
def load(path):
    try:
        modname = os.path.splitext(os.path.basename(path))[0]
        spec = importlib.util.spec_from_file_location(modname, path)
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
    except Exception:
        return None
try:
    akf_path = f"{root}/animation_key_frames.py"
    text = open(akf_path, "r", encoding="utf-8").read() if os.path.exists(akf_path) else ""
    keys = sorted(set(re.findall(r"cn_[0-9]+_[A-Za-z0-9_]+", text)))
    print("[deforum.inspect] animation_key_frames keys:", json.dumps(keys[:400], indent=2))
except Exception:
    print("[deforum.inspect] animation_key_frames parse failed:", traceback.format_exc()[-800:])
try:
    args_mod = load(f"{root}/args.py")
    names = [n for n in dir(args_mod)] if args_mod else []
    names = [n for n in names if ("Args" in n or "ControlNet" in n)]
    print("[deforum.inspect] args.py symbols:", json.dumps(names, indent=2))
except Exception:
    print("[deforum.inspect] args.py load failed")
PY

# -----------------------------
# A1111 venv + CLIP safety net
# (Primary installs in Dockerfile; this is a runtime guard.)
# -----------------------------
A1111_PY="$A1111_DIR/venv/bin/python"
if [[ ! -x "$A1111_PY" ]]; then
  echo "[start] Creating A1111 venv (fallback)…"
  python3 -m venv "$A1111_DIR/venv"
fi
"$A1111_PY" - <<'PY' || true
try:
    import clip  # type: ignore
    print("[start] clip already available in A1111 venv")
except Exception:
    print("[start] installing clip into A1111 venv (safety net)…")
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/openai/CLIP.git#egg=clip"])
PY

# -----------------------------
# Start a background A1111 on :3000 (ignore exit to not kill worker)
# -----------------------------
pushd "$A1111_DIR" >/dev/null
echo "[start] Launching A1111 (background)…"
( "$A1111_PY" launch.py ${COMMANDLINE_ARGS} || true ) &
A1111_PID=$!
popd >/dev/null

# -----------------------------
# Wait for REST API (longer, reliable)
# -----------------------------
python - <<'PY'
import time, requests, sys
url = "http://127.0.0.1:3000/sdapi/v1/sd-models"
for i in range(360):  # up to ~6 minutes
    try:
        r = requests.get(url, timeout=2)
        if r.status_code == 200:
            print("[start] A1111 API on :3000 is ready.")
            sys.exit(0)
    except Exception:
        pass
    time.sleep(1)
print("[start] WARNING: A1111 API on :3000 not ready after wait (continuing anyway).")
sys.exit(0)
PY

# -----------------------------
# Dump models via API (sanity: A1111 sees your ckpt)
# -----------------------------
echo "[debug] sd-models from API:"
( curl -s http://127.0.0.1:3000/sdapi/v1/sd-models \
  | jq -r 'try .[].title catch empty' \
  | sed 's/^/[sd-models] /' ) || true

# -----------------------------
# Dump available routes (helps confirm any deforum endpoints)
# -----------------------------
echo "[debug] Dumping /openapi.json route keys…"
( curl -s http://127.0.0.1:3000/openapi.json \
  | jq -r 'try .paths | keys[] catch empty' \
  | sed 's/^/[route] /' ) || echo "[debug] Could not fetch /openapi.json (jq/curl missing or server busy)."

# -----------------------------
# Start RunPod worker (BLOCKS). Use -u for unbuffered logs.
# -----------------------------
echo "[start] Starting RunPod worker…"
exec python -u /workspace/rp_handler.py
