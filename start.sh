#!/usr/bin/env bash
set -euo pipefail

cd /workspace

# -----------------------------
# Environment / passthrough
# -----------------------------
export COMMANDLINE_ARGS="${WEBUI_ARGS:-"--api --listen --xformers --enable-insecure-extension-access --port 3000"}"
export LAUNCH_BROWSER=0
export PYTHONUNBUFFERED=1

# New: CN envs you can set from the console/UI
# CN_MODEL_NAME     -> model stem, default: control_sd15_animal_openpose_fp16
# CN_MODEL_URL      -> full https URL to download the model from your blob (optional)
# CN_MODULE_NAME    -> preferred preprocessor module (optional)
echo "[start] COMMANDLINE_ARGS=${COMMANDLINE_ARGS}"
echo "[start] A1111_PORT(for CLI)=${A1111_PORT:-3001}"
echo "[start] CKPT_PATH=${CKPT_PATH:-<unset>}"
echo "[start] FORCE_DISABLE_CN=${FORCE_DISABLE_CN:-0}"
echo "[start] CN_MODEL_NAME=${CN_MODEL_NAME:-control_sd15_animal_openpose_fp16}"
echo "[start] CN_MODEL_URL=${CN_MODEL_URL:-<unset>}"
echo "[start] CN_MODULE_NAME=${CN_MODULE_NAME:-<auto>}"

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
# Ensure the ControlNet model is present BEFORE starting A1111
# Accepts .safetensors or .pth; also creates a convenience symlink for the other.
# Set CN_MODEL_URL to auto-download from your blob.
# -----------------------------
CN_DIR="$CN_EXT/models"
CN_MODEL_NAME="${CN_MODEL_NAME:-control_sd15_animal_openpose_fp16}"
CN_ST="$CN_DIR/${CN_MODEL_NAME}.safetensors"
CN_PTH="$CN_DIR/${CN_MODEL_NAME}.pth"

if [[ ! -f "$CN_ST" && ! -f "$CN_PTH" ]]; then
  if [[ -n "${CN_MODEL_URL:-}" ]]; then
    echo "[fetch] Downloading ControlNet model -> $CN_ST"
    curl -L --retry 5 --retry-all-errors -o "$CN_ST" "$CN_MODEL_URL" || echo "[fetch] Download failed (will continue; you can place the file manually)."
  else
    echo "[fetch] CN_MODEL_URL not set and model not found; skipping download."
  fi
fi

# Pair symlinks so either extension path works
if [[ -f "$CN_PTH" && ! -f "$CN_ST" ]]; then
  ln -sf "$(basename "$CN_PTH")" "$CN_ST"
fi
if [[ -f "$CN_ST" && ! -f "$CN_PTH" ]]; then
  ln -sf "$(basename "$CN_ST")" "$CN_PTH"
fi

# -----------------------------
# *** Hard-disable ControlNet if requested ***
# -----------------------------
if [[ "${FORCE_DISABLE_CN:-0}" == "1" ]]; then
  echo "[patch] Hard-disabling Deforum ControlNet via env toggle (FORCE_DISABLE_CN=1)…"
  python - <<'PY' || true
import re
p = "/workspace/stable-diffusion-webui/extensions/sd-webui-deforum/scripts/deforum_helpers/parseq_adapter.py"
try:
    s = open(p, "r", encoding="utf-8").read()
    s2 = re.sub(r"self\.cn_keys\s*=\s*ParseqControlNetKeysDecorator\(self,\s*ControlNetKeys\(anim_args,\s*controlnet_args\)\)\s*if\s*controlnet_args\s*else\s*None",
                "self.cn_keys = None", s, flags=re.M)
    if s2 != s:
        open(p, "w", encoding="utf-8").write(s2)
        print("[patch] parseq_adapter.py patched: ControlNet forcibly disabled.")
    else:
        print("[patch] CN patch already applied or target pattern not found (ok).")
except Exception as e:
    print("[patch] CN patch failed:", e)
PY
fi

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

echo "[check] ControlNet models dir: $CN_DIR"
ls -lh "$CN_DIR" || true

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
# Deforum schema introspection (prints exact cn_* key names used by THIS build)
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
PY

# -----------------------------
# A1111 venv + CLIP safety net
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
# Hotfix Deforum: guard against None in CN schedules + log the offending key
# -----------------------------
python - <<'PY'
import io, os, re

p = "/workspace/stable-diffusion-webui/extensions/sd-webui-deforum/scripts/deforum_helpers/animation_key_frames.py"
try:
    with open(p, "r", encoding="utf-8") as f:
        s = f.read()
    changed = False

    # Patch parse_inbetweens(...) to coerce None and log which CN key (filename) was None
    if "HOTFIX_NONE_PARSE_INBETWEENS" not in s:
        m = re.search(r"(def\s+parse_inbetweens\([^\)]*\):\n)([ \t]+)", s)
        if m:
            indent = m.group(2)
            inject = (
                m.group(1)
                + indent + "# HOTFIX_NONE_PARSE_INBETWEENS: default None to neutral schedule\n"
                + indent + "if value is None:\n"
                + indent + "    try:\n"
                + indent + "        print(f\"[deforum-hotfix] parse_inbetweens: {filename} was None → '0:(0)'\", flush=True)\n"
                + indent + "    except Exception:\n"
                + indent + "        pass\n"
                + indent + "    value = '0:(0)'\n"
            )
            s = s[:m.start()] + inject + s[m.end():]
            changed = True

    # Patch parse_key_frames(...) to coerce None and log
    if "HOTFIX_NONE_PARSE_KEY_FRAMES" not in s:
        m2 = re.search(r"(def\s+parse_key_frames\([^\)]*\):\n)([ \t]+)", s)
        if m2:
            indent2 = m2.group(2)
            inject2 = (
                m2.group(1)
                + indent2 + "# HOTFIX_NONE_PARSE_KEY_FRAMES: default None\n"
                + indent2 + "if string is None:\n"
                + indent2 + "    try:\n"
                + indent2 + "        print(\"[deforum-hotfix] parse_key_frames: received None → '0:(0)'\", flush=True)\n"
                + indent2 + "    except Exception:\n"
                + indent2 + "        pass\n"
                + indent2 + "    string = '0:(0)'\n"
            )
            s = s[:m2.start()] + inject2 + s[m2.end():]
            changed = True

    if changed:
        with open(p, "w", encoding="utf-8") as f:
            f.write(s)
        print("[hotfix] Deforum CN scheduler patched for None handling.")
    else:
        print("[hotfix] Deforum CN scheduler already patched or signatures not found (ok).")
except Exception as e:
    print("[hotfix] Patch failed:", e)
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
for i in range(360):
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
# Sanity: show CN model+module lists
# -----------------------------
echo "[debug] ControlNet model_list:"
curl -s http://127.0.0.1:3000/controlnet/model_list || true; echo
echo "[debug] ControlNet module_list:"
curl -s http://127.0.0.1:3000/controlnet/module_list || true; echo

# -----------------------------
# Start RunPod worker (BLOCKS)
# -----------------------------
echo "[start] Starting RunPod worker…"
exec python -u /workspace/rp_handler.py
