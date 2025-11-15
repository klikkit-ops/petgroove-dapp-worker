# rp_handler.py — Adds a "debug" mode that holds the worker open and prints diagnostics.
import os, json, time, glob, mimetypes, uuid, subprocess, re, traceback
from pathlib import Path
import requests
import runpod

A1111 = "http://127.0.0.1:3000"

def _tail(txt: str, n: int = 1600) -> str:
    return (txt or "")[-n:]

def _print_hdr(title: str):
    print(f"\n[debug] === {title} ===")

def _ls(path: str, pattern: str = "*", limit: int = 200):
    try:
        p = Path(path)
        if not p.exists():
            print(f"[debug] ls: {path} (missing)")
            return []
        items = sorted(p.glob(pattern))
        rows = []
        for i, item in enumerate(items[:limit]):
            try:
                sz = item.stat().st_size
            except Exception:
                sz = -1
            print(f"[debug] {path}: {item.name} ({sz} bytes)")
            rows.append({"name": item.name, "size": sz})
        if len(items) > limit:
            print(f"[debug] ... {len(items)-limit} more not shown")
        return rows
    except Exception as e:
        print(f"[debug] ls failed for {path}: {e}")
        return []

def _grep_deforum_cn_keys():
    """Greps Deforum scripts for cn_* names. This reflects EXACT keys used by the installed version."""
    root = "/workspace/stable-diffusion-webui/extensions/sd-webui-deforum/scripts"
    found = set()
    for patt in ["**/*.py", "**/*.json", "**/*.md"]:
        for f in Path(root).glob(patt):
            try:
                text = f.read_text(encoding="utf-8", errors="ignore")
                for m in re.findall(r"\bcn_[0-9]+_[A-Za-z0-9_]+", text):
                    found.add(m)
            except Exception:
                pass
    keys = sorted(found)
    for k in keys[:400]:
        print(f"[deforum.cn] {k}")
    if len(keys) > 400:
        print(f"[deforum.cn] ... {len(keys)-400} more not shown")
    return keys

def _probe_a1111_routes():
    routes = []
    try:
        r = requests.get(f"{A1111}/openapi.json", timeout=5)
        if r.status_code == 200:
            data = r.json()
            paths = sorted((data.get("paths") or {}).keys())
            for p in paths:
                print(f"[route] {p}")
            routes = paths
        else:
            print(f"[debug] openapi.json HTTP {r.status_code}")
    except Exception as e:
        print(f"[debug] openapi.json fetch failed: {e}")
    return routes

def _check_url(url: str, timeout=3):
    try:
        r = requests.get(url, timeout=timeout)
        return {"ok": r.status_code == 200, "status": r.status_code, "body": _tail(r.text, 300)}
    except Exception as e:
        return {"ok": False, "status": None, "error": str(e)}

def _wait_for_a1111(max_sec=360):
    """Wait up to max_sec for A1111 to answer /sdapi/v1/sd-models."""
    url = f"{A1111}/sdapi/v1/sd-models"
    t0 = time.time()
    while time.time() - t0 < max_sec:
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                print("[debug] A1111 API is ready.")
                return True
        except Exception:
            pass
        time.sleep(2)
    print("[debug] A1111 API did not become ready within wait window.")
    return False

def _env_summary():
    return {
        "python": os.popen("python -V").read().strip(),
        "pip_freeze_head": _tail(os.popen("pip freeze | head -n 30").read(), 1200),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        "ckpt_env": os.getenv("CKPT_PATH", "") or os.getenv("SD_CKPT_PATH", ""),
    }

def _find_ckpts():
    base = "/workspace/stable-diffusion-webui/models/Stable-diffusion"
    rows = _ls(base, "*.ckpt")
    rows += _ls(base, "*.safetensors")
    return [{"path": f"{base}/{r['name']}", "size": r["size"]} for r in rows]

def _find_cn_models():
    base = "/workspace/stable-diffusion-webui/extensions/sd-webui-controlnet/models"
    rows = _ls(base, "*")
    hit = [r for r in rows if "animal" in r["name"].lower() or "openpose" in r["name"].lower()]
    return {"all": [r["name"] for r in rows], "animal_openpose_hits": [r["name"] for r in hit]}

# --------------------------
# DEBUG MODE: hold the job open
# --------------------------
def run_debug(inp: dict):
    minutes = int(inp.get("minutes", 4))          # keep this below your endpoint job timeout
    heartbeat = float(inp.get("heartbeat", 5.0))  # seconds between log prints

    _print_hdr("Environment")
    env = _env_summary()
    print(json.dumps(env, indent=2))

    _print_hdr("List models")
    ckpts = _find_ckpts()
    cnmods = _find_cn_models()

    _print_hdr("Check A1111 readiness")
    ready = _wait_for_a1111(max_sec=360)

    _print_hdr("A1111 routes (/openapi.json)")
    routes = _probe_a1111_routes()

    _print_hdr("Deforum cn_* keys found in code")
    cn_keys = _grep_deforum_cn_keys()

    # Known deforum endpoints we’ve seen across versions (often not present):
    candidates = [
        f"{A1111}/deforum/run",
        f"{A1111}/sdapi/v1/deforum/run",
        f"{A1111}/deforum_api/run",
        f"{A1111}/sdapi/v1/deforum_api/run",
    ]
    _print_hdr("Probe likely Deforum endpoints")
    probes = {u: _check_url(u) for u in candidates}
    for u, res in probes.items():
        print(f"[probe] {u} -> {res}")

    # Heartbeat loop to keep worker open
    _print_hdr(f"Heartbeat for {minutes} minute(s)")
    end = time.time() + minutes * 60
    beats = 0
    while time.time() < end:
        beats += 1
        print(f"[debug] heartbeat #{beats} — worker alive; A1111_ready={ready}")
        time.sleep(heartbeat)

    # Return compact summary (full details are in logs)
    return {
        "retcode": 0,
        "summary": {
            "a1111_ready": ready,
            "routes_count": len(routes),
            "deforum_cn_keys": len(cn_keys),
            "ckpts_found": len(ckpts),
            "cn_models_found": cnmods,
            "probe_results": probes,
        }
    }

# --------------------------
# DEFAULT (non-debug) handler paths (no change to your generation code here)
# --------------------------
def handler(event):
    inp = (event or {}).get("input") or {}
    mode = (inp.get("mode") or "").lower()

    if mode == "debug":
        res = run_debug(inp)
        return {"status": "COMPLETED", "result": {"ok": True, "mode": "debug", **res}}

    # If you want to keep your generation path here, return a helpful message for now:
    return {
        "status": "COMPLETED",
        "result": {
            "ok": False,
            "message": "No generation performed. Run with input.mode='debug' to hold worker open.",
        },
    }

runpod.serverless.start({"handler": handler})
