# rp_handler.py — Deforum CLI runner (CN only when enabled) + timing + optional Vercel Blob upload
import os, json, glob, time, tempfile, subprocess, mimetypes, uuid
from pathlib import Path
import requests
import runpod

A1111 = "http://127.0.0.1:3000"

# ---------- helpers ----------
def _tail(txt: str, n: int = 1600) -> str:
    return (txt or "")[-n:]

def _sched(val, default_str: str) -> str:
    if val is None or val == "":
        return default_str
    if isinstance(val, (int, float)):
        return f"0:({val})"
    s = str(val).strip()
    return s if (":" in s and "(" in s and ")" in s) else f"0:({s})"

def newest_video(paths):
    cand = []
    for p in paths:
        cand.extend(glob.glob(os.path.join(p, "**", "*.mp4"),  recursive=True))
        cand.extend(glob.glob(os.path.join(p, "**", "*.webm"), recursive=True))
        cand.extend(glob.glob(os.path.join(p, "**", "*.mov"),  recursive=True))
    if not cand:
        return None
    cand.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
    return cand[0]

def upload_to_vercel_blob(file_path: str, run_id: str):
    """
    First try a user-provided proxy endpoint (VERCEL_BLOB_PROXY_URL) to avoid DNS issues.
    If not set or fails, try direct Blob API. Returns attempts for debugging.
    """
    attempts = []
    path = Path(file_path)
    if not path.exists():
        return {"ok": False, "reason": "file_missing"}

    # Inputs
    api_base  = os.getenv("VERCEL_BLOB_BASE") or os.getenv("BLOB_BASE") or "https://api.blob.vercel-storage.com"
    proxy_url = os.getenv("VERCEL_BLOB_PROXY_URL")  # e.g. https://petgroove.app/api/blob-upload
    token     = (os.getenv("VERCEL_BLOB_RW_TOKEN")
                 or os.getenv("VERCEL_BLOB_READ_WRITE_TOKEN")
                 or os.getenv("VERCEL_BLOB_TOKEN"))
    public_base = os.getenv("VERCEL_BLOB_PUBLIC_BASE")  # optional, to synthesize a public URL

    key = f"runs/{run_id}/{path.name}"
    ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"

    # 1) Try proxy first (POST binary to your domain)
    if proxy_url:
        try:
            with path.open("rb") as f:
                r = requests.post(
                    f"{proxy_url}?pathname={requests.utils.quote(key, safe='')}",
                    headers={"Content-Type": ctype},
                    data=f.read(),
                    timeout=180,
                )
            if r.status_code in (200, 201):
                body = r.json() if r.headers.get("content-type","").startswith("application/json") else {"url": r.text}
                return {"ok": True, "url": body.get("url"), "key": key, "via": "proxy"}
            attempts.append({"method": "proxy", "status": r.status_code, "body_tail": _tail(r.text, 400)})
        except Exception as e:
            attempts.append({"method": "proxy", "error": str(e)})

    # 2) Fall back to direct Blob API
    if not token:
        return {"ok": False, "reason": "missing_env", "attempts": attempts}

    for variant in ("PUT", "PUT-leading-slash"):
        try:
            url = f"{api_base.rstrip('/')}/?pathname={'/' if variant=='PUT-leading-slash' else ''}{requests.utils.quote(key, safe='')}"
            with path.open("rb") as f:
                r = requests.put(
                    url,
                    headers={"Authorization": f"Bearer {token}", "Content-Type": ctype},
                    data=f.read(),
                    timeout=180,
                )
            if r.status_code in (200, 201):
                # Prefer API JSON url; otherwise synthesize a public URL if available
                try:
                    body = r.json()
                except Exception:
                    body = {"url": r.text}
                url_out = body.get("url")
                if not url_out and public_base:
                    url_out = f"{public_base.rstrip('/')}/{key}"
                return {"ok": True, "url": url_out or url, "key": key, "via": "api"}
            attempts.append({"method": variant, "status": r.status_code, "body_tail": _tail(r.text, 400)})
        except Exception as e:
            attempts.append({"method": variant, "error": str(e)})

    return {
        "ok": False,
        "reason": "all_attempts_failed",
        "attempts": attempts,
        "suggest": {
            "ensure_base": "Set VERCEL_BLOB_BASE=https://api.blob.vercel-storage.com",
            "ensure_token": "Use a Read-Write token (VERCEL_BLOB_RW_TOKEN)",
            "proxy": "Set VERCEL_BLOB_PROXY_URL to a Vercel API route if DNS resolution keeps failing"
        },
    }
    
# ---------- Deforum job builder ----------
def build_deforum_job(inp: dict) -> dict:
    """
    Build a Deforum config using the key names THIS Deforum build expects.
    We fully populate cn_1_* so Deforum's schedule parser never sees None.
    """
    def S(x, default_str):  # schedule -> "0:(…)" string
        if x is None or x == "": return default_str
        if isinstance(x, (int, float)): return f"0:({x})"
        s = str(x).strip()
        return s if (":" in s and "(" in s and ")" in s) else f"0:({s})"

    prompt = inp.get("prompt", "a photorealistic orange tabby cat doing a simple dance, studio lighting")
    max_frames = int(inp.get("max_frames", 12))
    W = int(inp.get("width", 512))
    H = int(inp.get("height", 512))
    seed = int(inp.get("seed", 42))
    fps  = int(inp.get("fps", 8))

    # ---- ControlNet (enable via input.controlnet.enabled) ----
    cn = inp.get("controlnet") or {}
    cn_enabled = bool(cn.get("enabled", False))
    cn_vid = cn.get("vid_path") or inp.get("pose_video_path") or ""
    cn_model = cn.get("model", "control_sd15_animal_openpose_fp16")
    cn_module = cn.get("module", "openpose")  # works with Animal OpenPose

    # NOTE: keys below mirror what your grep showed Deforum expecting.
    controlnet_args = {
        # toggles / misc
        "cn_1_enabled": cn_enabled,
        "cn_1_low_vram": False,
        "cn_1_pixel_perfect": True,
        "cn_1_loopback_mode": False,
        "cn_1_overwrite_frames": True,
        "cn_1_invert_image": False,
        "cn_1_rgbbgr_mode": False,   # keep normal RGB

        # model + preprocessor
        "cn_1_module": cn_module,    # e.g. "openpose"
        "cn_1_model": cn_model,      # e.g. "control_sd15_animal_openpose_fp16"

        # video / mask paths (strings; OK if blank)
        "cn_1_vid_path": cn_vid,
        "cn_1_mask_vid_path": "",

        # scalar knobs (integers / enums)
        "cn_1_processor_res": int(cn.get("processor_res", 512)),
        "cn_1_threshold_a": int(cn.get("threshold_a", 64)),
        "cn_1_threshold_b": int(cn.get("threshold_b", 64)),
        "cn_1_resize_mode": cn.get("resize_mode", "Inner Fit (Scale to Fit)"),
        "cn_1_control_mode": cn.get("control_mode", "Balanced"),

        # schedules (must be strings)
        "cn_1_weight": S(cn.get("weight"), "0:(1.0)"),
        "cn_1_guidance_start": S(cn.get("guidance_start"), "0:(0.0)"),
        "cn_1_guidance_end": S(cn.get("guidance_end"), "0:(1.0)"),
        "cn_1_guess_mode": S(cn.get("guess_mode"), "0:(0)"),

        # some builds also look for a separate series string; keep it valid
        "cn_1_weight_schedule_series": S(cn.get("weight_schedule_series"), "0:(1.0)"),
    }

    job = {
        "prompt": {"0": prompt},
        "seed": seed,
        "max_frames": max_frames,
        "W": W,
        "H": H,
        "fps": fps,
        "sampler": "Euler a",
        "steps": 25,
        "cfg_scale": 7,
        "animation_mode": "2D",

        # transforms as schedules
        "angle": "0:(0)",
        "zoom": "0:(1.0)",
        "translation_x": "0:(0)",
        "translation_y": "0:(0)",
        "translation_z": "0:(0)",

        # init off for smoke tests
        "use_init": False,
        "init_image": "",
        "video_init_path": "",

        # not using Parseq in our flow
        "use_parseq": False,

        # outputs
        "make_video": True,
        "save_video": True,
        "outdir": "/workspace/outputs/deforum",
        "outdir_video": "/workspace/outputs/deforum",

        # attach full CN block
        "controlnet_args": controlnet_args,
    }

    return job

# ---------- run via launch.py ----------
def run_via_launch(job: dict, timings: list):
    t0 = time.time()
    out_dir = job.get("outdir", "/workspace/outputs/deforum")
    os.makedirs(out_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(job, f)
        cfg_path = f.name

    venv_py = "/workspace/stable-diffusion-webui/venv/bin/python"
    webui = "/workspace/stable-diffusion-webui"

    args = [
        venv_py, os.path.join(webui, "launch.py"),
        "--nowebui", "--skip-install",
        "--deforum-run-now", cfg_path,
        "--deforum-terminate-after-run-now",
        "--api", "--listen", "--xformers",
        "--enable-insecure-extension-access",
        "--port", "3000",
    ]
    env = os.environ.copy()
    proc = subprocess.run(args, cwd=webui, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=3600)
    timings.append({"step": "run_cli", "ms": int((time.time() - t0) * 1000)})

    return {"retcode": proc.returncode, "tail": _tail(proc.stdout, 4000), "outdir": out_dir}

# ---------- handler ----------
def handler(event):
    run_id = uuid.uuid4().hex[:8]
    timings = []
    inp = (event or {}).get("input") or {}

    t0 = time.time()
    job = build_deforum_job(inp)
    timings.append({"step": "build_job", "ms": int((time.time() - t0) * 1000)})

    res = run_via_launch(job, timings)

    t1 = time.time()
    picked = newest_video([
        "/workspace/outputs/deforum",
        "/workspace/stable-diffusion-webui/outputs/deforum",
        "/workspace/stable-diffusion-webui/outputs",
    ])
    timings.append({"step": "pick_video", "ms": int((time.time() - t1) * 1000)})

    uploaded = {"ok": False, "reason": "skipped"}
    if picked and inp.get("upload"):
        uploaded = upload_to_vercel_blob(picked, run_id)

    env_seen = {
        "blob_base_set": bool(os.getenv("VERCEL_BLOB_BASE") or os.getenv("BLOB_BASE")),
        "blob_token_set": bool(os.getenv("VERCEL_BLOB_RW_TOKEN") or os.getenv("VERCEL_BLOB_READ_WRITE_TOKEN") or os.getenv("VERCEL_BLOB_TOKEN")),
        "ckpt_path": os.getenv("SD_CKPT_PATH", "") or os.getenv("CKPT_PATH", ""),
    }

    ok = (res["retcode"] == 0) and bool(picked)
    out = {
        "ok": ok,
        "mode": "cli-launch",
        "run_id": run_id,
        "local_outdir": res.get("outdir"),
        "picked_file": picked,
        "uploaded": uploaded,
        "env_seen": env_seen,
        "timings": timings,
    }
    if not ok:
        out["launch_tail"] = res.get("tail")

    return {"status": "COMPLETED" if ok else "FAILED", "result": out}

runpod.serverless.start({"handler": handler})
