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
    Uploads a file to Vercel Blob, trying the official PUT flow first, then
    a POST multipart fallback if available.

    Env it understands:
      - VERCEL_BLOB_BASE (must be https://api.blob.vercel-storage.com)
      - VERCEL_BLOB_RW_TOKEN / VERCEL_BLOB_READ_WRITE_TOKEN / VERCEL_BLOB_TOKEN
      - VERCEL_BLOB_READ_WRITE_URL (legacy/alt POST endpoint; optional)
    """
    base = os.getenv("VERCEL_BLOB_BASE") or os.getenv("BLOB_BASE") or "https://api.blob.vercel-storage.com"
    token = (os.getenv("VERCEL_BLOB_RW_TOKEN")
             or os.getenv("VERCEL_BLOB_READ_WRITE_TOKEN")
             or os.getenv("VERCEL_BLOB_TOKEN"))
    post_url = os.getenv("VERCEL_BLOB_READ_WRITE_URL")  # optional signed URL variant

    # Normalize: we expect the API host for writes
    base = base.rstrip("/")
    if "vercel-storage.com" in base and not base.startswith("https://api.blob.vercel-storage.com"):
        # You likely pointed at a public/read host; fix it automatically
        base = "https://api.blob.vercel-storage.com"

    p = Path(file_path)
    if not p.exists():
        return {"ok": False, "reason": "file_missing"}

    if not token and not post_url:
        return {"ok": False, "reason": "missing_env", "need": ["VERCEL_BLOB_RW_TOKEN or VERCEL_BLOB_READ_WRITE_URL"]}

    # Reasonable, safe key (no spaces, no weird chars)
    safe_name = p.name.replace(" ", "-")
    key = f"runs/{run_id}/{safe_name}"
    ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    body = p.read_bytes()

    attempts = []

    # --- Attempt A: PUT with pathname no leading slash
    try:
        url = f"{base}?pathname={requests.utils.quote(key, safe='')}"
        r = requests.put(url, headers={"Authorization": f"Bearer {token}", "Content-Type": ctype}, data=body, timeout=180)
        attempts.append({"method": "PUT", "url": url, "status": r.status_code, "text_tail": (r.text or "")[-200:]})
        if r.status_code in (200, 201):
            try:
                jb = r.json()
            except Exception:
                jb = {"url": r.text}
            return {"ok": True, "url": jb.get("url") or jb.get("downloadUrl") or jb.get("pathname") or url,
                    "key": key, "attempts": attempts}
    except Exception as e:
        attempts.append({"method": "PUT", "error": str(e)})

    # --- Attempt B: PUT with leading slash pathname
    try:
        url2 = f"{base}?pathname=%2F{requests.utils.quote(key, safe='')}"
        r2 = requests.put(url2, headers={"Authorization": f"Bearer {token}", "Content-Type": ctype}, data=body, timeout=180)
        attempts.append({"method": "PUT", "url": url2, "status": r2.status_code, "text_tail": (r2.text or "")[-200:]})
        if r2.status_code in (200, 201):
            try:
                jb = r2.json()
            except Exception:
                jb = {"url": r2.text}
            return {"ok": True, "url": jb.get("url") or jb.get("downloadUrl") or jb.get("pathname") or url2,
                    "key": key, "attempts": attempts}
    except Exception as e:
        attempts.append({"method": "PUT-leading-slash", "error": str(e)})

    # --- Attempt C: POST multipart (if a signed RW URL is available)
    if post_url:
        try:
            files = {
                "file": (safe_name, body, ctype),
            }
            data = {"pathname": key}
            r3 = requests.post(post_url, headers={"Authorization": f"Bearer {token}"} if token else {}, files=files, data=data, timeout=180)
            attempts.append({"method": "POST-multipart", "url": post_url, "status": r3.status_code, "text_tail": (r3.text or "")[-200:]})
            if r3.status_code in (200, 201):
                try:
                    jb = r3.json()
                except Exception:
                    jb = {"url": r3.text}
                return {"ok": True, "url": jb.get("url") or jb.get("downloadUrl") or jb.get("pathname"), "key": key, "attempts": attempts}
        except Exception as e:
            attempts.append({"method": "POST-multipart", "error": str(e)})

    # If we got here, all attempts failed
    return {"ok": False, "reason": "all_attempts_failed", "attempts": attempts, "suggest": {
        "ensure_base": "Set VERCEL_BLOB_BASE=https://api.blob.vercel-storage.com",
        "ensure_token": "Use a Read-Write token (VERCEL_BLOB_RW_TOKEN)",
        "pathname_rules": "Avoid spaces; try without and with leading slash",
    }}

# ---------- job builder ----------
def build_deforum_job(inp: dict) -> dict:
    S = _sched

    prompt     = inp.get("prompt", "a photorealistic orange tabby cat doing a simple dance, studio lighting")
    max_frames = int(inp.get("max_frames", 12))
    W          = int(inp.get("width", 512))
    H          = int(inp.get("height", 512))
    seed       = int(inp.get("seed", 42))
    fps        = int(inp.get("fps", 8))

    cn         = inp.get("controlnet") or {}
    cn_enabled = bool(cn.get("enabled", False))
    cn_vid     = cn.get("vid_path") or inp.get("pose_video_path") or ""

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
        "angle": "0:(0)",
        "zoom": "0:(1.0)",
        "translation_x": "0:(0)",
        "translation_y": "0:(0)",
        "translation_z": "0:(0)",
        "use_init": False,
        "init_image": "",
        "video_init_path": "",
        "use_parseq": False,
        "make_video": True,
        "save_video": True,
        "outdir": "/workspace/outputs/deforum",
        "outdir_video": "/workspace/outputs/deforum",
    }

    # IMPORTANT: only include controlnet_args when explicitly enabled
    if cn_enabled:
        cn_ns = {
            "cn_1_enabled": True,
            "cn_1_model": cn.get("model", "None"),           # set to a real filename once the model is installed
            "cn_1_module": cn.get("module", "openpose_full"),
            "cn_1_weight": S(cn.get("weight"), "0:(1.0)"),
            "cn_1_weight_schedule_series": S(cn.get("weight_schedule_series"), "0:(1.0)"),
            "cn_1_guidance_start": S(cn.get("guidance_start"), "0:(0.0)"),
            "cn_1_guidance_end": S(cn.get("guidance_end"), "0:(1.0)"),
            "cn_1_processor_res": S(cn.get("processor_res"), "0:(512)"),
            "cn_1_threshold_a": S(cn.get("threshold_a"), "0:(64)"),
            "cn_1_threshold_b": S(cn.get("threshold_b"), "0:(64)"),
            "cn_1_guess_mode": S(cn.get("guess_mode"), "0:(0)"),
            "cn_1_invert_image": S(cn.get("invert_image"), "0:(0)"),
            "cn_1_rgbbgr_mode": S(cn.get("rgbbgr_mode"), "0:(0)"),
            "cn_1_pixel_perfect": bool(cn.get("pixel_perfect", True)),
            "cn_1_resize_mode": cn.get("resize_mode", "Inner Fit (Scale to Fit)"),
            "cn_1_control_mode": cn.get("control_mode", "Balanced"),
            "cn_1_low_vram": bool(cn.get("low_vram", False)),
            "cn_1_loopback_mode": bool(cn.get("loopback_mode", False)),
            "cn_1_overwrite_frames": True,
            "cn_1_mask_vid_path": "",
            "cn_1_vid_path": cn_vid,
        }
        job["controlnet_args"] = cn_ns
        # (Optionally) keep dupes at top level for older loaders:
        job.update(cn_ns)
    else:
        job["controlnet_args"] = None  # <- this is the key to avoid the schedule parser

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
