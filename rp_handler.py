# rp_handler.py — Deforum CLI runner with full top-level cn_1_* schedules + timing
import os, json, glob, time, tempfile, subprocess, mimetypes, uuid
from pathlib import Path
import requests
import runpod

A1111 = "http://127.0.0.1:3000"

# ---------- helpers ----------
def _tail(txt: str, n: int = 2000) -> str:
    return (txt or "")[-n:]

def _sched(val, default_str: str) -> str:
    """Always return a Deforum-style schedule string like '0:(0.75)'."""
    if val is None or val == "": return default_str
    if isinstance(val, (int, float)): return f"0:({val})"
    s = str(val).strip()
    return s if (":" in s and "(" in s and ")") else f"0:({s})"

def newest_video(paths):
    cand = []
    for p in paths:
        cand.extend(glob.glob(os.path.join(p, "**", "*.mp4"), recursive=True))
        cand.extend(glob.glob(os.path.join(p, "**", "*.webm"), recursive=True))
        cand.extend(glob.glob(os.path.join(p, "**", "*.mov"), recursive=True))
    if not cand: return None
    cand.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
    return cand[0]

def upload_to_vercel_blob(file_path: str, run_id: str):
    base = os.getenv("VERCEL_BLOB_BASE") or os.getenv("BLOB_BASE")
    token = (os.getenv("VERCEL_BLOB_RW_TOKEN")
             or os.getenv("VERCEL_BLOB_READ_WRITE_TOKEN")
             or os.getenv("VERCEL_BLOB_TOKEN"))
    if not base or not token:
        return {"ok": False, "reason": "missing_env"}
    path = Path(file_path)
    if not path.exists():
        return {"ok": False, "reason": "file_missing"}
    key = f"runs/{run_id}/{path.name}"
    url = f"{base.rstrip('/')}/?pathname={requests.utils.quote(key, safe='')}"
    ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    with path.open("rb") as f:
        r = requests.put(url, headers={"Authorization": f"Bearer {token}", "Content-Type": ctype},
                         data=f.read(), timeout=180)
    if r.status_code not in (200, 201):
        try:
            body = r.json()
        except Exception:
            body = _tail(r.text, 400)
        return {"ok": False, "reason": f"upload_http_{r.status_code}", "body": body}
    try:
        body = r.json()
    except Exception:
        body = {"url": r.text}
    return {"ok": True, "url": body.get("url") or url, "key": key}

# ---------- build a Deforum config that matches YOUR keys ----------

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

    # ---- REQUIRED: nested controlnet_args namespace ----
    cn_ns = {
        "cn_1_enabled": cn_enabled,
        "cn_1_model": cn.get("model", "None"),           # keep "None" until the real model is present
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

    # (Optional) keep a duplicate at top level too; harmless and sometimes helpful with older loaders:
    job.update(cn_ns)

    return job

# ---------- run Deforum through launch.py ----------
def run_via_launch(job: dict, timings: list):
    t0 = time.time()
    out_dir = job.get("outdir", "/workspace/outputs/deforum")
    os.makedirs(out_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(job, f)
        cfg_path = f.name

    venv_py = "/workspace/stable-diffusion-webui/venv/bin/python"
    webui   = "/workspace/stable-diffusion-webui"

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
    ckpt = os.getenv("CKPT_PATH") or os.getenv("SD_CKPT_PATH")
    if ckpt:
        env["SD_WEBUI_RESTARTING"] = "1"
        env["COMMANDLINE_ARGS"] = (env.get("COMMANDLINE_ARGS","") + f" --ckpt \"{ckpt}\"").strip()

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
        t2 = time.time()
        uploaded = upload_to_vercel_blob(picked, run_id)
        timings.append({"step": "maybe_upload", "ms": int((time.time() - t2) * 1000)})

    env_seen = {
        "blob_base_set": bool(os.getenv("VERCEL_BLOB_BASE") or os.getenv("BLOB_BASE")),
        "blob_token_set": bool(os.getenv("VERCEL_BLOB_RW_TOKEN") or os.getenv("VERCEL_BLOB_READ_WRITE_TOKEN") or os.getenv("VERCEL_BLOB_TOKEN")),
        "ckpt_path": os.getenv("CKPT_PATH", ""),
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
