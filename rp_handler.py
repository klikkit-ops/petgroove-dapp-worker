# rp_handler.py — Deforum CLI runner with full cn_* schedules + timing + optional Vercel Blob upload
import os, json, glob, time, tempfile, subprocess, mimetypes, uuid
from pathlib import Path
import requests
import runpod

A1111 = "http://127.0.0.1:3000"

# ---------- small helpers ----------
def _tail(txt: str, n: int = 1600) -> str:
    return (txt or "")[-n:]

def _schedule(val, default_str: str) -> str:
    """
    Deforum wants schedule STRINGS like '0:(0.75)' for many fields.
    - number -> "0:(number)"
    - blank/None -> default_str
    - string with schedule syntax -> as-is
    - any other string -> wrap as "0:(string)"
    """
    if val is None or val == "":
        return default_str
    if isinstance(val, (int, float)):
        return f"0:({val})"
    s = str(val).strip()
    if ":" in s and "(" in s and ")" in s:
        return s
    return f"0:({s})"

def newest_video(paths):
    cand = []
    for p in paths:
        cand.extend(glob.glob(os.path.join(p, "**", "*.mp4"), recursive=True))
        cand.extend(glob.glob(os.path.join(p, "**", "*.webm"), recursive=True))
        cand.extend(glob.glob(os.path.join(p, "**", "*.mov"), recursive=True))
    if not cand:
        return None
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

# ---------- Deforum job builder ----------
def _cn_block(idx: int, enabled: bool, module: str, model: str, user: dict):
    """
    Build a COMPLETE ControlNet slot (cn_#_*) with all schedule fields as STRINGS.
    Deforum commonly parses schedules for weight + guidance_window; processor_res/thresholds are scalars.
    """
    pfx = f"cn_{idx}_"
    # Defaults are safe/minimal; all schedule-y fields become strings
    d = {
        f"{pfx}overwrite_frames": True,
        f"{pfx}vid_path": user.get("pose_video_path", "") or "",
        f"{pfx}mask_vid_path": "",
        f"{pfx}enabled": bool(enabled),
        f"{pfx}low_vram": False,
        f"{pfx}pixel_perfect": True,
        f"{pfx}module": module,              # e.g. "openpose_full" (preprocessor); "none" if not using
        f"{pfx}model": model,                # e.g. "control_sd15_animal_openpose_fp16" or "None"
        f"{pfx}weight": _schedule(user.get("cn_weight"), "0:(1.0)"),
        f"{pfx}guidance_start": _schedule(user.get("cn_guidance_start"), "0:(0.0)"),
        f"{pfx}guidance_end": _schedule(user.get("cn_guidance_end"), "0:(1.0)"),
        f"{pfx}processor_res": int(user.get("cn_processor_res", 512)),
        f"{pfx}threshold_a": int(user.get("cn_threshold_a", 64)),
        f"{pfx}threshold_b": int(user.get("cn_threshold_b", 64)),
        f"{pfx}resize_mode": user.get("cn_resize_mode", "Inner Fit (Scale to Fit)"),
        f"{pfx}control_mode": user.get("cn_control_mode", "Balanced"),
        f"{pfx}loopback_mode": False,
    }
    return d

def build_deforum_job(inp: dict) -> dict:
    """
    Build a Deforum config using the key names Deforum actually expects.
    NOTE: We always send string schedules so nothing is None.
    """
    def S(x, default_str):
        # schedule helper: coerce to deforum-style string schedule
        if x is None or x == "":
            return default_str
        if isinstance(x, (int, float)):
            return f"0:({x})"
        s = str(x).strip()
        return s if (":" in s and "(" in s and ")" in s) else f"0:({s})"

    prompt = inp.get("prompt", "a photorealistic orange tabby cat doing a simple dance, studio lighting")
    max_frames = int(inp.get("max_frames", 12))
    W = int(inp.get("width", 512))
    H = int(inp.get("height", 512))
    seed = int(inp.get("seed", 42))
    fps  = int(inp.get("fps", 8))

    # ---- ControlNet (Deforum expects cn_1_* keys) ----
    # Keep disabled by default for smoke tests; you can flip with input.controlnet.enabled = true
    cn = inp.get("controlnet") or {}
    cn_enabled = bool(cn.get("enabled", False))

    # If you’re driving from a pose VIDEO, put its path (or URL Deforum can load) here:
    # We pass through a cn_1_vid_path only if provided by caller.
    cn_1_vid_path = cn.get("vid_path") or inp.get("pose_video_path") or ""

    controlnet_args = {
        "cn_1_enabled": cn_enabled,
        "cn_1_model": cn.get("model", "control_sd15_animal_openpose_fp16"),
        "cn_1_module": cn.get("module", "openpose"),   # 'openpose' works with Animal OpenPose
        "cn_1_weight": S(cn.get("weight"), "0:(1.0)"),
        "cn_1_guidance_start": S(cn.get("guidance_start"), "0:(0.0)"),
        "cn_1_guidance_end": S(cn.get("guidance_end"), "0:(1.0)"),
        "cn_1_pixel_perfect": bool(cn.get("pixel_perfect", True)),
        "cn_1_annotator_resolution": S(cn.get("annotator_resolution"), "0:(512)"),
    }
    if cn_1_vid_path:
        controlnet_args["cn_1_vid_path"] = cn_1_vid_path  # only include if provided

    # ---- Core Deforum params (lean & safe) ----
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

        # common transforms as schedules
        "angle": "0:(0)",
        "zoom": "0:(1.0)",
        "translation_x": "0:(0)",
        "translation_y": "0:(0)",
        "translation_z": "0:(0)",

        # init OFF for smoke tests
        "use_init": False,
        "init_image": "",
        "video_init_path": "",

        # not using Parseq
        "use_parseq": False,

        # outputs
        "make_video": True,
        "save_video": True,
        "outdir": "/workspace/outputs/deforum",
        "outdir_video": "/workspace/outputs/deforum",

        # attach controlnet args so Deforum parser sees cn_1_* keys
        "controlnet_args": controlnet_args,
    }

    return job

# ---------- run via launch.py (CLI) ----------
def run_via_launch(job: dict, timings: list):
    """
    Write a temp JSON config and ask A1111 to run it immediately via --deforum-run-now.
    """
    t0 = time.time()
    out_dir = job.get("outdir", "/workspace/outputs/deforum")
    os.makedirs(out_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(job, f)
        cfg_path = f.name

    venv_py = "/workspace/stable-diffusion-webui/venv/bin/python"
    webui = "/workspace/stable-diffusion-webui"

    # Use existing API server context; Deforum will run and then exit if requested
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
    tmarks = []
    inp = (event or {}).get("input") or {}

    t0 = time.time()
    job = build_deforum_job(inp)
    tmarks.append({"step": "build_job", "ms": int((time.time() - t0) * 1000)})

    # Run
    res = run_via_launch(job, tmarks)

    # Pick newest video
    t1 = time.time()
    picked = newest_video([
        "/workspace/outputs/deforum",
        "/workspace/stable-diffusion-webui/outputs/deforum",
        "/workspace/stable-diffusion-webui/outputs",
    ])
    tmarks.append({"step": "pick_video", "ms": int((time.time() - t1) * 1000)})

    # Optional upload
    t2 = time.time()
    uploaded = {"ok": False, "reason": "skipped"}
    if picked and inp.get("upload"):
        uploaded = upload_to_vercel_blob(picked, run_id)
    tmarks.append({"step": "maybe_upload", "ms": int((time.time() - t2) * 1000)})

    env_seen = {
        "blob_base_set": bool(os.getenv("VERCEL_BLOB_BASE") or os.getenv("BLOB_BASE")),
        "blob_token_set": bool(os.getenv("VERCEL_BLOB_RW_TOKEN") or os.getenv("VERCEL_BLOB_READ_WRITE_TOKEN") or os.getenv("VERCEL_BLOB_TOKEN")),
        "ckpt_path": os.getenv("SD_CKPT_PATH", ""),
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
        "timings": tmarks,
    }
    if not ok:
        out["launch_tail"] = res.get("tail")

    return {"status": "COMPLETED" if ok else "FAILED", "result": out}

runpod.serverless.start({"handler": handler})
