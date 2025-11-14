# rp_handler.py — Deforum CLI runner with timings, JSON logs, optional CKPT, and Vercel Blob upload
import os, json, glob, time, tempfile, subprocess, mimetypes, uuid
from pathlib import Path
import requests
import runpod

# -------------------- config --------------------
A1111_PORT = os.getenv("A1111_PORT", "3001")  # avoid clash with background 3000
JOB_TIMEOUT_SECS = int(os.getenv("DEFORUM_JOB_TIMEOUT", "900"))
A1111_ROOT = "/workspace/stable-diffusion-webui"
VENVPY = f"{A1111_ROOT}/venv/bin/python"
LAUNCH = f"{A1111_ROOT}/launch.py"

# -------------------- logging / timing --------------------
def _jsonlog(event: str, **fields):
    rec = {"event": event, **fields}
    try:
        print(json.dumps(rec, ensure_ascii=False))
    except Exception:
        print(f"[log-fallback] {event} {fields}")

def _tail(txt: str, n: int = 1600) -> str:
    return (txt or "")[-n:]

def make_timer():
    timings = []
    def timed(step_name):
        def deco(fn):
            def inner(*a, **k):
                t0 = time.perf_counter()
                try:
                    return fn(*a, **k)
                finally:
                    ms = int((time.perf_counter() - t0) * 1000)
                    timings.append({"step": step_name, "ms": ms})
                    _jsonlog("timing", step=step_name, elapsed_ms=ms)
            return inner
        return deco
    return timings, timed

# -------------------- schedules & outputs --------------------
def _schedule(val, default_str: str) -> str:
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

# -------------------- vercel blob --------------------
def upload_to_vercel_blob(file_path: str, run_id: str):
    base = os.getenv("VERCEL_BLOB_BASE") or os.getenv("BLOB_BASE")
    token = (os.getenv("VERCEL_BLOB_RW_TOKEN")
             or os.getenv("VERCEL_BLOB_READ_WRITE_TOKEN")
             or os.getenv("VERCEL_BLOB_TOKEN"))
    if not base or not token:
        return {"ok": False, "reason": "missing_env"}
    p = Path(file_path)
    if not p.exists():
        return {"ok": False, "reason": "file_missing"}
    key = f"runs/{run_id}/{p.name}"
    url = f"{base.rstrip('/')}/?pathname={requests.utils.quote(key, safe='')}"
    ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    with p.open("rb") as f:
        r = requests.put(url, headers={"Authorization": f"Bearer {token}", "Content-Type": ctype},
                         data=f.read(), timeout=180)
    if r.status_code not in (200, 201):
        body = _tail(r.text, 400)
        return {"ok": False, "reason": f"upload_http_{r.status_code}", "body": body}
    try:
        body = r.json()
    except Exception:
        body = {"url": r.text}
    return {"ok": True, "url": body.get("url") or url, "key": key}

# -------------------- job builder --------------------
def build_deforum_job(inp: dict) -> dict:
    """
    Minimal, safe Deforum config. For smoke tests, ControlNet is OMITTED entirely unless enabled.
    (If present but disabled, Deforum still tries to parse schedules and can crash.)
    """
    prompt = inp.get("prompt", "a simple colored shape on a plain background")
    max_frames = int(inp.get("max_frames", 8))
    W = int(inp.get("width", 512))
    H = int(inp.get("height", 512))
    seed = int(inp.get("seed", 1))
    steps = int(inp.get("steps", 15))
    fps = int(inp.get("fps", 8))

    job = {
        "prompt": {"0": prompt},
        "seed": seed,
        "max_frames": max_frames,
        "W": W,
        "H": H,
        "sampler": "Euler a",
        "steps": steps,
        "cfg_scale": 7,
        "animation_mode": "2D",
        "fps": fps,
        # No motion transforms for smoke
        "angle": "0:(0)",
        "zoom": "0:(1.0)",
        "translation_x": "0:(0)",
        "translation_y": "0:(0)",
        "translation_z": "0:(0)",
        # No init/pose for smoke test
        "use_init": False,
        "init_image": "",
        "video_init_path": "",
        "use_parseq": False,
        # Output
        "make_video": True,
        "save_video": True,
        "outdir": "/workspace/outputs/deforum",
        "outdir_video": "/workspace/outputs/deforum",
    }

    # --- Only include ControlNet block if explicitly enabled ---
    if bool(inp.get("controlnet_enabled", False)):
        # Use Deforum's expected key names:
        controlnet_args = {
            "enabled": True,
            "controlnet_module": inp.get("controlnet_module", "openpose_full"),           # preprocessor
            "controlnet_model": inp.get("controlnet_model", "control_sd15_animal_openpose_fp16"),
            "controlnet_pixel_perfect": True,
            # schedules:
            "controlnet_weight":            _schedule(inp.get("cn_weight"), "0:(1.0)"),
            "controlnet_guidance_start":    _schedule(inp.get("cn_start"),  "0:(0.0)"),
            "controlnet_guidance_end":      _schedule(inp.get("cn_end"),    "0:(1.0)"),
            "controlnet_detect_resolution": _schedule(inp.get("cn_res"),    "0:(512)"),
            "guess_mode":                   _schedule(inp.get("cn_guess"),  "0:(0)"),
            "threshold_a":                  _schedule(inp.get("cn_th_a"),   "0:(64)"),
            "threshold_b":                  _schedule(inp.get("cn_th_b"),   "0:(64)"),
            "softness":                     _schedule(inp.get("cn_soft"),   "0:(0.0)"),
        }
        job["controlnet_args"] = controlnet_args
    # else: omit controlnet_args entirely

    return job

# -------------------- CLI runner --------------------
def run_deforum_cli(job: dict, timeout_sec: int = None):
    timeout_sec = timeout_sec or JOB_TIMEOUT_SECS
    out_dir = job.get("outdir", "/workspace/outputs/deforum")
    os.makedirs(out_dir, exist_ok=True)

    cfg = tempfile.NamedTemporaryFile("w", delete=False, suffix=".json")
    json.dump(job, cfg)
    cfg.close()

    ckpt = os.getenv("CKPT_PATH", "").strip()
    ckpt_exists = bool(ckpt and os.path.exists(ckpt))

    cmd = [
        VENVPY, LAUNCH,
        "--nowebui",
        "--xformers",
        "--api",
        "--enable-insecure-extension-access",
        "--port", A1111_PORT,
        "--deforum-run-now", cfg.name,
        "--deforum-terminate-after-run-now",
    ]
    if ckpt_exists:
        cmd += ["--no-download-sd-model", "--ckpt", ckpt, "--skip-install"]

    _jsonlog("deforum.exec", cmd=" ".join(cmd), ckpt=ckpt if ckpt_exists else None)

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_sec,
            cwd=A1111_ROOT,
            env={**os.environ},
        )
        return {"retcode": proc.returncode, "tail": _tail(proc.stdout, 2000), "outdir": out_dir}
    except subprocess.TimeoutExpired as e:
        return {"retcode": 504, "tail": _tail((e.stdout or "") + "\n[TIMEOUT]", 2000), "outdir": out_dir}
    except Exception as e:
        return {"retcode": 500, "tail": f"[EXC] {type(e).__name__}: {e}", "outdir": out_dir}
    finally:
        try:
            os.unlink(cfg.name)
        except Exception:
            pass

# -------------------- handler --------------------
def handler(event):
    run_id = uuid.uuid4().hex[:8]
    inp = (event or {}).get("input") or {}
    _jsonlog("handler.start", run_id=run_id, input_keys=list(inp.keys()))

    timings, timed = make_timer()

    @timed("build_job")
    def _build():
        return build_deforum_job(inp)

    @timed("run_cli")
    def _run(job):
        req_timeout_ms = inp.get("timeout_ms")
        req_timeout = int(req_timeout_ms / 1000) if isinstance(req_timeout_ms, (int, float)) else None
        return run_deforum_cli(job, timeout_sec=req_timeout)

    @timed("pick_video")
    def _pick():
        return newest_video([
            "/workspace/outputs/deforum",
            "/workspace/stable-diffusion-webui/outputs/deforum",
            "/workspace/stable-diffusion-webui/outputs",
        ])

    @timed("maybe_upload")
    def _upload(picked_path: str):
        if picked_path and inp.get("upload"):
            return upload_to_vercel_blob(picked_path, run_id)
        return {"ok": False, "reason": "skipped"}

    job = _build()
    res = _run(job)
    picked = _pick()
    uploaded = _upload(picked)

    env_seen = {
        "blob_base_set": bool(os.getenv("VERCEL_BLOB_BASE") or os.getenv("BLOB_BASE")),
        "blob_token_set": bool(os.getenv("VERCEL_BLOB_RW_TOKEN") or os.getenv("VERCEL_BLOB_READ_WRITE_TOKEN") or os.getenv("VERCEL_BLOB_TOKEN")),
        "ckpt_path": os.getenv("CKPT_PATH") or "",
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

    _jsonlog("handler.end", run_id=run_id, ok=ok, picked=bool(picked))
    return {"status": "COMPLETED" if ok else "FAILED", "result": out}

runpod.serverless.start({"handler": handler})