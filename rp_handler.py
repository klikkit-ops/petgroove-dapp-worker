# rp_handler.py — Deforum runner with timing spans, JSON logs + heartbeat, safe ControlNet schedules, Vercel Blob upload
import os, json, glob, time, tempfile, subprocess, mimetypes, uuid, threading, traceback
from pathlib import Path
import requests
import runpod

A1111 = "http://127.0.0.1:3000"

# --------- timing / logging ----------
def _now_ms() -> int:
    return int(time.time() * 1000)

def _tail(txt: str, n: int = 1200) -> str:
    return (txt or "")[-n:]

def log_event(event: str, **fields):
    payload = {"ts_ms": _now_ms(), "event": event}
    payload.update(fields)
    print(json.dumps(payload, ensure_ascii=False), flush=True)

def log_exception(event: str, err: BaseException, **fields):
    tb = "".join(traceback.format_exception(type(err), err, err.__traceback__))
    fields.update({"error": str(err), "traceback_tail": _tail(tb, 2000)})
    log_event(event, **fields)

def timed(span_name: str):
    def _decorator(fn):
        def _wrapped(*args, **kwargs):
            t0 = _now_ms()
            log_event("span.start", span=span_name)
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                log_exception("span.error", e, span=span_name, elapsed_ms=_now_ms() - t0)
                raise
            finally:
                log_event("span.end", span=span_name, elapsed_ms=_now_ms() - t0)
        return _wrapped
    return _decorator

# --------- budgets / heartbeat ----------
DEFAULT_TIMEOUT_MS = int(os.getenv("RUNPOD_REQUEST_TIMEOUT_MS", "54000"))  # allow override per env
HEARTBEAT_MS = int(os.getenv("RUNPOD_HEARTBEAT_MS", "10000"))             # ping every 10s by default

try:
    # Available in runpod==1.7.x+
    from runpod.serverless.utils import rp_job
except Exception:  # graceful fallback if utils path changes
    rp_job = None

def remaining_ok(start_ms: int, budget_ms: int, safety_ms: int = 1500) -> bool:
    return (_now_ms() - start_ms) <= (budget_ms - safety_ms)

def _heartbeat_loop(job_id: str, stop_evt: threading.Event):
    """Emit periodic 'IN_PROGRESS' updates so RunPod doesn't assume we're wedged."""
    if not rp_job:
        # Fall back to log-only heartbeats (still useful for you, not for platform)
        while not stop_evt.wait(HEARTBEAT_MS / 1000.0):
            log_event("heartbeat.log", job_id=job_id)
        return

    while not stop_evt.wait(HEARTBEAT_MS / 1000.0):
        try:
            rp_job.update_job(job_id, status="IN_PROGRESS", progress=0.1, logs="rendering...")
            log_event("heartbeat.sent", job_id=job_id)
        except Exception as e:
            log_exception("heartbeat.error", e, job_id=job_id)

# ---------- existing helpers (kept, with tiny tweaks) ----------
def _schedule(val, default_str: str) -> str:
    """
    Deforum expects schedule STRINGS like '0:(0.75)'. Never None.
    """
    if val is None or val == "":
        return default_str
    if isinstance(val, (int, float)):
        return f"0:({val})"
    s = str(val).strip()
    if ":" in s and "(" in s and ")" in s:
        return s
    return f"0:({s})"

def newest_video(paths, since_ms: int = 0):
    cand = []
    for p in paths:
        for ext in ("*.mp4", "*.webm", "*.mov"):
            cand.extend(glob.glob(os.path.join(p, "**", ext), recursive=True))
    if not cand:
        return None
    def _mtime(p): 
        try: return int(Path(p).stat().st_mtime * 1000)
        except Exception: return 0
    cand = [p for p in cand if _mtime(p) >= since_ms]
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
        body = _tail(r.text, 400)
        return {"ok": False, "reason": f"upload_http_{r.status_code}", "body": body}
    try:
        body = r.json()
    except Exception:
        body = {"url": r.text}
    return {"ok": True, "url": body.get("url") or url, "key": key}

# ---------- job builder (your original, unchanged except for tiny safety) ----------
def build_deforum_job(inp: dict) -> dict:
    prompt = inp.get("prompt", "a photorealistic orange tabby cat doing a simple dance, studio lighting")
    max_frames = int(inp.get("max_frames", 12))
    W = int(inp.get("width", 512))
    H = int(inp.get("height", 512))
    seed = int(inp.get("seed", 42))

    controlnet_args = {
        "enabled": False,
        "controlnet_model": "control_sd15_animal_openpose_fp16",
        "controlnet_preprocessor": "openpose_full",
        "controlnet_pixel_perfect": True,
        "controlnet_strength":              _schedule(inp.get("cn_strength"), "0:(0.85)"),
        "image_strength":                   _schedule(inp.get("cn_image_strength"), "0:(0.75)"),
        "controlnet_start":                 _schedule(inp.get("cn_start"), "0:(0.0)"),
        "controlnet_end":                   _schedule(inp.get("cn_end"), "0:(1.0)"),
        "controlnet_annotator_resolution":  _schedule(inp.get("cn_annotator_res"), "0:(512)"),
        "guess_mode":                       _schedule(inp.get("cn_guess_mode"), "0:(0)"),
        "threshold_a":                      _schedule(inp.get("cn_threshold_a"), "0:(64)"),
        "threshold_b":                      _schedule(inp.get("cn_threshold_b"), "0:(64)"),
        "softness":                         _schedule(inp.get("cn_softness"), "0:(0.0)"),
        "processor_res":                    _schedule(inp.get("cn_processor_res"), "0:(512)"),
        "weight":                           _schedule(inp.get("cn_weight"), "0:(1.0)"),
        "prev_frame_controlnet": False,
    }

    job = {
        "prompt": {"0": prompt},
        "seed": seed,
        "max_frames": max_frames,
        "W": W,
        "H": H,
        "sampler": "Euler a",
        "steps": 25,
        "cfg_scale": 7,
        "animation_mode": "2D",
        "fps": int(inp.get("fps", 8)),
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
        "controlnet_args": controlnet_args,
    }

    cn = inp.get("controlnet") or {}
    if isinstance(cn, dict) and cn.get("enabled") is True:
        job["controlnet_args"]["enabled"] = True
        if cn.get("model"):
            job["controlnet_args"]["controlnet_model"] = cn["model"]
        if cn.get("preprocessor"):
            job["controlnet_args"]["controlnet_preprocessor"] = cn["preprocessor"]

    return job

# ---------- deforum API runner (with budget + heartbeats) ----------
@timed("generate")
def run_deforum(job: dict, job_id: str, start_ms: int, budget_ms: int):
    out_dir = job.get("outdir", "/workspace/outputs/deforum")
    os.makedirs(out_dir, exist_ok=True)

    api_url = f"{A1111}/sdapi/v1/deforum/run"
    payload = job

    # Heartbeat thread while request runs
    stop_evt = threading.Event()
    t = threading.Thread(target=_heartbeat_loop, args=(job_id, stop_evt), daemon=True)
    t.start()

    try:
        # requests timeout should not exceed remaining budget
        remain = max(5, int((budget_ms - (_now_ms() - start_ms)) / 1000))
        log_event("deforum.call", url=api_url, timeout_s=remain, max_frames=job.get("max_frames"))
        r = requests.post(api_url, json=payload, timeout=remain)
        log_event("deforum.resp", status=r.status_code)

        if r.status_code == 200:
            return {"retcode": 0, "tail": _tail(r.text, 2000), "outdir": out_dir}
        else:
            return {"retcode": r.status_code, "tail": _tail(r.text, 2000), "outdir": out_dir}
    except requests.exceptions.RequestException as e:
        log_exception("deforum.http_error", e)
        return {"retcode": 503, "tail": f"RequestException: {str(e)}", "outdir": out_dir}
    finally:
        stop_evt.set()
        t.join(timeout=1)

# ---------- handler ----------
@timed("handler")
def handler(event):
    # Identify job + budget
    job_id = (event or {}).get("id") or (event or {}).get("requestId") or uuid.uuid4().hex[:8]
    inp = (event or {}).get("input") or {}
    run_id = uuid.uuid4().hex[:8]
    start_ms = _now_ms()
    budget_ms = int(inp.get("timeout_ms") or os.getenv("RUNPOD_REQUEST_TIMEOUT_MS") or DEFAULT_TIMEOUT_MS)

    log_event("job.start", job_id=job_id, run_id=run_id, budget_ms=budget_ms, input_keys=list(inp.keys()))

    # Smoke-test path
    if inp.get("engine") == "smoke_test":
        log_event("job.smoke_ok", job_id=job_id)
        return {"status": "COMPLETED", "result": {"ok": True, "run_id": run_id, "message": "smoke_ok"}}

    # Build job
    job = build_deforum_job(inp)

    # Generate
    gen = run_deforum(job, job_id, start_ms, budget_ms)

    # Pick a fresh output video (after start_ms)
    picked = newest_video([
        "/workspace/outputs/deforum",
        "/workspace/stable-diffusion-webui/outputs/deforum",
        "/workspace/stable-diffusion-webui/outputs",
    ], since_ms=start_ms)

    # Upload (optional)
    uploaded = {"ok": False, "reason": "no_file_or_missing_env"}
    if picked and inp.get("upload") and remaining_ok(start_ms, budget_ms):
        log_event("upload.start", path=picked)
        try:
            uploaded = upload_to_vercel_blob(picked, run_id)
            log_event("upload.end", ok=bool(uploaded.get("ok")), status=uploaded.get("status"))
        except Exception as e:
            log_exception("upload.error", e)

    env_seen = {
        "blob_base_set": bool(os.getenv("VERCEL_BLOB_BASE") or os.getenv("BLOB_BASE")),
        "blob_token_set": bool(os.getenv("VERCEL_BLOB_RW_TOKEN") or os.getenv("VERCEL_BLOB_READ_WRITE_TOKEN") or os.getenv("VERCEL_BLOB_TOKEN")),
    }

    ok = (gen["retcode"] == 0) and bool(picked)
    out = {
        "ok": ok,
        "mode": "api-call",
        "run_id": run_id,
        "elapsed_ms": _now_ms() - start_ms,
        "local_outdir": gen.get("outdir"),
        "picked_file": picked,
        "uploaded": uploaded,
        "env_seen": env_seen,
    }
    if not ok:
        out["launch_tail"] = gen.get("tail")

    log_event("job.end", job_id=job_id, ok=ok, elapsed_ms=out["elapsed_ms"])
    return {"status": "COMPLETED" if ok else "FAILED", "result": out}

runpod.serverless.start({"handler": handler})