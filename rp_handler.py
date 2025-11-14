# rp_handler.py — Deforum CLI runner with timings + optional Vercel Blob upload
import os, json, glob, time, tempfile, subprocess, mimetypes, uuid
from pathlib import Path
import requests
import runpod

# ---------- tiny helpers ----------
def _tail(txt: str, n: int = 1600) -> str:
    return (txt or "")[-n:]

def _schedule(val, default_str: str) -> str:
    """
    Deforum expects schedule STRINGS like '0:(0.75)'. Never None.
    - number -> "0:(number)"
    - blank/None -> default_str
    - string without schedule syntax -> wrap as "0:(string)"
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
        body = _tail(r.text, 400)
        return {"ok": False, "reason": f"upload_http_{r.status_code}", "body": body}
    try:
        body = r.json()
    except Exception:
        body = {"url": r.text}
    return {"ok": True, "url": body.get("url") or url, "key": key}

# ---------- timing decorator ----------
def make_timer():
    timings = []
    def timed(step_name):
        def deco(fn):
            def inner(*a, **k):
                t0 = time.time()
                try:
                    return fn(*a, **k)
                finally:
                    timings.append({"step": step_name, "ms": int((time.time() - t0) * 1000)})
            return inner
        return deco
    return timings, timed

# ---------- job builder ----------
def build_deforum_job(inp: dict) -> dict:
    """
    Minimal Deforum config that ALWAYS includes a fully-populated controlnet_args
    with string schedules (and enabled=False) to prevent 'NoneType.split' crashes.
    """
    prompt = inp.get("prompt", "a photorealistic orange tabby cat doing a simple dance, studio lighting")
    max_frames = int(inp.get("max_frames", 12))
    W = int(inp.get("width", 512))
    H = int(inp.get("height", 512))
    seed = int(inp.get("seed", 42))

    controlnet_args = {
        "enabled": bool(inp.get("controlnet_enabled", False)),
        "controlnet_model": inp.get("controlnet_model", "control_sd15_animal_openpose_fp16"),
        "controlnet_preprocessor": inp.get("controlnet_preprocessor", "openpose_full"),
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
        "fps": 8,
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

    return job

# ---------- CLI runner ----------
def run_deforum_cli(job: dict, timeout_sec: int = None):
    """
    Launches Deforum via launch.py --deforum-run-now <job.json> and returns {retcode, tail, outdir}
    Uses port 3001 to avoid clashing with any background A1111 that start.sh may have on 3000.
    """
    timeout_sec = timeout_sec or int(os.getenv("DEFORUM_CLI_TIMEOUT", "900"))
    out_dir = job.get("outdir", "/workspace/outputs/deforum")
    os.makedirs(out_dir, exist_ok=True)

    # Write job to temp file
    tmp = tempfile.NamedTemporaryFile("w", delete=False, suffix=".json")
    json.dump(job, tmp)
    tmp.close()

    py = "/workspace/stable-diffusion-webui/venv/bin/python"
    launch_py = "/workspace/stable-diffusion-webui/launch.py"

    cmd = [
        py, launch_py,
        "--nowebui",
        "--xformers",
        "--api",
        "--enable-insecure-extension-access",
        "--port", os.getenv("A1111_PORT", "3001"),
        "--deforum-run-now", tmp.name,
        "--deforum-terminate-after-run-now",
    ]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_sec,
            cwd="/workspace/stable-diffusion-webui",
            env={**os.environ},
        )
        return {
            "retcode": proc.returncode,
            "tail": _tail(proc.stdout, 2000),
            "outdir": out_dir,
        }
    except subprocess.TimeoutExpired as e:
        return {
            "retcode": 504,
            "tail": _tail((e.stdout or "") + "\n[TIMEOUT]", 2000),
            "outdir": out_dir,
        }
    except Exception as e:
        return {
            "retcode": 500,
            "tail": f"[EXC] {type(e).__name__}: {e}",
            "outdir": out_dir,
        }
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

# ---------- handler ----------
def handler(event):
    timings, timed = make_timer()
    run_id = uuid.uuid4().hex[:8]
    inp = (event or {}).get("input") or {}

    @timed("build_job")
    def _build():
        return build_deforum_job(inp)

    @timed("run_cli")
    def _run(job):
        return run_deforum_cli(job)

    @timed("pick_video")
    def _pick():
        return newest_video([
            "/workspace/outputs/deforum",
            "/workspace/stable-diffusion-webui/outputs/deforum",
            "/workspace/stable-diffusion-webui/outputs",
        ])

    @timed("maybe_upload")
    def _upload(picked):
        if picked and inp.get("upload"):
            return upload_to_vercel_blob(picked, run_id)
        return {"ok": False, "reason": "skipped"}

    job = _build()
    res = _run(job)
    picked = _pick()
    uploaded = _upload(picked)

    env_seen = {
        "blob_base_set": bool(os.getenv("VERCEL_BLOB_BASE") or os.getenv("BLOB_BASE")),
        "blob_token_set": bool(os.getenv("VERCEL_BLOB_RW_TOKEN") or os.getenv("VERCEL_BLOB_READ_WRITE_TOKEN") or os.getenv("VERCEL_BLOB_TOKEN")),
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