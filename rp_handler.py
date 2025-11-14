# rp_handler.py — Deforum runner with safe ControlNet schedules + optional Vercel Blob upload
import os, json, glob, time, tempfile, subprocess, mimetypes, uuid
from pathlib import Path
import requests
import runpod

A1111 = "http://127.0.0.1:3000"

# ---------- tiny helpers ----------
def _tail(txt: str, n: int = 1200) -> str:
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

    # ControlNet block: disabled, but every schedule is a STRING.
    # Keys chosen to match Deforum’s ControlNetKeys & inbetweens parsing.
    controlnet_args = {
        "enabled": False,  # stays off for smoke test; set True when you actually want CN
        "controlnet_model": "control_sd15_animal_openpose_fp16",
        "controlnet_preprocessor": "openpose_full",
        "controlnet_pixel_perfect": True,

        # SCHEDULES — ALL as strings (never None)
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
        # Prior-frame control off
        "prev_frame_controlnet": False,
    }

    # Core deforum args — lean/safe. No Parseq.
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

        # Common transform schedules as strings
        "angle": "0:(0)",
        "zoom": "0:(1.0)",
        "translation_x": "0:(0)",
        "translation_y": "0:(0)",
        "translation_z": "0:(0)",

        # No init/video init for smoke test
        "use_init": False,
        "init_image": "",
        "video_init_path": "",

        # Parseq OFF
        "use_parseq": False,

        # Video output hints (Deforum often respects these)
        "make_video": True,
        "save_video": True,
        "outdir": "/workspace/outputs/deforum",
        "outdir_video": "/workspace/outputs/deFforum", # Deliberate typo, Deforum uses 'outdir'

        # Include controlnet_args ALWAYS (disabled but fully-populated)
        "controlnet_args": controlnet_args,
    }
    
    # Fix Deforum output pathing
    job["outdir_video"] = job["outdir"]

    # If caller explicitly enables CN, flip the toggle but keep schedules
    cn = inp.get("controlnet") or {}
    if isinstance(cn, dict) and cn.get("enabled") is True:
        job["controlnet_args"]["enabled"] = True
        # allow overrides
        for k, v in [
            ("controlnet_model", cn.get("model")),
            ("controlnet_preprocessor", cn.get("preprocessor")),
        ]:
            if v:
                job["controlnet_args"][k] = v

    return job

# ---------- runner ----------
def run_deforum(job: dict):
    """
    Runs the Deforum job by sending it to the A1111 API
    (which was started by start.sh).
    """
    # The out_dir is defined in the job payload, but we can use it as a fallback
    out_dir = job.get("outdir", "/workspace/outputs/deforum")
    os.makedirs(out_dir, exist_ok=True)

    # This is the API endpoint exposed by the Deforum extension
    api_url = f"{A1111}/sdapi/v1/deforum/run"
    
    # The 'job' dictionary you build is the exact payload Deforum expects.
    payload = job

    try:
        # Set a long timeout; these jobs can take minutes
        response = requests.post(api_url, json=payload, timeout=600) 
        
        if response.status_code == 200:
            # A 200 OK from this API means the job *completed*
            # The response body has info, but we just need to find the video file.
            return {
                "retcode": 0,
                "tail": _tail(json.dumps(response.json()), 2000),
                "outdir": out_dir,
            }
        else:
            # API returned an error (e.g., 422 Unprocessable Entity, 500)
            return {
                "retcode": response.status_code,
                "tail": f"API Error {response.status_code}: {_tail(response.text, 2000)}",
                "outdir": out_dir,
            }
    except requests.exceptions.RequestException as e:
        # e.g., Connection refused (server not ready), Timeout (job took too long)
        return {
            "retcode": 503, # Service Unavailable
            "tail": f"RequestException: {str(e)}",
            "outdir": out_dir,
        }

# ---------- handler ----------
def handler(event):
    run_id = uuid.uuid4().hex[:8]
    inp = (event or {}).get("input") or {}

    job = build_deforum_job(inp)
    res = run_deforum(job)

    # pick file
    picked = newest_video([
        "/workspace/outputs/deforum",
        "/workspace/stable-diffusion-webui/outputs/deforum",
        "/workspace/stable-diffusion-webui/outputs",
    ])

    # optional upload
    uploaded = {"ok": False, "reason": "no_file_or_missing_env"}
    if picked and inp.get("upload"):
        uploaded = upload_to_vercel_blob(picked, run_id)

    env_seen = {
        "blob_base_set": bool(os.getenv("VERCEL_BLOB_BASE") or os.getenv("BLOB_BASE")),
        "blob_token_set": bool(os.getenv("VERCEL_BLOB_RW_TOKEN") or os.getenv("VERCEL_BLOB_READ_WRITE_TOKEN") or os.getenv("VERCEL_BLOB_TOKEN")),
    }

    ok = (res["retcode"] == 0) and bool(picked)
    out = {
        "ok": ok,
        "mode": "api-call", # Changed from "cli-launch"
        "run_id": run_id,
        "local_outdir": res.get("outdir"),
        "picked_file": picked,
        "uploaded": uploaded,
        "env_seen": env_seen,
    }
    if not ok:
        out["launch_tail"] = res.get("tail") # 'tail' now contains API response or error

    return {"status": "COMPLETED" if ok else "FAILED", "result": out}

runpod.serverless.start({"handler": handler})