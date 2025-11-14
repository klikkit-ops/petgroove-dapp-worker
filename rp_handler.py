# rp_handler.py — Deforum runner (API-first) with timing + structured logs + route discovery + optional Vercel Blob upload
import os, json, glob, time, tempfile, requests, mimetypes, uuid
from pathlib import Path
import runpod

A1111 = os.getenv("A1111_BASE_URL", "http://127.0.0.1:3000")
JOB_TIMEOUT_SECS = int(os.getenv("DEFORUM_JOB_TIMEOUT", "900"))  # up to 15m

# -------------------- logging / timing --------------------
def log(run_id: str, event: str, **fields):
    """Single-line JSON log for grep/observability."""
    rec = {"run_id": run_id, "event": event, **fields}
    try:
        print(json.dumps(rec, ensure_ascii=False))
    except Exception:
        # never crash logging
        print(str(rec))

def time_block(run_id: str, label: str):
    class _Timer:
        def __enter__(self):
            self.t0 = time.perf_counter()
            return self
        def __exit__(self, exc_type, exc, tb):
            self.ms = int((time.perf_counter() - self.t0) * 1000)
            log(run_id, "timing", label=label, elapsed_ms=self.ms)
    return _Timer()

def _tail(txt: str, n: int = 1200) -> str:
    return (txt or "")[-n:]

# -------------------- schedules --------------------
def _schedule(val, default_str: str) -> str:
    if val is None or val == "":
        return default_str
    if isinstance(val, (int, float)):
        return f"0:({val})"
    s = str(val).strip()
    if ":" in s and "(" in s and ")" in s:
        return s
    return f"0:({s})"

# -------------------- outputs --------------------
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

# -------------------- job builder --------------------
def build_deforum_job(inp: dict) -> dict:
    prompt = inp.get("prompt", "a photorealistic orange tabby cat doing a simple dance, studio lighting")
    max_frames = int(inp.get("max_frames", 12))
    W = int(inp.get("width", 512))
    H = int(inp.get("height", 512))
    seed = int(inp.get("seed", 42))

    controlnet_args = {
        "enabled": bool((inp.get("controlnet") or {}).get("enabled", False)),
        "controlnet_model": (inp.get("controlnet") or {}).get("model", "control_sd15_animal_openpose_fp16"),
        "controlnet_preprocessor": (inp.get("controlnet") or {}).get("preprocessor", "openpose_full"),
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

# -------------------- API discovery --------------------
def discover_deforum_run_url(run_id: str):
    """Try to find the correct /deforum/... run path from /openapi.json."""
    try:
        with time_block(run_id, "fetch_openapi_ms"):
            r = requests.get(f"{A1111}/openapi.json", timeout=10)
        if r.ok:
            paths = (r.json() or {}).get("paths", {}) or {}
            candidates = [p for p in paths.keys() if "deforum" in p.lower() and p.lower().endswith("/run")]
            if candidates:
                url = f"{A1111}{candidates[0]}"
                log(run_id, "discover_ok", discovered=url)
                return url
            log(run_id, "discover_none", note="no deforum /run in openapi.json", sample=list(paths.keys())[:10])
        else:
            log(run_id, "discover_http", status=r.status_code, body=_tail(r.text, 200))
    except Exception as e:
        log(run_id, "discover_error", error=str(e))
    return None

def candidate_run_urls():
    # Known variants across Deforum versions/forks
    return [
        "/deforum/run",
        "/sdapi/v1/deforum/run",
        "/deforum_api/run",
        "/sdapi/v1/deforum_api/run",
    ]

# -------------------- runner (API) --------------------
def run_deforum_via_api(run_id: str, job: dict):
    tried = []
    discovered = discover_deforum_run_url(run_id)
    urls = [discovered] if discovered else []
    urls += [f"{A1111}{p}" for p in candidate_run_urls()]

    for url in urls:
        if not url:
            continue
        tried.append(url)
        try:
            with time_block(run_id, f"post_deforum_ms:{url}"):
                r = requests.post(url, json=job, timeout=JOB_TIMEOUT_SECS)
            if r.status_code == 404:
                log(run_id, "deforum_404", url=url)
                continue
            if r.ok:
                # API implementations differ; we don’t rely on response body contents.
                return {"retcode": 0, "tail": _tail(r.text, 2000), "outdir": job.get("outdir")}
            else:
                log(run_id, "deforum_http_error", url=url, status=r.status_code, body=_tail(r.text, 400))
                # for non-404, still continue to try other candidates
        except requests.exceptions.RequestException as e:
            log(run_id, "deforum_req_exc", url=url, error=str(e))
            continue

    return {
        "retcode": 404,
        "tail": f"No working Deforum API route. Tried: {tried}",
        "outdir": job.get("outdir")
    }

# -------------------- handler --------------------
def handler(event):
    run_id = uuid.uuid4().hex[:8]
    inp = (event or {}).get("input") or {}
    log(run_id, "handler_start", input_keys=list(inp.keys()))

    with time_block(run_id, "build_job_ms"):
        job = build_deforum_job(inp)

    with time_block(run_id, "api_total_ms"):
        res = run_deforum_via_api(run_id, job)

    picked = newest_video([
        "/workspace/outputs/deforum",
        "/workspace/stable-diffusion-webui/outputs/deforum",
        "/workspace/stable-diffusion-webui/outputs",
    ])

    uploaded = {"ok": False, "reason": "no_file_or_missing_env"}
    if picked and inp.get("upload"):
        with time_block(run_id, "upload_ms"):
            uploaded = upload_to_vercel_blob(picked, run_id)

    env_seen = {
        "blob_base_set": bool(os.getenv("VERCEL_BLOB_BASE") or os.getenv("BLOB_BASE")),
        "blob_token_set": bool(os.getenv("VERCEL_BLOB_RW_TOKEN") or os.getenv("VERCEL_BLOB_READ_WRITE_TOKEN") or os.getenv("VERCEL_BLOB_TOKEN")),
    }

    ok = (res["retcode"] == 0) and bool(picked)
    out = {
        "ok": ok,
        "mode": "api-call",
        "run_id": run_id,
        "local_outdir": res.get("outdir"),
        "picked_file": picked,
        "uploaded": uploaded,
        "env_seen": env_seen,
        "elapsed_ms": None,  # see timing logs for phase breakdowns
    }
    if not ok:
        out["launch_tail"] = res.get("tail")

    log(run_id, "handler_end", ok=ok, picked=bool(picked))
    return {"status": "COMPLETED" if ok else "FAILED", "result": out}

runpod.serverless.start({"handler": handler})