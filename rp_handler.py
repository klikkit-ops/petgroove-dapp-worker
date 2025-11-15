# rp_handler.py — Deforum CLI runner (CN enabled via vid_path) + timing + Vercel Blob upload + CN debug + init_image & negatives
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
    attempts = []
    path = Path(file_path)
    if not path.exists():
        return {"ok": False, "reason": "file_missing"}

    api_base  = os.getenv("VERCEL_BLOB_BASE") or os.getenv("BLOB_BASE") or "https://api.blob.vercel-storage.com"
    proxy_url = os.getenv("VERCEL_BLOB_PROXY_URL")
    token     = (os.getenv("VERCEL_BLOB_RW_TOKEN")
                 or os.getenv("VERCEL_BLOB_READ_WRITE_TOKEN")
                 or os.getenv("VERCEL_BLOB_TOKEN"))
    public_base = os.getenv("VERCEL_BLOB_PUBLIC_BASE")

    key = f"runs/{run_id}/{path.name}"
    ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"

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

# ---------- CN: discover & resolve ----------
def _get_cn_lists():
    models, modules = set(), set()
    try:
        j = requests.get(f"{A1111}/controlnet/model_list", timeout=5).json()
        lst = j.get("model_list") if isinstance(j, dict) else j
        if isinstance(lst, list):
            models.update(lst)
    except Exception:
        pass
    try:
        j = requests.get(f"{A1111}/controlnet/module_list", timeout=5).json()
        lst = j.get("module_list") if isinstance(j, dict) else j
        if isinstance(lst, list):
            modules.update(lst)
    except Exception:
        pass
    return sorted(models), sorted(modules)

def _resolve_cn(model_in: "str | None", module_in: "str | None"):
    prefer_model = os.getenv("CN_MODEL_NAME") or model_in or "control_sd15_animal_openpose_fp16"
    prefer_module = os.getenv("CN_MODULE_NAME") or module_in or "animal_openpose"
    avail_models, avail_modules = _get_cn_lists()

    model = prefer_model
    if avail_models and model not in avail_models:
        for cand in (prefer_model, "control_sd15_animal_openpose_fp16", "control_v11p_sd15_openpose"):
            if cand in avail_models:
                model = cand
                break

    module = prefer_module
    if avail_modules and module not in avail_modules:
        for cand in (prefer_module, "animal_openpose", "openpose_full", "openpose"):
            if cand in avail_modules:
                module = cand
                break

    return model, module, avail_models, avail_modules

# ---------- Deforum job builder ----------
def build_deforum_job(inp: dict) -> dict:
    def S(x, default_str):
        if x is None or x == "":
            return default_str
        if isinstance(x, (int, float)):
            return f"0:({x})"
        s = str(x).strip()
        return s if (":" in s and "(" in s and ")" in s) else f"0:({s})"

    # basics
    prompt = inp.get("prompt", "photorealistic cat dancing on the spot, consistent anatomy, one head, four legs, one tail, natural balance")
    negative = inp.get("negative_prompt") or inp.get("negative") or "text, letters, logo, watermark, border, poster, extra limbs, deformed, low quality, blurry"
    W = int(inp.get("width", 512))
    H = int(inp.get("height", 512))
    fps  = int(inp.get("fps", 12))
    seconds = int(inp.get("seconds", 8))
    max_frames = int(inp.get("max_frames", seconds * fps))
    seed = int(inp.get("seed", 42))

    # init image (pet photo)
    init_image = inp.get("init_image") or inp.get("image_url") or ""
    use_init = bool(init_image)
    image_strength = float(inp.get("image_strength", 0.60))

    # ControlNet
    cn = inp.get("controlnet") or {}
    cn_enabled = bool(cn.get("enabled", False))
    cn_1_vid_path = cn.get("vid_path") or inp.get("pose_video_path") or ""
    resolved_model, resolved_module, _models, _modules = _resolve_cn(
        cn.get("model"), cn.get("module")
    )

    controlnet_args = {
        "cn_1_enabled": cn_enabled,
        "cn_1_model": resolved_model,
        "cn_1_module": resolved_module,
        "cn_1_weight": S(cn.get("weight"), "0:(0.95)"),
        "cn_1_weight_schedule_series": S(cn.get("weight_schedule_series"), "0:(1.0)"),
        "cn_1_guidance_start": S(cn.get("guidance_start"), "0:(0.0)"),
        "cn_1_guidance_end": S(cn.get("guidance_end"), "0:(1.0)"),
        "cn_1_processor_res": S(cn.get("processor_res"), "0:(640)"),
        "cn_1_annotator_resolution": S(cn.get("annotator_resolution"), "0:(640)"),
        "cn_1_threshold_a": S(cn.get("threshold_a"), "0:(64)"),
        "cn_1_threshold_b": S(cn.get("threshold_b"), "0:(64)"),
        "cn_1_guess_mode": S(cn.get("guess_mode"), "0:(0)"),
        "cn_1_resize_mode": cn.get("resize_mode", "Inner Fit (Scale to Fit)"),
        "cn_1_control_mode": cn.get("control_mode", "Balanced"),
        "cn_1_low_vram": bool(cn.get("low_vram", False)),
        "cn_1_pixel_perfect": bool(cn.get("pixel_perfect", True)),
        "cn_1_loopback_mode": False,
        "cn_1_overwrite_frames": True,
        "cn_1_invert_image": False,
        "cn_1_rgbbgr_mode": False,
        "cn_1_mask_vid_path": ""
    }
    if cn_1_vid_path:
        controlnet_args["cn_1_vid_path"] = cn_1_vid_path

    # core deforum
    job = {
        # Prompts (support both key names across Deforum versions)
        "prompts": {"0": prompt},
        "prompt": {"0": prompt},
        "negative_prompts": {"0": negative},
        "negative_prompt": {"0": negative},

        "seed": seed,
        "max_frames": max_frames,
        "W": W,
        "H": H,
        "fps": fps,
        "sampler": "Euler a",
        "steps": int(inp.get("steps", 28)),
        "cfg_scale": float(inp.get("cfg_scale", 6.5)),
        "animation_mode": "2D",

        # transforms
        "angle": "0:(0)",
        "zoom": "0:(1.0)",
        "translation_x": "0:(0)",
        "translation_y": "0:(0)",
        "translation_z": "0:(0)",

        # init (pet identity)
        "use_init": use_init,
        "init_image": init_image,
        "image_strength_schedule": S(inp.get("image_strength_schedule"), f"0:({image_strength})"),
        "strength_schedule": S(inp.get("strength_schedule"), f"0:({image_strength})"),

        # parseq off
        "use_parseq": False,

        # outputs
        "make_video": True,
        "save_video": True,
        "outdir": "/workspace/outputs/deforum",
        "outdir_video": "/workspace/outputs/deforum",

        # ControlNet
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
    handler_start = time.time()
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
    models_list, modules_list = _get_cn_lists()

    out = {
        "ok": ok,
        "mode": "cli-launch",
        "run_id": run_id,
        "local_outdir": res.get("outdir"),
        "picked_file": picked,
        "uploaded": uploaded,
        "env_seen": env_seen,
        "timings": timings,
        "total_elapsed_ms": int((time.time() - handler_start) * 1000),
        "debug_available_cn_models": models_list,
        "debug_available_cn_modules": modules_list,
    }
    if not ok:
        out["launch_tail"] = res.get("tail")
        cn = job.get("controlnet_args")
        if cn:
            out["debug_cn_keys"] = sorted(list(cn.keys()))
            out["debug_cn_values"] = {
                k: (cn[k] if isinstance(cn[k], (str, int, float, bool)) else str(cn[k]))
                for k in out["debug_cn_keys"]
            }
    elif inp.get("debug"):
        cn = job.get("controlnet_args")
        if cn:
            out["debug_cn_keys"] = sorted(list(cn.keys()))
            out["debug_cn_values"] = {
                k: (cn[k] if isinstance(cn[k], (str, int, float, bool)) else str(cn[k]))
                for k in out["debug_cn_keys"]
            }

    return {"status": "COMPLETED" if ok else "FAILED", "result": out}

runpod.serverless.start({"handler": handler})
