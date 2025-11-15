# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# ---------- Build args for pinning (optional) ----------
# Fill these with concrete commit SHAs to lock versions.
ARG A1111_COMMIT=
ARG DEFORUM_COMMIT=
ARG CONTROLNET_COMMIT=
# Animal OpenPose model URL (override if you mirror)
ARG CN_ANIMAL_OPENPOSE_URL="https://huggingface.co/f5aiteam/Controlnet/resolve/main/control_sd15_animal_openpose_fp16.pth?download=true"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=on \
    SHELL=/bin/bash \
    WEBUI_ARGS="--api --listen --xformers --enable-insecure-extension-access --port 3000"

SHELL ["/bin/bash","-lc"]
WORKDIR /workspace

# ---------- Base system ----------
RUN set -eux; \
    rm -rf /var/lib/apt/lists/*; \
    for i in 1 2 3 4 5; do \
      apt-get update -o Acquire::Retries=3 -o Acquire::http::No-Cache=true && break || { \
        echo "apt-get update failed (attempt $i) – retrying..."; \
        sleep 5; \
        rm -rf /var/lib/apt/lists/*; \
      }; \
    done; \
    apt-get install -y --no-install-recommends \
      python3.10 python3.10-venv python3-pip \
      git wget curl aria2 rsync jq moreutils \
      libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg \
      fonts-dejavu-core ca-certificates; \
    rm -rf /var/lib/apt/lists/*

# Convenience symlink for scripts that call "python"
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# ---------- Clone A1111 ----------
RUN git clone --depth 1 https://github.com/AUTOMATIC1111/stable-diffusion-webui /workspace/stable-diffusion-webui && \
    if [ -n "$A1111_COMMIT" ]; then \
      cd /workspace/stable-diffusion-webui && git fetch --depth 1 origin "$A1111_COMMIT" && git checkout "$A1111_COMMIT"; \
    fi

# ---------- Clone extensions (Deforum + ControlNet) ----------
RUN git clone --depth 1 https://github.com/deforum-art/sd-webui-deforum \
      /workspace/stable-diffusion-webui/extensions/sd-webui-deforum && \
    if [ -n "$DEFORUM_COMMIT" ]; then \
      cd /workspace/stable-diffusion-webui/extensions/sd-webui-deforum && git fetch --depth 1 origin "$DEFORUM_COMMIT" && git checkout "$DEFORUM_COMMIT"; \
    fi && \
    git clone --depth 1 https://github.com/Mikubill/sd-webui-controlnet \
      /workspace/stable-diffusion-webui/extensions/sd-webui-controlnet && \
    if [ -n "$CONTROLNET_COMMIT" ]; then \
      cd /workspace/stable-diffusion-webui/extensions/sd-webui-controlnet && git fetch --depth 1 origin "$CONTROLNET_COMMIT" && git checkout "$CONTROLNET_COMMIT"; \
    fi && \
    mkdir -p /workspace/stable-diffusion-webui/extensions/sd-webui-controlnet/models

# ---------- ControlNet Animal OpenPose (fp16) baked into image ----------
# 🔴 Adds the required ControlNet model so CN "animal openpose" works out of the box.
RUN set -eux; \
    CN_DIR="/workspace/stable-diffusion-webui/extensions/sd-webui-controlnet/models"; \
    CN_FILE="$CN_DIR/control_sd15_animal_openpose_fp16.pth"; \
    if [ ! -f "$CN_FILE" ]; then \
      echo "Downloading Animal OpenPose model to $CN_FILE"; \
      curl -L --fail -o "$CN_FILE" "$CN_ANIMAL_OPENPOSE_URL"; \
    else \
      echo "Model already present: $CN_FILE"; \
    fi; \
    ls -lh "$CN_FILE"

# ---------- Worker venv (rp_handler uses this) ----------
RUN python3.10 -m venv /workspace/venv && \
    /workspace/venv/bin/pip install --upgrade pip setuptools wheel && \
    /workspace/venv/bin/pip install runpod==1.7.7 requests huggingface_hub

# ---------- A1111 venv + deps preinstall ----------
RUN cd /workspace/stable-diffusion-webui && \
    python3.10 -m venv venv && \
    source venv/bin/activate && \
    pip install --upgrade pip setuptools wheel && \
    # CUDA 12.1 wheels for PyTorch:
    pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121 && \
    # A1111 base requirements (versions file first if present):
    (pip install -r requirements_versions.txt || pip install -r requirements.txt) && \
    # Deforum + ControlNet requirements (ignore if file paths change):
    (pip install -r extensions/sd-webui-deforum/requirements.txt || true) && \
    (pip install -r extensions/sd-webui-controlnet/requirements.txt || true) && \
    # Ensure CLIP is present to avoid ModuleNotFoundError:
    pip install "git+https://github.com/openai/CLIP.git#egg=clip" || pip install clip || true

# ---------- Copy worker files ----------
COPY start.sh /workspace/start.sh
COPY rp_handler.py /workspace/rp_handler.py
COPY schemas /workspace/schemas

RUN chmod +x /workspace/start.sh

# ---------- Diagnostics on build (optional) ----------
RUN echo "A1111_COMMIT=${A1111_COMMIT}" && \
    echo "DEFORUM_COMMIT=${DEFORUM_COMMIT}" && \
    echo "CONTROLNET_COMMIT=${CONTROLNET_COMMIT}"

EXPOSE 3000
ENTRYPOINT ["/bin/bash","/workspace/start.sh"]
