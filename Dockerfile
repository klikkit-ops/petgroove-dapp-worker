FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# ---- Base env ----
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=on \
    PIP_PREFER_BINARY=1 \
    SHELL=/bin/bash
# WebUI flags (Deforum API comes from the extension; no --deforum-api flag needed)
ENV WEBUI_ARGS="--api --listen --xformers --enable-insecure-extension-access --port 3000"

SHELL ["/bin/bash","-lc"]
WORKDIR /workspace

# ---- System deps (robust apt) ----
# Adds ffmpeg + GUI libs + build tooling + Cairo stack needed by ControlNet deps (svglib/pycairo)
RUN set -eux; \
    rm -rf /var/lib/apt/lists/*; \
    for i in 1 2 3 4 5; do \
        apt-get update -o Acquire::Retries=3 -o Acquire::http::No-Cache=true && break || { \
          echo "apt-get update failed (attempt $i) – retrying..."; \
          sleep 5; \
          rm -rf /var/lib/apt/lists/*; \
        }; \
    done; \
    sed -i 's|http://archive.ubuntu.com/ubuntu/|http://us.archive.ubuntu.com/ubuntu/|g' /etc/apt/sources.list || true; \
    apt-get update -o Acquire::Retries=3 -o Acquire::http::No-Cache=true || true; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3-pip \
        git wget curl aria2 rsync jq moreutils \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg fonts-dejavu-core \
        build-essential pkg-config libcairo2-dev libglib2.0-dev libpango1.0-dev libffi-dev \
        python3-dev meson ninja-build; \
    rm -rf /var/lib/apt/lists/*

# Convenience symlink for scripts that call "python"
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# ---- Worker venv (for RunPod handler only) ----
RUN python3.10 -m venv /workspace/venv && \
    /workspace/venv/bin/pip install --upgrade pip setuptools wheel && \
    /workspace/venv/bin/pip install runpod==1.7.7 requests huggingface_hub imageio imageio-ffmpeg pillow

# ---- Clone A1111 ----
RUN git clone --depth 1 https://github.com/AUTOMATIC1111/stable-diffusion-webui /workspace/stable-diffusion-webui

# ---- Extensions: Deforum + ControlNet ----
RUN git clone --depth 1 https://github.com/deforum-art/sd-webui-deforum \
      /workspace/stable-diffusion-webui/extensions/sd-webui-deforum && \
    git clone --depth 1 https://github.com/Mikubill/sd-webui-controlnet \
      /workspace/stable-diffusion-webui/extensions/sd-webui-controlnet

# Prepare ControlNet models folder (harmless if empty)
RUN mkdir -p /workspace/stable-diffusion-webui/extensions/sd-webui-controlnet/models

# ---- Prewarm A1111 venv with Torch + base reqs + extension reqs + tricky wheels + CLIP ----
# This keeps cold starts fast and avoids runtime compiles (pycairo etc.)
RUN cd /workspace/stable-diffusion-webui && \
    python3.10 -m venv venv && \
    source venv/bin/activate && \
    pip install --upgrade pip setuptools wheel && \
    # CUDA 12.1 wheels for Torch:
    pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121 && \
    # A1111 base requirements
    (pip install -r requirements_versions.txt || pip install -r requirements.txt) && \
    # Extension requirements (ignore if files move)
    pip install -r extensions/sd-webui-deforum/requirements.txt || true && \
    pip install -r extensions/sd-webui-controlnet/requirements.txt || true && \
    # Tricky wheels that often build at runtime if missing
    pip install svglib==1.6.0 reportlab==4.4.4 tinycss2==1.4.0 cssselect2==0.8.0 \
                pycairo==1.29.0 rlpycairo==0.4.0 && \
    # Some revs of k-diffusion import `clip`; install OpenAI CLIP from GitHub
    pip install --no-cache-dir "git+https://github.com/openai/CLIP.git#egg=clip" && \
    # Helpers frequently used by Deforum/video export
    pip install imageio imageio-ffmpeg

# ---- Copy worker files ----
COPY start.sh /workspace/start.sh
COPY rp_handler.py /workspace/rp_handler.py
COPY schemas /workspace/schemas
RUN chmod +x /workspace/start.sh

EXPOSE 3000
ENTRYPOINT ["/bin/bash","/workspace/start.sh"]