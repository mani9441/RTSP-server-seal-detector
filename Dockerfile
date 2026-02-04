FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# ---------- SYSTEM DEPENDENCIES (CRITICAL FOR VIDEO) ----------
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    && rm -rf /var/lib/apt/lists/*

# ---------- PYTHON ENV ----------
RUN pip3 install --upgrade pip

# Install PyTorch (keep yours)
RUN pip3 install --no-cache-dir \
    torch==2.0.1 \
    torchvision==0.15.2 \
    torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

# Install MMCV (your critical line — keep it)
RUN pip3 install --no-cache-dir \
    mmcv==2.0.1 \
    -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html

# Install rest of your requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy your app code
COPY . .

# Expose FastAPI port
EXPOSE 5000

# Run FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
