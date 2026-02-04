FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# upgrade pip
RUN pip3 install --upgrade pip

# install torch first
RUN pip3 install --no-cache-dir \
    torch==2.0.1 \
    torchvision==0.15.2 \
    torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118


# 🔥 THIS IS THE CRITICAL LINE 🔥
RUN pip3 install --no-cache-dir \
    mmcv==2.0.1 \
    -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html

# now install rest
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]

