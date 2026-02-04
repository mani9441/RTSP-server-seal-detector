import cv2
import torch
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager

from mmdet.apis import init_detector, inference_detector

# ------------------------------------------------------
# FASTAPI LIFESPAN
# ------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n🚀 Server running at: http://localhost:5000\n")
    yield
    print("🛑 Server shutting down")

app = FastAPI(title="Camera-Proof Vision API", lifespan=lifespan)

# ------------------------------------------------------
# FRONTEND
# ------------------------------------------------------
app.mount("/static", StaticFiles(directory="frontend"), name="static")
templates = Jinja2Templates(directory="frontend")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ------------------------------------------------------
# LOAD MODEL (ONCE)
# ------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG = "configs/ppyoloe_localised/ppyoloeloc.py"
CHECKPOINT = "models/ppyoloe_localised/epoch_100.pth"

model = init_detector(CONFIG, CHECKPOINT, device=DEVICE)

# ------------------------------------------------------
# FRAME INFERENCE
# ------------------------------------------------------
def predict_frame(frame):
    with torch.no_grad():
        result = inference_detector(model, frame)
        inst = result.pred_instances
        boxes = inst.bboxes.cpu().numpy()
        scores = inst.scores.cpu().numpy()
        labels = inst.labels.cpu().numpy()
        classes = model.dataset_meta["classes"]

        for box, score, label in zip(boxes, scores, labels):
            if score < 0.3:
                continue
            x1, y1, x2, y2 = box.astype(int)
            name = classes[label]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{name} {score:.2f}",
                (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
    return frame

# ------------------------------------------------------
# UNIVERSAL CAMERA HANDLER (CAMERA-PROOF)
# ------------------------------------------------------
def open_video_source(source: str):
    """
    Accepts:
    - '0', '1', '2'  -> local USB webcams
    - rtsp://...     -> IP/port cameras
    - http://...     -> MJPEG cameras
    - video.mp4      -> video files
    """

    # Case 1 — Local USB camera (e.g. "0")
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        return cap

    # Case 2 — RTSP (IP/Port cameras)
    if source.startswith("rtsp"):
        safe_url = (
            "rtsp_transport=tcp buffer_size=2048000 " + source
        )
        cap = cv2.VideoCapture(safe_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        return cap

    # Case 3 — HTTP MJPEG or video file
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    return cap

# ------------------------------------------------------
# STREAMING GENERATOR (WORKS FOR ANY CAMERA)
# ------------------------------------------------------
def generate_frames(source: str):
    cap = open_video_source(source)

    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1

        # Skip every other frame if model is heavy
        if frame_count % 2 != 0:
            continue

        frame = predict_frame(frame)

        ret, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )

    cap.release()

# ------------------------------------------------------
# SINGLE UNIVERSAL ENDPOINT
# ------------------------------------------------------
@app.get("/predict/stream")
async def predict_stream(source: str):
    """
    Examples:
    /predict/stream?source=0
    /predict/stream?source=rtsp://192.168.1.50:554/stream
    /predict/stream?source=http://192.168.1.50/video.mjpg
    /predict/stream?source=video.mp4
    """
    return StreamingResponse(
        generate_frames(source),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
