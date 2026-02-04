import cv2
import os
import uuid
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, Response
from mmdet.apis import init_detector, inference_detector
from fastapi.responses import JSONResponse

from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi import Request
from fastapi.templating import Jinja2Templates


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    print("\n🚀 Server running at:")
    print(" http://localhost:5000\n")

    yield

    # shutdown (optional)
    print("🛑 Server shutting down")



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(
    title="Unified Vision Inference API",
    lifespan=lifespan
)


# frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")
templates = Jinja2Templates(directory="frontend")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/results", response_class=HTMLResponse)
async def results(request: Request):
    return templates.TemplateResponse("results.html", {"request": request})




CONFIG = "configs/ppyoloe_localised/ppyoloeloc.py"
CHECKPOINT = "models/ppyoloe_localised/epoch_100.pth"
OUTPUT_DIR = "outputs"

model = init_detector(CONFIG, CHECKPOINT, device=DEVICE)


# ======================================================
# FRAME INFERENCE (UNCHANGED)
# ======================================================
def predict_frame(frame):
    with torch.no_grad():
        result = inference_detector(model, frame)
        inst = result.pred_instances
        boxes = inst.bboxes.cpu().numpy()
        scores = inst.scores.cpu().numpy()
        labels = inst.labels.cpu().numpy()
        classes = model.dataset_meta["classes"]

        for box, score, label in zip(boxes, scores, labels):
            if score < 0.3: continue
            x1, y1, x2, y2 = box.astype(int)
            name = classes[label]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} {score:.2f}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame

# ======================================================
# LIVE STREAMING LOGIC (The "Smart" part for RTSP)
# ======================================================
def generate_rtsp_frames(rtsp_url: str):
    cap = cv2.VideoCapture(rtsp_url)
    # Optional: Reduce buffer size for lower latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # If your model is heavy, you could skip frames here
        # e.g., if frame_count % 2 == 0:
        
        frame = predict_frame(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    cap.release()

@app.get("/predict/rtsp")
async def predict_rtsp(url: str):
    """
    Returns a live MJPEG stream that an <img> tag can consume directly.
    """
    return StreamingResponse(
        generate_rtsp_frames(url),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )




