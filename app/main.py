import base64
import io
import time
from fastapi import FastAPI, HTTPException
from PIL import Image
import numpy as np
import httpx
from schemas import PredictRequest, PredictResponse, HealthResponse, MetricsResponse, Detection
from model import load_model, get_default_model_name

app = FastAPI(title="YOLO Inference API", version="1.0.0")
_metrics = {"total": 0, "success": 0, "total_ms": 0.0}

def _decode_image(image_base64: str) -> np.ndarray:
    raw = base64.b64decode(image_base64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.array(img)

def _load_image_from_request(request: PredictRequest) -> np.ndarray:
    if not request.image_base64 and not request.image_url:
        raise HTTPException(status_code=422, detail="Forneça image_base64 ou image_url.")
    if request.image_base64:
        return _decode_image(request.image_base64)
    else:
        resp = httpx.get(request.image_url, timeout=15.0, follow_redirects=True)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        return np.array(img)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    model_name = get_default_model_name()
    try:
        load_model(model_name)
        loaded = True
    except Exception:
        loaded = False
    return HealthResponse(status="ok", model_loaded=loaded, model_name=model_name)

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    _metrics["total"] += 1
    try:
        img = _load_image_from_request(request)
        model = load_model(request.model_name)

        t0 = time.perf_counter()
        results = model(img, conf=request.confidence, verbose=False)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        detections = []
        for r in results:
            for box in r.boxes:
                coords = box.xyxy[0].tolist()
                cls_id = int(box.cls[0].item())
                conf_val = float(box.conf[0].item())
                detections.append(Detection(
                    label=model.names[cls_id],
                    confidence=round(conf_val, 4),
                    bbox=[round(float(c), 2) for c in coords],
                ))
        h, w = img.shape[:2]
        _metrics["success"] += 1
        _metrics["total_ms"] += elapsed_ms
        return PredictResponse(
            detections=detections,
            inference_ms=round(elapsed_ms, 2),
            model_used=request.model_name,
            image_width=w,
            image_height=h,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

