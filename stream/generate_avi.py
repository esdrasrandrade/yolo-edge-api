import cv2
from ultralytics import YOLO
import time
import subprocess
import numpy as np
import sys
import torch

# Patch de segurança obrigatório para PyTorch 2.6+ com Ultralytics
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# Inicializa o modelo
model = YOLO("models/yolov8n.pt")

# Abre o rpicam-vid em subprocesso igual ao v3_optimized
cmd = [
    "rpicam-vid", "-t", "0", "-n", "--codec", "mjpeg",
    "--camera", "0", "--width", "640", "--height", "480",
    "--framerate", "30", "-o", "-"
]
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

# Configura o gravador de vídeo .avi
fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter('/tmp/stream_anotado.avi', fourcc, 20.0, (640, 480))

print("[INFO] Gravando /tmp/stream_anotado.avi por 10 segundos...")
start_time = time.time()
buf = b""

while time.time() - start_time < 10:
    chunk = proc.stdout.read(4096)
    if not chunk:
        break
    buf += chunk
    end = buf.rfind(b"\xff\xd9")
    if end == -1:
        continue
    start = buf.rfind(b"\xff\xd8", 0, end)
    if start == -1:
        continue
    jpg = buf[start:end + 2]
    buf = buf[end + 2:]

    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        continue

    # Inferência YOLO e OSD simples
    results = model(frame, conf=0.4, verbose=False)
    annotated = results[0].plot()

    # Escreve no arquivo .avi
    writer.write(annotated)
    print(".", end="", flush=True)

proc.terminate()
writer.release()
print("\n[OK] Vídeo gravado com sucesso em /tmp/stream_anotado.avi!")
