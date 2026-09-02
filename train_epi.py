# %%
# Célula 1 --- Patch do torch.load e confirmação da GPU
import torch

_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
  if "weights_only" not in kwargs:
    kwargs["weights_only"] = False
  return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load
print("CUDA disponível:", torch.cuda.is_available())
if torch.cuda.is_available():
  print("GPU:", torch.cuda.get_device_name(0))

# %%
# Célula 2 --- Treinamento rápido (ajustado para economizar tempo)
from ultralytics import YOLO

if __name__ == "__main__":
  model = YOLO("yolov8n.pt")

  # Reduzimos para 5 épocas apenas para validar o fluxo de ponta a ponta
  results = model.train(
      data="dataset/exports/epi-v1/data.yaml",
      epochs=5,
      imgsz=320,  # Reduzimos a resolução de treino para acelerar ainda mais
      device=0,
      project="runs",
      name="epi-v1",
  )
  print("Pesos salvos em:", results.save_dir)
