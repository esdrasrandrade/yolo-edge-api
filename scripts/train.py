import argparse
import torch
from ultralytics import YOLO

# Sobrescreve o torch.load globalmente no script para evitar erros de weights_only no PyTorch 2.6+
_original_torch_load = torch.load
def _safe_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _safe_torch_load

def main():
    parser = argparse.ArgumentParser(description="Treinamento YOLOv8 para Detecção de EPIs")
    parser.add_argument("--data", type=str, required=True, help="Caminho para o arquivo data.yaml")
    parser.add_argument("--epochs", type=int, default=10, help="Número de épocas")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamanho da imagem")
    args = parser.parse_args()

    print(f"Iniciando treinamento com o dataset: {args.data}")
    model = YOLO("yolov8n.pt")
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz)
    print("Treinamento finalizado com sucesso!")

if __name__ == "__main__":
    main()
