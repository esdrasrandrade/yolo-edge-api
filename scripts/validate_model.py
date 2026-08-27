import argparse
import sys
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/yolov8n.pt")
    parser.add_argument("--threshold", type=float, default=0.50)
    args = parser.parse_args()
    
    model = YOLO(args.model)
    metrics = model.val(data="coco128.yaml", split="val", verbose=False)
    map50 = float(metrics.box.map50)
    print(f"mAP@0.5 = {map50:.4f} | Limiar: {args.threshold}")
    if map50 < args.threshold:
        print("[FALHA] mAP abaixo do limiar.")
        sys.exit(1)
    print("[OK] Quality gate aprovado.")

if __name__ == "__main__":
    main()
