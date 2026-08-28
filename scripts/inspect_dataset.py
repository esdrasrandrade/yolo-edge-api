#!/usr/bin/env python3
"""
scripts/inspect_dataset.py
Valida a integridade e balanceamento de um dataset no formato YOLOv8.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import sys
import yaml

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="Caminho para data.yaml")
    p.add_argument("--min-per-class", type=int, default=30, help="Mínimo de instâncias por classe no treino")
    return p.parse_args()

def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def count_labels(labels_dir: Path, num_classes: int) -> tuple[dict, int]:
    counts = defaultdict(int)
    missing = 0
    for img_path in labels_dir.parent.glob("images/*"):
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            missing += 1
            continue
        with open(label_path) as f:
            for line in f:
                cls = int(line.split()[0])
                counts[cls] += 1
    return dict(counts), missing

def main():
    args = parse_args()
    cfg = load_yaml(args.dataset)
    base = Path(args.dataset).parent
    names = cfg.get("names", [])
    nc = cfg.get("nc", len(names))
    print(f"\n{'='*55}\n Inspeção do Dataset: {base.name}\n{'='*55}")
    print(f" Classes ({nc}): {names}")
    issues = 0
    for split in ["train", "valid", "test"]:
        labels_dir = base / split / "labels"
        if not labels_dir.exists():
            continue
        counts, missing = count_labels(labels_dir, nc)
        total = sum(counts.values())
        imgs = len(list((base / split / "images").glob("*")))
        print(f"\n  [{split.upper()}]  {imgs} imagens  |  {total} anotações  |  {missing} sem label")
        for cls_id, cls_name in enumerate(names):
            n = counts.get(cls_id, 0)
            warn = "  ← ABAIXO DO MÍNIMO" if split == "train" and n < args.min_per_class else ""
            print(f"    {cls_name:15s} {n:5d} {warn}")
            if warn:
                issues += 1
    if issues:
        sys.exit(1)
    else:
        print("\n Dataset aprovado para treinamento.")

if __name__ == "__main__":
    main()
