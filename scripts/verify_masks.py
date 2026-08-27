#!/usr/bin/env python3
"""
F3.5 — Verificacion visual de alineacion imagen<->mascara y de mascaras degeneradas.

Comprueba que, para el indice i, la imagen del zip de dataset_tool y la mascara
del zip de preprocess_masks.py se corresponden (misma lesion), y saca a disco
solapamientos (imagen + mascara en rojo) de:
  - todas las mascaras con area de lesion < --deg-thresh (degeneradas)
  - una muestra aleatoria de --n-random imagenes (control de alineacion)

Uso (venv del TFG activo):
    python verify_masks.py \
        --image-zip ~/TFG/datos/ham7_train_256.zip \
        --mask-zip  ~/TFG/datos/ham7_train_masks_256.zip \
        --mask-index ~/TFG/datos/ham7_train_masks_256.zip.index.csv \
        --out ~/TFG/experimentos/f3_verificacion_mascaras

Abre luego las imagenes en --out para revisarlas a ojo.
"""
import argparse
import csv
import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-zip", required=True)
    ap.add_argument("--mask-zip", required=True)
    ap.add_argument("--mask-index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-random", type=int, default=12)
    ap.add_argument("--deg-thresh", type=float, default=0.01)
    args = ap.parse_args()

    out = Path(os.path.expanduser(args.out))
    out.mkdir(parents=True, exist_ok=True)

    iz = zipfile.ZipFile(os.path.expanduser(args.image_zip))
    mz = zipfile.ZipFile(os.path.expanduser(args.mask_zip))
    labels = json.loads(iz.read("dataset.json"))["labels"]  # orden por idx

    with open(os.path.expanduser(args.mask_index)) as f:
        idx_to_id = {int(r["idx"]): r["image_id"] for r in csv.DictReader(f)}

    n = len(labels)
    assert n == len(idx_to_id), f"desajuste: {n} labels vs {len(idx_to_id)} mascaras"

    def load_img(i):
        return np.array(Image.open(io.BytesIO(iz.read(labels[i][0]))).convert("RGB"))

    def load_mask(i):
        return np.array(Image.open(io.BytesIO(mz.read(f"img{i:08d}.png"))).convert("L"))

    def overlay(i, tag):
        img = load_img(i).astype(np.float32)
        m = load_mask(i) > 127
        red = img.copy()
        red[m] = 0.5 * red[m] + 0.5 * np.array([255, 0, 0])
        area = float(m.mean())
        cls = labels[i][1]
        name = f"{tag}_i{i:05d}_{idx_to_id[i]}_cls{cls}_area{area:.3f}.png"
        Image.fromarray(red.astype(np.uint8)).save(out / name)

    # degeneradas
    areas = np.array([load_mask(i).mean() / 255.0 for i in range(n)])
    deg = np.where(areas < args.deg_thresh)[0]
    print(f"Mascaras degeneradas (<{args.deg_thresh:.0%} area): {len(deg)}")
    for i in deg:
        print(f"  idx {i:5d}  {idx_to_id[i]}  area={areas[i]:.4f}")
        overlay(i, "DEG")

    # muestra aleatoria de control
    rng = np.random.default_rng(SEED)
    sample = rng.choice(n, size=min(args.n_random, n), replace=False)
    for i in sample:
        overlay(int(i), "OK")

    print(f"Solapamientos guardados en: {out}")
    print("Revisa que en las 'OK_*' la zona roja cae sobre la lesion (alineacion correcta).")


if __name__ == "__main__":
    main()
