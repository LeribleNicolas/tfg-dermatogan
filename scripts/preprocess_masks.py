#!/usr/bin/env python3
"""
F3.3 — Preprocesa las mascaras de segmentacion de TRAIN, alineadas 1:1 con el
zip de imagenes generado por dataset_tool.py.

Aplica EXACTAMENTE el mismo recorte geometrico que dataset_tool.py con
--transform center-crop (recorte central min(H,W) x min(H,W)), pero reescala
con interpolacion NEAREST y binariza {0,255}, para no introducir valores
intermedios en una mascara binaria.

Alineacion: se enumera la MISMA carpeta-fuente de train (train_src) con el
mismo orden (sorted por ruta) que usa dataset_tool.py, y cada mascara se guarda
como img{idx:08d}.png con el MISMO indice que la imagen correspondiente en el
zip. Asi, en F4, imagen[i] <-> mascara[i] sin ambiguedad.

Uso (venv del TFG activo):
    python preprocess_masks.py \
        --train-src ~/TFG/datos/train_src \
        --masks     ~/TFG/datos/ham10000-masks/HAM10000_segmentations_lesion_tschandl \
        --dest      ~/TFG/datos/ham7_train_masks_256.zip \
        --mask-suffix _segmentation.png

Salida:
    <dest>.zip con img{idx:08d}.png (modo L, 0/255)
    <dest>.index.csv con idx,image_id (trazabilidad / verificacion)
"""
import argparse
import csv
import io
import os
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


def center_crop_square(arr):
    """Recorte central min(H,W) x min(H,W). Identico a dataset_tool.py."""
    h, w = arr.shape[:2]
    crop = min(h, w)
    top = (h - crop) // 2
    left = (w - crop) // 2
    return arr[top:top + crop, left:left + crop]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-src", required=True,
                    help="Carpeta con los symlinks de train (misma que dataset_tool.py)")
    ap.add_argument("--masks", required=True, help="Carpeta con las mascaras .png")
    ap.add_argument("--dest", required=True, help="Zip de salida")
    ap.add_argument("--mask-suffix", default="_segmentation.png")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--img-ext", default=".jpg")
    args = ap.parse_args()

    src = Path(os.path.expanduser(args.train_src))
    masks_dir = Path(os.path.expanduser(args.masks))
    dest = Path(os.path.expanduser(args.dest))

    # MISMO orden que dataset_tool.py: sorted por ruta completa
    imgs = sorted(str(p) for p in src.rglob(f"*{args.img_ext}"))
    print(f"Imagenes de train enumeradas: {len(imgs)}")

    faltan, areas = [], []
    index_rows = []
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_STORED) as zf:
        for idx, img_path in enumerate(imgs):
            image_id = Path(img_path).stem
            mask_path = masks_dir / f"{image_id}{args.mask_suffix}"
            if not mask_path.exists():
                faltan.append(image_id)
                continue

            m = np.array(Image.open(mask_path).convert("L"))
            m = center_crop_square(m)
            m = Image.fromarray(m).resize((args.size, args.size), Image.NEAREST)
            m = (np.array(m) > 127).astype(np.uint8) * 255

            areas.append(float((m > 0).mean()))
            buf = io.BytesIO()
            Image.fromarray(m, mode="L").save(buf, format="PNG")
            zf.writestr(f"img{idx:08d}.png", buf.getvalue())
            index_rows.append((idx, image_id))

    assert not faltan, f"{len(faltan)} mascaras no encontradas, ej: {faltan[:5]}"

    with open(str(dest) + ".index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "image_id"])
        w.writerows(index_rows)

    areas = np.array(areas)
    print(f"Mascaras escritas: {len(index_rows)} -> {dest}")
    print(f"Fraccion de lesion (area>0): media={areas.mean():.3f} "
          f"min={areas.min():.3f} max={areas.max():.3f}")
    n_deg = int((areas < 0.01).sum())
    if n_deg:
        print(f"AVISO: {n_deg} mascaras con <1% de area de lesion (revisar).")


if __name__ == "__main__":
    main()
