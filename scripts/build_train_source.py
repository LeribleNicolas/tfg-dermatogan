#!/usr/bin/env python3
"""
F3.2 (parte 1) — Prepara la carpeta-fuente de TRAIN para dataset_tool.py.

Crea un directorio con symlinks a las imagenes del split de train (no copia,
para no duplicar ~2 GB) y escribe el dataset.json con las etiquetas de clase
(7 clases, indices 0..6) que dataset_tool.py leera para el conditioning.

Uso (venv del TFG activo):
    python build_train_source.py \
        --train-csv ~/TFG/datos/splits/train.csv \
        --images    ~/TFG/datos/ham10000/HAM10000_images_part_1 \
                    ~/TFG/datos/ham10000/HAM10000_images_part_2 \
        --out       ~/TFG/datos/train_src

Despues, empaquetar a 256x256:
    python ~/TFG/stylegan2-ada-pytorch/dataset_tool.py \
        --source ~/TFG/datos/train_src \
        --dest   ~/TFG/datos/ham7_train_256.zip \
        --transform center-crop \
        --resolution 256x256
"""
import argparse
import json
import os
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--images", nargs="+", required=True,
                    help="Una o mas carpetas donde buscar los .jpg")
    ap.add_argument("--out", required=True, help="Carpeta-fuente a crear")
    ap.add_argument("--ext", default=".jpg")
    args = ap.parse_args()

    out = Path(os.path.expanduser(args.out))
    out.mkdir(parents=True, exist_ok=True)

    # indice image_id -> ruta absoluta real, recorriendo las carpetas dadas
    index = {}
    for folder in args.images:
        folder = Path(os.path.expanduser(folder))
        for p in folder.rglob(f"*{args.ext}"):
            index[p.stem] = p.resolve()
    print(f"Imagenes encontradas en las carpetas fuente: {len(index)}")

    df = pd.read_csv(os.path.expanduser(args.train_csv))
    labels = []
    faltan = []
    for image_id, label in zip(df["image_id"], df["label"]):
        src = index.get(image_id)
        if src is None:
            faltan.append(image_id)
            continue
        fname = f"{image_id}{args.ext}"
        link = out / fname
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(src)
        labels.append([fname, int(label)])

    assert not faltan, f"{len(faltan)} imagenes de train no encontradas, ej: {faltan[:5]}"
    assert len(labels) == len(df), f"{len(labels)} != {len(df)}"

    # dataset.json en la raiz de la carpeta-fuente (formato que espera dataset_tool.py)
    labels.sort(key=lambda x: x[0])  # orden estable por nombre de fichero
    with open(out / "dataset.json", "w") as f:
        json.dump({"labels": labels}, f)

    print(f"Symlinks creados: {len(labels)} -> {out}")
    print(f"dataset.json escrito con {len(labels)} etiquetas")
    # recuento por clase (control)
    vc = df["label"].value_counts().sort_index()
    print("Etiquetas por clase (indice: n):", dict(vc))


if __name__ == "__main__":
    main()
