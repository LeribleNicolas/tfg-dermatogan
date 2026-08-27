#!/usr/bin/env python3
"""
F3.1 — Split de HAM10000 agrupado por lesion_id y estratificado por clase (dx).

Genera particiones train/val/test (80/10/10 por defecto) garantizando que
todas las imagenes de una misma lesion caen en la MISMA particion (sin fuga
entre train y test) y manteniendo aproximadamente las proporciones por clase.

Uso (en WSL2, con el venv del TFG activo):
    python split_ham10000.py \
        --metadata ~/TFG/datos/HAM10000_metadata.csv \
        --out      ~/TFG/datos/splits

Requisitos: pandas, numpy, scikit-learn>=0.24 (StratifiedGroupKFold).

Salida en --out:
    train.csv, val.csv, test.csv   (columnas: image_id, lesion_id, dx, label)
    class_index.json               (mapeo clase -> indice, FIJO para el conditioning)

IMPORTANTE: el orden de clases define los indices 0..6 del modelo condicional.
Es FIJO y debe ser el mismo en todo el proyecto (dataset.json, generacion, metricas).
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# Orden ALFABETICO fijo -> indices 0..6. No cambiar una vez entrenado un modelo.
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
SEED = 42  # reproducibilidad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True, help="Ruta a HAM10000_metadata.csv")
    ap.add_argument("--out", required=True, help="Directorio de salida para los splits")
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.10)
    args = ap.parse_args()

    meta_path = os.path.expanduser(args.metadata)
    out_dir = Path(os.path.expanduser(args.out))
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(meta_path)
    needed = {"lesion_id", "image_id", "dx"}
    assert needed.issubset(df.columns), f"Faltan columnas {needed - set(df.columns)}"

    df = df.drop_duplicates(subset="image_id").reset_index(drop=True)
    assert len(df) == 10015, f"Esperaba 10015 imagenes unicas, hay {len(df)}"

    df["label"] = df["dx"].map(CLASS_TO_IDX)
    assert df["label"].notna().all(), "Hay valores de dx fuera de CLASSES"
    df["label"] = df["label"].astype(int)

    # n_splits folds ~iguales, estratificados por clase y agrupados por lesion.
    # fold 0 -> test, fold 1 -> val, resto -> train.  (test-frac=val-frac=0.10 -> 80/10/10)
    n_splits = int(round(1.0 / args.test_frac))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    folds = [test_idx for _, test_idx in sgkf.split(df, df["label"], groups=df["lesion_id"])]

    test_idx = folds[0]
    val_idx = folds[1]
    used = set(test_idx) | set(val_idx)
    train_idx = np.array([i for i in range(len(df)) if i not in used])

    splits = {"train": train_idx, "val": val_idx, "test": test_idx}

    # --- Integridad: ninguna lesion aparece en dos particiones ---
    lesion_sets = {k: set(df.iloc[v]["lesion_id"]) for k, v in splits.items()}
    parts = list(splits)
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            a, b = parts[i], parts[j]
            overlap = lesion_sets[a] & lesion_sets[b]
            assert not overlap, f"FUGA de {len(overlap)} lesiones entre {a} y {b}"

    # cobertura total
    assert sum(len(v) for v in splits.values()) == len(df)

    # --- Resumen por clase ---
    print(f"{'clase':6} {'train':>7} {'val':>6} {'test':>6}")
    for c, idx in CLASS_TO_IDX.items():
        counts = [int((df.iloc[splits[s]]["label"] == idx).sum()) for s in parts]
        print(f"{c:6} {counts[0]:7d} {counts[1]:6d} {counts[2]:6d}")
    print(f"{'TOTAL':6} {len(train_idx):7d} {len(val_idx):6d} {len(test_idx):6d}")

    # --- Escritura ---
    for s, v in splits.items():
        sub = df.iloc[v][["image_id", "lesion_id", "dx", "label"]].sort_values("image_id")
        sub.to_csv(out_dir / f"{s}.csv", index=False)
        print(f"  {s}: {len(v):5d} imagenes -> {out_dir / f'{s}.csv'}")

    with open(out_dir / "class_index.json", "w") as f:
        json.dump(CLASS_TO_IDX, f, indent=2)
    print(f"  mapeo de clases -> {out_dir / 'class_index.json'}")


if __name__ == "__main__":
    main()
