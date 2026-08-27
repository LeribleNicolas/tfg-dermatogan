"""Verificacion del WeightedInfiniteSampler: compara la fraccion muestreada por
clase con la esperada, para varios valores de alpha.

Ejecutar desde la raiz del repo (con el venv activo):
    python test_weighted_sampler.py --data ~/TFG/datos/ham7_train_256.zip

Esperado:
    alpha= 0.00 -> fracciones ~ proporciones reales del dataset (nv domina)
    alpha=-1.00 -> fracciones ~ 1/7 = 0.143 para las 7 clases (rebalanceo completo)
"""
import argparse
import collections

import numpy as np

from training.dataset import ImageFolderDataset
from training.weighted_sampler import WeightedInfiniteSampler, _per_index_classes

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--draws", type=int, default=200000)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, -0.5, -1.0])
    args = ap.parse_args()

    ds = ImageFolderDataset(path=args.data, use_labels=True)
    cls = _per_index_classes(ds)
    counts = np.bincount(cls, minlength=len(CLASSES))
    print("Imagenes por clase:", {CLASSES[i]: int(counts[i]) for i in range(len(CLASSES))})
    print(f"(total {int(counts.sum())} imagenes, {args.draws} extracciones por alpha)\n")

    for a in args.alphas:
        sampler = WeightedInfiniteSampler(ds, alpha=a, seed=0)
        it = iter(sampler)
        c = collections.Counter()
        for _ in range(args.draws):
            c[int(cls[next(it)])] += 1
        frac = {CLASSES[i]: round(c[i] / args.draws, 3) for i in range(len(CLASSES))}
        print(f"alpha={a:+.2f} -> {frac}")


if __name__ == "__main__":
    main()
