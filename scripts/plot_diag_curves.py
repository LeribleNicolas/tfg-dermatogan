#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figuras de la corrida de entrenamiento (StyleGAN2-ADA) para la memoria.

Lee, del directorio de una corrida:
  - stats.jsonl              -> r_t (Loss/signs/real), p (Progress/augment),
                                perdidas G y D, frente a kimg.
  - metric-fid50k_full.jsonl -> FID global frente a kimg.

Genera:
  - fid_curve.png            -> FID vs kimg (con el mejor valor marcado).
  - training_curves.png      -> panel 2x2: FID, r_t (con linea de target),
                                p, y perdidas G/D.

No usa red, ni el detector, ni GPU: solo matplotlib.

Uso:
    python ~/TFG/scripts/plot_diag_curves.py \
        --run-dir=~/TFG/experimentos/00001-ham7_train_256-cond-mirror-auto1-kimg5000-ada-target0.6-bgc \
        --target=0.6
"""

import os
import re
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _val(entry, key):
    """Devuelve el valor de una clave de stats.jsonl (puede ser dict {'mean':..} o escalar)."""
    if key not in entry:
        return None
    v = entry[key]
    return v['mean'] if isinstance(v, dict) and 'mean' in v else v


def load_stats(path):
    """Lee stats.jsonl -> dict de series {kimg, rt, p, gloss, dloss} (listas alineadas)."""
    series = {k: [] for k in ('kimg', 'rt', 'p', 'gloss', 'dloss')}
    if not os.path.isfile(path):
        return series
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            kimg = _val(d, 'Progress/kimg')
            if kimg is None:
                continue
            series['kimg'].append(kimg)
            series['rt'].append(_val(d, 'Loss/signs/real'))
            series['p'].append(_val(d, 'Progress/augment'))
            series['gloss'].append(_val(d, 'Loss/G/loss'))
            series['dloss'].append(_val(d, 'Loss/D/loss'))
    return series


def load_fid(path):
    """Lee metric-fid50k_full.jsonl -> (kimg[], fid[])."""
    kimg, fid = [], []
    if not os.path.isfile(path):
        return np.array(kimg), np.array(fid)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            m = re.search(r'network-snapshot-(\d+)\.pkl', d.get('snapshot_pkl', ''))
            if not m:
                continue
            kimg.append(int(m.group(1)))
            fid.append(d['results']['fid50k_full'])
    order = np.argsort(kimg)
    return np.array(kimg)[order], np.array(fid)[order]


def _clean(x, y):
    """Filtra pares con y no nulo."""
    x = np.asarray(x, float)
    y = np.asarray([np.nan if v is None else v for v in y], float)
    m = ~np.isnan(y)
    return x[m], y[m]


def main():
    ap = argparse.ArgumentParser(description='Figuras de la corrida de entrenamiento.')
    ap.add_argument('--run-dir', required=True, help='Directorio de la corrida (contiene stats.jsonl y metric-*.jsonl).')
    ap.add_argument('--target', type=float, default=0.6, help='Valor objetivo de ADA (linea de referencia en r_t).')
    ap.add_argument('--outdir', default=None, help='Directorio de salida (por defecto, el de la corrida).')
    args = ap.parse_args()

    outdir = args.outdir or args.run_dir
    os.makedirs(outdir, exist_ok=True)

    stats = load_stats(os.path.join(args.run_dir, 'stats.jsonl'))
    fk, fv = load_fid(os.path.join(args.run_dir, 'metric-fid50k_full.jsonl'))

    # --- Figura 1: FID solo ---
    if len(fk):
        best_i = int(np.argmin(fv))
        plt.figure(figsize=(7, 4.5))
        plt.plot(fk, fv, marker='o', ms=3, lw=1.4, color='#1f77b4')
        plt.scatter([fk[best_i]], [fv[best_i]], color='#d62728', zorder=5,
                    label=f'mejor: {fv[best_i]:.2f} @ {fk[best_i]} kimg')
        plt.xlabel('kimg'); plt.ylabel('FID (fid50k_full)')
        plt.title('FID global vs. kimg')
        plt.grid(alpha=0.3); plt.legend()
        plt.tight_layout()
        p1 = os.path.join(outdir, 'fid_curve.png')
        plt.savefig(p1, dpi=150); plt.close()
        print(f'Guardada: {p1}  (mejor FID {fv[best_i]:.2f} @ {fk[best_i]} kimg)')

    # --- Figura 2: panel 2x2 ---
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    # FID
    if len(fk):
        ax[0, 0].plot(fk, fv, marker='o', ms=3, lw=1.4, color='#1f77b4')
        ax[0, 0].set_title('FID (fid50k_full)')
        ax[0, 0].set_xlabel('kimg'); ax[0, 0].set_ylabel('FID'); ax[0, 0].grid(alpha=0.3)

    # r_t
    x, y = _clean(stats['kimg'], stats['rt'])
    if len(x):
        ax[0, 1].plot(x, y, lw=1.0, color='#2ca02c')
        ax[0, 1].axhline(args.target, ls='--', color='#d62728', label=f'target = {args.target}')
        ax[0, 1].set_title('r_t = E[sign(D_logits_real)]  (sobreajuste de D)')
        ax[0, 1].set_xlabel('kimg'); ax[0, 1].set_ylabel('r_t'); ax[0, 1].grid(alpha=0.3); ax[0, 1].legend()

    # p (augment)
    x, y = _clean(stats['kimg'], stats['p'])
    if len(x):
        ax[1, 0].plot(x, y, lw=1.0, color='#9467bd')
        ax[1, 0].set_title('p (probabilidad de augmentation ADA)')
        ax[1, 0].set_xlabel('kimg'); ax[1, 0].set_ylabel('p'); ax[1, 0].grid(alpha=0.3)

    # perdidas G/D
    xg, yg = _clean(stats['kimg'], stats['gloss'])
    xd, yd = _clean(stats['kimg'], stats['dloss'])
    if len(xg):
        ax[1, 1].plot(xg, yg, lw=1.0, label='G', color='#1f77b4')
    if len(xd):
        ax[1, 1].plot(xd, yd, lw=1.0, label='D', color='#ff7f0e')
    ax[1, 1].set_title('Pérdidas G y D')
    ax[1, 1].set_xlabel('kimg'); ax[1, 1].set_ylabel('loss'); ax[1, 1].grid(alpha=0.3); ax[1, 1].legend()

    fig.suptitle('Curvas de entrenamiento — corrida diagnóstica F4', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p2 = os.path.join(outdir, 'training_curves.png')
    fig.savefig(p2, dpi=150); plt.close(fig)
    print(f'Guardada: {p2}')


if __name__ == '__main__':
    main()
