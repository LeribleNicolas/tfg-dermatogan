#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analisis de vecino mas cercano (memorizacion) por clase para StyleGAN2-ADA condicional.

Objetivo: comprobar si el generador esta MEMORIZANDO (copiando) las pocas imagenes
reales de las clases minoritarias (df, vasc, akiec) en lugar de generalizar.

Como funciona, por clase:
  1. Genera N imagenes condicionadas a la clase.
  2. Para cada generada, busca su imagen REAL mas cercana (distancia L2 en pixeles,
     sobre una version reescalada) dentro de esa clase.
  3. Compara la distribucion de distancias "generada -> real mas cercana" con la
     linea base "real -> real mas cercana" (leave-one-out). Si las generadas NO
     estan sistematicamente mas cerca de las reales de lo que las reales estan
     entre si, NO hay evidencia de memorizacion.
  4. Guarda un MONTAJE visual por clase: para las generadas MAS sospechosas (las de
     menor distancia a una real), muestra [generada | k reales mas cercanas], para
     inspeccion manual -- la prueba definitiva de si son copias o no.

Se usa distancia en pixeles (no features de Inception) porque para detectar COPIAS
casi identicas una metrica de bajo nivel es mas adecuada, y ademas evita depender
del detector Inception (y de la red).

Uso tipico:
    cd ~/TFG/stylegan2-ada-pytorch
    source ../venv/bin/activate
    python ~/TFG/scripts/nn_memorization.py \
        --network=~/TFG/experimentos/00001-*/network-snapshot-004000.pkl \
        --data=~/TFG/datos/ham7_train_256.zip \
        --class-index=~/TFG/datos/splits/class_index.json \
        --classes=df,vasc,akiec \
        --outdir=~/TFG/experimentos/nn_memorization_004000
"""

import sys
import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def to_small(uint8_chw, nn_res, device):
    """uint8 [N,3,H,W] -> float [0,1] reescalado a nn_res -> aplanado [N, D]."""
    x = torch.from_numpy(uint8_chw).to(device).float() / 255.0
    x = F.interpolate(x, size=(nn_res, nn_res), mode='area')
    return x.reshape(x.shape[0], -1)


def make_montage(rows, gap=4, bg=255):
    """rows: lista de filas; cada fila es lista de imagenes uint8 CHW (mismo HxW)."""
    rows_hwc = [[np.transpose(im, (1, 2, 0)) for im in row] for row in rows]
    H, W = rows_hwc[0][0].shape[:2]
    ncol = max(len(r) for r in rows_hwc)
    nrow = len(rows_hwc)
    canvas = np.full((nrow * H + (nrow - 1) * gap, ncol * W + (ncol - 1) * gap, 3), bg, np.uint8)
    for r, row in enumerate(rows_hwc):
        for c, im in enumerate(row):
            y, x = r * (H + gap), c * (W + gap)
            canvas[y:y + H, x:x + W] = im
    return canvas


def main():
    p = argparse.ArgumentParser(description='Analisis de vecino mas cercano (memorizacion) por clase.')
    p.add_argument('--network', required=True, help='Checkpoint .pkl (usa G_ema).')
    p.add_argument('--data', required=True, help='Zip de entrenamiento (imagenes reales).')
    p.add_argument('--class-index', required=True, help='JSON nombre_clase -> indice.')
    p.add_argument('--classes', default='df,vasc,akiec', help='Clases a analizar (por nombre, separadas por coma).')
    p.add_argument('--num-gen', type=int, default=2000, help='Imagenes generadas por clase (default 2000).')
    p.add_argument('--montage-samples', type=int, default=8, help='Nº de generadas (las mas cercanas) en el montaje.')
    p.add_argument('--topk', type=int, default=5, help='Reales mas cercanas por generada en el montaje.')
    p.add_argument('--nn-res', type=int, default=64, help='Resolucion para calcular la distancia (default 64).')
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--repo-dir', default=os.environ.get("SG2_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    p.add_argument('--outdir', default='.', help='Directorio de salida (montajes + json).')
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    sys.path.insert(0, args.repo_dir)
    import dnnlib
    import legacy
    from training.dataset import ImageFolderDataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.class_index, 'r', encoding='utf-8') as f:
        name_to_idx = json.load(f)
    num_classes = len(name_to_idx)
    wanted = [(n.strip(), name_to_idx[n.strip()]) for n in args.classes.split(',') if n.strip()]
    wanted_idxs = {idx for _, idx in wanted}

    print(f'Cargando red: {args.network}')
    with dnnlib.util.open_url(args.network) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device).eval()
    assert G.c_dim == num_classes

    # --- Cargar imagenes reales de las clases pedidas (iterando en el proceso
    #     principal, sin DataLoader -> sin problemas de concurrencia con el zip) ---
    print(f'Leyendo imagenes reales de las clases {[n for n, _ in wanted]}...')
    dataset = ImageFolderDataset(path=args.data, use_labels=True, max_size=None, xflip=False)
    reals = {idx: [] for idx in wanted_idxs}
    for i in range(len(dataset)):
        img, lab = dataset[i]                 # img uint8 CHW, lab one-hot
        c = int(np.argmax(lab))
        if c in reals:
            reals[c].append(img)
        if (i + 1) % 1000 == 0:
            print(f'  escaneadas {i + 1}/{len(dataset)}', end='\r')
    print()

    summary = {}
    for name, idx in wanted:
        real_u8 = np.stack(reals[idx]).astype(np.uint8)         # [R,3,H,W]
        R = real_u8.shape[0]
        real_small = to_small(real_u8, args.nn_res, device)      # [R, D]

        # --- Generar y quedarnos con las imagenes (uint8 en CPU) + version pequena ---
        gen_u8 = np.empty((args.num_gen, *real_u8.shape[1:]), np.uint8)
        gen_small = torch.empty((args.num_gen, real_small.shape[1]), device=device)
        n = 0
        with torch.no_grad():
            while n < args.num_gen:
                b = min(args.batch, args.num_gen - n)
                z = torch.randn([b, G.z_dim], device=device)
                c = torch.zeros([b, G.c_dim], device=device); c[:, idx] = 1
                img = G(z, c)                                    # float ~[-1,1]
                img_u8 = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                gen_u8[n:n + b] = img_u8.cpu().numpy()
                gen_small[n:n + b] = to_small(img_u8.cpu().numpy(), args.nn_res, device)
                n += b
                print(f'  [{name}] generadas {n}/{args.num_gen}', end='\r')
        print()

        # --- Distancias generada -> real mas cercana ---
        d_gen = torch.cdist(gen_small, real_small)               # [G, R]
        gen_nn_dist, gen_nn_idx = d_gen.min(dim=1)               # [G]

        # --- Linea base: real -> real mas cercana (leave-one-out) ---
        d_rr = torch.cdist(real_small, real_small)
        d_rr.fill_diagonal_(float('inf'))
        real_nn_dist = d_rr.min(dim=1).values                    # [R]

        g = gen_nn_dist.cpu().numpy()
        r = real_nn_dist.cpu().numpy()
        pct = lambda a, q: float(np.percentile(a, q))
        # Veredicto heuristico: si el minimo de las generadas no baja del percentil 1
        # de las distancias reales-reales, no hay señal de copia.
        flag = bool(g.min() < pct(r, 1))
        veredicto = ('POSIBLE memorizacion: revisa el montaje' if flag
                     else 'sin evidencia de memorizacion')

        summary[name] = dict(
            index=idx, n_real=int(R), n_gen=int(args.num_gen),
            gen_nn_dist=dict(min=float(g.min()), p1=pct(g, 1), median=pct(g, 50)),
            real_nn_dist=dict(min=float(r.min()), p1=pct(r, 1), median=pct(r, 50)),
            possible_memorization=flag,
        )
        print(f'  -> {name:6s}  gen->real  min={g.min():.4f} p1={pct(g,1):.4f} med={pct(g,50):.4f}  |  '
              f'real->real p1={pct(r,1):.4f} med={pct(r,50):.4f}  =>  {veredicto}')

        # --- Montaje: las generadas MAS sospechosas (menor distancia a una real) ---
        order = np.argsort(g)[:args.montage_samples]
        rows = []
        for gi in order:
            nn = torch.argsort(d_gen[gi])[:args.topk].cpu().numpy()
            row = [gen_u8[gi]] + [real_u8[j] for j in nn]        # [generada | k reales cercanas]
            rows.append(row)
        montage = make_montage(rows)
        out_png = os.path.join(args.outdir, f'nn_montage_{name}.png')
        Image.fromarray(montage).save(out_png)
        print(f'     montaje guardado: {out_png}  (col 1 = generada; resto = reales mas cercanas)')

        del gen_small, d_gen, d_rr
        torch.cuda.empty_cache()

    out_json = os.path.join(args.outdir, 'nn_memorization.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(dict(network=args.network, seed=args.seed, nn_res=args.nn_res,
                       num_gen=args.num_gen, per_class=summary), f, indent=2, ensure_ascii=False)
    print(f'\nResumen guardado en: {out_json}')
    print('Interpretacion: abre los nn_montage_*.png. Si las reales mas cercanas NO son')
    print('casi identicas a la generada, no hay memorizacion (aunque la clase sea de baja calidad).')


if __name__ == '__main__':
    main()
