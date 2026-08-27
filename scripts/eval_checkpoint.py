#!/usr/bin/env python
"""
eval_checkpoint.py
==================

Evaluacion por clase (FID + KID) y chequeo de memorizacion (vecino mas cercano, NN)
de un checkpoint StyleGAN2-ADA *condicional*, en una sola pasada.

Autor: TFG GAN Dermatologia (Nicolas Lerible Garcia).

Contexto
--------
La corrida definitiva usa muestreo ponderado con alpha = -0.85, que sobre-muestrea
fuertemente las clases minoritarias (df ~8x). Por eso:
  1. El FID *global* enga\xf1a: lo domina `nv`. La metrica que decide es el FID+KID
     POR CLASE, en especial de las minoritarias (akiec, vasc, df).
  2. Con ese sobre-muestreo hay que descartar MEMORIZACION en las minoritarias.

Este script cubre ambas cosas de una vez.

Consistencia con tus numeros previos
-------------------------------------
Reutiliza el MISMO detector Inception del repo (metrics/inception-2015-12-05.pkl)
y replica exactamente las formulas de FID y de KID de `metrics/` (KID con
num_subsets=100 y max_subset_size=1000). Asi los resultados son comparables con
tu `per_class_004000.json` y con `calc_metrics.py`.

Uso (dentro del repo stylegan2-ada-pytorch, con el entorno `sg2` activo)
------------------------------------------------------------------------
    python eval_checkpoint.py \\
        --network=/ruta/network-snapshot-004800.pkl \\
        --data=/ruta/ham7_train_256.zip \\
        --outdir=/ruta/experimentos/eval_004800 \\
        --class-names=nv,mel,bkl,bcc,akiec,vasc,df \\
        --num-gen=50000 \\
        --mem-classes=df,vasc,akiec \\
        --detector=$HOME/.cache/dnnlib/downloads/inception-2015-12-05.pkl

Notas importantes
-----------------
* `--class-names` DEBE ir en el mismo orden que los enteros de etiqueta usados al
  construir el dataset (0..C-1). Si no lo pasas, se etiquetan como class0..classN
  y tendras que renombrar despues. Verificalo contra tu per_class_004000.json.
* `--detector` acepta una ruta local (tu CDN de NVIDIA devuelve 403). `open_url`
  interpreta como fichero local cualquier cadena que no empiece por `xxx://`.
* El chequeo de memorizacion es una CRIBA: la distancia NN en espacio Inception
  se\xf1ala candidatos, pero la evidencia decisiva son los montajes PNG (imagen
  generada + sus vecinos reales) que hay que inspeccionar a ojo.
* Reproducible: semillas fijas (--seed). Todo se registra en `resultados.json`.
"""

import argparse
import json
import os
import sys

import numpy as np
import PIL.Image
import scipy.linalg
import torch

# En Ampere (p.ej. RTX 3060) + torch 1.9 + el detector Inception TorchScript, el
# fuser TensorExpr del JIT provoca `RuntimeError: MALFORMED INPUT: lanes dont match`
# (y a veces segfault). Mismo parche que se aplico a train.py el 03/08: desactivar
# el fuser antes de ejecutar el detector.
try:
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
except Exception:  # por si alguna API no existe en otra version de torch
    pass

# Modulos del propio repo stylegan2-ada-pytorch (ejecutar desde su raiz).
import dnnlib
import legacy
from training.dataset import ImageFolderDataset
from metrics import metric_utils

# Detector Inception oficial del repo (misma red que usa calc_metrics.py).
DETECTOR_URL = ('https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/'
                'pretrained/metrics/inception-2015-12-05.pt')
DETECTOR_KWARGS = dict(return_features=True)  # -> features de 2048-d


# --------------------------------------------------------------------------- #
#  Utilidades                                                                  #
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(
        description='FID+KID por clase y chequeo de memorizacion de un checkpoint '
                    'StyleGAN2-ADA condicional.')
    p.add_argument('--network', required=True, help='Ruta al .pkl del checkpoint.')
    p.add_argument('--data', required=True, help='Dataset de referencia (.zip).')
    p.add_argument('--outdir', required=True, help='Carpeta de salida.')
    p.add_argument('--detector', default=DETECTOR_URL,
                   help='Ruta/URL del detector Inception (usa ruta local si la CDN da 403).')
    p.add_argument('--class-names', default=None,
                   help='Nombres de clase separados por coma, en orden 0..C-1 '
                        '(p.ej. nv,mel,bkl,bcc,akiec,vasc,df).')
    p.add_argument('--num-gen', type=int, default=50000,
                   help='Imagenes generadas por clase para FID/KID (bajalo para pruebas).')
    p.add_argument('--batch', type=int, default=64, help='Tama\xf1o de lote.')
    p.add_argument('--mem-classes', default='df,vasc,akiec',
                   help='Clases (nombres o indices) para el chequeo de memorizacion.')
    p.add_argument('--mem-num-gen', type=int, default=500,
                   help='Imagenes generadas por clase para la criba de memorizacion.')
    p.add_argument('--mem-topk', type=int, default=16,
                   help='N generadas mas cercanas a un real que se guardan en montaje.')
    p.add_argument('--mem-neighbors', type=int, default=3,
                   help='N vecinos reales por generada en el montaje.')
    p.add_argument('--seed', type=int, default=42, help='Semilla global.')
    p.add_argument('--kid-subsets', type=int, default=100)
    p.add_argument('--kid-subset-size', type=int, default=1000)
    return p.parse_args()


def resolve_class_selection(sel, class_names):
    """Convierte 'df,vasc' o '6,5' en una lista de indices de clase."""
    out = []
    for tok in sel.split(','):
        tok = tok.strip()
        if tok == '':
            continue
        if tok.lstrip('-').isdigit():
            out.append(int(tok))
        else:
            if tok not in class_names:
                raise ValueError(f'Clase desconocida: {tok!r}. Nombres: {class_names}')
            out.append(class_names.index(tok))
    return out


@torch.no_grad()
def features_from_uint8(batch_uint8, detector):
    """Extrae features Inception (2048-d) de un lote uint8 NCHW en un device."""
    return detector(batch_uint8, **DETECTOR_KWARGS).cpu().numpy()


@torch.no_grad()
def real_features_for_class(dataset, idxs, detector, device, batch):
    """Features Inception de todas las imagenes reales de una clase."""
    feats = []
    for start in range(0, len(idxs), batch):
        sub = idxs[start:start + batch]
        imgs = np.stack([dataset[int(i)][0] for i in sub])  # uint8 CHW
        t = torch.from_numpy(imgs).to(device)
        feats.append(features_from_uint8(t, detector))
    return np.concatenate(feats, axis=0)


@torch.no_grad()
def gen_features_for_class(G, class_idx, num_gen, detector, device, batch, seed,
                           keep_images=False):
    """Genera `num_gen` imagenes condicionadas a `class_idx` y devuelve sus features.

    Si keep_images=True, devuelve tambien las imagenes uint8 (para montajes).
    """
    gen = torch.Generator(device=device).manual_seed(seed + 1000 * class_idx)
    feats, kept = [], []
    remaining = num_gen
    c = torch.zeros([batch, G.c_dim], device=device)
    c[:, class_idx] = 1
    while remaining > 0:
        n = min(batch, remaining)
        z = torch.randn([n, G.z_dim], device=device, generator=gen)
        img = G(z, c[:n], truncation_psi=1, noise_mode='random')  # como en metrics/
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        feats.append(features_from_uint8(img, detector))
        if keep_images:
            kept.append(img.cpu().numpy())
        remaining -= n
    feats = np.concatenate(feats, axis=0)
    images = np.concatenate(kept, axis=0) if keep_images else None
    return feats, images


def compute_fid(real, gen):
    """FID entre dos conjuntos de features (misma formula que metrics/)."""
    mu_r, mu_g = real.mean(0), gen.mean(0)
    sig_r = np.cov(real, rowvar=False)
    sig_g = np.cov(gen, rowvar=False)
    covmean, _ = scipy.linalg.sqrtm(sig_g.dot(sig_r), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu_g - mu_r
    return float(diff.dot(diff) + np.trace(sig_g + sig_r - 2 * covmean))


def compute_kid(real, gen, num_subsets, max_subset_size, rng):
    """KID polinomico (misma formula y constantes que metrics/kernel_inception_distance.py)."""
    n = real.shape[1]
    m = min(min(real.shape[0], gen.shape[0]), max_subset_size)
    total = 0.0
    for _ in range(num_subsets):
        x = gen[rng.choice(gen.shape[0], m, replace=False)]
        y = real[rng.choice(real.shape[0], m, replace=False)]
        a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
        b = (x @ y.T / n + 1) ** 3
        total += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
    return float(total / num_subsets / m)


def nn_min_distances(query, bank, exclude_self=False):
    """Para cada fila de `query`, distancia L2 al vecino mas cercano en `bank`.

    Devuelve (dist_min, idx_min). Con exclude_self=True ignora el propio indice
    (para el baseline real-vs-real, asumiendo query is bank).
    """
    # ||q-b||^2 = |q|^2 + |b|^2 - 2 q.b   (estable y suficiente para ranking)
    q2 = (query ** 2).sum(1, keepdims=True)
    b2 = (bank ** 2).sum(1, keepdims=True).T
    d2 = q2 + b2 - 2.0 * query @ bank.T
    d2 = np.maximum(d2, 0.0)
    if exclude_self:
        np.fill_diagonal(d2, np.inf)
    idx = d2.argmin(1)
    dmin = np.sqrt(d2[np.arange(d2.shape[0]), idx])
    return dmin, idx


def topk_neighbors(query_row, bank, k):
    """Indices de los k reales mas cercanos a una generada (para montaje)."""
    d2 = ((bank - query_row[None, :]) ** 2).sum(1)
    return np.argsort(d2)[:k]


def save_montage(rows, path, pad=2, bg=255):
    """rows: lista de listas de imagenes uint8 CHW (misma longitud por fila)."""
    h, w = rows[0][0].shape[1], rows[0][0].shape[2]
    ncol = max(len(r) for r in rows)
    nrow = len(rows)
    canvas = np.full((nrow * h + (nrow + 1) * pad,
                      ncol * w + (ncol + 1) * pad, 3), bg, dtype=np.uint8)
    for i, row in enumerate(rows):
        for j, im in enumerate(row):
            y = pad + i * (h + pad)
            x = pad + j * (w + pad)
            canvas[y:y + h, x:x + w] = np.transpose(im, (1, 2, 0))
    PIL.Image.fromarray(canvas).save(path)


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print('[AVISO] No hay CUDA disponible: sera MUY lento.', file=sys.stderr)

    # --- Cargar generador (G_ema) ---------------------------------------- #
    print(f'Cargando red: {args.network}')
    with dnnlib.util.open_url(args.network) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device).eval()
    num_classes = int(G.c_dim)
    if num_classes == 0:
        sys.exit('ERROR: el modelo no es condicional (c_dim=0).')

    # --- Nombres de clase ------------------------------------------------- #
    if args.class_names:
        class_names = [s.strip() for s in args.class_names.split(',')]
        if len(class_names) != num_classes:
            sys.exit(f'ERROR: --class-names tiene {len(class_names)} nombres pero '
                     f'el modelo tiene {num_classes} clases.')
    else:
        class_names = [f'class{i}' for i in range(num_classes)]
        print('[AVISO] Sin --class-names: se usan etiquetas genericas. Verifica el '
              'orden contra tu per_class_004000.json.', file=sys.stderr)

    # --- Detector Inception ---------------------------------------------- #
    print(f'Cargando detector: {args.detector}')
    detector = metric_utils.get_feature_detector(
        url=args.detector, device=device, num_gpus=1, rank=0, verbose=True)

    # --- Dataset real y mapa de indices por clase ------------------------ #
    print(f'Abriendo dataset: {args.data}')
    dataset = ImageFolderDataset(path=args.data, use_labels=True, max_size=None, xflip=False)
    labels = np.array([int(dataset.get_label(i).argmax()) for i in range(len(dataset))])
    idx_by_class = {c: np.where(labels == c)[0] for c in range(num_classes)}

    results = {
        'network': os.path.abspath(args.network),
        'data': os.path.abspath(args.data),
        'num_gen_per_class': args.num_gen,
        'seed': args.seed,
        'per_class': {},
        'macro': {},
        'memorization': {},
    }

    # ================================================================== #
    #  1) FID + KID por clase                                            #
    # ================================================================== #
    fids, kids = [], []
    real_feats_cache = {}
    for c in range(num_classes):
        name = class_names[c]
        n_real = len(idx_by_class[c])
        if n_real == 0:
            print(f'[AVISO] Clase {name}: 0 imagenes reales, se omite.', file=sys.stderr)
            continue
        print(f'\n=== Clase {name} (idx {c}) | reales={n_real} ===')
        real_feats = real_features_for_class(dataset, idx_by_class[c], detector, device, args.batch)
        real_feats_cache[c] = real_feats  # reutilizado en memorizacion

        gen_feats, _ = gen_features_for_class(
            G, c, args.num_gen, detector, device, args.batch, args.seed)

        fid = compute_fid(real_feats, gen_feats)
        kid = compute_kid(real_feats, gen_feats, args.kid_subsets, args.kid_subset_size, rng)
        fids.append(fid)
        kids.append(kid)
        results['per_class'][name] = {
            'n_real': int(n_real), 'fid': fid, 'kid': kid, 'kid_x1000': kid * 1000.0}
        print(f'  FID={fid:.2f}  KID={kid:.6f}  (KIDx1000={kid*1000:.2f})')

    results['macro'] = {
        'fid_macro': float(np.mean(fids)) if fids else None,
        'kid_macro': float(np.mean(kids)) if kids else None,
        'kid_macro_x1000': float(np.mean(kids) * 1000) if kids else None,
    }

    # ================================================================== #
    #  2) Chequeo de memorizacion (NN en espacio Inception)             #
    # ================================================================== #
    mem_classes = resolve_class_selection(args.mem_classes, class_names)
    mem_dir = os.path.join(args.outdir, 'memorizacion')
    os.makedirs(mem_dir, exist_ok=True)

    for c in mem_classes:
        name = class_names[c]
        if c not in real_feats_cache:
            print(f'[AVISO] Sin reales para {name}, se omite memorizacion.', file=sys.stderr)
            continue
        print(f'\n=== Memorizacion clase {name} (idx {c}) ===')
        real_feats = real_feats_cache[c]

        # Baseline real-vs-real: cada real vs su vecino real mas cercano (excluyendose).
        rr_dmin, _ = nn_min_distances(real_feats, real_feats, exclude_self=True)

        # Generadas (pocas, con imagenes guardadas para montaje).
        gen_feats, gen_imgs = gen_features_for_class(
            G, c, args.mem_num_gen, detector, device, args.batch,
            args.seed + 777, keep_images=True)
        gr_dmin, gr_idx = nn_min_distances(gen_feats, real_feats)

        # Estadisticos: si las generadas estuvieran "copiando", sus distancias NN
        # caerian dentro (o por debajo) de la banda real-vs-real. Si estan por
        # encima, el modelo generaliza (no copia).
        stat = {
            'real_vs_real': {
                'min': float(rr_dmin.min()), 'p05': float(np.percentile(rr_dmin, 5)),
                'median': float(np.median(rr_dmin)), 'mean': float(rr_dmin.mean())},
            'gen_vs_real': {
                'min': float(gr_dmin.min()), 'p05': float(np.percentile(gr_dmin, 5)),
                'median': float(np.median(gr_dmin)), 'mean': float(gr_dmin.mean())},
            # Fraccion de generadas mas cercanas a un real que el p05 real-real:
            'frac_gen_below_real_p05': float(
                (gr_dmin < np.percentile(rr_dmin, 5)).mean()),
        }
        results['memorization'][name] = stat
        print(f'  real-real  min={stat["real_vs_real"]["min"]:.2f}  '
              f'median={stat["real_vs_real"]["median"]:.2f}')
        print(f'  gen-real   min={stat["gen_vs_real"]["min"]:.2f}  '
              f'median={stat["gen_vs_real"]["median"]:.2f}')
        print(f'  frac gen < p05(real-real) = {stat["frac_gen_below_real_p05"]:.4f} '
              f'(cerca de 0 = sin memorizacion)')

        # Montaje de las mas sospechosas: generadas con menor distancia NN.
        order = np.argsort(gr_dmin)[:args.mem_topk]
        rows = []
        for gi in order:
            neigh = topk_neighbors(gen_feats[gi], real_feats, args.mem_neighbors)
            row = [gen_imgs[gi]]  # 1a columna: generada
            for nj in neigh:      # siguientes: vecinos reales
                real_global_idx = int(idx_by_class[c][nj])
                row.append(dataset[real_global_idx][0])
            rows.append(row)
        montage_path = os.path.join(mem_dir, f'memorizacion_{name}.png')
        save_montage(rows, montage_path)
        print(f'  Montaje (col 1 = generada, resto = reales NN): {montage_path}')

    # --- Guardar resultados ---------------------------------------------- #
    out_json = os.path.join(args.outdir, 'resultados.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\nOK. Resultados en {out_json}')
    if results['macro']['fid_macro'] is not None:
        print(f"Media macro  FID={results['macro']['fid_macro']:.2f}  "
              f"KIDx1000={results['macro']['kid_macro_x1000']:.2f}")


if __name__ == '__main__':
    main()