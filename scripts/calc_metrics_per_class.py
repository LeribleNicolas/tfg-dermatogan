#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FID + KID por clase para un checkpoint de StyleGAN2-ADA condicional.

Reutiliza la maquinaria del repositorio oficial (dnnlib, legacy, metrics.metric_utils)
para extraer features de Inception-v3, y calcula, CLASE A CLASE:

  - FID  (Frechet Inception Distance): mismo estimador que fid50k_full.
  - KID  (Kernel Inception Distance, MMD polinomico insesgado): mismo estimador
         que kid50k_full. Es el que importa en las clases minoritarias (df, vasc),
         donde el FID esta sesgado por el bajo numero de imagenes reales.

Decision del TFG (14/07/2026): reportar FID + KID conjuntamente por clase.

Uso tipico:
    cd ~/TFG/stylegan2-ada-pytorch
    source ../venv/bin/activate
    python ~/TFG/scripts/calc_metrics_per_class.py \
        --network=~/TFG/experimentos/00001-*/network-snapshot-004000.pkl \
        --data=~/TFG/datos/ham7_train_256.zip \
        --class-index=~/TFG/datos/splits/class_index.json \
        --num-gen=50000 \
        --outfile=~/TFG/experimentos/per_class_metrics_004000.json

Notas:
    - Semilla fija (--seed, por defecto 42) para reproducibilidad.
    - --num-gen es el nº de imagenes generadas por clase (50000 = rigor tipo
      "50k_full"; baja a 10000 para una pasada mas rapida).
    - El KID crudo puede ser negativo (estimador insesgado); se reporta tambien
      x1000, como es habitual en la literatura.
"""

import sys
import os
import json
import argparse
import numpy as np
import scipy.linalg
import torch

# --- Parche de compatibilidad: desactivar el fuser TE del JIT ---------------
# El detector Inception es un ScriptModule; en Ampere + este PyTorch el fuser
# TensorExpr genera codigo vectorizado invalido ("MALFORMED INPUT: lanes dont
# match"). Desactivarlo no afecta a la correccion, solo a una optimizacion JIT.
torch._C._jit_set_texpr_fuser_enabled(False)
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)

# URL y kwargs del detector, identicos a los de metrics/frechet_inception_distance.py
DETECTOR_URL = os.environ.get("SG2_INCEPTION", "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt")
DETECTOR_KWARGS = dict(return_features=True)


def compute_fid(mu_real, sigma_real, mu_gen, sigma_gen):
    """FID a partir de medias/covarianzas (mismo calculo que el repo oficial)."""
    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False)
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    return float(fid)


def compute_kid(real_features, gen_features, num_subsets=100, max_subset_size=1000):
    """KID (MMD polinomico insesgado), identico a metrics/kernel_inception_distance.py."""
    n = real_features.shape[1]
    m = min(real_features.shape[0], gen_features.shape[0], max_subset_size)
    t = 0.0
    for _ in range(num_subsets):
        x = gen_features[np.random.choice(gen_features.shape[0], m, replace=False)]
        y = real_features[np.random.choice(real_features.shape[0], m, replace=False)]
        a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
        b = (x @ y.T / n + 1) ** 3
        t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
    return float(t / num_subsets / m)


def main():
    parser = argparse.ArgumentParser(description='FID + KID por clase para StyleGAN2-ADA condicional.')
    parser.add_argument('--network', required=True, help='Ruta al checkpoint .pkl (usa G_ema).')
    parser.add_argument('--data', required=True, help='Ruta al zip de entrenamiento (imagenes reales).')
    parser.add_argument('--class-index', required=True, help='JSON con el mapeo nombre_clase -> indice.')
    parser.add_argument('--num-gen', type=int, default=50000, help='Imagenes generadas por clase (default 50000).')
    parser.add_argument('--batch', type=int, default=64, help='Tamano de lote (default 64).')
    parser.add_argument('--seed', type=int, default=42, help='Semilla (default 42).')
    parser.add_argument('--repo-dir', default=os.environ.get("SG2_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        help='Directorio del repo StyleGAN2-ADA (para importar dnnlib/legacy/metrics).')
    parser.add_argument('--outfile', default=None, help='(Opcional) JSON de salida con los resultados.')
    args = parser.parse_args()

    # Reproducibilidad
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Poner el repo en el path e importar su maquinaria
    sys.path.insert(0, args.repo_dir)
    import dnnlib
    import legacy
    from metrics import metric_utils
    from training.dataset import ImageFolderDataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Mapeo de clases (indice -> nombre), ordenado por indice
    with open(args.class_index, 'r', encoding='utf-8') as f:
        name_to_idx = json.load(f)
    idx_to_name = {v: k for k, v in name_to_idx.items()}
    num_classes = len(name_to_idx)
    class_names = [idx_to_name[i] for i in range(num_classes)]

    # Cargar generador (media EMA)
    print(f'Cargando red: {args.network}')
    with dnnlib.util.open_url(args.network) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device).eval()
    assert G.c_dim == num_classes, \
        f'c_dim del modelo ({G.c_dim}) != nº de clases del class_index ({num_classes})'

    # Detector Inception (mismo que usa fid50k_full/kid50k_full)
    print('Cargando detector Inception-v3...')
    detector = metric_utils.get_feature_detector(DETECTOR_URL, device=device, num_gpus=1, rank=0, verbose=True)

    # ------------------------------------------------------------------ #
    # 1) Features de las imagenes REALES, agrupadas por clase (una pasada)
    # ------------------------------------------------------------------ #
    # xflip=False: evaluamos las imagenes reales tal cual (el mirror es una
    # augmentation de entrenamiento, no parte de la distribucion a evaluar).
    print(f'Extrayendo features de imagenes reales: {args.data}')
    dataset = ImageFolderDataset(path=args.data, use_labels=True, max_size=None, xflip=False)
    real_stats = {c: metric_utils.FeatureStats(capture_all=True, capture_mean_cov=True)
                  for c in range(num_classes)}

    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=False,
                                         num_workers=0, pin_memory=True)
    seen = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)                 # uint8 [B,C,H,W], en [0,255]
            cls = labels.argmax(dim=1).to(device)       # etiquetas one-hot -> indice (torch)
            feats = detector(images, **DETECTOR_KWARGS) # [B, 2048]
            for c in range(num_classes):
                mask = (cls == c)
                if mask.any():
                    real_stats[c].append_torch(feats[mask], num_gpus=1, rank=0)
            seen += images.shape[0]
            print(f'  reales procesadas: {seen}/{len(dataset)}', end='\r')
    print()

    n_real = {c: real_stats[c].num_items for c in range(num_classes)}

    # ------------------------------------------------------------------ #
    # 2) Por clase: generar num_gen imagenes condicionadas y medir FID/KID
    # ------------------------------------------------------------------ #
    results = {}
    for c in range(num_classes):
        name = class_names[c]
        gen_stats = metric_utils.FeatureStats(capture_all=True, capture_mean_cov=True,
                                              max_items=args.num_gen)
        with torch.no_grad():
            while not gen_stats.is_full():
                z = torch.randn([args.batch, G.z_dim], device=device)
                lab = torch.zeros([args.batch, G.c_dim], device=device)
                lab[:, c] = 1
                img = G(z, lab)                                   # truncation_psi=1 (sin truncar)
                img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                feats = detector(img, **DETECTOR_KWARGS)
                gen_stats.append_torch(feats, num_gpus=1, rank=0)
                print(f'  [{name}] generadas: {gen_stats.num_items}/{args.num_gen}', end='\r')
        print()

        mu_r, sig_r = real_stats[c].get_mean_cov()
        mu_g, sig_g = gen_stats.get_mean_cov()
        fid = compute_fid(mu_r, sig_r, mu_g, sig_g)
        kid = compute_kid(real_stats[c].get_all(), gen_stats.get_all())

        results[name] = dict(index=c, n_real=int(n_real[c]), n_gen=int(gen_stats.num_items),
                             fid=fid, kid=kid, kid_x1000=kid * 1000.0)
        print(f'  -> {name:6s}  n_real={n_real[c]:5d}  FID={fid:8.3f}  KID={kid:.6f}  (KIDx1000={kid*1000:.3f})')

    # ------------------------------------------------------------------ #
    # 3) Tabla resumen
    # ------------------------------------------------------------------ #
    print('\n================= FID + KID por clase =================')
    print(f'{"clase":8s} {"n_real":>7s} {"n_gen":>7s} {"FID":>10s} {"KIDx1000":>10s}')
    print('-' * 46)
    for name in class_names:
        r = results[name]
        flag = '  <- pocas reales' if r['n_real'] < 500 else ''
        print(f'{name:8s} {r["n_real"]:7d} {r["n_gen"]:7d} {r["fid"]:10.3f} {r["kid_x1000"]:10.3f}{flag}')
    print('-' * 46)
    macro_fid = float(np.mean([results[n]['fid'] for n in class_names]))
    macro_kid = float(np.mean([results[n]['kid_x1000'] for n in class_names]))
    print(f'{"MEDIA":8s} {"":7s} {"":7s} {macro_fid:10.3f} {macro_kid:10.3f}  (macro por clase)')
    print('=======================================================')
    print('Nota: en clases con n_real bajo (df, vasc) el FID esta sesgado por')
    print('el tamano de muestra; el KID (insesgado) es la lectura fiable ahi.')

    if args.outfile:
        payload = dict(network=args.network, data=args.data, num_gen=args.num_gen,
                       seed=args.seed, per_class=results,
                       macro_fid=macro_fid, macro_kid_x1000=macro_kid)
        with open(args.outfile, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f'\nResultados guardados en: {args.outfile}')


if __name__ == '__main__':
    main()
