#!/usr/bin/env python3
"""
Genera figuras de monitorización a partir del `stats.jsonl` de una corrida de
StyleGAN2-ADA: r_t (con la línea de target de ADA), p (augment) y pérdidas G/D.
Reproducible y sin depender de la UI de TensorBoard — apto para la memoria (F6).

Uso:
    python plot_training_stats.py --run ~/TFG/experimentos/runs/00001-... --target 0.6

Si existe(n) `metric-*.jsonl` en la corrida, añade además la curva de FID/KID.
Guarda `monitor.png` dentro de la carpeta de la corrida (o en --out).
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def series(recs, key):
    xs, ys = [], []
    for r in recs:
        if key in r and "Progress/kimg" in r:
            xs.append(r["Progress/kimg"]["mean"])
            ys.append(r[key]["mean"])
    return xs, ys


def load_metric(run):
    """Lee curvas de métricas (FID/KID) de los metric-*.jsonl si existen."""
    out = {}
    for f in glob.glob(os.path.join(run, "metric-*.jsonl")):
        recs = [json.loads(l) for l in open(f)]
        for r in recs:
            for name, val in r.get("results", {}).items():
                # kimg embebido en el nombre del snapshot: network-snapshot-000200.pkl
                snap = r.get("snapshot_pkl", "")
                digits = "".join(c for c in snap if c.isdigit())
                kimg = int(digits) if digits else len(out.get(name, []))
                out.setdefault(name, []).append((kimg, val))
    for name in out:
        out[name].sort()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--target", type=float, default=0.6, help="ada_target usado")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run = os.path.expanduser(args.run)
    recs = [json.loads(l) for l in open(os.path.join(run, "stats.jsonl"))]
    metrics = load_metric(run)

    panels = 3 + (1 if metrics else 0)
    fig, axes = plt.subplots(panels, 1, figsize=(9, 3 * panels), sharex=False)

    # r_t
    x, y = series(recs, "Loss/signs/real")
    axes[0].plot(x, y, color="tab:blue")
    axes[0].axhline(args.target, ls="--", color="tab:red", label=f"target={args.target}")
    axes[0].set_ylabel(r"$r_t$ = E[sign(D_real)]"); axes[0].legend(); axes[0].grid(alpha=.3)
    axes[0].set_title("Sobreajuste de D y augmentation adaptativa (ADA)")

    # p
    x, y = series(recs, "Progress/augment")
    axes[1].plot(x, y, color="tab:green")
    axes[1].set_ylabel("p (augment)"); axes[1].grid(alpha=.3)

    # losses G/D
    for key, c, lab in [("Loss/G/loss", "tab:orange", "G"), ("Loss/D/loss", "tab:purple", "D")]:
        x, y = series(recs, key)
        axes[2].plot(x, y, color=c, label=lab)
    axes[2].set_ylabel("pérdida"); axes[2].legend(); axes[2].grid(alpha=.3)
    axes[2].set_xlabel("kimg")

    # métricas (FID/KID) si hay
    if metrics:
        for name, pts in metrics.items():
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            axes[3].plot(xs, ys, marker="o", label=name)
        axes[3].set_ylabel("métrica"); axes[3].set_xlabel("kimg")
        axes[3].legend(); axes[3].grid(alpha=.3)

    out = args.out or os.path.join(run, "monitor.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"Figura guardada en: {out}")
    print(f"Ticks leídos: {len(recs)} | métricas: {list(metrics)}")


if __name__ == "__main__":
    main()