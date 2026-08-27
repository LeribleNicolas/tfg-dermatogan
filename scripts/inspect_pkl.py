"""
Inspección empírica de cifar10.pkl para verificar la arquitectura
del embedding condicional en StyleGAN2-ADA.

Objetivo: confirmar que la matriz de pesos del embedding de clase
tiene una dimensión igual al número de clases (c_dim), y por tanto
cambiar c_dim implica reinicializar esa capa.

Uso:
    python inspect_pkl.py <ruta_al_pkl>
"""

import sys
import pickle
import pprint

import torch


def main(pkl_path: str) -> None:
    print(f"Cargando modelo desde: {pkl_path}\n")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    print(f"Claves presentes en el pickle: {list(data.keys())}\n")

    # G_ema es la media exponencial móvil del generador, es la que se usa
    # en inferencia. G es el generador durante entrenamiento y D el discriminador.
    G = data["G_ema"]
    D = data["D"]

    print("=" * 70)
    print("GENERATOR (G_ema)")
    print("=" * 70)
    print(f"  c_dim (número de clases):        {G.c_dim}")
    print(f"  z_dim (dimensión del latente):   {G.z_dim}")
    print(f"  w_dim (dimensión de estilos):    {G.w_dim}")
    print(f"  img_resolution:                  {G.img_resolution}")
    print(f"  img_channels:                    {G.img_channels}")
    print()

    print("  MappingNetwork.embed:")
    print(f"    Tipo: {type(G.mapping.embed).__name__}")
    print(f"    Módulo completo: {G.mapping.embed}")
    print(f"    Shape del peso (weight): {G.mapping.embed.weight.shape}")
    if hasattr(G.mapping.embed, "bias") and G.mapping.embed.bias is not None:
        print(f"    Shape del bias:          {G.mapping.embed.bias.shape}")
    print()

    print("=" * 70)
    print("DISCRIMINATOR (D)")
    print("=" * 70)
    print(f"  c_dim (número de clases):        {D.c_dim}")
    print(f"  img_resolution:                  {D.img_resolution}")
    print()

    if D.c_dim > 0:
        print("  MappingNetwork.embed (proyección de la clase en D):")
        print(f"    Tipo: {type(D.mapping.embed).__name__}")
        print(f"    Módulo completo: {D.mapping.embed}")
        print(f"    Shape del peso (weight): {D.mapping.embed.weight.shape}")
        if hasattr(D.mapping.embed, "bias") and D.mapping.embed.bias is not None:
            print(f"    Shape del bias:          {D.mapping.embed.bias.shape}")
    else:
        print("  D no tiene embedding condicional (c_dim == 0).")
    print()

    print("=" * 70)
    print("VERIFICACIÓN")
    print("=" * 70)
    embed_weight = G.mapping.embed.weight
    dims = list(embed_weight.shape)
    if G.c_dim in dims:
        idx = dims.index(G.c_dim)
        print(f"  El eje {idx} del peso del embedding tiene tamaño {G.c_dim},")
        print(f"  que coincide con c_dim (número de clases).")
        print(f"  → Cambiar c_dim modifica la forma de esta matriz.")
        print(f"  → Un checkpoint entrenado con c_dim={G.c_dim} NO puede")
        print(f"    cargarse en un modelo con c_dim distinto sin reinicializar")
        print(f"    esta capa.")
    else:
        print(f"  ⚠ Advertencia: c_dim={G.c_dim} no aparece en la shape "
              f"{tuple(dims)} del embedding. Revisar el código.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python inspect_pkl.py <ruta_al_pkl>")
        sys.exit(1)
    main(sys.argv[1])