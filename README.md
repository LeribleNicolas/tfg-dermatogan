# DermatoGAN — Generación sintética de imágenes dermatoscópicas con StyleGAN2-ADA

Modelo generativo condicional (StyleGAN2-ADA) para la síntesis de imágenes
dermatoscópicas de las **siete clases de lesiones cutáneas** del conjunto
HAM10000, a resolución 256×256. Desarrollado como Trabajo de Fin de Grado de
Ingeniería de Software (Universidad de Málaga).

- **Autor:** Nicolás Lerible García
- **Tutor:** Enrique Domínguez Merino
- **Arquitectura:** StyleGAN2-ADA condicional (implementación oficial de NVIDIA, adaptada)
- **Dataset:** HAM10000 (Tschandl et al., 2018) + máscaras de segmentación (Tschandl et al., 2019)

Este repositorio contiene **únicamente lo imprescindible para reproducir la
funcionalidad** del proyecto: el código de entrenamiento, generación y
evaluación, y los scripts propios de preprocesado y análisis. No incluye la
memoria, las fichas bibliográficas ni los materiales de trabajo, que se
mantienen aparte.

---

## Aviso de licencia y atribución

Este repositorio integra y adapta el código oficial de NVIDIA
[`stylegan2-ada-pytorch`](https://github.com/NVlabs/stylegan2-ada-pytorch),
distribuido bajo la **NVIDIA Source Code License** (uso no comercial). Se
conserva el fichero `LICENSE.txt` original y sus avisos de copyright.

- El **código base** (`train.py`, `generate.py`, `calc_metrics.py`,
  `dataset_tool.py`, `legacy.py`, `dnnlib/`, `torch_utils/`, `training/`,
  `metrics/`) pertenece a NVIDIA y se ha modificado en puntos concretos
  (ver «Modificaciones sobre el código de NVIDIA»).
- Los **scripts propios** (`scripts/`) y el módulo `training/weighted_sampler.py`
  son obra del autor del TFG.
- El **conjunto de datos HAM10000** se distribuye bajo licencia **CC BY-NC 4.0**;
  **no se incluye** en este repositorio y debe descargarse por separado.

El uso previsto es **académico y de investigación**, sin fines comerciales.

---

## Estructura del repositorio

```
tfg-dermatogan/
├── train.py                 # Entrenamiento (modificado: --sampling-alpha, fix JIT Ampere)
├── generate.py              # Generación de imágenes desde un checkpoint
├── calc_metrics.py          # Métricas oficiales FID/KID/IS/PPL (modificado: fix JIT Ampere)
├── dataset_tool.py          # Empaquetado de imágenes a formato StyleGAN2-ADA
├── legacy.py                # Carga de checkpoints .pkl
├── dnnlib/  torch_utils/    # Utilidades del framework (NVIDIA)
├── training/
│   ├── training_loop.py     # Bucle de entrenamiento (modificado: enchufa el sampler ponderado)
│   ├── weighted_sampler.py  # [PROPIO] WeightedInfiniteSampler (muestreo por clase n_i^alpha)
│   └── ...                  # networks, dataset, loss, augment (NVIDIA)
├── metrics/
│   ├── perceptual_path_length.py  # (modificado: ruta de vgg16 por variable de entorno)
│   └── ...                  # FID, KID, IS, metric_utils (NVIDIA)
├── scripts/                 # [PROPIOS] Preprocesado, evaluación por clase y figuras
│   ├── split_ham10000.py            # F3.1 · split agrupado por lesion_id
│   ├── build_train_source.py        # F3.2 · carpeta-fuente + dataset.json (7 clases)
│   ├── preprocess_masks.py          # F3.3 · máscaras alineadas 1:1
│   ├── verify_masks.py              # F3.5 · verificación de alineación
│   ├── calc_metrics_per_class.py    # FID + KID por clase
│   ├── eval_checkpoint.py           # FID + KID por clase + memorización (una pasada)
│   ├── nn_memorization.py           # Análisis de vecino más cercano (memorización)
│   ├── is_torchvision.py            # Inception Score alternativo (torchvision)
│   ├── plot_training_stats.py       # Curvas r_t / p / pérdidas G-D
│   ├── plot_diag_curves.py          # Curva de FID + curvas de entrenamiento
│   ├── inspect_pkl.py               # Inspección de la arquitectura de un .pkl
│   └── test_weighted_sampler.py     # Test del sampler ponderado
├── notebooks/
│   └── train_colab.ipynb    # Notebook de entrenamiento en Colab/Kaggle (entorno fijado)
├── requirements.txt
├── LICENSE.txt              # Licencia original de NVIDIA
└── README.md
```

Herramientas del repositorio oficial no usadas en este proyecto (`projector.py`,
`style_mixing.py`, `docs/`, `Dockerfile`) se han omitido deliberadamente; están
disponibles en el repositorio de NVIDIA enlazado arriba.

---

## Requisitos

**Hardware.** GPU NVIDIA con arquitectura Ampere o compatible con CUDA 11.1.
El proyecto se desarrolló en una **RTX 3060 (12 GB)**; a 256×256 con `batch=16`
el consumo ronda los ~5 GB.

**Software.** El código de NVIDIA (2020) exige un entorno **fijado**; las
versiones modernas de PyTorch/CUDA/Python **no son compatibles**.

- **Linux** (probado en WSL2 · Ubuntu 22.04). En Windows nativo la compilación de
  los kernels CUDA personalizados es más frágil.
- **Python 3.9**
- **PyTorch 1.9.1+cu111** y **Torchvision 0.10.1+cu111** (mínimo con soporte pleno
  para Ampere / compute capability 8.6)
- **CUDA Toolkit 11.x** con compilador `nvcc` (para compilar `bias_act` y `upfirdn2d`)

---

## Instalación

```bash
# 1) Clonar
git clone <url-de-este-repositorio> tfg-dermatogan
cd tfg-dermatogan

# 2) Entorno virtual con Python 3.9
python3.9 -m venv ../venv
source ../venv/bin/activate

# 3) PyTorch fijado (rueda cu111) + dependencias
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 \
    -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

Verificación rápida de la GPU:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

La primera vez que se ejecute `train.py`/`generate.py`, StyleGAN2-ADA compila
los kernels CUDA con `ninja` (tarda un par de minutos; se cachea después).

### Variables de entorno (opcional pero recomendado)

La CDN de NVIDIA que sirve los detectores de las métricas puede devolver **403**.
Para trabajar sin conexión o evitar ese fallo, descarga los ficheros una vez y
apunta a ellos con estas variables (si no se definen, se usa la URL oficial):

```bash
export SG2_INCEPTION=/ruta/local/inception-2015-12-05.pt   # detector FID/KID/IS
export SG2_VGG16=/ruta/local/vgg16.pt                       # LPIPS para PPL
# export SG2_REPO=/ruta/a/tfg-dermatogan                    # sólo si ejecutas los scripts desde fuera del repo
```

---

## Datos

El conjunto de datos **no se versiona**. Descárgalo y colócalo bajo `datos/`
(fuera del control de versiones, ya excluido en `.gitignore`):

- **Imágenes:** HAM10000 — 10.015 `.jpg` + `HAM10000_metadata.csv`.
- **Máscaras:** [`tschandl/ham10000-lesion-segmentations`](https://www.kaggle.com/datasets/tschandl/ham10000-lesion-segmentations)
  — 10.015 `.png` (`*_segmentation.png`).

Licencias: HAM10000 y sus máscaras son **CC BY-NC 4.0** (uso no comercial, con
atribución). Cítalas en cualquier trabajo derivado.

---

## Uso

Las siete clases se indexan en **orden alfabético** (así las asigna
`dataset_tool.py`), tal y como recoge `datos/splits/class_index.json`:

| Índice | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|--------|-----|-----|-----|-----|-----|-----|------|
| Clase  | akiec | bcc | bkl | df | mel | nv | vasc |

### 1. Preprocesado de datos (fase F3)

```bash
# 1.1 · Split train/val/test agrupado por lesión (evita fuga entre particiones)
python scripts/split_ham10000.py \
    --metadata datos/HAM10000_metadata.csv \
    --out      datos/splits
# -> datos/splits/{train,val,test}.csv  +  class_index.json

# 1.2 · Carpeta-fuente de train (symlinks) + dataset.json con las etiquetas
python scripts/build_train_source.py \
    --train-csv datos/splits/train.csv \
    --images    datos/HAM10000_images_part_1 datos/HAM10000_images_part_2 \
    --out       datos/train_src

# 1.3 · Empaquetado a formato StyleGAN2-ADA (recorte central 256×256)
python dataset_tool.py \
    --source=datos/train_src \
    --dest=datos/ham7_train_256.zip \
    --transform=center-crop --width=256 --height=256

# 1.4 · Máscaras de segmentación alineadas 1:1 (opcional, para análisis en F8)
python scripts/preprocess_masks.py \
    --train-src datos/train_src \
    --masks     datos/masks \
    --dest      datos/ham7_train_masks_256.zip

# 1.5 · Verificación de alineación imagen<->máscara
python scripts/verify_masks.py \
    --image-zip  datos/ham7_train_256.zip \
    --mask-zip   datos/ham7_train_masks_256.zip \
    --mask-index datos/ham7_train_masks_256.index.csv \
    --out        datos/verify_out
```

### 2. Entrenamiento (fases F4–F7)

Configuración **definitiva** del TFG (modelo condicional, 7 clases, muestreo
ponderado `α = −0.5`):

```bash
python train.py \
    --outdir=experimentos/runs \
    --data=datos/ham7_train_256.zip \
    --gpus=1 \
    --cond=1 \
    --mirror=1 \
    --cfg=auto \
    --aug=ada --target=0.6 --augpipe=bgc \
    --sampling-alpha=-0.5 \
    --metrics=fid50k_full \
    --seed=42 \
    --snap=50 \
    --kimg=5000
```

`--sampling-alpha` es la extensión propia (`0` = uniforme, `−1` = rebalanceo
completo; `−0.5` = rebalanceo parcial). El FID global satura hacia
~4000 kimg; el checkpoint de referencia es `network-snapshot-004000.pkl`.

Reanudar desde un snapshot: añadir `--resume=experimentos/runs/<run>/network-snapshot-XXXXXX.pkl`.

**Monitorización** (figuras reproducibles a partir de `stats.jsonl`, sin TensorBoard):

```bash
python scripts/plot_training_stats.py --run experimentos/runs/<run> --target 0.6
python scripts/plot_diag_curves.py    --run-dir experimentos/runs/<run> --target 0.6
```

### 3. Generación de imágenes

Generar 16 imágenes de una clase (p. ej. melanoma, índice 4) desde un checkpoint:

```bash
python generate.py \
    --network=experimentos/runs/<run>/network-snapshot-004000.pkl \
    --seeds=0-15 \
    --class=4 \
    --trunc=1 \
    --outdir=out/mel
```

`--trunc` es el *truncation psi* (1 = máxima variedad; valores <1 = más
calidad y menos diversidad). `--class` toma un índice de 0 a 6 según la tabla
anterior.

### 4. Evaluación (fase F8)

```bash
# 4.1 · Métricas globales oficiales
python calc_metrics.py \
    --metrics=fid50k_full,kid50k_full,is50k \
    --network=experimentos/runs/<run>/network-snapshot-004000.pkl \
    --data=datos/ham7_train_256.zip --gpus=1

# PPL (requiere vgg16; usa SG2_VGG16 si la CDN da 403)
python calc_metrics.py --metrics=ppl2_wend \
    --network=.../network-snapshot-004000.pkl \
    --data=datos/ham7_train_256.zip --gpus=1

# 4.2 · FID + KID por clase (clave para las clases minoritarias)
python scripts/calc_metrics_per_class.py \
    --network=.../network-snapshot-004000.pkl \
    --data=datos/ham7_train_256.zip \
    --class-index=datos/splits/class_index.json \
    --outfile=experimentos/per_class_004000.json

# 4.3 · FID+KID por clase + chequeo de memorización en una sola pasada
python scripts/eval_checkpoint.py \
    --network=.../network-snapshot-004000.pkl \
    --data=datos/ham7_train_256.zip \
    --class-index=datos/splits/class_index.json \
    --outdir=experimentos/eval_004000

# 4.4 · Memorización (vecino más cercano) sobre las clases minoritarias
python scripts/nn_memorization.py \
    --network=.../network-snapshot-004000.pkl \
    --data=datos/ham7_train_256.zip \
    --class-index=datos/splits/class_index.json \
    --classes=df,vasc,akiec \
    --outdir=experimentos/nn_memorization_004000
```

---

## Modificaciones sobre el código de NVIDIA

Para trazabilidad, estos son los únicos cambios respecto al repositorio oficial:

1. **`train.py` + `training/training_loop.py`** — nuevo argumento `--sampling-alpha`
   que enchufa el `WeightedInfiniteSampler` propio (muestreo ponderado por clase,
   `p_i ∝ n_i^α`) en sustitución del `InfiniteSampler` uniforme.
2. **`training/weighted_sampler.py`** — módulo nuevo con el sampler ponderado.
3. **`train.py`, `calc_metrics.py`** — se desactiva el *fuser* TensorExpr del JIT
   (`torch._C._jit_set_texpr_fuser_enabled(False)` y afines) para evitar el error
   `MALFORMED INPUT: lanes dont match` del detector Inception en TorchScript sobre
   GPUs Ampere con torch 1.9.
4. **`metrics/perceptual_path_length.py`** — la ruta del modelo `vgg16` se lee de la
   variable de entorno `SG2_VGG16` (con la URL oficial como valor por defecto), para
   sortear el 403 de la CDN de NVIDIA.
5. **`scripts/`** — los detectores Inception/VGG y el directorio del repo se
   resuelven mediante variables de entorno (`SG2_INCEPTION`, `SG2_VGG16`, `SG2_REPO`)
   o autodetección, de modo que los scripts son portables.

---

## Notas de reproducibilidad

- **Semillas fijas** (`--seed`) en entrenamiento, generación y evaluación.
- **GPUs Ampere:** el parche del *fuser* JIT es obligatorio; sin él, las métricas
  que usan el detector Inception fallan en TorchScript.
- **`--resume`** carga sólo los pesos (no restaura el optimizador Adam, `cur_nimg`
  ni la `p` de ADA): el contador de kimg reinicia desde 0 y se crea una carpeta de
  run nueva.
- Las clases minoritarias (`akiec`, `vasc`, `df`) presentan menor calidad por
  escasez de datos reales, no por el muestreo; es una **limitación conocida** del
  conjunto de datos.

---

## Créditos

- Karras, T., Aittala, M., Hellsten, J., Laine, S., Lehtinen, J., & Aila, T. (2020).
  *Training generative adversarial networks with limited data.* NeurIPS. (StyleGAN2-ADA)
- Tschandl, P., Rosendahl, C., & Kittler, H. (2018). *The HAM10000 dataset.*
  Scientific Data, 5, 180161. https://doi.org/10.1038/sdata.2018.161
- Tschandl, P., Sinz, C., & Kittler, H. (2019). *Domain-specific classification-pretrained
  fully convolutional network encoders for skin lesion segmentation.* Computers in
  Biology and Medicine, 104, 111-116. https://doi.org/10.1016/j.compbiomed.2018.11.010
