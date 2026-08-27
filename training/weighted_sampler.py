"""WeightedInfiniteSampler — muestreo ponderado por clase para StyleGAN2-ADA.

Sustituye a `torch_utils.misc.InfiniteSampler` (muestreo uniforme) para compensar
el desbalance de clases de HAM10000 sin duplicar imagenes en disco.

Convencion de ponderacion (decision del 14/07/2026):
    p_i proporcional a n_i^alpha,  donde n_i = numero de imagenes de la clase de la imagen i.
        alpha =  0   -> uniforme por imagen (comportamiento original, SIN ponderar)
        alpha = -1   -> rebalanceo completo (todas las clases equiprobables)
        -1 < alpha < 0 -> rebalanceo parcial ("temperatura")

El calculo de clases se hace sobre `dataset._raw_idx`, de modo que respeta
`max_size` (subconjunto) y `xflip` (espejado) igual que el pipeline original.
"""
import numpy as np
import torch


def _per_index_classes(dataset):
    """Devuelve la clase (int) de cada indice del dataset, respetando _raw_idx."""
    labels = np.asarray(dataset._get_raw_labels())
    if labels.ndim > 1:                      # one-hot -> indice de clase
        labels = np.argmax(labels, axis=1)
    return labels[dataset._raw_idx].astype(np.int64)


def class_probabilities(dataset, alpha):
    """Probabilidad de muestreo por indice y conteos por clase. Util para logging/test."""
    cls = _per_index_classes(dataset)
    counts = np.bincount(cls)
    w = counts[cls].astype(np.float64) ** float(alpha)   # n_i^alpha
    p = w / w.sum()
    return p, cls, counts


class WeightedInfiniteSampler(torch.utils.data.Sampler):
    """Sampler infinito con muestreo i.i.d. ponderado por clase (con reemplazo)."""

    def __init__(self, dataset, alpha=0.0, rank=0, num_replicas=1, seed=0):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        super().__init__(dataset)
        p, _, _ = class_probabilities(dataset, alpha)
        self._cdf = np.cumsum(p)
        self._cdf[-1] = 1.0                  # blindaje frente a error de redondeo
        self._n = len(dataset)
        self._rank = rank
        self._num_replicas = num_replicas
        self._seed = seed

    def __iter__(self):
        rnd = np.random.RandomState(self._seed)
        idx = 0
        while True:
            i = int(np.searchsorted(self._cdf, rnd.random_sample(), side="right"))
            if i >= self._n:
                i = self._n - 1
            if idx % self._num_replicas == self._rank:
                yield i
            idx += 1
