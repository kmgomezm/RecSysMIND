"""Bootstrap pareado por impresión para comparar dos modelos."""
import numpy as np


def paired_bootstrap(a, b, n_boot=2000, seed=42):
    """a, b: métrica por impresión de dos modelos, alineadas por impresión.
    Devuelve la diferencia media, el IC 95 % y las medias bootstrap."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(d)
    medias = np.empty(n_boot)
    for k in range(n_boot):
        medias[k] = d[rng.integers(0, n, n)].mean()
    return d.mean(), np.percentile(medias, [2.5, 97.5]), medias
    