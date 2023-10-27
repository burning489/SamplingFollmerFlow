import numpy as np

import ot


def w2(
    x_s,
    x_t,
    metric="euclidean",
    *args, **kwargs
):
    M = ot.dist(x_s, x_t, metric=metric)
    return ot.emd2([], [], M, *args, **kwargs)
    


def mmd(x_s, x_t):
    n, m = x_s.shape[0], x_t.shape[0]
    term1 = x_s @ x_s.T
    np.fill_diagonal(term1, 0)
    term2 = x_t @ x_t.T
    np.fill_diagonal(term2, 0)
    term3 = 2 * (x_s @ x_t.T).mean()
    return term1.sum() / (n * (n - 1)) + term2.sum() / (m * (m - 1)) - term3
