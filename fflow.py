from functools import partial

import numpy as np
import torch

from misc import pbar


def fflow(velocity_fn, x, time_grid, n_mc=None, mc_scheduler_fn=None):
    """Föllmer Flow prototype.

    Args:
        velocity (callable): Föllmer velocity at :math:`(x, t)`

        x (array-like): of shape (1, dim), initial value of ODE flow

        time_grid (array-like): of shape ( `time_steps` +1, )
            time grid ranging from :math:`t_0` to :math:`t_1`

    Returns:
        array-like: of shape (1, dim), Föllmer sample
    """
    for t in range(time_grid.size - 1):
        if n_mc:
            velocity_fn = partial(velocity_fn, n_mc=mc_scheduler_fn(n_mc, t))
        dt = time_grid[t + 1] - time_grid[t]
        x = x + velocity_fn(x, time_grid[t]) * dt
    return x


def fflow_sampler(
    velocity_fn,
    x0,
    time_grid,
    parallel=True,
    n_job=-1,
    n_mc=None,
    mc_scheduler_fn=None,
    *args,
    **kwargs,
):
    """Föllmer flow with parallelization.

    By default, use multithreading to accelerate, thread number set to maximum capacity.

    Args:
        velocity (callable): Föllmer velocity at :math:`(x, t)`, may be a python function or `torch.nn.Module`

        x0 (array-like): of shape (nsample, dim), initial value of ODE flow

        time_steps (int): number of time discretization intervals

        time_span (array-like): containing :math:`t_0, t_1`

        grid_type (str): "uniform" or "ununiform"

            if "uniform", :math:`\Delta t` will be constant

            if "ununiform", transform :math:`f(T) = 1 - \exp(-T)` will be applied to the uniform grid

        parallel (bool): whether call all `joblib` for multithreading

        n_job (int): number of threads, -1 means maximum possible

    Returns:
        array-like: of shape (nsample, dim), same type as `x0`,  Föllmer flow samples

    Raises:
        TypeError: type of `x0` must be `numpy.ndarray` or `torch.Tensor`
    """
    nsample = x0.shape[0]
    my_pbar = pbar(stop=nsample, update_freq=nsample // 100)
    if parallel:
        from joblib import Parallel, delayed

        xt = Parallel(n_jobs=n_job)(
            delayed(fflow)(velocity_fn,
                           x0[i: i + 1, ...], time_grid, n_mc, mc_scheduler_fn)
            for i in my_pbar
        )
        xt = np.squeeze(np.array(xt), axis=-2)
    else:
        if isinstance(x0, np.ndarray):
            xt = np.empty_like(x0)
        elif isinstance(x0, torch.Tensor):
            xt = torch.empty_like(x0)
        else:
            raise TypeError(
                f"invalid type of x0: {type(x0)}, expected np.ndarrry or torch.Tensor"
            )
        for i in my_pbar:
            xt[i: i + 1] = fflow(
                velocity_fn, x0[i: i +
                                1, ...], time_grid, n_mc, mc_scheduler_fn
            )
    return xt
