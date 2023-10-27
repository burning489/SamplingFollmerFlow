import sys
import os
import time
from math import floor, ceil
import logging
from contextlib import contextmanager
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import numpy as np


@contextmanager
def my_logging(header="", handler=None, level=logging.INFO):
    """Personalized logging decorator

    Args:
        level (logging.level): level of logger
        header (str): header for each logging record
        handler (logging.handler): specify whether to flush to stdout or redirect to log file
    """
    logging.basicConfig()
    logger = logging.getLogger()

    if logger.hasHandlers():
        logger.handlers.clear()

    if handler:
        logger.addHandler(handler)
    else:
        logger.addHandler(logging.StreamHandler(sys.stdout))

    formatter = logging.Formatter(
        fmt=f"[{header}] " + "%(asctime)s %(message)s", datefmt="%y-%m-%d.%H:%M:%S"
    )
    logger.handlers[0].setFormatter(formatter)
    old_level = logger.getEffectiveLevel()
    logger.setLevel(level)
    logger.propagate = True

    try:
        yield logger
    finally:
        logger.setLevel(old_level)


def gen_grid(time_span, time_steps, grid_type, *args, **kwargs):
    """Generate time grid on `time_span`.

    Args:
        time_span (array-like): containing :math:`t_0, t_1`
        time_steps (int): number of intervals `time_span` will be discretized into
        grid_type (str): "uniform" or "ununiform"

            if "uniform", :math:`\Delta t` will be constant

            if "ununiform", transform :math:`f(T) = 1 - \exp(-T)` will be applied to the uniform grid

    Returns:
        array-like: of shape ( `time_steps` +1, )
            time grid ranging from :math:`t_0` to :math:`t_1`

    """
    if grid_type == "uniform":
        return np.linspace(time_span[0], time_span[1], time_steps + 1)
    elif grid_type == "ununiform":
        tmp = np.linspace(time_span[0], time_span[1], time_steps + 1)
        return 1 - np.exp(-tmp)
    else:
        raise ValueError("unkonwn grid type")


def format_sec(seconds):
    """Reformat seconds to readable formats.

    Args:
        seconds (float): number of seconds

    Returns:
        str: in format %H:%M:%S, where H, M and S represents hour, minite and seconds, respectively

    """
    hour = floor(seconds / 3600)
    minute = floor((seconds - hour * 3600) / 60)
    second = floor(seconds - hour * 3600 - minute * 60)
    return f"{hour:02}:{minute:02}:{second:02}"


def eta(elapsed: float, current: int, stop: int):
    """Compute estimated time arrival from time elapsed and iteration passed.

    Args:
        elapsed (float): seconds elapsed
        current (int): current iteration number
        stop (int): total iteration number

    Returns:
        float: estimated remaining seconds

    """
    return elapsed * (stop / current - 1)


def pbar(stop, update_freq=1):
    """Personalied progress bar. Modifiled from `range`.

    Args:
        stop (int): total iterations
        updated_freq (int): frequency of flushing progress information to root logger

    Yield:
        int: current iteration number

    """

    tic = time.perf_counter()
    logger = logging.getLogger()
    update_freq = max(update_freq, 1)
    for i in range(stop):
        if (i + 1) % update_freq == 0:
            toc = time.perf_counter()
            logger.info(
                f"progress: {(i+1)/stop*100:05.2f}% [{i + 1}/{stop}], elapsed:{format_sec(toc - tic)}, "
                f"eta:{format_sec(eta(toc - tic, i + 1, stop))}"
            )
        yield i


def sample_gm1d_truth(target, nsample):
    weights = target.weights
    kap = weights.shape[0]
    dim = target.dimension
    mean = target.mean_array
    var = target.var_array

    choices = np.random.choice(
        np.arange(0, kap, 1), size=(nsample,), p=weights)
    unique, counts = np.unique(choices, return_counts=True)
    ret = np.empty((nsample, dim))
    start = 0
    for i, n in zip(unique, counts):
        ret[start: start + n, ...] = np.random.normal(
            loc=mean[i], scale=np.sqrt(var[i]), size=(n, 1)
        )
        start += n
    return ret


def sample_gmnd_truth(target, nsample):
    weights = target.weights
    kap = weights.shape[0]
    dim = target.dimension
    mean = target.mean_array
    cov = target.cov_array

    choices = np.random.choice(
        np.arange(0, kap, 1), size=(nsample,), p=weights)
    unique, counts = np.unique(choices, return_counts=True)
    ret = np.empty((nsample, dim))
    start = 0
    for i, n in zip(unique, counts):
        ret[start: start + n, ...] = np.random.multivariate_normal(
            mean=mean[i], cov=cov[i], size=(n,)
        )
        start += n
    return ret


def get_mc_scheduler_fn(mc_scheduler):
    if mc_scheduler == "static":
        def mc_scheduler_fn(n_mc, t): return n_mc
    elif mc_scheduler == "dynamic":
        def mc_scheduler_fn(n_mc, t): return ceil(n_mc / (t + 1))
    else:
        mc_scheduler_fn = None
    return mc_scheduler_fn


def get_velocity_fn(target, velocity_form):
    if velocity_form == "closed":
        velocity_fn = target.velocity_closed
    elif velocity_form == "mc":
        velocity_fn = target.velocity_mc_stable
    else:
        raise ValueError(f"Unknown velocity form {velocity_form}")
    return velocity_fn


def get_time_grid(time_span, time_steps, uniform, *args, **kwargs):
    """Generate time grid on `time_span`.

    Args:
        time_span (array-like): containing :math:`t_0, t_1`
        time_steps (int): number of intervals `time_span` will be discretized into
        grid_type (str): "uniform" or "ununiform"

            if "uniform", :math:`\Delta t` will be constant

            if "ununiform", transform :math:`f(T) = 1 - \exp(-T)` will be applied to the uniform grid

    Returns:
        array-like: of shape ( `time_steps` +1, )
            time grid ranging from :math:`t_0` to :math:`t_1`

    """
    if uniform:
        return np.linspace(time_span[0], time_span[1], time_steps + 1)
    else:
        tmp = np.linspace(time_span[0], time_span[1], time_steps + 1)
        return 1 - np.exp(-tmp)


def prior_sampling(dimension, mu, sigma, nsample):
    p_mean = np.ones(dimension) * mu
    p_cov = np.eye(dimension) * sigma**2
    return np.random.multivariate_normal(p_mean, p_cov, size=(nsample,))


def mkdir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass


# reference: https://stackoverflow.com/questions/35042255/how-to-plot-multiple-seaborn-jointplot-in-subplot
class SeabornFig2Grid():

    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())

def plot2d(x, lim):
    g = sns.JointGrid(x=x[:, 0], y=x[:, 1], height=6)
    g = g.plot_joint(plt.scatter, s=0.05, alpha=0.5)
    g = g.plot_marginals(sns.kdeplot, bw_adjust=0.15)
    g.set_axis_labels(xlabel='', ylabel='')
    g.figure.axes[0].grid(alpha=0.5, linestyle='-.')
    g.figure.axes[0].set_xlim([-lim, lim])
    g.figure.axes[0].set_ylim([-lim, lim])
    g.figure.axes[0].set_xticks([])
    g.figure.axes[0].set_yticks([])
    return g