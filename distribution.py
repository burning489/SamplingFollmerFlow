import numpy as np
from scipy.special import logsumexp


_DISTRIBUTIONS = {}


def register_distribution(cls=None, *, name=None):
    """A decorator for registering distribution classes."""

    def _register(cls):
        local_name = cls.__name__ if name is None else name
        if local_name in _DISTRIBUTIONS:
            raise ValueError(
                f"Already registered class with name: {local_name}")
        _DISTRIBUTIONS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def _mc_stable(z, ln_g):
    """Monte Carlo simulation of Föllmer drift/velocity with logsumexp transformation.

    Args:
        z (array-like): of shape (n_mc, dimension), random variable drawn from standard Gaussian
        ln_g (array-like): of shape (n_mc, 1),

            .. math::

                \\text{drift:} \quad & \log (g(x + \sqrt{1 - t} z)),

                \\text{velocity:} \quad & \log (g(tx + \sqrt{1 - t^2} z)),

            where :math:`g` is the scaled Radon-Nikodym derivative of :math:`{\\nu}` w.r.t. :math:`{\\gamma_d}`.

    Returns:
        array-like: MC estimation of Föllmer drift/velocity
    """
    ln_de = logsumexp(ln_g, axis=0)
    z_plus = np.maximum(z, 0)
    z_minus = np.maximum(-z, 0)
    with np.errstate(divide="ignore"):
        ln_nu_plus = logsumexp(np.log(z_plus) + ln_g, axis=0)
        ln_nu_minus = logsumexp(np.log(z_minus) + ln_g, axis=0)
    ln_r_plus = ln_nu_plus - ln_de
    ln_r_minus = ln_nu_minus - ln_de
    r_hat = np.exp(ln_r_plus) - np.exp(ln_r_minus)
    return r_hat


class BaseDistribution:
    """Abstract base class for distributions.

    Attributes:
            dimension (int): dimension of distribution

            velocity_forms (dict): Föllmer velocity in different forms

            drift_forms (dict): Föllmer drift in different forms

    Args:
            dimension (int): dimension of distribution

    """

    def __init__(self, dimension, *args, **kwargs):
        pass

    def potential(self, x):
        """Abstract base method."""
        raise NotImplementedError

    def potential_grad(self, x):
        """Abstract base method."""
        raise NotImplementedError

    def scaled_density(self, x):
        """Abstract base method."""
        raise NotImplementedError

    def scaled_ratio(self, x):
        """Abstract base method."""
        raise NotImplementedError

    def ln_scaled_ratio(self, x):
        """Abstract base method."""
        raise NotImplementedError

    def velocity_closed(self, x, t, *args, **kwargs):
        """Abstract base method."""
        raise NotImplementedError

    def velocity_mc_stable(self, x, t, n_mc, preconditioner=None, *args, **kwargs):
        """Monte Carlo Föllmer velocity with logsumexp trick for numerical stability.

                Parallelization should be done outside this method.

        Args:
                x (array-like): of shape (1, d)

                t (float): time

                n_mc (int): number of Monte Carlo simulations

        Returns:
                array-like: of shape (1, d), Föllmer drift at :math:`(x, t)`
        """
        if t == 0:
            return np.atleast_2d(self.mean)
        else:
            if preconditioner:
                # (d, )
                p_mu = preconditioner["offset"] * np.ones(self.dimension)
                # (d, d)
                A = preconditioner["deviation"] * np.eye(self.dimension)
            else:
                p_mu = np.zeros(self.dimension)
                A = np.eye(self.dimension)
            z = np.random.randn(n_mc, self.dimension)
            ln_g = self.ln_scaled_ratio(
                t * x + (1 - t) * p_mu + np.sqrt(1 - t**2) *
                z @ A, preconditioner
            )
            r_hat = _mc_stable(z, ln_g)
            return r_hat @ A.T / np.sqrt(1 - t**2)


@register_distribution(name="GM1d")
class GaussianMixture1d(BaseDistribution):
    """1-dimensional Gaussian mixture distribition.

    Attributes:
        dimension (int): dimension of distribution, 1 in this class
        theta (array-like): of shape (k, ), weight of each mode
        mean_array (array-like): of shape (k, ), mean of each mode
        var_array(array-like): of shape (k, ), variance of mode
        n_mode (int): number of modes
        mean (float): mean of distribution, weight sum of mean_array

    Args:
        dimension (int): dimension of distribution
        theta (array-like): of shape (k, ), weight of each mode
        mean_array (array-like): of shape (k, ), mean of each mode
        var_array(array-like): of shape (k, ), variance of mode
    """

    def __init__(self, dimension, weights, mean_array, var_array, *args, **kwargs):
        self.dimension = dimension
        self.weights = np.array(weights)
        self.n_mode = len(self.weights)
        self.mean_array = np.array(mean_array)
        self.var_array = np.array(var_array)
        self.mean = self.mean_array @ self.weights

    def potential(self, x, *args, **kwargs):
        """Potential function :math:`U(x)`.

        Args:
            x (array-like): of shape (n, 1), n observations of 1-d data

        Returns:
            array-like: of shape (n, 1), potential at x
        """
        # (n, k)
        offset = x - self.mean_array
        # (n, k)
        expo = -0.5 * offset**2 / self.var_array
        # (k, )
        weight = self.weights / np.sqrt(self.var_array) / np.sqrt(2 * np.pi)
        # (n, 1)
        return -logsumexp(a=expo, axis=-1, b=weight, keepdims=True)

    def potential_grad(self, x, *args, **kwargs):
        """Gradient of potential function :math:`{\\nabla}U(x)`.

        Args:
            x (array-like): of shape (n, 1), n observations of 1-d data

        Returns:
            array-like: of shape (n, 1), gradient of potential at x
        """
        # (n, k)
        offset = x - self.mean_array
        # (n, k)
        expo = -0.5 * offset**2 / self.var_array
        # (k, )
        weight = self.weights / np.sqrt(self.var_array) / np.sqrt(2 * np.pi)
        # (n, )
        density = np.exp(expo) @ weight
        # (n, k)
        weight = -weight * offset / self.var_array
        # (n, )
        density_grad = np.sum(np.exp(expo) * weight, axis=-1)
        # (n, 1)
        return (-density_grad / density)[..., None]

    def scaled_density(self, x):
        """Scaled density function at x, scaled by 1/np.sqrt(2*np.pi).

        Args:
            x (array-like): of shape (n, 1), n observations of 1-d data

        Returns:
            array-like: of shape (n, 1), scaled density at x
        """
        # (n, k)
        offset = x - self.mean_array
        # (n, k)
        expo = -0.5 * offset**2 / self.var_array
        # (k, )
        weight = self.weights / np.sqrt(self.var_array)
        # (n, 1)
        return np.sum(np.exp(expo) * weight, axis=-1, keepdims=True)

    def scaled_ratio(self, x, preconditioner=None):
        """Scaled Radon-Nikodym derivative function at x, scaled by 1/np.sqrt(2*np.pi)

        Args:
            x (array-like): of shape (n, 1), n observations of 1-d data

        Returns:
            array-like: of shape (n, 1), scaled Radon-Nikodym derivative at x
        """
        if preconditioner:
            p_mu, p_var = preconditioner["offset"], preconditioner["deviation"] ** 2
        else:
            p_mu, p_var = 0.0, 1.0
        # (n, k)
        offset = x - self.mean_array
        # (n, k)
        expo = -0.5 * offset**2 / self.var_array + \
            0.5 * (x - p_mu) ** 2 / p_var
        # (k, )
        weight = self.weights / np.sqrt(self.var_array) * p_var
        return np.sum(np.exp(expo) * weight, axis=-1, keepdims=True)

    def ln_scaled_ratio(self, x, preconditioner=None):
        """Log of scaled Radon-Nikodym derivative function at x.

        Args:
            x (array-like): of shape (n, 1), n observations of 1-d data

        Returns:
            array-like: of shape (n, 1), log of scaled Radon-Nikodym derivative at x
        """
        if preconditioner:
            p_mu, p_var = preconditioner["offset"], preconditioner["deviation"] ** 2
        else:
            p_mu, p_var = 0.0, 1.0
        # (n, k)
        offset = x - self.mean_array
        # (n, k)
        expo = -0.5 * offset**2 / self.var_array + \
            0.5 * (x - p_mu) ** 2 / p_var
        # (k, )
        weight = self.weights / np.sqrt(self.var_array) * p_var
        # (n, 1)
        return logsumexp(a=expo, axis=-1, b=weight, keepdims=True)

    def velocity_closed(self, x, t, preconditioner=None, *args, **kwargs):
        """Closed-form velocity filed for Föllmer flow.

        Args:
            x (array-like): of shape (n, 1), n observations of 1-d data
            t (float): time

        Returns:
            array-like: of shape (n, 1), closed-form Föllmer velocity :math:`V(x, t)`
        """
        if t == 0:
            return np.repeat(np.atleast_2d(self.mean), x.shape[0], axis=0)
        if preconditioner:
            p_mu, p_var = preconditioner["offset"], preconditioner["deviation"] ** 2
        else:
            p_mu, p_var = 0.0, 1.0
        # (k, )
        var = t**2 * self.var_array + (1 - t**2) * p_var
        # (n, k)
        offset = x - t * self.mean_array - (1 - t) * p_mu
        # (n, k)
        p = np.exp(-0.5 * offset**2 / var) / np.sqrt(var)
        # (n, k)
        g = p * self.weights
        # (n, )
        deno = np.sum(g, axis=-1, keepdims=True)
        # (n, )
        nume = np.sum(-offset / var * g, axis=-1, keepdims=True)
        # (n, 1)
        return (x - p_mu + p_var * nume / deno) / t


@register_distribution(name="GMNd")
class GaussianMixtureNd(BaseDistribution):
    """N-dimensional Gaussian mixture distribition.

    Attributes:
        dimension (int): dimension of distribution

        theta (array-like): of shape (k, ), weight of each mode

        mean_array (array-like): of shape (k, d), array of mean of each mode

        var_array (array-like): of shape (k, d, d), array of covariance of each mode

        n_mode (int): number of modes

        mean (array-like): of shape (d, ), mean of distribution, weighted sum of mean_array

        invcov_array (array-like): of shape (k, d, d), array of inverse of covariance of each mode

        detcov_array (array-like): of shape (k, ), array of determinant of covariance of each mode

    Args:
        dimension (int): dimension of distribution

        theta (array-like): of shape (k, ), weight of each mode

        mean_array (array-like): of shape (k, d), array of mean of each mode

        var_array (array-like): of shape (k, d, d), array of covariance of each mode
    """

    def __init__(self, dimension, weights, mean_array, cov_array, *args, **kwargs):
        self.dimension = dimension
        self.weights = np.array(weights)
        self.n_mode = len(self.weights)
        self.mean_array = np.array(mean_array)
        self.cov_array = np.array(cov_array)
        self.invcov_array = np.linalg.inv(self.cov_array)
        self.sqrt_detcov_array = np.sqrt(np.linalg.det(self.cov_array))
        self.mean = self.weights @ self.mean_array

    def potential(self, x, *args, **kwargs):
        """Potential function :math:`U(x)`.

        Args:
            x (array-like): of shape (n, d), n observations of d-dimensional data

        Returns:
            array-like: of shape (n, 1), potential at x
        """
        # (n, k, 1, d)
        offset = np.expand_dims(np.expand_dims(
            x, axis=-2) - self.mean_array, axis=-2)
        # (n, k)
        expo = -0.5 * np.squeeze(
            np.sum(offset @ self.invcov_array * offset, axis=-1), axis=-1
        )
        # (k, )
        weight = (
            self.weights / self.sqrt_detcov_array /
            np.sqrt(2 * np.pi) ** self.dimension
        )
        potential, sgn = logsumexp(
            a=expo, axis=-1, b=weight, keepdims=True, return_sign=True
        )
        return -potential * sgn

    def potential_grad(self, x, *args, **kwargs):
        """Gradient of potential function :math:`{\\nabla}U(x)`.

        Args:
            x (array-like): of shape (n, d), n observations of d-dimensional data

        Returns:
            array-like: of shape (n, d), gradient of potential at x
        """
        # (n, k, 1, d)
        offset = np.expand_dims(np.expand_dims(
            x, axis=-2) - self.mean_array, axis=-2)
        # (n, k, 1, d)
        tmp = offset @ self.invcov_array
        # (n, k)
        expo = np.exp(-0.5 * np.squeeze(np.sum(tmp * offset, axis=-1), axis=-1))
        # (k, )
        weight = (
            self.weights / self.sqrt_detcov_array /
            np.sqrt(2 * np.pi) ** self.dimension
        )
        # (n, )
        density = np.sum(expo * weight, axis=-1)
        # (n, k, d)
        weight = -weight[..., None] * np.squeeze(tmp, axis=-2)
        # (n, d)
        density_grad = np.sum(expo[..., None] * weight, axis=-2)
        return -density_grad / density

    def scaled_density(self, x):
        """Scaled density function at x.

        Args:
            x (array-like): of shape (n, d), n observations of d-dimensional data

        Returns:
            array-like: of shape (n, 1), scaled density at x
        """
        # (n, k, 1, d)
        offset = np.expand_dims(np.expand_dims(
            x, axis=-2) - self.mean_array, axis=-2)
        return np.sum(
            np.exp(
                -0.5
                * np.squeeze(
                    np.sum(offset @ self.invcov_array * offset, axis=-1), axis=-1
                )
            )
            * (self.weights / self.sqrt_detcov_array),
            axis=-1,
            keepdims=True,
        )

    def scaled_ratio(self, x, preconditioner=None):
        """Scaled Radon-Nikodym derivative function at x.

        Args:
            x (array-like): of shape (n, d), n observations of d-dimensional data

        Returns:
            array-like: of shape (n, 1), scaled Radon-Nikodym derivative at x
        """
        if preconditioner:
            # (d, )
            p_mu = preconditioner["offset"] * np.ones(self.dimension)
            p_std = np.sqrt(preconditioner["deviation"])
            # (d, d)
            A = preconditioner["deviation"] * np.eye(self.dimension)
        else:
            p_mu = np.zeros(self.dimension)
            p_std = 1.0
            A = np.eye(self.dimension)
        # (n, k, 1, d)
        offset = np.expand_dims(np.expand_dims(
            x, axis=-2) - self.mean_array, axis=-2)
        return np.sum(
            np.exp(
                -0.5
                * np.squeeze(
                    np.sum(offset @ self.invcov_array * offset, axis=-1), axis=-1
                )
                + 0.5 * np.sum((x - p_mu) @ A * (x - p_mu),
                               axis=-1, keepdims=True)
            )
            * (self.weights / self.sqrt_detcov_array * p_std),
            axis=-1,
            keepdims=True,
        )

    def ln_scaled_ratio(self, x, preconditioner=None):
        """Log of scaled Radon-Nikodym derivative function at x.

        Args:
            x (array-like): of shape (n, d), n observations of d-dimensional data

        Returns:
            array-like: of shape (n, 1), log of scaled Radon-Nikodym derivative at x
        """
        if preconditioner:
            # (d, )
            p_mu = preconditioner["offset"] * np.ones(self.dimension)
            p_std = np.sqrt(preconditioner["deviation"])
            # (d, d)
            A = preconditioner["deviation"] * np.eye(self.dimension)
        else:
            p_mu = np.zeros(self.dimension)
            p_std = 1.0
            A = np.eye(self.dimension)
        # (n, k, 1, d)
        offset = np.expand_dims(np.expand_dims(
            x, axis=-2) - self.mean_array, axis=-2)
        # (n, k)
        expo = -0.5 * np.squeeze(
            np.sum(offset @ self.invcov_array * offset, axis=-1), axis=-1
        ) + 0.5 * np.sum((x - p_mu) @ A * (x - p_mu), axis=-1, keepdims=True)
        lse, sgn = logsumexp(
            a=expo,
            axis=-1,
            b=self.weights / self.sqrt_detcov_array * p_std,
            keepdims=True,
            return_sign=True,
        )
        return lse * sgn

    def velocity_closed(self, x, t, preconditioner=None, *args, **kwargs):
        """Preconditioned Föllmer velocity.

        Args:
            x (array-like): of shape (n, d), n observations of d-dimensional data
            t (float): time
        """
        if preconditioner:
            # (d, )
            p_mu = preconditioner["offset"] * np.ones(self.dimension)
            # (d, d)
            p_cov = preconditioner["deviation"] ** 2 * np.eye(self.dimension)
        else:
            p_mu = np.zeros(self.dimension)
            p_cov = np.eye(self.dimension)
        if t == 0:
            return np.repeat(np.atleast_2d(self.mean), x.shape[0], axis=0) - p_mu
        # (k, d, d)
        cov = t**2 * self.cov_array + (1 - t**2) * p_cov
        invcov = np.linalg.inv(cov)
        # (k, )
        detcov = np.linalg.det(cov)
        # (n, k, 1, d)
        offset = np.expand_dims(
            np.expand_dims(x, axis=-2) - t * self.mean_array - (1 - t) * p_mu, axis=-2
        )
        # (n, k, 1, d)
        offset_invcov = offset @ invcov
        # (n, k)
        p = np.exp(
            -0.5 * np.sum(offset_invcov * offset, axis=-1).squeeze(axis=-1)
        ) / np.sqrt(detcov)
        # (n, k)
        g = p * self.weights
        # (n, 1)
        deno = np.sum(g, axis=-1, keepdims=True)
        # (n, d)
        nume = np.sum(-offset_invcov.squeeze(axis=-2) * g[..., None], axis=-2)
        # (n, d)
        return (x - p_mu + (nume / deno) @ p_cov) / t


def get_distribution(name):
    return _DISTRIBUTIONS[name]
