import numpy as np
from joblib import Parallel, delayed

from misc import pbar


class mcmc_sampler:
    def __init__(self, target, *args, **kwargs):
        self.potential_fn = target.potential
        self.potential_grad_fn = target.potential_grad
        self.dimension = target.dimension
        self.methods = {
            'RWMH': self.RWMH,
            'ULA': self.ULA,
            'tULA': self.tULA,
            'tULAc': self.tULAc,
            'MALA': self.MALA,
            'tMALA': self.tMALA,
            'tMALAc': self.tMALAc,
            'LM': self.LM,
            'tLM': self.tLM,
            'tLMc': self.tLMc,
        }

    def sample(self, x0, method, nsample, n_burnin, step, parallel=True, n_job=-1, *args, **kwargs):
        n_chain = x0.shape[0]
        assert nsample % n_chain == 0
        nsample_per_chain = nsample // n_chain
        self.step = step
        my_pbar = pbar(stop=n_chain)

        def one_chain_sampling(x0, *args, **kwargs):
            ret = np.empty((nsample_per_chain, self.dimension))
            sampler = self.methods[method](x0)
            burned = False
            for total in [n_burnin, nsample_per_chain]:
                cnt = 0
                while cnt < total:
                    x = next(sampler)
                    if burned:
                        ret[cnt] = x
                    cnt += 1
                burned = True
            return ret

        if parallel:
            ret = np.array(Parallel(n_jobs=n_job)(
                delayed(one_chain_sampling)(x0[i:i+1, ...]) for i in my_pbar))
        else:
            ret = np.empty((n_chain, nsample_per_chain, self.dimension))
            for i_chain in my_pbar:
                sampler = self.methods[method](
                    x0[i_chain:i_chain+1, ...], *args, **kwargs)
                burned = False
                for total in [n_burnin, nsample_per_chain]:
                    cnt = 0
                    while cnt < total:
                        x = next(sampler)
                        if burned:
                            ret[i_chain, cnt] = x
                        cnt += 1
                    burned = True
        return ret.reshape(-1, self.dimension)

    def RWMH(self, x0, *args, **kwargs):
        x = x0
        sqrtstep = np.sqrt(2*self.step)
        p_x = self.potential_fn(x)
        while True:
            yield x
            prop = x + sqrtstep * np.random.normal(size=(1, self.dimension))
            p_prop = self.potential_fn(prop)
            logratio = p_x - p_prop
            if np.log(np.random.uniform()) <= logratio:
                x, p_x = prop, p_prop

    def ULA(self, x0, *args, **kwargs):
        x = x0
        sqrtstep = np.sqrt(2*self.step)
        while True:
            yield x
            x = x - self.step * \
                self.potential_grad_fn(x) + sqrtstep * \
                np.random.normal(size=(1, self.dimension))

    def tULA(self, x0,  taming=(lambda g, step: g/(1. + step*np.linalg.norm(g))), *args, **kwargs):
        x = x0
        sqrtstep = np.sqrt(2*self.step)
        while True:
            yield x
            x = x - self.step * taming(self.potential_grad_fn(x), self.step) + \
                sqrtstep * np.random.normal(size=(1, self.dimension))

    def tULAc(self, x0, *args, **kwargs):
        return self.tULA(x0, lambda g, step: np.divide(g, 1. + step*np.absolute(g)), *args, **kwargs)

    def MALA(self, x0, *args, **kwargs):
        x = x0
        sqrtstep = np.sqrt(2*self.step)
        p_x, g_x = self.potential_fn(x), self.potential_grad_fn(x)
        while True:
            yield x
            prop = x - self.step * g_x + sqrtstep * \
                np.random.normal(size=(1, self.dimension))
            p_prop, g_prop = self.potential_fn(
                prop), self.potential_grad_fn(prop)
            logratio = -p_prop + p_x + 1./(4*self.step) * (np.linalg.norm(
                prop - x + self.step*g_x)**2 - np.linalg.norm(x - prop + self.step*g_prop)**2)
            if np.log(np.random.uniform()) <= logratio:
                x, p_x, g_x = prop, p_prop, g_prop

    def tMALA(self, x0, taming=(lambda g, step: g/(1. + step*np.linalg.norm(g))), *args, **kwargs):
        x = x0
        sqrtstep = np.sqrt(2*self.step)
        p_x, g_x = self.potential_fn(x), self.potential_grad_fn(x)
        tamed_g_x = taming(g_x, self.step)
        while True:
            yield x
            prop = x - self.step * tamed_g_x + sqrtstep * \
                np.random.normal(size=(1, self.dimension))
            p_prop, g_prop = self.potential_fn(
                prop), self.potential_grad_fn(prop)
            tamed_g_prop = taming(g_prop, self.step)
            logratio = -p_prop + p_x + 1./(4*self.step) * (np.linalg.norm(
                prop - x + self.step*tamed_g_x)**2 - np.linalg.norm(x - prop + self.step*tamed_g_prop)**2)
            if np.log(np.random.uniform()) <= logratio:
                x, p_x, g_x, tamed_g_x = prop, p_prop, g_prop, tamed_g_prop

    def tMALAc(self, x0, *args, **kwargs):
        return self.tMALA(x0, lambda g, step: np.divide(g, 1. + step*np.absolute(g)), *args, **kwargs)

    def LM(self, x0, *args, **kwargs):
        x = x0
        sqrtstep = np.sqrt(0.5 * self.step)
        eps1 = np.random.normal(size=self.dim)
        while True:
            yield x
            eps2 = np.random.normal(size=(1, self.dimension))
            x = x - self.step * \
                self.potential_grad_fn(x) + sqrtstep * (eps1 + eps2)
            eps1 = eps2

    def tLM(self, x0, taming=(lambda g, step: g/(1. + step * np.linalg.norm(g))), *args, **kwargs):
        x = x0
        sqrtstep = np.sqrt(0.5 * self.step)
        eps1 = np.random.normal(size=(1, self.dimension))
        while True:
            yield x
            eps2 = np.random.normal(size=(1, self.dimension))
            x = x - self.step * \
                taming(self.potential_grad_fn(x), self.step) + \
                sqrtstep * (eps1 + eps2)
            eps1 = eps2

    def tLMc(self, x0, *args, **kwargs):
        return self.tLM(x0, lambda g, step: np.divide(g, 1. + step*np.absolute(g)))
