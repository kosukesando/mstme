import enum
from typing import Iterable

import numpy as np
import openturns as ot
import xarray as xr
from numpy.typing import ArrayLike
from scipy.stats import laplace, rv_continuous
from scipy.stats._continuous_distns import genpareto
from scipy.stats.distributions import rv_frozen
from statsmodels.distributions.empirical_distribution import ECDF


class MixDist(rv_continuous):
    def __init__(self, pd_ext: rv_frozen, data: Iterable[float]):
        super().__init__()
        self._pd_nrm: ECDF = ECDF(data)
        self._pd_ext: rv_frozen = pd_ext
        self._data = data

    def _cdf(self, x):
        x = np.asarray(x)
        X_shape = x.shape
        scalar_input = False
        X_flat = x.flatten()
        if X_flat.ndim == 0:
            X_flat = X_flat[None]  # Makes x 1D
            scalar_input = True
        mu = self._pd_ext.args[1]  # args -> ((shape, loc, scale),)
        mu_ecdf = self._pd_nrm(mu)
        out_flat = np.where(
            X_flat > mu,
            1 - (1 - mu_ecdf) * (1 - self._pd_ext.cdf(X_flat)),
            self._pd_nrm(X_flat),
        )
        out = out_flat.reshape(X_shape)
        if scalar_input:
            return np.squeeze(out)
        return out

    def _ppf(self, x):
        x = np.asarray(x)
        X_shape = x.shape
        scalar_input = False
        X_flat = x.flatten()

        if X_flat.ndim == 0:
            X_flat = X_flat[None]  # Makes x 1D
            scalar_input = True
        out_flat = np.zeros(X_flat.size)
        mu = self._pd_ext.args[1]  # args -> ((shape, loc, scale),)
        for i, x in enumerate(X_flat):
            if x > self._pd_nrm(mu):
                out_flat[i] = self._pd_ext.ppf(1 - (1 - x) / (1 - self._pd_nrm(mu)))
            else:
                out_flat[i] = np.quantile(self._data, x)
        out = out_flat.reshape(X_shape)
        if scalar_input:
            return np.squeeze(out)
        return out

    def transform_to_laplace(self, data_original_scale: float | Iterable[float]):
        return laplace.ppf(self.cdf(data_original_scale))

    def transform_from_laplace(self, data_laplace_scale: float | Iterable[float]):
        return self.ppf(laplace.cdf(data_laplace_scale))


class GPPAR(enum.Enum):
    XI = (0, r"$\xi$")
    MU = (1, r"$\mu$")
    SIGMA = (2, r"$\sigma$")

    def idx(self) -> int:
        return self.value[0]

    def name(self) -> str:
        return self.value[1]


def genpar_estimation(
    data: Iterable[float],
    threshold: float,
    N_gp: int = 100,
    method: str = "ot_build",
    **kwargs,
) -> tuple[rv_frozen, list[tuple[float]]]:
    """
    Estimates GPD using bootstrap
    ## Returns
    - gp: GP object using the mean of genpar_params
    - genpar_params: In order of xi,mu,sigma shape(N_gp, 3)
    """
    NUM_ATTEMPTS = 100
    _rng = kwargs.get("rng", np.random.default_rng(0))

    _data = np.fromiter(data, dtype=float)
    if np.count_nonzero(_data > threshold) < 5:
        raise ValueError(f"Not enough events(<5) above marginal threshold: {threshold}")

    genpar_params = []
    for i in range(N_gp):
        for j in range(NUM_ATTEMPTS):
            _bootstrap = _rng.choice(_data, size=data.shape[0])
            _pot = _bootstrap[_bootstrap > threshold]
            _sample = ot.Sample(_pot[:, np.newaxis])
            try:
                match method:
                    case "ot_build":
                        distribution = ot.GeneralizedParetoFactory().build(_sample)
                        _sp, _xp, _mp = distribution.getParameter()  # sigma,xi,mu

                    case "ot_build_mom":
                        distribution: ot.GeneralizedPareto = (
                            ot.GeneralizedParetoFactory().buildMethodOfMoments(_sample)
                        )
                        _sp, _xp, _mp = distribution.getParameter()  # sigma,xi,mu
                        _xp = (
                            -_xp
                        )  # openTURNS buildMethodOfMoments has bug where the shape parameter is estimated as k(=-xi)

                    case "ot_build_er":
                        distribution: ot.GeneralizedPareto = ot.GeneralizedParetoFactory().buildMethodOfExponentialRegression(
                            _sample
                        )
                        _sp, _xp, _mp = distribution.getParameter()  # sigma,xi,mu

                    case "ot_build_pwm":
                        distribution: ot.GeneralizedPareto = ot.GeneralizedParetoFactory().buildMethodOfProbabilityWeightedMoments(
                            _sample
                        )
                        _sp, _xp, _mp = distribution.getParameter()  # sigma,xi,mu

                    case "scipy":
                        _xp, _mp, _sp = genpareto.fit(
                            _pot,
                            floc=threshold,
                            method=kwargs["method_scipy"],
                        )
                    case _:
                        distribution: ot.GeneralizedPareto = (
                            ot.GeneralizedParetoFactory().build(_sample)
                        )
                        _sp, _xp, _mp = distribution.getParameter()  # sigma,xi,mu
                break
            except:
                if j == 99:
                    raise (ValueError("Genpar estimation failed"))
        genpar_params.append((_xp, _mp, _sp))
    xp, mp, sp = np.mean(np.array(genpar_params), axis=0)
    gp = genpareto(xp, mp, sp)
    return gp, genpar_params


def main():
    import matplotlib.pyplot as plt

    sample = genpareto.rvs(c=0.5, size=1000)
    gp, _ = genpar_estimation(sample, 2.0)
    print(gp.args)

    hoge = MixDist(gp, sample)
    x = np.linspace(0, 5, 100)
    plt.plot(x, hoge.cdf(x))
    plt.show()
    return


if __name__ == "__main__":
    main()
