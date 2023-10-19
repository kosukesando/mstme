from __future__ import annotations

import concurrent.futures as cf
import enum
import time
import warnings
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Iterable

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import openturns as ot
import xarray as xr
from numpy.typing import ArrayLike
from pathos.multiprocessing import ProcessPool
from scipy.optimize import minimize
from scipy.spatial import KDTree
from scipy.stats import genextreme, kendalltau, laplace, rv_continuous
from scipy.stats._continuous_distns import genpareto
from scipy.stats.distributions import rv_frozen
from statsmodels.distributions.empirical_distribution import ECDF
from tqdm import trange

# define constants and functions

pos_color = plt.rcParams["axes.prop_cycle"].by_key()["color"]
rng = np.random.default_rng(9999)
G = 9.8
G_F = 1.11


class SubsampleException(Exception):
    pass


class STM(enum.Enum):
    H = (0, "hs", "$H_s$", "$\hat H_s$", "m")
    U = (1, "UV_10m", "$U_{10}$", "$\hat U$", "m/s")

    def idx(self) -> int:
        return self.value[0]

    def key(self) -> str:
        return self.value[1]

    def name(self) -> str:
        return self.value[2]

    def name_norm(self) -> str:
        return self.value[3]

    def unit(self) -> str:
        return self.value[4]

    @classmethod
    def size(cls) -> int:
        return len(list(cls))


class GPPAR(enum.Enum):
    XI = (0, "$\\xi$")
    MU = (1, "$\\mu$")
    SIGMA = (2, "$\\sigma$")

    def idx(self) -> int:
        return self.value[0]

    def name(self) -> str:
        return self.value[1]


@dataclass
class SIMSET:
    region: str
    rf: str
    depth: int

    def __post_init__(self):
        ds_path = Path(f"./ds_filtered_{self.region}.pickle")
        ds_track_path = Path(f"./ds_track_{self.region}.pickle")
        match self.region:
            case "guadeloupe":
                self.dir_data = "./ww3_meteo"
                self.dir_bathy = "./Bathy.nc"
                self.min_lon = -62.00
                self.min_lat = 15.80
                self.max_lon = -60.80
                self.max_lat = 16.60
                self.dir_tracks = "./tracks"
                self.occur_freq = 44 / (2021 - 1971 + 1)
            # the cyclones were selected as passing at distance of 200km from Guadeloupe.
            # According to IBTrACS, there were 44 storms of class 0~5 during 1971-2021
            case "caribbean":
                self.dir_data = "./ww3_meteo_slim"
                self.dir_bathy = "./Bathy.nc"
                self.min_lon = -65.00
                self.min_lat = 12.00
                self.max_lon = -58.00
                self.max_lat = 18.00
                self.dir_tracks = "./tracks"
                self.occur_freq = 44 / (2021 - 1971 + 1)
            case "guadeloupe-wide":
                self.dir_data = "./data/ww3_meteo_max/*.nc"
                self.dir_bathy = "./data/Bathy.nc"
                self.min_lon = -62.50
                self.min_lat = 15.00
                self.max_lon = -60.50
                self.max_lat = 17.00
                self.dir_tracks = "./data/tracks"
                self.occur_freq = 44 / (2021 - 1971 + 1)
            # the cyclones were selected as passing at distance of 200km from Guadeloupe.
            # According to IBTrACS, there were 44 storms of class 0~5 during 1971-2021


@dataclass
class Area:
    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float


###########################################################################################################


def get_cluster_mask(
    mstme,
    region_filter: str,
) -> tuple[np.ndarray, np.ndarray]:
    mask = []
    # Filter by STM location
    is_east = mstme.ds.longitude[mstme.stm_node_idx].values > -61.5
    is_west = np.logical_not(is_east)
    is_north = mstme.ds.latitude[mstme.stm_node_idx].values > 16.2
    is_south = np.logical_not(is_north)
    # is_pos = mstme.tval[:, :, mstme.stm_node_idx] > 0
    # is_neg = np.logical_not(is_pos)
    match region_filter:
        case "h-east":
            mask = is_east[0]
        case "h-west":
            mask = is_west[0]
        case "u-east":
            mask = is_east[1]
        case "u-west":
            mask = is_west[1]
        case "h-north":
            mask = is_north[0]
        case "h-south":
            mask = is_south[0]
        case "u-north":
            mask = is_north[1]
        case "u-south":
            mask = is_south[1]
        case "none":
            mask = np.full((mstme.num_events,), True)
        case _:
            mask = None
            is_pos = None
            raise (ValueError("idk"))

    return mask


def _cost_func(p: list, x: np.ndarray, y: np.ndarray) -> float:
    """
    cost(p,data,vi)->float
    p: parameter; [a,b,mu,sigma]
    x: ndarray with shape(num_events,) conditioning
    y: ndarray with shape(num_vars-1, num_events) conditioned
    """
    q = 0
    a = p[0]
    b = p[1]
    mu = p[2]
    sg = p[3]
    if y.ndim < 2:
        y = np.expand_dims(y, axis=0)
    if (x < 0).any():
        raise (ValueError())
    for vj in range(y.shape[0]):
        _qj = np.sum(
            np.log(sg * x**b)
            + 0.5 * ((y[vj] - (a * x + mu * x**b)) / (sg * x**b)) ** 2
        )
        if np.isnan(_qj):
            print(f"Qj is NaN a:{a:0.5f}, {b:0.5f}, {mu:0.5f}, {sg:0.5f}")
            print(f"{x}, {y[vj]}")
        q += _qj
    return q


def _jacobian_custom(p, x, y) -> np.ndarray:
    a = p[0]
    b = p[1]
    mu = p[2]
    sg = p[3]
    da = np.sum(-(x ** (1 - 2 * b) * (-a * x - mu * x**b + y)) / sg**2)
    db = np.sum(
        (
            x ** (-2 * b)
            * np.log(x)
            * (
                -(a**2) * x**2
                + a * x * (2 * y - mu * x**b)
                + sg**3 * x ** (3 * b)
                + mu * y * x**b
                - y**2
            )
        )
        / sg**2
    )
    dm = np.sum(-(x ** (-b) * (-a * x - mu * x**b + y)) / sg**2)
    ds = np.sum(x**b - (x ** (-2 * b) * (a * x + mu * x**b - y) ** 2) / sg**3)
    return np.array([da, db, dm, ds])


def _genpar_estimation(
    stm: xr.DataArray,
    thr_mar: ArrayLike,
    N_gp: int = 100,
    method: str = "ot_build",
    **kwargs,
) -> tuple[list[rv_frozen], np.ndarray] | tuple[None, None]:
    """
    Estimates GPD of multiple variables using bootstrap
    ## Returns
    - gp: GP object using the mean of genpar_params
    - genpar_params: In order of xi,mu,sigma shape(num_vars, N_gp, 3)
    """
    assert thr_mar.ndim == 1
    assert stm.ndim == 2
    global rng
    if thr_mar is None:
        raise (ValueError("Threshold is None"))
    thr_mar = np.array(thr_mar)
    is_e_mar = stm.values > thr_mar[:, np.newaxis]
    num_vars = stm.shape[0]
    num_events = stm.shape[1]
    if (np.count_nonzero(is_e_mar, axis=1) == 0).any():
        print(f"No events above marginal threshold: {thr_mar}")
        return None, None
    if thr_mar.shape[0] != num_vars:
        raise ValueError("Number of thresholds do not match number of variables")
    genpar_params = np.zeros((num_vars, N_gp, 3))
    gp: list[rv_frozen] = []
    for S in STM:
        vi = S.idx()
        for i in range(N_gp):
            for j in range(100):
                _stm_bootstrap = rng.choice(stm[vi], size=num_events)
                _stm_pot = _stm_bootstrap[_stm_bootstrap > thr_mar[vi]]
                _sample = ot.Sample(_stm_pot[:, np.newaxis])
                try:
                    match method:
                        case "ot_build":
                            distribution = ot.GeneralizedParetoFactory().build(_sample)
                            _sp, _xp, _mp = distribution.getParameter()  # sigma,xi,mu

                        case "ot_build_mom":
                            distribution: ot.GeneralizedPareto = (
                                ot.GeneralizedParetoFactory().buildMethodOfMoments(
                                    _sample
                                )
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
                                _stm_pot, floc=thr_mar[vi], method=kwargs["method"]
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
            genpar_params[vi, i, :] = [_xp, _mp, _sp]
        xp, mp, sp = np.mean(genpar_params[vi, :, :], axis=0)
        # print(f"GENPAR{xp, mp, sp}")
        gp.append(genpareto(xp, mp, sp))
    return gp, genpar_params


def _kendall_tau_mv(stm_g, exp, is_e):
    num_vars = exp.shape[0]
    num_nodes = exp.shape[2]
    tval = np.zeros(((num_vars, num_vars, num_nodes)))
    pval = np.zeros((num_vars, num_vars, num_nodes))
    for Si in STM:
        vi = Si.idx()
        for Sj in STM:
            vj = Sj.idx()
            _stm = stm_g[vi, is_e[vi]]
            _exp = exp[vj, is_e[vi], :]
            t0 = time.time()
            for ni in range(num_nodes):
                _tval, _pval = kendalltau(_stm, _exp[:, ni])
                tval[vi, vj, ni] = _tval
                pval[vi, vj, ni] = _pval
            t1 = time.time()
            # print(f"kt:{vi},{vj},{t1-t0}")
    return pval, tval


def _ndist_replacement(
    stm_g: np.ndarray, ndist: rv_continuous, N_rep: int
) -> np.ndarray:
    # Laplace replacement
    num_vars = stm_g.shape[0]
    num_events = stm_g.shape[1]
    stm_g_rep = np.zeros((N_rep, num_vars, num_events))
    for i in range(N_rep):
        _idx = rng.choice(num_events, size=num_events)
        _stm = stm_g[:, _idx]
        for S in STM:
            vi = S.idx()

            _laplace_sample = ndist.rvs(size=num_events)
            _laplace_sample_sorted = np.sort(_laplace_sample)
            _arg = np.argsort(_stm[vi])
            stm_g_rep[i, vi, _arg] = _laplace_sample_sorted
    return stm_g_rep


def _estimate_conmul_params(stm_g_rep: np.ndarray, thr_com: float):
    N_rep = stm_g_rep.shape[0]
    num_vars = stm_g_rep.shape[1]
    num_events = stm_g_rep.shape[2]
    # Estimate conditional model parameters
    lb = [-1, None, -5, 0.1]
    ub = [1, 1, 5, None]
    params_uc = np.zeros((num_vars, N_rep, 4))
    costs = np.zeros((num_vars, N_rep))
    a0 = np.random.uniform(low=lb[0], high=ub[0])
    b0 = np.random.uniform(low=-1, high=ub[1])
    m0 = np.random.uniform(low=-1, high=1)
    s0 = 1
    _p0 = np.array([a0, b0, m0, s0])
    for S in STM:
        vi = S.idx()
        for i in range(N_rep):
            _stm = stm_g_rep[i]
            evt_mask = np.logical_and((_stm[vi, :] > thr_com), (~np.isinf(_stm[vi, :])))
            x = _stm[vi, evt_mask]  # conditioning
            y = np.delete(_stm[:, evt_mask], vi, axis=0)  # conditioned
            optres = minimize(
                _cost_func,
                _p0,
                args=(x, y),
                jac=_jacobian_custom,
                method="L-BFGS-B",
                # method="trust-constr",
                bounds=(
                    (lb[0], ub[0]),
                    (lb[1], ub[1]),
                    (lb[2], ub[2]),
                    (lb[3], ub[3]),
                ),
            )
            _param = optres.x
            _cost = optres.fun
            # if np.isnan(_cost):
            #     print(_param)
            #     print(_cost(_param, x, y))
            #     raise (ValueError("Cost is NaN"))
            params_uc[vi, i, :] = _param
            costs[vi, i] = _cost
    return params_uc


def _calculate_residual(stm_g: np.ndarray, params_mean: np.ndarray, thr_com: float):
    num_vars = stm_g.shape[0]
    num_events = stm_g.shape[1]
    residual = []
    eps = 1e-10
    for S in STM:
        vi = S.idx()
        _is_e = stm_g[vi] > thr_com
        _x = stm_g[vi, _is_e]  # conditioning(extreme)
        _y = np.delete(stm_g[:, _is_e], vi, axis=0)  # conditioned
        _a = params_mean[vi, 0]
        _b = params_mean[vi, 1]
        _z = (_y - _a * _x) / (_x**_b + eps)
        residual.append(_z)
    return residual


def _sample_stm_g(
    stm_g: np.ndarray,
    ndist: rv_continuous,
    params_mean: np.ndarray,
    residual: list,
    thr_com: float,
    size=1000,
):
    # Sample from model
    N_sample = size
    num_vars = stm_g.shape[0]
    num_events = stm_g.shape[1]
    stm_g_largest = stm_g.argmax(axis=0)
    is_me = np.empty((num_vars, num_events))
    is_e = stm_g > thr_com
    for S in STM:
        vi = S.idx()
        is_me[vi] = np.logical_and(stm_g_largest == vi, is_e[vi])
    is_e_any = is_e.any(axis=0)
    stm_most_extreme_ratio = np.count_nonzero(is_me, axis=1) / np.count_nonzero(
        is_e_any
    )
    thr_uni = ndist.cdf(thr_com)
    # std_gum = ndist.ppf(rng.uniform(thr_uni, 1, size=N_sample))
    vi_list = rng.choice(num_vars, size=N_sample, p=stm_most_extreme_ratio)

    sample_full_g = np.zeros((num_vars, N_sample))
    for i, vi in enumerate(vi_list):
        _a = np.asarray(params_mean[vi, 0])
        _b = np.asarray(params_mean[vi, 1])
        # _z_max = max(residual[vi])
        # _y_given_z_max = std_gum[i] * _a + (std_gum[i] ** _b) * _z_max
        count = 0
        while True:
            count += 1
            std_gum = ndist.ppf(rng.uniform(thr_uni, 1, size=1))
            _z = rng.choice(residual[vi], axis=1)
            _y_given_x = std_gum * _a + (std_gum**_b) * _z
            if count > 10000:
                raise ValueError(
                    f"No points found:\n\t_y:{_y_given_x}\n\tz:{_z}\n\ta:{_a}\n\tb:{_b}"
                )
            if (_y_given_x < std_gum).all():
                _samples = np.insert(np.asarray(_y_given_x), vi, std_gum)
                sample_full_g[:, i] = _samples
                break
    return sample_full_g


def _calc_norm_fetch(vm, vf) -> float:
    """
    Calculate the normalized fetch as described by that australian dude i forgot the name of
    """
    a = -2.175e-3
    b = 1.506e-2
    c = -1.223e-1
    d = 2.190e-1
    e = 6.737e-1
    f = 7.980e-1
    return 1 * (a * vm**2 + b * vm * vf + c * vf**2 + d * vm + e * vf + f)


def _calc_eq_fetch(vm, vf, r) -> float:
    """
    Calculate the equivalent fetch as described by that australian dude i forgot the name of
    """
    return _calc_norm_fetch(vm, vf) * (22.5e3 * np.log10(r) - 70.8e3)


def _get_ss_pool(mstme: MSTME, num_ss, num_events_ss: int):
    # make event masks for subsampling shared by mstme and pwe
    mask_ss = np.full((num_ss, mstme.get_root().num_events), False)
    # indices where mask is true
    _idx_cluster_mask = np.flatnonzero(mstme.mask)
    # print("_idx_cluster_mask.size:", _idx_cluster_mask.size)
    for ssi in range(num_ss):
        # while True:
        _idx_ss = rng.choice(_idx_cluster_mask, size=num_events_ss, replace=False)
        _idx_is_e_any = np.flatnonzero(mstme.is_e_any)
        # if :
        #     raise (SubsampleException("Not enough samples"))
        # if np.count_nonzero(np.isin(_idx_ss, _idx_is_e_any)) > 50:
        #     break
        mask_ss[ssi, _idx_ss] = True
    return mask_ss


def _subsample_worker(mask, mstme: MSTME, N_sample: int, pos_list: list[int]):
    subcluster = MSTME(
        mask=mask,
        parent=mstme,
    )
    subcluster.sample(N_sample)
    subcluster.sample_PWE(pos_list, N_sample)
    tm_MSTME_ss = subcluster.tm_sample[:, :, pos_list]
    tm_PWE_ss = subcluster.tm_sample_PWE
    stm_MSTME_ss = subcluster.stm_sample
    del subcluster
    return tm_MSTME_ss, tm_PWE_ss, stm_MSTME_ss


def sample_stm(mstme: MSTME, N_sample=1000):
    # Sample from model
    thr_uni = mstme.ndist.cdf(mstme.thr_com)
    std_gum = mstme.ndist.ppf(mstme.rng.uniform(thr_uni, 1, size=N_sample))
    mstme.vi_list = mstme.rng.choice(
        mstme.num_vars, size=N_sample, p=mstme.stm_most_extreme_ratio
    )

    stm_sample_g = np.zeros((mstme.num_vars, N_sample))
    for i, vi in enumerate(mstme.vi_list):
        _a = np.asarray(mstme.params_mean[vi, 0])
        _b = np.asarray(mstme.params_mean[vi, 1])
        while True:
            std_gum = mstme.ndist.ppf(mstme.rng.uniform(thr_uni, 1, size=1))
            _z = mstme.rng.choice(mstme.residual[vi], axis=1)
            _y_given_x = std_gum * _a + (std_gum**_b) * _z
            if (_y_given_x < std_gum).all():
                _samples = np.insert(np.asarray(_y_given_x), vi, std_gum)
                stm_sample_g[:, i] = _samples
                break

    # Transform back to original scale
    stm_sample = np.zeros(stm_sample_g.shape)
    sample_uni = mstme.ndist.cdf(stm_sample_g)
    for S in STM:
        vi = S.idx()
        stm_sample[vi] = mstme.mix_dist[vi].ppf(sample_uni[vi])
    mstme.stm_sample = stm_sample
    mstme.stm_sample_g = stm_sample_g
    return stm_sample


def sample_MSTME(mstme: MSTME, size) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ## Returns
    tm_sample, stm_sample, exp_sample
    """
    # Sample STM from conmul model
    stm_sample = sample_stm(mstme, size)
    # Sample Exposure sets from events where STM is extreme in either variable
    _idx_evt = mstme.rng.choice(np.nonzero(mstme.is_e_any)[0], size=size)
    exp_sample = mstme.exp[:, _idx_evt, :]
    # factor
    tm_sample: np.ndarray = np.einsum("ven,ve->ven", exp_sample, stm_sample)
    return tm_sample, stm_sample, exp_sample


def sample_PWE(idx_node, mstme: MSTME, size: int = 1000) -> np.ndarray:
    """
    ## Returns
    tm_sample
    """
    ## takes about 1.7sec for N_rep=100,size=1000
    tm = mstme.tm[:, :, idx_node]
    thr_mar = np.percentile(tm, mstme.thr_pct_mar * 100, axis=1)
    gp, _ = _genpar_estimation(tm, thr_mar, method="MLE", N_gp=1)
    mix_dist: list[MixDist] = []
    tm_g = np.zeros(tm.shape)
    for S in STM:
        vi = S.idx()
        mix_dist.append(MixDist(gp[vi], tm[vi]))
        tm_g[vi, :] = mstme.ndist.ppf(mix_dist[vi].cdf(tm[vi]))
    thr_com: float = np.percentile(tm_g.max(axis=0), mstme.thr_pct_com * 100)
    N_rep = 100
    stm_g_rep = _ndist_replacement(tm_g, mstme.ndist, N_rep)
    params_mean = np.mean(_estimate_conmul_params(stm_g_rep, thr_com), axis=1)
    residual = _calculate_residual(tm_g, params_mean, thr_com)
    tm_sample_g = _sample_stm_g(tm_g, mstme.ndist, params_mean, residual, thr_com, size)
    tm_sample = np.zeros(tm_sample_g.shape)
    for S in STM:
        vi = S.idx()
        tm_sample[vi, :] = mix_dist[vi].ppf(mstme.ndist.cdf(tm_sample_g[vi]))

    return tm_sample


def subsample_MSTME(
    mstme: MSTME,
    num_ss: int,
    N_year_pool: int,
    N_sample: int = 1000,
    pos_list: list = None,
):
    if pos_list is None:
        pos_list = range(mstme.num_nodes)

    # make event masks for subsampling shared by mstme and pwe
    _num_events_ss = round(N_year_pool * mstme.occur_freq)
    _mask_ss = _get_ss_pool(mstme, num_ss, _num_events_ss)

    # prepare container variables
    tm_MSTME_ss = np.zeros((num_ss, mstme.num_vars, N_sample, len(pos_list)))
    stm_MSTME_ss = np.zeros((num_ss, mstme.num_vars, N_sample))

    worker_partial = partial(
        _subsample_MSTME_worker, mstme=mstme, N_sample=N_sample, pos_list=pos_list
    )
    pool = ProcessPool()
    results = pool.imap(worker_partial, _mask_ss)
    for ssi, (_tm_MSTME_ss, _stm_MSTME_ss) in zip(range(num_ss), results):
        print(ssi)
        tm_MSTME_ss[ssi, :, :, :] = _tm_MSTME_ss
        stm_MSTME_ss[ssi, :, :] = _stm_MSTME_ss
        del _tm_MSTME_ss, _stm_MSTME_ss

    return tm_MSTME_ss, stm_MSTME_ss


def _subsample_MSTME_worker(
    mask,
    mstme: MSTME,
    N_sample: int,
    pos_list: list[int],
):
    subcluster = MSTME(
        mask=mask,
        parent=mstme,
    )
    tm_sample, stm_sample, _ = sample_MSTME(subcluster, N_sample)
    tm_MSTME_ss = tm_sample[:, :, pos_list]
    stm_MSTME_ss = stm_sample
    del subcluster
    return tm_MSTME_ss, stm_MSTME_ss


def _subsample_PWE(
    mstme: MSTME,
    num_ss: int,
    N_year_pool: int,
    N_sample: int = 1000,
    pos_list: list = None,
    thr_pct_com=None,
):
    if pos_list is None:
        pos_list = range(mstme.num_nodes)
    if len(pos_list) > 500:
        warnings.warn(
            f"Attempting to do PWE on {len(pos_list)} locations. May take a while."
        )
    _num_events_ss = round(N_year_pool * mstme.occur_freq)

    # prepare container variables
    tm_PWE_ss = np.zeros((num_ss, mstme.num_vars, N_sample, len(pos_list)))

    worker_partial = partial(
        _subsample_PWE_worker,
        mstme=mstme,
        N_sample=N_sample,
        pos_list=pos_list,
    )
    pool = ProcessPool()
    num_done = 0
    results = []
    i = 0
    while num_done < num_ss:
        # print(num_done)
        # make new mask
        if len(results) < pool.ncpus:
            print(f"Started:{i}")
            _mask_ss = _get_ss_pool(mstme, 1, _num_events_ss)
            results.append(pool.apipe(worker_partial, _mask_ss[0]))
            i += 1
        # print(results)
        idx_got = []
        # for result in reversed(results):
        for j in range(len(results)):
            result = results[j]
            if num_done < num_ss:
                if result.ready():
                    val = result.get()
                    idx_got.append(j)
                    if val is not None:
                        print(f"\t{num_done}")
                        tm_PWE_ss[num_done, :, :, :] = val
                        num_done += 1
        for j in reversed(range(len(results))):
            if j in idx_got:
                results.pop(j)
    return tm_PWE_ss
    # # make event masks for subsampling shared by mstme and pwe
    # _num_events_ss = round(N_year_pool * mstme.occur_freq)
    # _mask_ss = _get_ss_pool(mstme, num_ss, _num_events_ss)

    # # prepare container variables
    # tm_PWE_ss = np.zeros((num_ss, mstme.num_vars, N_sample, len(pos_list)))

    # worker_partial = partial(
    #     _subsample_PWE_worker,
    #     mstme=mstme,
    #     N_sample=N_sample,
    #     pos_list=pos_list,
    # )
    # pool = ProcessPool()
    # results = pool.imap(worker_partial, _mask_ss)
    # for ssi, _tm_PWE_ss in zip(
    #     range(num_ss),
    #     results,
    # ):
    #     print(ssi)
    #     tm_PWE_ss[ssi, :, :, :] = _tm_PWE_ss
    # return tm_PWE_ss


def _subsample_PWE_worker(
    mask,
    mstme: MSTME,
    N_sample: int,
    pos_list: list[int],
):
    subcluster = MSTME(
        mask=mask,
        parent=mstme,
    )
    tm_PWE_ss = np.empty((mstme.num_vars, N_sample, len(pos_list)))
    for i, ni in enumerate(pos_list):
        print(i)
        try:
            tm_PWE_ss[:, :, i] = sample_PWE(ni, subcluster, N_sample)
        except ValueError:
            return None
    del subcluster
    return tm_PWE_ss


def _subsample_PWE_sequential(
    mstme: MSTME,
    num_ss: int,
    N_year_pool: int,
    N_sample: int = 1000,
    pos_list: list = None,
    thr_pct_com=None,
):
    if pos_list is None:
        pos_list = range(mstme.num_nodes)
    if thr_pct_com is None:
        thr_pct_com = mstme.thr_pct_com
    if len(pos_list) > 500:
        warnings.warn(
            f"Attempting to do PWE on {len(pos_list)} locations. May take a while."
        )
    # make event masks for subsampling shared by mstme and pwe
    _num_events_ss = round(N_year_pool * mstme.occur_freq)
    _mask_ss = _get_ss_pool(mstme, num_ss, _num_events_ss)

    # prepare container variables
    tm_PWE_ss = np.zeros((num_ss, mstme.num_vars, N_sample, len(pos_list)))

    for ssi in range(num_ss):
        subcluster = MSTME(
            mask=_mask_ss[ssi],
            parent=mstme,
            thr_pct_com=thr_pct_com,
        )
        print(ssi)
        # for i in range(len(pos_list)):
        #     print(f"\t{i:2d}")
        #     _tm_PWE_ss = sample_PWE(pos_list[i], subcluster, N_sample)
        #     tm_PWE_ss[ssi, :, :, i] = _tm_PWE_ss
        pool = ProcessPool()
        worker_partial_seq = partial(sample_PWE, mstme=subcluster, size=N_sample)
        print(isinstance(pos_list, Iterable))
        results = pool.imap(worker_partial_seq, pos_list)
        for i, _tm_PWE_ss in zip(range(len(pos_list)), results):
            print(f"\t{i:2d}")
            tm_PWE_ss[ssi, :, :, i] = _tm_PWE_ss
    return tm_PWE_ss


#####################################################################################


class MixDist:
    def __init__(self, pd_ext: rv_frozen, stm: xr.DataArray):
        self.pd_nrm: ECDF = ECDF(stm)
        self.pd_ext: rv_frozen = pd_ext
        self.stm = stm

    def cdf(self, X):
        X = np.asarray(X)
        X_shape = X.shape
        scalar_input = False
        X_flat = X.flatten()
        if X_flat.ndim == 0:
            X_flat = X_flat[None]  # Makes x 1D
            scalar_input = True
        out_flat = np.zeros(X_flat.size)

        mu = self.pd_ext.args[1]  # args -> ((shape, loc, scale),)
        mu_ecdf = self.pd_nrm(mu)

        # list comprehension
        # def f(x):
        #     if x > mu:
        #         out = 1 - (1 - self.pd_nrm(mu)) * (1 - self.pd_ext.cdf(x))
        #     else:
        #         out = self.pd_nrm(x)

        # out_flat = [f(x) for x in X_flat]

        # np.where
        out_flat = np.where(
            X_flat > mu,
            1 - (1 - mu_ecdf) * (1 - self.pd_ext.cdf(X_flat)),
            self.pd_nrm(X_flat),
        )
        out = out_flat.reshape(X_shape)
        if scalar_input:
            return np.squeeze(out)
        return out

    def ppf(self, X):
        X = np.asarray(X)
        X_shape = X.shape
        scalar_input = False
        X_flat = X.flatten()

        if X_flat.ndim == 0:
            X_flat = X_flat[None]  # Makes x 1D
            scalar_input = True
        out_flat = np.zeros(X_flat.size)
        mu = self.pd_ext.args[1]  # args -> ((shape, loc, scale),)
        for i, x in enumerate(X_flat):
            if x > self.pd_nrm(mu):
                out_flat[i] = self.pd_ext.ppf(1 - (1 - x) / (1 - self.pd_nrm(mu)))
            else:
                out_flat[i] = np.quantile(self.stm, x)
        out = out_flat.reshape(X_shape)
        if scalar_input:
            return np.squeeze(out)
        return out


class MSTME:
    def __init__(
        self,
        **kwargs,
    ):
        t0 = time.time()
        # Static attributes
        self.rng: np.random.Generator = np.random.default_rng()

        ds = kwargs.get("ds", None)
        mask = kwargs.get("mask", None)
        parent: MSTME = kwargs.get("parent", None)
        # Determine if this instance is parent or child
        if ds is None:  # Child
            if mask is None:
                raise (ValueError("This is a child instance. Mask should not be None!"))
            if parent is None:
                raise (
                    ValueError("This is a child instance. Parent should not be None!")
                )
            if mask.ndim != 1:
                raise (ValueError("Mask needs to be 1-D"))
                # print('Mask size should match root cluster event count!\nInterpreting mask as mask from child.')
                # if mask.shape[0] == self.parent.num_events:

            self.is_child = True
            self.parent = parent
            self.num_vars = self.parent.num_vars
            self.num_nodes: int = self.parent.num_nodes
            self.ndist = self.parent.ndist
            self.area: Area = self.parent.area
            self.thr_pct_mar = kwargs.get("thr_pct_mar", self.parent.thr_pct_mar)
            self.thr_pct_com = kwargs.get("thr_pct_com", self.parent.thr_pct_com)
            self.dir_out = kwargs.get("dir_out", None)
            self.mask = np.logical_and(mask, self.parent.mask)
            self.ds = self.get_root().ds.isel(event=self.mask)
            self.num_events: int = np.count_nonzero(self.mask)
            self.draw_fig = kwargs.get("draw_fig", False)
            self.occur_freq = self.parent.occur_freq * (
                self.num_events / self.parent.num_events
            )
            if mask.shape[0] != self.get_root().num_events:
                raise (
                    ValueError(
                        "Mask size should match root cluster event count!\nInterpreting mask as mask from child."
                    )
                )

        else:  # Parent
            if mask is not None:
                raise (ValueError("This is a parent instance. Mask should be None!"))
            if parent is not None:
                raise (ValueError("This is a parent instance. Parent should be None!"))

            self.ds = ds
            self.is_child = False
            self.num_events: int = self.ds.event.size
            self.num_nodes: int = self.ds.node.size
            self.num_vars: int = STM.size()
            self.mask = np.full((self.num_events,), True)
            self.thr_pct_mar = kwargs.get("thr_pct_mar", None)
            self.thr_pct_com = kwargs.get("thr_pct_com", None)
            self.ndist = kwargs.get("ndist", laplace)
            self.occur_freq = kwargs.get("occur_freq", None)
            self.area: Area = kwargs.get("area", None)
            self.dir_out = kwargs.get("dir_out", None)
            self.draw_fig = kwargs.get("draw_fig", False)

        self.gpe_method = kwargs.get("gpe_method", "MLE")
        t1 = time.time()
        self.rf = kwargs.get("rf", "none")
        self.tracks: xr.DataArray = self.ds.Tracks
        self.tm = self.ds[[v.key() for v in STM]].to_array()
        self.stm = self.ds[[f"STM_{v.key()}" for v in STM]].to_array()
        self.exp = self.ds[[f"EXP_{v.key()}" for v in STM]].to_array()
        t2 = time.time()
        self.stm_node_idx = self.exp.argmax(axis=2)
        self.latlon = np.array([self.ds.latitude, self.ds.longitude]).T
        self.thr_mar = np.percentile(self.stm, self.thr_pct_mar * 100, axis=1)
        self.is_e_mar: np.ndarray = self.stm.values > self.thr_mar[:, np.newaxis]
        self.gp, self.gp_params = _genpar_estimation(self.stm, self.thr_mar)
        t3 = time.time()
        self.mix_dist: list[MixDist] = []
        _stm_g = np.zeros(self.stm.shape)
        self.thr_mar_in_com = np.zeros((self.num_vars,))
        for S in STM:
            vi = S.idx()
            self.mix_dist.append(MixDist(self.gp[vi], self.stm[vi]))
            _stm_g[vi, :] = self.ndist.ppf(self.mix_dist[vi].cdf(self.stm[vi]))
            self.thr_mar_in_com[vi] = self.ndist.ppf(
                self.mix_dist[vi].cdf(self.thr_mar[vi])
            )
        self.stm_g: np.ndarray = _stm_g
        # self.thr_com: float = max(
        #     np.percentile(self.stm_g.max(axis=0), self.thr_pct_com * 100),
        #     self.thr_mar_in_com.max(),
        # )
        self.thr_com: float = np.percentile(
            self.stm_g.max(axis=0), self.thr_pct_com * 100
        )
        self.is_e: np.ndarray = self.stm_g > self.thr_com
        t4 = time.time()
        self.tree = KDTree(self.latlon)
        _, self.idx_pos_list = self.tree.query(
            [[16.150 - i * 0.05, -61.493] for i in range(4)]
        )
        if not isinstance(self.idx_pos_list, Iterable):
            self.idx_pos_list = [self.idx_pos_list]
        t5 = time.time()

        N_rep = 100
        self.stm_g_rep = _ndist_replacement(self.stm_g, self.ndist, N_rep)
        self.params_uc = _estimate_conmul_params(self.stm_g_rep, self.thr_com)
        self.params_mean = np.mean(self.params_uc, axis=1)
        self.residual = _calculate_residual(self.stm_g, self.params_mean, self.thr_com)
        t5_1 = time.time()
        stm_g_largest = self.stm_g.argmax(axis=0)
        self.is_me = np.empty((self.num_vars, self.num_events))

        for S in STM:
            vi = S.idx()
            self.is_me[vi] = np.logical_and(stm_g_largest == vi, self.is_e[vi])
        self.is_e_any = self.is_e.any(axis=0)
        self.stm_most_extreme_ratio = np.count_nonzero(
            self.is_me, axis=1
        ) / np.count_nonzero(self.is_e_any)
        t6 = time.time()

    def calc_kendall_tau(self):
        self.pval, self.tval = _kendall_tau_mv(self.stm_g, self.exp, self.is_e)

    def get_root(self) -> MSTME:
        if self.is_child:
            return self.parent.get_root()
        else:
            return self

    def get_region_filter(
        self,
        region_filter: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        mask = []
        # Filter by STM location
        is_east = self.ds.longitude[self.stm_node_idx].values > -61.5
        is_west = np.logical_not(is_east)
        is_north = self.ds.latitude[self.stm_node_idx].values > 16.2
        is_south = np.logical_not(is_north)
        # is_pos = self.tval[:, :, self.stm_node_idx] > 0
        # is_neg = np.logical_not(is_pos)
        match region_filter:
            case "h-east":
                mask = is_east[0]
            case "h-west":
                mask = is_west[0]
            case "u-east":
                mask = is_east[1]
            case "u-west":
                mask = is_west[1]
            case "h-north":
                mask = is_north[0]
            case "h-south":
                mask = is_south[0]
            case "u-north":
                mask = is_north[1]
            case "u-south":
                mask = is_south[1]
            case "h-tau-pos":
                mask = is_pos[0, 0, 0, :]
            case "h-tau-neg":
                mask = is_neg[0, 0, 0, :]
            case "u-tau-pos":
                mask = is_pos[1, 1, 1, :]
            case "u-tau-neg":
                mask = is_neg[1, 1, 1, :]
            case "none":
                mask = np.full((self.num_events,), True)
            case _:
                mask = None
                is_pos = None
                raise (ValueError("idk"))

        return mask

    def estimate_conmul(self):
        """
        params_uc: a,b,mu,sigma (num_var, N_rep, num_param)
        """
        # Laplace replacement
        N_rep = 100
        stm_g_rep = np.zeros((N_rep, self.num_vars, self.num_events))
        self.stm_g_rep = stm_g_rep
        for i in range(N_rep):
            _idx = self.rng.choice(self.num_events, size=self.num_events)
            _stm = self.stm_g[:, _idx]
            for S in STM:
                vi = S.idx()

                _laplace_sample = self.ndist.rvs(size=self.num_events)
                _laplace_sample_sorted = np.sort(_laplace_sample)
                _arg = np.argsort(_stm[vi])
                stm_g_rep[i, vi, _arg] = _laplace_sample_sorted
        # Estimate conditional model parameters
        lb = [0, None, -5, 0.1]
        ub = [1, 1, 5, 10]
        params_uc = np.zeros((self.num_vars, N_rep, 4))
        costs = np.zeros((self.num_vars, N_rep))
        for S in STM:
            vi = S.idx()

            for i in range(N_rep):
                _stm = self.stm_g_rep[i]
                a0 = np.random.uniform(low=lb[0], high=ub[0])
                b0 = np.random.uniform(low=-1, high=ub[1])
                m0 = np.random.uniform(low=-1, high=1)
                # s0 = np.random.uniform(low=0.01, high=0.99)
                s0 = 1
                _p0 = np.array([a0, b0, m0, s0])
                if np.isnan(_p0).any():
                    raise (ValueError("WTF"))
                evt_mask = np.logical_and(
                    (_stm[vi, :] > self.thr_com), (~np.isinf(_stm[vi, :]))
                )
                x = _stm[vi, evt_mask]  # conditioning
                # conditioned
                y = np.delete(_stm[:, evt_mask], vi, axis=0)
                optres = minimize(
                    _cost_func,
                    _p0,
                    args=(x, y),
                    jac=_jacobian_custom,
                    method="L-BFGS-B",
                    bounds=(
                        (lb[0], ub[0]),
                        (lb[1], ub[1]),
                        (lb[2], ub[2]),
                        (lb[3], ub[3]),
                    ),
                )
                _param = optres.x
                _cost = optres.fun
                if np.isnan(_cost):
                    print(_param)
                    print(_cost(_param, x, y))
                    raise (ValueError("Cost is NaN"))
                params_uc[vi, i, :] = _param
                costs[vi, i] = _cost
        params_mean = np.mean(params_uc, axis=1)
        residual = []
        for S in STM:
            vi = S.idx()
            var_name = S.name()
            _x = self.stm_g[vi, self.is_e[vi]]  # conditioning(extreme)
            _y = np.delete(self.stm_g[:, self.is_e[vi]], vi, axis=0)  # conditioned
            _a = params_mean[vi, 0]
            _b = params_mean[vi, 1]
            _z = (_y - _a * _x) / (_x**_b)
            residual.append(_z)
            # print(_z)
            for i, __z in enumerate(_z.squeeze()):
                if __z > 5:
                    print(f"{var_name}a,b,x,y", _a, _b, _x[i], _y[0, i])
            # print(f"{var_name} min, max: {_z.min()},{_z.max()}")

        return params_mean, residual, params_uc

    def kendallpval_cost(self, k, stm_norm, exp):
        _t, _p = kendalltau(stm_norm, exp**stm_norm**k)
        return -_p

    def sample_stm(self, N_sample=1000):
        # Sample from model

        thr_uni = self.ndist.cdf(self.thr_com)
        self.vi_list = self.rng.choice(
            self.num_vars, size=N_sample, p=self.stm_most_extreme_ratio
        )

        sample_full_g = np.zeros((self.num_vars, N_sample))
        for i, vi in enumerate(self.vi_list):
            _a = np.asarray(self.params_mean[vi, 0])
            _b = np.asarray(self.params_mean[vi, 1])
            while True:
                std_gum = self.ndist.ppf(self.rng.uniform(thr_uni, 1, size=1))
                _z = self.rng.choice(self.residual[vi], axis=1)
                _y_given_x = std_gum * _a + (std_gum**_b) * _z
                if (_y_given_x < std_gum).all():
                    _samples = np.insert(np.asarray(_y_given_x), vi, std_gum)
                    sample_full_g[:, i] = _samples
                    break
                # else:
                #     print(_y_given_x, std_gum)

        # Transform back to original scale
        sample_full = np.zeros(sample_full_g.shape)
        sample_uni = self.ndist.cdf(sample_full_g)
        for S in STM:
            vi = S.idx()
            sample_full[vi] = self.mix_dist[vi].ppf(sample_uni[vi])
        self.sample_full = sample_full
        self.sample_full_g = sample_full_g
        return sample_full

    def sample(self, size):
        self.N_sample = size
        # Sample STM from conmul model
        self.stm_sample = self.sample_stm(size)
        # Sample Exposure sets from events where STM is extreme in either variable
        _idx_evt = self.rng.choice(np.nonzero(self.is_e_any)[0], size=size)
        self.exp_sample = self.exp[:, _idx_evt, :]

        # factor
        self.tm_sample: np.ndarray = np.einsum(
            "ven,ve->ven", self.exp_sample, self.stm_sample
        )

    def sample_PWE(self, pos_list: list, N_sample: int = 1000):
        tm_sample = np.zeros((self.num_vars, N_sample, len(pos_list)))
        tm_original = np.zeros((self.num_vars, self.num_events, len(pos_list)))
        for i, ni in enumerate(pos_list):
            tm_original[:, :, i] = self.tm[:, :, ni]
            tm_sample[:, :, i] = self.sample_tm_PWE(ni, N_sample)
        # self.plot_isocontour_PWE(tm_original, tm_sample)
        self.tm_sample_PWE = tm_sample
        self.tm_original_PWE = tm_original

    def sample_tm_PWE(self, idx_node: int, N_sample: int):
        _tm = self.tm[:, :, idx_node]
        _thr_mar = np.percentile(_tm, self.thr_pct_mar * 100, axis=1)
        _gp, _ = _genpar_estimation(_tm, _thr_mar, method="MLE", N_gp=1)
        # print(_gp)
        _mix_dist: list[MixDist] = []
        _tm_g = np.zeros(_tm.shape)
        for S in STM:
            vi = S.idx()
            _mix_dist.append(MixDist(_gp[vi], _tm[vi]))
            _tm_g[vi, :] = self.ndist.ppf(_mix_dist[vi].cdf(_tm[vi]))
        _thr_com: float = np.percentile(_tm_g.max(axis=0), self.thr_pct_com * 100)
        N_rep = 100
        num_vars = _tm_g.shape[0]
        stm_g_rep = _ndist_replacement(_tm_g, self.ndist, N_rep)
        params_mean = np.mean(_estimate_conmul_params(stm_g_rep, _thr_com), axis=1)
        residual = _calculate_residual(_tm_g, params_mean, _thr_com)
        _tm_sample_g = _sample_stm_g(
            _tm_g, self.ndist, params_mean, residual, _thr_com, N_sample
        )
        _tm_sample = np.zeros(_tm_sample_g.shape)
        for S in STM:
            vi = S.idx()
            _tm_sample[vi, :] = _mix_dist[vi].ppf(self.ndist.cdf(_tm_sample_g[vi]))

        return _tm_sample

    def search_marginal(self, thr_start, thr_end, N_gp=100, N_THR=10):
        # Generalized Pareto estimation over threshold range
        thr_list = np.linspace(thr_start, thr_end, N_THR)
        genpar_params = []
        for ti, _thr in enumerate(thr_list):
            # print(_thr)
            _, _genpar_params = _genpar_estimation(
                self.stm, _thr, N_gp=N_gp
            )  # [vi,N_gp,3]
            if _genpar_params is None:
                thr_list = thr_list[:ti]
                N_THR = ti + 1
                break
            else:
                genpar_params.append(_genpar_params)
        genpar_params = np.array(genpar_params)  # shape=(N_THR,num_var,N_gp,3)
        # Shape parameter
        fig, ax = plt.subplots(
            1,
            self.num_vars,
            sharey=True,
            figsize=(4 * self.num_vars, 3),
            facecolor="white",
            squeeze=False,
        )
        u95 = np.percentile(genpar_params, 97.5, axis=2)
        l95 = np.percentile(genpar_params, 2.5, axis=2)
        med = np.percentile(genpar_params, 50.0, axis=2)
        # [N_THR, vi, 3]
        for S in STM:
            vi = S.idx()
            var_name = S.name()
            ax[0, vi].set_title(var_name)
            ax[0, vi].set_ylim(-1, 1)
            ax[0, 0].set_ylabel(GPPAR.XI.name())
            ax[0, vi].set_xlabel(f"Threshold[{S.unit()}]")
            ax[0, vi].plot(thr_list[:, vi], med[:, vi, GPPAR.XI.idx()])
            ax[0, vi].fill_between(
                thr_list[:, vi],
                u95[:, vi, GPPAR.XI.idx()],
                l95[:, vi, GPPAR.XI.idx()],
                alpha=0.5,
            )

        plt.savefig(
            f"{self.dir_out}/Marginal_param_vs_threshold.pdf", bbox_inches="tight"
        )
        plt.savefig(
            f"{self.dir_out}/Marginal_param_vs_threshold.png", bbox_inches="tight"
        )

    def subsample(
        self,
        pos_list: list[int],
        num_ss: int,
        N_year_pool: int,
        N_sample: int = 1000,
    ):
        # make event masks for subsampling shared by mstme and pwe
        _num_events_ss = round(N_year_pool * self.occur_freq)
        _mask_ss = _get_ss_pool(self, num_ss, _num_events_ss)

        # prepare container variables
        tm_shape = (num_ss, self.num_vars, N_sample, len(pos_list))
        tm_MSTME_ss = np.zeros(tm_shape)
        tm_PWE_ss = np.zeros(tm_shape)
        stm_MSTME_ss = np.zeros((num_ss, self.num_vars, N_sample))

        worker_partial = partial(
            _subsample_worker, mstme=self, N_sample=N_sample, pos_list=pos_list
        )
        pool = ProcessPool()
        results = pool.imap(worker_partial, _mask_ss)
        for ssi, (_tm_MSTME_ss, _tm_PWE_ss, _stm_MSTME_ss) in zip(
            range(num_ss),
            results,
        ):
            print(ssi)
            tm_MSTME_ss[ssi, :, :, :] = _tm_MSTME_ss
            tm_PWE_ss[ssi, :, :, :] = _tm_PWE_ss
            stm_MSTME_ss[ssi, :, :] = _stm_MSTME_ss
        self.tm_MSTME_ss = tm_MSTME_ss
        self.tm_PWE_ss = tm_PWE_ss
        self.stm_MSTME_ss = stm_MSTME_ss

        return tm_MSTME_ss, tm_PWE_ss, stm_MSTME_ss
