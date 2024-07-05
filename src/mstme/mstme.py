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

import mstme.conmul as conmul
import mstme.marginal as marginal
from mstme.marginal import MixDist

# define constants and functions

pos_color = plt.rcParams["axes.prop_cycle"].by_key()["color"]
rng = np.random.default_rng(9999)
G = 9.8
G_F = 1.11


@dataclass
class SimulationConfig:
    dir_data: Path
    area: Area
    occur_freq: float
    stm: list[STM]


@dataclass
class Area:
    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float


@dataclass
class STM:
    key: str
    name: str
    name_laplace: str
    unit: str


class MSTME:
    def __init__(
        self,
        ds: xr.Dataset,
        sim_config: SimulationConfig,
        thr_pct_com: Iterable,
        thr_pct_mar: Iterable,
        **kwargs,
    ):
        """
        - data
        """
        # Arguments
        self._ds = ds
        self._sim_config = sim_config
        self._thr_pct_com = thr_pct_com
        self._thr_pct_mar = thr_pct_mar
        self._rng: np.random.Generator = kwargs.get("rng", np.random.default_rng())
        self._mask = kwargs.get("mask", np.full((self._num_events,), True))
        self._dir_out = kwargs.get("dir_out", None)
        self._draw_fig = kwargs.get("draw_fig", False)
        self._gpe_method = kwargs.get("gpe_method", "MLE")
        # Data
        self._num_events: int = self._ds.event.size
        self._num_nodes: int = self._ds.node.size
        self._num_vars = len(self._sim_config.stm)
        self._tm = self._ds[[v.key for v in self._sim_config.stm]].to_array()
        self._stm = self._ds[[f"STM_{v.key}" for v in self._sim_config.stm]].to_array()
        self._exp = self._ds[[f"EXP_{v.key}" for v in self._sim_config.stm]].to_array()
        self._stm_node_idx = self._exp.argmax(axis=2)
        # Marginal
        self._thr_mar = np.percentile(self._stm, self._thr_pct_mar * 100, axis=1)
        self._is_e_mar: np.ndarray = self._stm.values > self._thr_mar[:, np.newaxis]
        self._gp: list[rv_frozen] = []
        self._gp_params: list[tuple] = []
        self._mix_dist: list[MixDist] = []
        self._stm_g: np.ndarray = np.zeros(self._stm.shape)
        self._thr_mar_in_com = np.zeros((self._num_vars,))
        for vi in range(self._num_vars):
            _gp, _gp_params = marginal.genpar_estimation(self._stm, self._thr_mar)
            self._gp.append(_gp)
            self._gp_params.append(_gp_params)
            _mix_dist = MixDist(_gp, self._stm[vi])
            self._mix_dist.append(_mix_dist)
            self._stm_g[vi, :] = _mix_dist.transform_to_laplace(self._stm[vi])
            self._thr_mar_in_com[vi] = _mix_dist.transform_to_laplace(self._thr_mar[vi])

        # Conmul
        self._thr_com: float = np.percentile(
            self._stm_g.max(axis=0), self._thr_pct_com * 100
        )
        _cme_estimator = conmul.ConmulExtremeEstimator(
            self._stm_g, N_rep=kwargs.get("N_rep_cme", 100)
        )
        self._params_uc = _cme_estimator.estimate(self._thr_com)
        self._params_mean = np.mean(np.array(self._params_uc), axis=0)
        self._cme_model = conmul.ConmulExtremeModel(
            self._stm_g, self._thr_com, self._params_mean
        )

    @property
    def latlon(self):
        return np.array([self._ds.latitude, self._ds.longitude]).T

    @property
    def is_e(self):
        return self._cme_model.is_extreme()

    @property
    def is_e_any(self):
        return self._cme_model.is_extreme_any()

    @property
    def is_me(self):
        return self._cme_model.is_most_extreme()

    @property
    def stm(self):
        return self._stm

    @property
    def exp(self):
        return self._exp

    @property
    def num_vars(self):
        return self._num_vars

    @property
    def num_events(self):
        return self._num_events


###########################################################################################################


def get_ss_pool(
    num_events: int, num_ss: int, num_events_ss: int, **kwargs
) -> list[np.ndarray]:
    """make event masks for subsampling shared by mstme and pwe"""
    _masks_ss = []
    # indices where mask is true if kwargs["mask"] exists
    _idx_cluster_mask = np.flatnonzero(
        kwargs.get("mask", np.arange(0, num_events, 1, int))
    )
    for ssi in range(num_ss):
        _mask_ss = np.full((num_events), False)
        _idx_ss = rng.choice(_idx_cluster_mask, size=num_events_ss, replace=False)
        # Will raise ValueError if num_events_ss < _idx_cluster_mask.size
        _mask_ss[_idx_ss] = True
        _masks_ss.append(_mask_ss)
    return _masks_ss


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


def sample_MSTME(mstme: MSTME, size) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ## Returns
    tm_sample, stm_sample, exp_sample
    """
    # Sample STM from conmul model
    stm_sample = sample_stm(mstme, size)
    # Sample Exposure sets from events where STM is extreme in either variable
    _idx_evt = mstme._rng.choice(np.nonzero(mstme.is_e_any)[0], size=size)
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
    gp, _ = genpar_estimation(tm, thr_mar, method="MLE", N_gp=1)
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
    _mask_ss = get_ss_pool(mstme, num_ss, _num_events_ss)

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
            rf"Attempting to do PWE on {len(pos_list)} locations. May take a while."
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
        # make new mask
        if len(results) < pool.ncpus:
            print(f"Started:{i}")
            _mask_ss = get_ss_pool(mstme, 1, _num_events_ss)
            results.append(pool.apipe(worker_partial, _mask_ss[0]))
            i += 1
        idx_got = []
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
