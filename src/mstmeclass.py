from multiprocessing.heap import Arena
from multiprocessing.sharedctypes import Value
import cartopy.crs as ccrs
from tkinter.tix import Tree
from typing import Iterable
from unicodedata import numeric
import numpy as np
from scipy.stats._continuous_distns import genpareto
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
from scipy.stats import laplace
from scipy.stats import genextreme
from scipy.stats import kendalltau
from scipy.optimize import minimize
from scipy.stats import rv_continuous
import openturns as ot

# import src.threshold_search as threshold_search
import xarray as xr
from tqdm import trange
from scipy.spatial import KDTree

# from stme import genpar_estimation


pos_color = plt.rcParams["axes.prop_cycle"].by_key()["color"]
rng = np.random.default_rng(9999)


def _savefig(*args, **kwargs):
    plt.savefig(*args, **kwargs)
    plt.close(plt.gcf())


# def plot_map():
#     fig, ax = plt.subplots(
#         1,
#         2,
#         subplot_kw={"projection": ccrs.PlateCarree()},
#         figsize=(8 * 2, 6),
#     )
#     for vi in range(2):
#         ax[vi].set_extent([min_lon, max_lon, min_lat, max_lat])
#         ax[vi].coastlines()


# def f_hat_cdf(pd_nrm, pd_ext, X):
#     X = np.asarray(X)
#     scalar_input = False
#     if X.ndim == 0:
#         X = X[None]  # Makes x 1D
#         scalar_input = True
#     val = np.zeros(X.shape)
#     mu = pd_ext.args[1]  # args -> ((shape, loc, scale),)
#     for i, x in enumerate(X):
#         if x > mu:
#             val[i] = 1 - (1 - pd_nrm(mu)) * (1 - pd_ext.cdf(x))
#         else:
#             val[i] = pd_nrm(x)
#     if scalar_input:
#         return np.squeeze(val)
#     return val

# def _f_hat_ppf(pd_nrm, pd_ext, _stm, X_uni):
#     _X_uni = np.asarray(X_uni)
#     _scalar_input = False
#     if _X_uni.ndim == 0:
#         _X_uni = _X_uni[None]  # Makes x 1D
#         _scalar_input = True
#     _val = np.zeros(_X_uni.shape)
#     _mu = pd_ext.args[1]  # args -> ((shape, loc, scale),)
#     for i, x in enumerate(_X_uni):
#         if x > pd_nrm(_mu):
#             _val[i] = pd_ext.ppf(1 - (1 - x) / (1 - pd_nrm(_mu)))
#         else:
#             _val[i] = np.quantile(_stm, x)
#     if _scalar_input:
#         return np.squeeze(_val)
#     return _val


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
            # raise (ValueError("Qj is NaN"))
            # print()
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


def _search_isocontour(scatter, n):
    coords = []
    scatter = np.unique(scatter, axis=1)
    _num_events = scatter.shape[1]
    _xg, _yg = np.sort(scatter[0]), np.sort(scatter[1])
    _xi, _yi = _num_events - 1, 0
    # Search isocontour
    while True:
        # Keep searching until the edge
        if _xi - 1 < 0 or _yi + 1 >= _num_events:
            break

        _x = _xg[_xi]
        _y = _yg[_yi]

        # Check if you can go up, in which case you do
        _count_up = np.count_nonzero(
            np.logical_and(scatter[0] >= _x, scatter[1] >= _yg[_yi + 1])
        )
        if _count_up == n:
            _yi += 1
        # If you can't, look directly to the left
        else:
            _dxi = 1
            _success = False
            while not _success:
                _dyi = 0
                # Keep moving up from there until you reach a viable point
                while True:
                    # If you reach the leftmost-1 column or the topmost-1 row, exit
                    if _yi + _dyi > _num_events or _xi - _dxi < 0:
                        break
                    _count_left = np.count_nonzero(
                        np.logical_and(
                            scatter[0] >= _xg[_xi - _dxi],
                            scatter[1] >= _yg[_yi + _dyi],
                        )
                    )
                    if _count_left == n:
                        _xi -= _dxi
                        _yi += _dyi
                        _success = True
                        break
                    elif _count_left > n:
                        _dyi += 1
                    else:
                        # No point found along the vertical transect
                        # So move to the left and search again
                        _success = False
                        _dxi += 1
                        break
        coords.append([_xg[_xi], _yg[_yi]])

    return np.array(coords).T


def _genpar_estimation(
    stm: np.ndarray, thr_mar: np.ndarray, N_gp: int = 100, **kwargs
) -> tuple[list[rv_continuous], np.ndarray]:
    global rng
    is_e_marginal = stm > thr_mar[:, np.newaxis]
    num_vars = stm.shape[0]
    num_events = stm.shape[1]
    if (np.count_nonzero(is_e_marginal, axis=1) == 0).any():
        raise (ValueError("No events above marginal threshold"))
    if len(thr_mar) != num_vars:
        raise ValueError("Number of thresholds do not match number of variables")
    genpar_params = np.zeros((num_vars, N_gp, 3))
    gp = [None] * num_vars
    for vi in range(num_vars):
        for i in range(N_gp):
            j = 0
            while True:
                _stm_bootstrap = rng.choice(stm[vi], size=num_events)
                _stm_pot = _stm_bootstrap[_stm_bootstrap > thr_mar[vi]]
                _sample = ot.Sample(_stm_pot[:, np.newaxis])
                if j > 100:
                    raise (ValueError("Cannot find any points above threshold"))
                try:
                    distribution: ot.GeneralizedPareto = (
                        ot.GeneralizedParetoFactory().build(_sample)
                    )
                    _sp, _xp, _mp = distribution.getParameter()  # sigma,xi,mu
                    # _xp = (
                    #     -_xp
                    # )  # openTURNS buildMethodOfMoments has bug where the shape parameter is estimated as k(=-xi)
                    # _xp, _mp, _sp = genpareto.fit(
                    #     _stm_pot, floc=thr_mar[vi], method=kwargs["method"]
                    # )
                except:
                    j += 1
                    continue

                break
            genpar_params[vi, i, :] = [_xp, _mp, _sp]
            # genpar_params[vi, i, :] = [_xp, _mp, _sp]
        xp, mp, sp = np.median(genpar_params[vi, :, :], axis=0)
        print(f"GENPAR{xp, mp, sp}")
        gp[vi] = genpareto(xp, mp, sp)
    return gp, genpar_params


# def _ndist_transform(stm: xr.DataArray, gp: list[rv_continuous], ndist: rv_continuous):
#     stm_g = np.zeros(stm.shape)
#     num_vars = stm.shape[0]
#     _uniform = np.zeros(stm_g.shape)
#     for vi in range(num_vars):
#         _stm = stm[vi]
#         _uniform[vi] = f_hat_cdf[vi](_stm)
#         stm_g[vi] = ndist.ppf(_uniform[vi])
#     # print("H_hat min, max:", stm_g[0].min(), stm_g[0].max())
#     # print("U_hat min, max:", stm_g[1].min(), stm_g[1].max())
#     return stm_g, f_hat_cdf


def _kendall_tau_mv(stm_g, exp, is_e):
    num_vars = exp.shape[0]
    num_nodes = exp.shape[2]
    tval = np.zeros(((num_vars, num_vars, num_nodes)))
    pval = np.zeros((num_vars, num_vars, num_nodes))
    for vi in range(num_vars):
        for vj in range(num_vars):
            _stm = stm_g[vi, is_e[vi]]
            _exp = exp[vj, is_e[vi], :]
            for ni in range(num_nodes):
                _tval, _pval = kendalltau(_stm, _exp[:, ni])
                tval[vi, vj, ni] = _tval
                pval[vi, vj, ni] = _pval
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
        for vi in range(num_vars):
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
    lb = [0, None, -5, 0.1]
    ub = [1, 1, 5, None]
    params_uc = np.zeros((num_vars, N_rep, 4))
    costs = np.zeros((num_vars, N_rep))
    for vi in range(num_vars):
        for i in range(N_rep):
            _stm = stm_g_rep[i]
            a0 = np.random.uniform(low=lb[0], high=ub[0])
            b0 = np.random.uniform(low=-1, high=ub[1])
            m0 = np.random.uniform(low=-1, high=1)
            s0 = 1
            _p0 = np.array([a0, b0, m0, s0])
            if np.isnan(_p0).any():
                raise (ValueError("WTF"))
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
            if np.isnan(_cost):
                print(_param)
                print(_cost(_param, x, y))
                raise (ValueError("Cost is NaN"))
            params_uc[vi, i, :] = _param
            costs[vi, i] = _cost
    # print(f"costs:{costs}")
    params_median = np.median(params_uc, axis=1)
    # print("Params_median:", params_median)
    return params_median


def _calculate_residual(stm_g: np.ndarray, params_median: np.ndarray, thr_com: float):
    num_vars = stm_g.shape[0]
    num_events = stm_g.shape[1]
    residual = []
    for vi in range(num_vars):
        _is_e = stm_g[vi] > thr_com
        _x = stm_g[vi, _is_e]  # conditioning(extreme)
        _y = np.delete(stm_g[:, _is_e], vi, axis=0)  # conditioned
        _a = params_median[vi, 0]
        _b = params_median[vi, 1]
        _z = (_y - _a * _x) / (_x**_b)
        residual.append(_z)
    return residual


def _sample_stm_g(
    stm_g: np.ndarray,
    ndist: rv_continuous,
    params_median: np.ndarray,
    residual: list,
    thr_com: float,
    size=1000,
):
    # Sample from model
    N_sample = size
    num_vars = stm_g.shape[0]
    num_events = stm_g.shape[1]
    vi_largest = stm_g.argmax(axis=0)
    is_me = np.empty((num_vars, num_events))
    is_e = stm_g > thr_com
    for vi in range(num_vars):
        is_me[vi] = np.logical_and(vi_largest == vi, is_e[vi])
    is_e_any = is_e.any(axis=0)
    v_me_ratio = np.count_nonzero(is_me, axis=1) / np.count_nonzero(is_e_any)
    # print(
    #     num_vars,
    #     num_events,
    #     vi_largest,
    #     np.count_nonzero(is_me, axis=1),
    #     np.count_nonzero(is_e_any),
    # )
    thr_uni = ndist.cdf(thr_com)
    std_gum = ndist.ppf(rng.uniform(thr_uni, 1, size=N_sample))
    vi_list = rng.choice(num_vars, size=N_sample, p=v_me_ratio)

    sample_full_g = np.zeros((num_vars, N_sample))
    for i, vi in enumerate(vi_list):
        _a = np.asarray(params_median[vi, 0])
        _b = np.asarray(params_median[vi, 1])
        while True:
            _z = rng.choice(residual[vi], axis=1)
            _y_given_x = std_gum[i] * _a + (std_gum[i] ** _b) * _z
            if (_y_given_x < std_gum[i]).all():
                _samples = np.insert(np.asarray(_y_given_x), vi, std_gum[i])
                sample_full_g[:, i] = _samples
                break
    return sample_full_g


class MixDist:
    def __init__(self, pd_ext: rv_continuous, stm: np.ndarray):
        self.pd_nrm: ECDF = ECDF(stm)
        self.pd_ext: rv_continuous = pd_ext
        self.stm = stm

    def cdf(self, X):
        X = np.asarray(X)
        scalar_input = False
        if X.ndim == 0:
            X = X[None]  # Makes x 1D
            scalar_input = True
        val = np.zeros(X.shape)
        mu = self.pd_ext.args[1]  # args -> ((shape, loc, scale),)

        for i, x in enumerate(X):
            if x > mu:
                val[i] = 1 - (1 - self.pd_nrm(mu)) * (1 - self.pd_ext.cdf(x))
            else:
                val[i] = self.pd_nrm(x)
        if scalar_input:
            return np.squeeze(val)
        return val

    def ppf(self, X_uni):
        _X_uni = np.asarray(X_uni)
        _scalar_input = False
        if _X_uni.ndim == 0:
            _X_uni = _X_uni[None]  # Makes x 1D
            _scalar_input = True
        _val = np.zeros(_X_uni.shape)
        _mu = self.pd_ext.args[1]  # args -> ((shape, loc, scale),)
        for i, x in enumerate(_X_uni):
            if x > self.pd_nrm(_mu):
                _val[i] = self.pd_ext.ppf(1 - (1 - x) / (1 - self.pd_nrm(_mu)))
            else:
                _val[i] = np.quantile(self.stm, x)
        if _scalar_input:
            return np.squeeze(_val)
        return _val


class MSTME:
    def __init__(
        self,
        ds: xr.Dataset,
        occur_freq: float,
        area: list,
        thr_pct_mar: float = 0.75,
        thr_pct_com: float = 0.75,
        tracks: Iterable = None,
        ndist: rv_continuous = laplace,
        dir_out: str = None,
        draw_fig: bool = False,
        gpe_method: str = "MLE",
    ):
        self.ds: xr.Dataset = ds
        self.occur_freq: float = occur_freq
        self.area = area
        self.tracks = tracks
        self.gpe_method = gpe_method
        self.num_events: int = self.ds.event.size
        self.num_nodes: int = self.ds.node.size
        self.num_vars: int = 2
        self.stm: xr.DataArray = self.ds[["hs", "UV_10m"]].max(dim="node").to_array()
        self.exp: xr.DataArray = self.ds[["hs", "UV_10m"]].to_array() / self.stm
        self.lonlat = np.array([self.ds.longitude, self.ds.latitude]).T
        self.var_name: list = ["$H_s$", "$U$"]
        self.var_name_g: list = ["$\hat H_s$", "$\hat U$"]
        self.par_name: list = ["$\\xi$", "$\\mu$", "$\\sigma$"]
        self.unit: list = ["[m]", "[m/s]"]
        self.dir_out = dir_out
        self.draw_fig = draw_fig
        self.thr_pct_mar: float = thr_pct_mar
        self.thr_pct_com: float = thr_pct_com
        self.thr_mar = np.percentile(self.stm, self.thr_pct_mar * 100, axis=1)
        self.is_e_marginal: np.ndarray = self.stm > self.thr_mar[:, np.newaxis]
        _gp, _genpar_params = _genpar_estimation(
            self.stm, self.thr_mar, method=self.gpe_method
        )
        self.gp: list[rv_continuous] = _gp
        self.genpar_params: np.ndarray = _genpar_params
        self.ndist: rv_continuous = ndist
        self.mix_dist: list[MixDist] = [None, None]
        _stm_g = np.zeros(self.stm.shape)
        self.thr_mar_in_com = np.zeros((self.num_vars,))
        for vi in range(self.num_vars):
            self.mix_dist[vi] = MixDist(self.gp[vi], self.stm[vi])
            _stm_g[vi, :] = self.ndist.ppf(self.mix_dist[vi].cdf(self.stm[vi]))
            self.thr_mar_in_com[vi] = self.ndist.ppf(
                self.mix_dist[vi].cdf(self.thr_mar[vi])
            )
        self.stm_g: np.ndarray = _stm_g
        self.thr_com: float = max(
            np.percentile(self.stm_g.max(axis=0), self.thr_pct_com * 100),
            self.thr_mar_in_com.max(),
        )
        self.is_e: np.ndarray = self.stm_g > self.thr_com
        _pval, _tval = _kendall_tau_mv(self.stm_g, self.exp, self.is_e)
        self.pval: np.ndarray = _pval
        self.tval: np.ndarray = _tval
        self.tree = KDTree(self.lonlat)
        _, _idx_pos_list = self.tree.query(
            [[-61.493, 16.150 - i * 0.05] for i in range(4)]
        )
        if not isinstance(_idx_pos_list, Iterable):
            _idx_pos_list = [_idx_pos_list]
        self.idx_pos_list: list[int] = _idx_pos_list

    def kendall_tau_mv(self):
        self.pval, self.tval = _kendall_tau_mv(self.stm_g, self.exp, self.is_e)

        return self.pval, self.tval

    def get_region_filter(
        self,
        region_filter: str,
    ) -> np.ndarray:
        mask = []
        # Filter by STM location
        self.stm_idx = self.exp.argmax(axis=2)
        is_east = self.lonlat[self.stm_idx, 0] > -61.5
        is_west = np.logical_not(is_east)
        is_north = self.lonlat[self.stm_idx, 1] > 16.2
        is_south = np.logical_not(is_north)
        is_pos = self.tval[:, :, self.stm_idx] > 0
        is_neg = np.logical_not(is_pos)
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
                raise (ValueError("idk"))
        self.draw("STM_Histogram_filtered", mask=mask, region_filter=region_filter)
        self.draw("STM_location", mask=mask)
        return mask, is_pos

    def draw(self, fig_name: str, **kwargs):
        match fig_name:
            case "STM_Histogram_filtered":
                _mask = kwargs["mask"]
                fig, ax = plt.subplots(
                    2, self.num_vars, figsize=(8 * self.num_vars, 6 * 2)
                )
                stm_min = np.floor(self.stm.min(axis=1) / 5) * 5
                stm_max = np.ceil(self.stm.max(axis=1) / 5) * 5
                for vi in range(self.num_vars):
                    for i, b in enumerate([True, False]):
                        ax[i, vi].set_xlabel(f"{self.var_name[vi]}{self.unit[vi]}")
                        ax[i, vi].hist(
                            self.stm[vi, (_mask == b)],
                            bins=np.arange(stm_min[vi], stm_max[vi], 1),
                        )
                        ax[i, vi].set_title(
                            f'{"is" if b else "not"} {kwargs["region_filter"]}'
                        )
            case "STM_location":
                _mask = kwargs["mask"]
                fig, ax = plt.subplots(1, self.num_vars, figsize=(8 * self.num_vars, 6))
                for vi in range(self.num_vars):
                    ax[vi].scatter(
                        self.lonlat[:, 0],
                        self.lonlat[:, 1],
                        c="black",
                        s=2,
                    )
                    ax[vi].scatter(
                        self.lonlat[self.stm_idx[vi, _mask], 0],
                        self.lonlat[self.stm_idx[vi, _mask], 1],
                        c="red",
                        s=20,
                        alpha=0.1,
                    )
                    ax[vi].scatter(
                        self.lonlat[self.stm_idx[vi, ~_mask], 0],
                        self.lonlat[self.stm_idx[vi, ~_mask], 1],
                        c="blue",
                        s=20,
                        alpha=0.1,
                    )
            case "Tracks_vs_STM":
                fig, ax = plt.subplots(
                    1,
                    self.num_vars,
                    subplot_kw={"projection": ccrs.PlateCarree()},
                    figsize=(8 * self.num_vars, 6),
                )
                for vi in range(self.num_vars):
                    ax[vi].set_extent(self.area)
                    cmap = plt.get_cmap("viridis", 100)
                    for ei in range(self.num_events):
                        ax[vi].plot(
                            self.tracks[ei][:, 0],
                            self.tracks[ei][:, 1],
                            c=cmap(self.stm[vi, ei] / self.stm[vi].max()),
                            lw=10,
                            alpha=0.4,
                        )
                    ax[vi].coastlines(lw=5)
                    cax = fig.add_axes(
                        [
                            ax[vi].get_position().x1 + 0.01,
                            ax[vi].get_position().y0,
                            0.02,
                            ax[vi].get_position().height,
                        ]
                    )
                    sm = plt.cm.ScalarMappable(
                        cmap=cmap,
                        norm=plt.Normalize(
                            vmin=self.stm[vi].min(), vmax=self.stm[vi].max()
                        ),
                    )
                    plt.colorbar(sm, cax=cax)
                    gl = ax[vi].gridlines(draw_labels=True)
                    gl.top_labels = False
                    gl.right_labels = False
                    gl.xlines = False
                    gl.ylines = False
            case "Kendall_Tau_all_var_pval":
                fig, ax = plt.subplots(
                    self.num_vars,
                    self.num_vars,
                    sharey=True,
                    figsize=(8 * self.num_vars, 6 * self.num_vars),
                    facecolor="white",
                    squeeze=False,
                )

                for vi in range(self.num_vars):
                    for vj in range(self.num_vars):
                        ax[vi, vj].set_xlabel("Longitude")
                        ax[vi, vj].set_ylabel("Latitude")
                        _c = [
                            "red" if p < 0.05 else "black" for p in self.pval[vi, vj, :]
                        ]
                        im = ax[vi, vj].scatter(
                            self.lonlat[:, 0], self.lonlat[:, 1], s=5, c=_c
                        )
                        ax[vi, vj].set_title(
                            f"STM:{self.var_name[vi]} E:{self.var_name[vj]}"
                        )
            case "Kendall_Tau_all_var_tval":
                fig, ax = plt.subplots(
                    self.num_vars,
                    self.num_vars,
                    sharey=True,
                    figsize=(8 * self.num_vars, 6 * self.num_vars),
                    facecolor="white",
                    squeeze=False,
                )
                for vi in range(self.num_vars):
                    for vj in range(self.num_vars):
                        ax[vi, vj].set_xlabel("Longitude")
                        ax[vi, vj].set_ylabel("Latitude")
                        im = ax[vi, vj].scatter(
                            self.lonlat[:, 0],
                            self.lonlat[:, 1],
                            s=5,
                            c=self.tval[vi, vj, :],
                            cmap="seismic",
                            vmax=np.abs(self.tval[vi]).max(),
                            vmin=-np.abs(self.tval[vi]).max(),
                        )
                        plt.colorbar(im, ax=ax[vi, vj])
                        ax[vi, vj].set_title(
                            f"STM:{self.var_name[vi]} E:{self.var_name[vj]}"
                        )
            case "Kendall_Tau_marginal_pval":
                fig, ax = plt.subplots(
                    1,
                    self.num_vars,
                    sharey=True,
                    figsize=(8, 6 * self.num_vars),
                    facecolor="white",
                    squeeze=False,
                )

                for vi in range(self.num_vars):
                    ax[vi].set_xlabel("Longitude")
                    ax[vi].set_ylabel("Latitude")
                    _c = ["red" if p < 0.05 else "black" for p in self.pval[vi, vi, :]]
                    im = ax[vi].scatter(self.lonlat[:, 0], self.lonlat[:, 1], s=5, c=_c)
                    ax[vi].set_title(f"STM:{self.var_name[vi]} E:{self.var_name[vj]}")
            case "Kendall_Tau_marginal_tval":
                fig, ax = plt.subplots(
                    1,
                    self.num_vars,
                    sharey=True,
                    figsize=(8 * self.num_vars, 6 * self.num_vars),
                    facecolor="white",
                    squeeze=False,
                )

                for vi in range(self.num_vars):
                    ax[vi].set_xlabel("Longitude")
                    ax[vi].set_ylabel("Latitude")
                    im = ax[vi].scatter(
                        self.lonlat[:, 0],
                        self.lonlat[:, 1],
                        s=5,
                        c=self.tval[vi, vj, :],
                        cmap="seismic",
                        vmax=np.abs(self.tval[vi]).max(),
                        vmin=-np.abs(self.tval[vi]).max(),
                    )
                    ax[vi].set_title(f"STM:{self.var_name[vi]} E:{self.var_name[vj]}")
            case _:
                raise (ValueError(f"No figure defined with the name {fig_name}"))
        if self.dir_out != None:
            plt.savefig(f"{self.dir_out}/{fig_name}.pdf", bbox_inches="tight")
            plt.savefig(f"{self.dir_out}/{fig_name}.png", bbox_inches="tight")
        if not self.draw_fig:
            plt.close()


class Cluster:
    def __init__(
        self,
        mask: np.ndarray,
        parent: MSTME,
        draw_fig: bool = False,
    ):
        self.mask = mask
        self.parent: MSTME = parent
        self.ds: xr.Dataset = self.parent.ds.sel({"event": self.mask})
        self.tracks = self.parent.tracks[mask]
        self.gpe_method = self.parent.gpe_method
        self.num_vars: int = self.parent.num_vars
        self.num_nodes: int = self.parent.num_nodes
        self.num_events: int = np.count_nonzero(self.mask)
        self.stm: xr.DataArray = self.ds[["hs", "UV_10m"]].max(dim="node").to_array()
        self.exp: xr.DataArray = self.ds[["hs", "UV_10m"]].to_array() / self.stm
        self.tm: xr.DataArray = self.ds[["hs", "UV_10m"]].to_array()
        self.lonlat: np.ndarray = self.parent.lonlat
        self.thr_pct_mar: float = self.parent.thr_pct_mar
        self.thr_pct_com: float = self.parent.thr_pct_com
        print(self.stm.shape)
        self.thr_mar = np.percentile(self.stm, self.thr_pct_mar * 100, axis=1)
        self.is_e_marginal: np.ndarray = self.stm > self.thr_mar[:, np.newaxis]
        _gp, _genpar_params = _genpar_estimation(
            self.stm, self.thr_mar, method=self.gpe_method
        )
        self.gp: list[rv_continuous] = _gp
        self.genpar_params = _genpar_params
        self.occur_freq: float = self.parent.occur_freq * (
            self.num_events / self.parent.num_events
        )
        self.dir_out = self.parent.dir_out
        self.draw_fig = draw_fig
        self.ndist = self.parent.ndist
        self.mix_dist: list[MixDist] = [None, None]
        _stm_g = np.zeros(self.stm.shape)
        self.thr_mar_in_com = np.zeros((self.num_vars,))
        for vi in range(self.num_vars):
            self.mix_dist[vi] = MixDist(self.gp[vi], self.stm[vi])
            _stm_g[vi, :] = self.ndist.ppf(self.mix_dist[vi].cdf(self.stm[vi]))
            self.thr_mar_in_com[vi] = self.ndist.ppf(
                self.mix_dist[vi].cdf(self.thr_mar[vi])
            )
        self.stm_g: np.ndarray = _stm_g
        self.thr_com: float = max(
            np.percentile(self.stm_g.max(axis=0), self.thr_pct_com * 100),
            self.thr_mar_in_com.max(),
        )
        self.is_e: np.ndarray = self.stm_g > self.thr_com
        _pval, _tval = _kendall_tau_mv(self.stm_g, self.exp, self.is_e)
        self.pval: np.ndarray = _pval
        self.tval: np.ndarray = _tval
        self.rng: np.random.Generator = np.random.default_rng()
        self.tree = self.parent.tree
        self.idx_pos_list = self.parent.idx_pos_list
        self.var_name: list[str] = self.parent.var_name
        self.var_name_g: list[str] = self.parent.var_name_g
        self.par_name: list[str] = self.parent.par_name
        self.unit: list[str] = self.parent.unit

    def estimate_conmul(self):
        # Laplace replacement
        N_rep = 100
        stm_g_rep = np.zeros((N_rep, self.num_vars, self.num_events))
        for i in range(N_rep):
            _idx = self.rng.choice(self.num_events, size=self.num_events)
            _stm = self.stm_g[:, _idx]
            for vi in range(self.num_vars):
                _laplace_sample = self.ndist.rvs(size=self.num_events)
                _laplace_sample_sorted = np.sort(_laplace_sample)
                _arg = np.argsort(_stm[vi])
                stm_g_rep[i, vi, _arg] = _laplace_sample_sorted
        self.stm_g_rep = stm_g_rep
        # Estimate conditional model parameters
        lb = [0, None, -5, 0.1]
        ub = [1, 1, 5, 10]
        params_uc = np.zeros((self.num_vars, N_rep, 4))
        costs = np.zeros((self.num_vars, N_rep))
        for vi in range(self.num_vars):
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
        # print(f"costs:{costs}")
        params_median = np.median(params_uc, axis=1)
        print("Params_median:", params_median)
        # # Threshold search
        # if SEARCH:
        #     threshold_search.search_conditional(stm_g_rep, 1.0, 3.0)
        # Calculating residuals
        residual = []
        print("Residuals")
        for vi in range(self.num_vars):
            _x = self.stm_g[vi, self.is_e[vi]]  # conditioning(extreme)
            _y = np.delete(self.stm_g[:, self.is_e[vi]], vi, axis=0)  # conditioned
            _a = params_median[vi, 0]
            _b = params_median[vi, 1]
            _z = (_y - _a * _x) / (_x**_b)
            residual.append(_z)
            # print(_z.flatten())
            for i, __z in enumerate(_z.squeeze()):
                if __z > 5:
                    print(f"{self.var_name[vi],}a,b,x,y", _a, _b, _x[i], _y[0, i])
            print(f"{self.var_name[vi]} min, max: {_z.min()},{_z.max()}")

        self.params_median = params_median
        self.residual = residual
        self.params_uc = params_uc
        return params_median, residual

    def sample_stm(self, size=1000):
        # Sample from model
        N_sample = size
        vi_largest = self.stm_g.argmax(axis=0)
        is_me = np.empty((self.num_vars, self.num_events))
        for vi in range(self.num_vars):
            is_me[vi] = np.logical_and(vi_largest == vi, self.is_e[vi])
        self.is_e_any = self.is_e.any(axis=0)
        v_me_ratio = np.count_nonzero(is_me, axis=1) / np.count_nonzero(self.is_e_any)

        thr_uni = self.ndist.cdf(self.thr_com)
        std_gum = self.ndist.ppf(self.rng.uniform(thr_uni, 1, size=N_sample))
        self.vi_list = self.rng.choice(self.num_vars, size=N_sample, p=v_me_ratio)

        sample_full_g = np.zeros((self.num_vars, N_sample))
        for i, vi in enumerate(self.vi_list):
            _a = np.asarray(self.params_median[vi, 0])
            _b = np.asarray(self.params_median[vi, 1])
            while True:
                _z = self.rng.choice(self.residual[vi], axis=1)
                _y_given_x = std_gum[i] * _a + (std_gum[i] ** _b) * _z
                if (_y_given_x < std_gum[i]).all():
                    _samples = np.insert(np.asarray(_y_given_x), vi, std_gum[i])
                    sample_full_g[:, i] = _samples
                    break

        # Transform back to original scale
        sample_full = np.zeros(sample_full_g.shape)
        sample_uni = self.ndist.cdf(sample_full_g)
        for vi in range(self.num_vars):
            sample_full[vi] = self.mix_dist[vi].ppf(sample_uni[vi])
        self.sample_full = sample_full
        self.sample_full_g = sample_full_g
        # self.draw("Simulated_Conmul_vs_Back_Transformed")
        return sample_full

    def kendallpval_cost(self, k, stm_norm, exp):
        _t, _p = kendalltau(stm_norm, exp**stm_norm**k)
        return -_p

    # def calculate_kval(self):
    #     exp_ext = np.zeros(self.exp.shape)
    #     kval = np.zeros((self.num_vars, self.num_nodes))
    #     pval = np.zeros((self.num_vars, self.num_nodes))
    #     tval = np.zeros((self.num_vars, self.num_nodes))
    #     # fig, ax = plt.subplots(1, self.num_vars, figsize=(16, 8))
    #     for ni in trange(self.num_nodes):
    #         for vi in range(self.num_vars):
    #             _mask = self.is_e[vi]
    #             k0 = 0
    #             _optres = minimize(
    #                 self.kendallpval_cost,
    #                 k0,
    #                 args=(self.stm_g[vi, _mask], self.exp[vi, _mask, ni]),
    #                 # method='Powell'
    #             )
    #             _k = _optres.x
    #             _p = _optres.fun
    #             exp_ext[vi, :, ni] = self.exp[vi, :, ni] ** (
    #                 (self.stm_g[vi]) ** kval[vi, ni]
    #             )
    #             _t, _p = kendalltau(self.stm_g[vi, _mask], exp_ext[vi, _mask, ni])
    #             kval[vi, ni] = _k
    #             pval[vi, ni] = _p
    #             tval[vi, ni] = _t
    #     return kval, pval, tval, exp_ext

    def sample(self, size):
        self.estimate_conmul()
        N_samples = size
        # Sample STM from conmul model
        self.stm_sample = self.sample_stm(N_samples)
        # Sample Exposure sets from events where STM is extreme in either variable
        _idx_evt = self.rng.choice(np.nonzero(self.is_e_any)[0], size=N_samples)
        self.exp_sample = self.exp[:, _idx_evt, :]

        # factor
        self.tm_sample: np.ndarray = np.einsum(
            "ven,ve->ven", self.exp_sample, self.stm_sample
        )

    def sample_PWE(self, N_sample: int = 1000):
        tm_sample = np.zeros((len(self.idx_pos_list), self.num_vars, N_sample))
        tm_original = np.zeros((len(self.idx_pos_list), self.num_vars, self.num_events))
        for i, ni in enumerate(self.idx_pos_list):
            tm_original[i, :, :] = self.tm[:, :, ni]
            tm_sample[i, :, :] = self.sample_tm_PWE(ni, N_sample)
        # self.plot_isocontour_PWE(tm_original, tm_sample)
        self.tm_PWE = tm_sample
        self.tm_original_PWE = tm_original

    def sample_tm_PWE(self, idx_node: int, N_sample: int):
        # print(f"{idx_node}")
        # self.draw("General Map", idx_location=idx_node)
        _tm = self.tm[:, :, idx_node]
        _thr_mar = np.percentile(_tm, self.thr_pct_mar * 100, axis=1)
        # self.draw("PWE_histogram_tm", idx_location=idx_node)
        _gp, _ = _genpar_estimation(_tm, _thr_mar, method="MLE")
        # print(_gp)
        _mix_dist: list[MixDist] = [None, None]
        _tm_g = np.zeros(_tm.shape)
        for vi in range(self.num_vars):
            _mix_dist[vi] = MixDist(_gp[vi], _tm[vi])
            _tm_g[vi, :] = self.ndist.ppf(_mix_dist[vi].cdf(_tm[vi]))
        _thr_com = np.percentile(_tm_g.max(axis=0), self.thr_pct_com * 100)
        N_rep = 100
        num_vars = _tm_g.shape[0]
        stm_g_rep = _ndist_replacement(_tm_g, self.ndist, N_rep)
        params_median = _estimate_conmul_params(stm_g_rep, _thr_com)
        residual = _calculate_residual(_tm_g, params_median, _thr_com)
        _tm_sample_g = _sample_stm_g(
            _tm_g, self.ndist, params_median, residual, _thr_com, size=N_sample
        )
        _tm_sample = np.zeros(_tm_sample_g.shape)
        for vi in range(num_vars):
            _tm_sample[vi, :] = _mix_dist[vi].ppf(self.ndist.cdf(_tm_sample_g[vi]))

        # draw("Simulated_Conmul_vs_Back_Transformed")
        return _tm_sample

    def search_marginal(self, thr_start, thr_end, N_gp=100, N_THR=10):
        # Generalized Pareto estimation over threshold range
        thr_list = np.linspace(thr_start, thr_end, N_THR)
        genpar_params = np.zeros((N_THR, self.num_vars, 3, N_gp))
        num_samples = np.zeros((N_THR, self.num_vars, N_gp))
        for ti, _thr in enumerate(thr_list):
            # for vi in range(self.num_vars):
            #     _stm_bootstrap = rng.choice(self.stm, size=(N_gp, self.num_events))
            _, _genpar_params = _genpar_estimation(
                self.stm, _thr, N_gp=N_gp
            )  # [vi,pi,N_gp]
            genpar_params[ti] = _genpar_params

        # Shape parameter
        fig, ax = plt.subplots(
            1,
            self.num_vars,
            sharey=True,
            figsize=(8 * self.num_vars, 6),
            facecolor="white",
            squeeze=False,
        )
        u95 = np.percentile(genpar_params, 97.5, axis=3)
        l95 = np.percentile(genpar_params, 2.5, axis=3)
        med = np.percentile(genpar_params, 50.0, axis=3)
        var_name = ["$H_s$", "$U$"]
        par_name = ["$\\xi$", "$\\mu$", "$\\sigma$"]
        for vi in range(self.num_vars):
            ax[0, vi].set_title(var_name[vi])
            ax[0, 0].set_ylabel(par_name[0])
            ax[0, vi].set_xlabel(f"Threshold{self.unit[vi]}")
            ax[0, vi].plot(thr_list[vi], med[:, 0, vi])
            ax[0, vi].fill_between(
                thr_list[vi], u95[:, vi, 0], l95[:, vi, 0], alpha=0.5
            )

        plt.savefig(
            f"{self.dir_out}/Marginal_param_vs_threshold.pdf", bbox_inches="tight"
        )
        plt.savefig(
            f"{self.dir_out}/Marginal_param_vs_threshold.png", bbox_inches="tight"
        )

    def draw(self, fig_name: str, **kwargs):
        """
        Genpar_Params
        Genpar_CDF
        """
        match fig_name:
            case "PWE_histogram_tm":
                fig, ax = plt.subplots(1, self.num_vars, figsize=(8 * self.num_vars, 6))
                ni = kwargs["idx_location"]
                for vi in range(self.num_vars):
                    _ax: plt.Axes = ax[vi]
                    _ax.hist(self.tm[vi, :, ni], bins=20)
                    _ax.set_title(f"{self.var_name[vi]}")
            case "General Map":
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                ax.scatter(self.lonlat[:, 0], self.lonlat[:, 1], c="black", s=5)
                # idx = np.array(kwargs["idx_location"])
                # for ni in idx:
                ni = kwargs["idx_location"]
                ax.scatter(self.lonlat[ni, 0], self.lonlat[ni, 1], s=20)
            case "Genpar_Params":
                fig, ax = plt.subplots(
                    len(self.par_name),
                    self.num_vars,
                    figsize=(8 * self.num_vars, 6 * len(self.par_name)),
                )
                for vi in range(self.num_vars):
                    ax[0, vi].set_title(self.var_name[vi])
                    for pi, p in enumerate(self.par_name):
                        ax[pi, 0].set_ylabel(self.par_name[pi])
                        ax[pi, vi].hist(self.genpar_params[vi, :, pi])
            case "Genpar_CDF":
                fig, ax = plt.subplots(1, self.num_vars, figsize=(8 * self.num_vars, 6))
                fig.set_facecolor("white")
                # ax.set_ylabel("CDF")
                N_gp = self.genpar_params.shape[1]
                _res = 100
                for vi in range(self.num_vars):
                    _cdf_all = np.zeros((N_gp, _res))
                    _x = np.linspace(self.thr_mar[vi], self.stm[vi].max(), _res)
                    for i in range(N_gp):
                        _xp = self.genpar_params[vi, i, 0]
                        _mp = self.genpar_params[vi, i, 1]
                        _sp = self.genpar_params[vi, i, 2]
                        _cdf_all[i, :] = genpareto(_xp, _mp, _sp).cdf(_x)

                    _y = self.gp[vi].cdf(_x)
                    _u95 = np.percentile(_cdf_all, 97.5, axis=0)
                    _l95 = np.percentile(_cdf_all, 2.5, axis=0)
                    ax[vi].plot(_x, _y, c="blue", lw=2, alpha=1)
                    ax[vi].fill_between(_x, _u95, _l95, alpha=0.5)
                    _ecdf = ECDF(self.stm[vi, self.is_e_marginal[vi]])
                    _x = np.linspace(self.thr_mar[vi], self.stm[vi].max(), _res)
                    ax[vi].plot(_x, _ecdf(_x), lw=2, color="black")
                    ax[vi].set_xlabel(f"{self.var_name[vi]}{self.unit[vi]}")
            case "Original_vs_Normalized":
                fig, ax = plt.subplots(1, 2, figsize=(7, 3))

                ax[0].scatter(self.stm[0], self.stm[1], s=5)
                ax[0].set_xlabel(f"{self.var_name[0]}{self.unit[0]}")
                ax[0].set_ylabel(f"{self.var_name[1]}{self.unit[1]}")
                ax[0].set_xlim(0, 20)
                ax[0].set_ylim(0, 60)

                ax[1].set_aspect(1)
                ax[1].scatter(self.stm_g[0], self.stm_g[1], s=5)
                ax[1].set_xlabel(r"$\hat H_s$")
                ax[1].set_ylabel(r"$\hat U$")
                ax[1].set_xlim(-5, 15)
                ax[1].set_ylim(-5, 15)
                ax[1].set_xticks([-2 + 2 * i for i in range(6)])
                ax[1].set_yticks([-2 + 2 * i for i in range(6)])
            case "Kendall_Tau_all_var_pval":
                fig, ax = plt.subplots(
                    self.num_vars,
                    self.num_vars,
                    sharey=True,
                    figsize=(8 * self.num_vars, 6 * self.num_vars),
                    facecolor="white",
                    squeeze=False,
                )

                for vi in range(self.num_vars):
                    for vj in range(self.num_vars):
                        ax[vi, vj].set_xlabel("Longitude")
                        ax[vi, vj].set_ylabel("Latitude")
                        _c = [
                            "red" if p < 0.05 else "black" for p in self.pval[vi, vj, :]
                        ]
                        im = ax[vi, vj].scatter(
                            self.lonlat[:, 0], self.lonlat[:, 1], s=5, c=_c
                        )
                        ax[vi, vj].set_title(
                            f"STM:{self.var_name[vi]} E:{self.var_name[vj]}"
                        )
            case "Kendall_Tau_all_var_tval":
                fig, ax = plt.subplots(
                    self.num_vars,
                    self.num_vars,
                    sharey=True,
                    figsize=(8 * self.num_vars, 6 * self.num_vars),
                    facecolor="white",
                    squeeze=False,
                )
                for vi in range(self.num_vars):
                    for vj in range(self.num_vars):
                        ax[vi, vj].set_xlabel("Longitude")
                        ax[vi, vj].set_ylabel("Latitude")
                        im = ax[vi, vj].scatter(
                            self.lonlat[:, 0],
                            self.lonlat[:, 1],
                            s=5,
                            c=self.tval[vi, vj, :],
                            cmap="seismic",
                            vmax=np.abs(self.tval[vi]).max(),
                            vmin=-np.abs(self.tval[vi]).max(),
                        )
                        plt.colorbar(im, ax=ax[vi, vj])
                        ax[vi, vj].set_title(
                            f"STM:{self.var_name[vi]} E:{self.var_name[vj]}"
                        )
            case "Kendall_Tau_marginal_pval":
                fig, ax = plt.subplots(
                    1,
                    self.num_vars,
                    sharey=True,
                    figsize=(8, 6 * self.num_vars),
                    facecolor="white",
                    squeeze=False,
                )

                for vi in range(self.num_vars):
                    ax[vi].set_xlabel("Longitude")
                    ax[vi].set_ylabel("Latitude")
                    _c = ["red" if p < 0.05 else "black" for p in self.pval[vi, vi, :]]
                    im = ax[vi].scatter(self.lonlat[:, 0], self.lonlat[:, 1], s=5, c=_c)
                    ax[vi].set_title(f"STM:{self.var_name[vi]} E:{self.var_name[vj]}")
            case "Kendall_Tau_marginal_tval":
                fig, ax = plt.subplots(
                    1,
                    self.num_vars,
                    sharey=True,
                    figsize=(8 * self.num_vars, 6 * self.num_vars),
                    facecolor="white",
                    squeeze=False,
                )

                for vi in range(self.num_vars):
                    ax[vi].set_xlabel("Longitude")
                    ax[vi].set_ylabel("Latitude")
                    im = ax[vi].scatter(
                        self.lonlat[:, 0],
                        self.lonlat[:, 1],
                        s=5,
                        c=self.tval[vi, vj, :],
                        cmap="seismic",
                        vmax=np.abs(self.tval[vi]).max(),
                        vmin=-np.abs(self.tval[vi]).max(),
                    )
                    ax[vi].set_title(f"STM:{self.var_name[vi]} E:{self.var_name[vj]}")
            case "Replacement":
                fig, ax = plt.subplots(1, 1, figsize=(8, 6), facecolor="white")
                ax.scatter(self.stm_g_rep[:, 0, :], self.stm_g_rep[:, 1, :], alpha=0.1)
                ax.scatter(self.stm_g[0], self.stm_g[1], color="blue")
                ax.set_xlabel(r"$\hat H_s$")
                ax.set_ylabel(r"$\hat U$")
                ax.set_xlim(-3, 15)
                ax.set_ylim(-3, 15)
            case "Conmul_Estimates":
                fig, ax = plt.subplots(
                    4, self.num_vars, figsize=(8 * self.num_vars, 6 * 4)
                )
                fig.tight_layout()
                ax[0, 0].set_ylabel("a")
                ax[1, 0].set_ylabel("b")
                ax[2, 0].set_ylabel("$\mu$")
                ax[3, 0].set_ylabel("$\sigma$")
                ax[3, 0].set_xlabel(self.var_name[0])
                ax[3, 1].set_xlabel(self.var_name[1])
                for vi in range(self.num_vars):
                    ax[0, vi].hist(self.params_uc[vi, :, 0])
                    ax[1, vi].hist(self.params_uc[vi, :, 1])
                    ax[2, vi].hist(self.params_uc[vi, :, 2])
                    ax[3, vi].hist(self.params_uc[vi, :, 3])
            case "ab_Estimates":
                fig, ax = plt.subplots(
                    1, self.num_vars, figsize=(8 * self.num_vars, 6), facecolor="white"
                )
                fig.supxlabel("$a$")
                fig.supylabel("$b$")
                params_ml = np.zeros((4, self.num_vars))
                for vi in range(self.num_vars):
                    ax[vi].scatter(
                        self.params_uc[vi, :, 0],
                        self.params_uc[vi, :, 1],
                        s=5,
                        label="Generated samples",
                    )
                    ax[vi].set_title(self.var_name[vi])
            case "amu_Estimates":
                fig, ax = plt.subplots(
                    1, self.num_vars, figsize=(8 * self.num_vars, 6), facecolor="white"
                )
                fig.supxlabel("$a$")
                fig.supylabel("$mu$")
                params_ml = np.zeros((4, self.num_vars))
                for vi in range(self.num_vars):
                    ax[vi].scatter(
                        self.params_uc[vi, :, 0],
                        self.params_uc[vi, :, 2],
                        s=5,
                        label="Generated samples",
                    )
                    ax[vi].set_title(self.var_name[vi])
            case "Residuals":
                fig, ax = plt.subplots(
                    1, self.num_vars, figsize=(8 * self.num_vars, 6), facecolor="white"
                )
                # fig.tight_layout()
                for vi in range(self.num_vars):
                    ax[vi].scatter(
                        self.ndist.cdf(self.stm_g[vi, self.is_e[vi]]),
                        self.residual[vi],
                        s=5,
                    )
                    ax[vi].set_xlabel(f"F({self.var_name[vi]})")
                ax[0].set_ylabel("$Z_{-j}$")
            case "Simulated_Conmul_vs_Back_Transformed":
                fig, ax = plt.subplots(
                    1, self.num_vars, figsize=(8 * self.num_vars, 6), facecolor="white"
                )

                ax[0].set_aspect(1)

                a_h, b_h, mu_h, sg_h = self.params_median[0, :]
                a_u, b_u, mu_u, sg_u = self.params_median[1, :]
                sample_given_h = []
                sample_given_u = []
                sample_given_hg = []
                sample_given_ug = []
                for i, vi in enumerate(self.vi_list):
                    if vi == 0:
                        sample_given_h.append(self.sample_full[:, i])
                        sample_given_hg.append(self.sample_full_g[:, i])
                    if vi == 1:
                        sample_given_u.append(self.sample_full[:, i])
                        sample_given_ug.append(self.sample_full_g[:, i])
                sample_given_h = np.array(sample_given_h).T
                sample_given_u = np.array(sample_given_u).T
                sample_given_hg = np.array(sample_given_hg).T
                sample_given_ug = np.array(sample_given_ug).T

                x_h = np.linspace(self.thr_com, 10, 100)
                y_h = x_h * a_h + (x_h**b_h) * mu_h
                ax[0].plot(x_h, y_h, color="orange", label="U|H")

                y_u = np.linspace(self.thr_com, 10, 100)
                x_u = y_u * a_u + (y_u**b_u) * mu_u
                ax[0].plot(x_u, y_u, color="teal", label="H|U")

                ax[0].scatter(
                    self.stm_g[0], self.stm_g[1], s=5, color="black", label="original"
                )
                ax[0].axvline(self.thr_com, color="black")
                ax[0].axhline(self.thr_com, color="black")

                ax[0].set_xlabel(r"$\hat H_s$")
                ax[0].set_ylabel(r"$\hat U$")
                ax[0].set_xlim(-2, 10)
                ax[0].set_ylim(-2, 10)
                ax[0].scatter(
                    sample_given_hg[0],
                    sample_given_hg[1],
                    s=1,
                    color="orange",
                    label="U|H",
                )
                ax[0].scatter(
                    sample_given_ug[0],
                    sample_given_ug[1],
                    s=1,
                    color="teal",
                    label="H|U",
                )
                # print(sample_given_hg.max(), sample_given_ug.max())
                # print(sample_given_hg.min(), sample_given_ug.min())

                ax[1].scatter(self.stm[0], self.stm[1], color="black", s=5)
                ax[1].scatter(sample_given_h[0], sample_given_h[1], color="orange", s=1)
                ax[1].scatter(sample_given_u[0], sample_given_u[1], color="teal", s=1)
                ax[1].set_xlabel(f"{self.var_name[0]}{self.unit[0]}")
                ax[1].set_ylabel(f"{self.var_name[1]}{self.unit[1]}")

                res = 100
                _x = np.linspace(0, self.stm[0].max(), res)
                _y = np.linspace(0, self.stm[1].max(), res)
                _x_mg, _y_mg = np.meshgrid(_x, _y)
                _z_mg_sample = np.zeros((res, res))
                _z_mg = np.zeros((res, res))
                _exceedance_prob = 1 - self.thr_pct_com
                for xi in range(res):
                    for yi in range(res):
                        _count_sample = np.count_nonzero(
                            np.logical_and(
                                self.sample_full[0] > _x[xi],
                                self.sample_full[1] > _y[yi],
                            )
                        )
                        _count = np.count_nonzero(
                            np.logical_and(self.stm[0] > _x[xi], self.stm[1] > _y[yi])
                        )
                        _z_mg_sample[xi, yi] = _count_sample
                        _z_mg[xi, yi] = _count
                return_periods = [100]

                _levels_original = [
                    self.num_events / (rp * self.occur_freq) for rp in return_periods
                ]
                _levels_sample = [
                    self.sample_full.shape[1]
                    / (rp * self.occur_freq * _exceedance_prob)
                    for rp in return_periods
                ]
                _linestyles = ["-", "--"]
                ax[1].contour(
                    _x_mg,
                    _y_mg,
                    _z_mg.T,
                    levels=_levels_original,
                    linestyles=_linestyles,
                    colors="black",
                )
                ax[1].contour(
                    _x_mg,
                    _y_mg,
                    _z_mg_sample.T,
                    levels=_levels_sample,
                    linestyles=_linestyles,
                    colors="red",
                )
            case "RV":
                return_period = kwargs["return_period"]
                tm_sample = self.tm_sample
                tm_original = self.tm
                stm_min = [0, 0]
                stm_max = [25, 60]
                # stm_min = np.floor(tm_sample[:, :, self.idx_pos_list].min(axis=(1, 2)) / 5) * 5
                # stm_max = np.ceil(tm_sample[:, :, self.idx_pos_list].max(axis=(1, 2)) / 5) * 5
                #########################################################
                fig, axes = plt.subplots(
                    2,
                    2,
                    figsize=(8 * 2, 6 * 2),
                    facecolor="white",
                )
                fig.supxlabel(r"$H_s$[m]")
                fig.supylabel(r"$U$[m/s]")
                for i, ax in enumerate(axes.flatten()):
                    ax.set_xlim(stm_min[0], stm_max[0])
                    ax.set_ylim(stm_min[1], stm_max[1])
                    _linestyles = ["-", "--"]
                    _idx_pos = self.idx_pos_list[i]
                    # sample
                    _num_events_extreme = tm_sample.shape[1]
                    _exceedance_prob = 1 - self.thr_pct_com
                    _count_sample = round(
                        _num_events_extreme
                        / (return_period * self.occur_freq * _exceedance_prob)
                    )
                    _ic_sample = _search_isocontour(
                        tm_sample[:, :, _idx_pos], _count_sample
                    )

                    # original
                    _ic_original = []
                    _count_original = round(
                        self.num_events / (return_period * self.occur_freq)
                    )
                    _ic_original = _search_isocontour(
                        tm_original[:, :, _idx_pos], _count_original
                    )

                    ax.scatter(
                        tm_sample[0, :, _idx_pos],
                        tm_sample[1, :, _idx_pos],
                        s=2,
                        c=pos_color[i],
                        label=f"Simulated",
                    )
                    ax.scatter(
                        tm_original[0, :, _idx_pos],
                        tm_original[1, :, _idx_pos],
                        s=10,
                        c="black",
                        label=f"Original",
                    )
                    _ic_original[1, 0] = 0
                    _ic_original[0, -1] = 0
                    _ic_sample[1, 0] = 0
                    _ic_sample[0, -1] = 0
                    ax.plot(
                        _ic_original[0],
                        _ic_original[1],
                        c="black",
                        lw=2,
                    )
                    ax.plot(
                        _ic_sample[0],
                        _ic_sample[1],
                        c=pos_color[i],
                        lw=2,
                    )
                    ax.set_title(f"Coord.{i+1}")
            case "RV_PWE":
                # tm_sample(#ofLoc(=4), num_vars, num_events)
                return_period = kwargs["return_period"]
                tm_sample = self.tm_PWE
                tm_original = self.tm_original_PWE
                stm_min = [0, 0]
                stm_max = [25, 60]
                #########################################################
                fig, axes = plt.subplots(
                    2,
                    2,
                    figsize=(8 * 2, 6 * 2),
                    facecolor="white",
                )
                fig.supxlabel(r"$H_s$[m]")
                fig.supylabel(r"$U$[m/s]")
                for i, ax in enumerate(axes.flatten()):
                    ax.set_xlim(stm_min[0], stm_max[0])
                    ax.set_ylim(stm_min[1], stm_max[1])
                    _linestyles = ["-", "--"]
                    # sample
                    _num_events_extreme = tm_sample.shape[2]
                    _exceedance_prob = 1 - self.thr_pct_com
                    _count_sample = round(
                        _num_events_extreme
                        / (return_period * self.occur_freq * _exceedance_prob)
                    )
                    _ic_sample = _search_isocontour(tm_sample[i, :, :], _count_sample)

                    # original
                    _ic_original = []
                    _count_original = round(
                        self.num_events / (return_period * self.occur_freq)
                    )
                    _ic_original = _search_isocontour(
                        tm_original[i, :, :], _count_original
                    )

                    ax.scatter(
                        tm_sample[i, 0, :],
                        tm_sample[i, 1, :],
                        s=2,
                        c=pos_color[i],
                        label=f"Simulated",
                    )
                    ax.scatter(
                        tm_original[i, 0, :],
                        tm_original[i, 1, :],
                        s=10,
                        c="black",
                        label=f"Original",
                    )
                    ax.plot(
                        _ic_original[0],
                        _ic_original[1],
                        c="black",
                        lw=2,
                    )
                    ax.plot(
                        _ic_sample[0],
                        _ic_sample[1],
                        c=pos_color[i],
                        lw=2,
                    )
                    ax.set_title(f"Coord.{i+1}")
            case _:
                raise (ValueError(f"No figure defined with the name {fig_name}"))
        if self.dir_out != None:
            plt.savefig(f"{self.dir_out}/{fig_name}.pdf", bbox_inches="tight")
            plt.savefig(f"{self.dir_out}/{fig_name}.png", bbox_inches="tight")
        if not self.draw_fig:
            plt.close()
