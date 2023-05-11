from __future__ import annotations
import cartopy.crs as ccrs
from typing import Iterable
import numpy as np
from numpy.typing import ArrayLike
from scipy.stats._continuous_distns import genpareto
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
from scipy.stats import laplace
from scipy.stats import genextreme
from scipy.stats import kendalltau
from scipy.optimize import minimize
from scipy.stats import rv_continuous
from scipy.stats.distributions import rv_frozen
import openturns as ot
import enum
import xarray as xr
from scipy.spatial import KDTree
from shapely.geometry import LineString, Point, MultiLineString
from tqdm import trange
from dataclasses import dataclass
from pathlib import Path

# define constants and functions

pos_color = plt.rcParams["axes.prop_cycle"].by_key()["color"]
rng = np.random.default_rng(9999)
G = 9.8
G_F = 1.11


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


# class PLOT_NAME(str,enum.Enum):

###########################################################################################################


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
                    # If you reach the 2nd column from left or the 2nd row from top, exit
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


def _get_interp_band(contours, scale, res=11):
    """
    Returns the upper band and lower band
    contours:list of contours, each contour is a list of (x,y) with differing length
    scale: scale factor of plot ylim/xlim
    upper:np.ndarray((2,res))
    lower:np.ndarray((2,res))
    """
    upper = np.empty((2, res))
    lower = np.empty((2, res))
    means = np.empty((2, res))

    # Make contours into list of MultiLineString objects
    mls = MultiLineString([LineString(c.T) for c in contours])

    # possible multiprocessing
    for i, rad in enumerate(np.linspace(0, np.pi / 2, res, endpoint=True)):
        a = np.tan(rad) * scale
        line = LineString([(0, 0), (100, a * 100)])
        if not np.isinf(a):
            intersections = line.intersection(mls)
            l_array = [point.distance(Point(0, 0)) for point in intersections.geoms]
            lu = np.percentile(l_array, 90)
            ll = np.percentile(l_array, 10)
            xu = lu * np.cos(np.arctan(a))
            yu = lu * np.sin(np.arctan(a))
            xl = ll * np.cos(np.arctan(a))
            yl = ll * np.sin(np.arctan(a))
            upper[:, i] = [xu, yu]
            lower[:, i] = [xl, yl]
            lm = np.mean(l_array)
            xm = lm * np.cos(np.arctan(a))
            ym = lm * np.sin(np.arctan(a))
            means[:, i] = [xm, ym]
        else:
            continue
    return upper, lower, means


def _genpar_estimation(
    stm: xr.DataArray,
    thr_mar: ArrayLike,
    N_gp: int = 100,
    method: str = "ot_build",
    **kwargs,
) -> tuple[list[rv_frozen], np.ndarray] | tuple[None, None]:
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
        xp, mp, sp = np.median(genpar_params[vi, :, :], axis=0)
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
    lb = [0, None, -5, 0.1]
    ub = [1, 1, 5, None]
    params_uc = np.zeros((num_vars, N_rep, 4))
    costs = np.zeros((num_vars, N_rep))
    for S in STM:
        vi = S.idx()

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
    params_median = np.median(params_uc, axis=1)
    return params_median


def _calculate_residual(stm_g: np.ndarray, params_median: np.ndarray, thr_com: float):
    num_vars = stm_g.shape[0]
    num_events = stm_g.shape[1]
    residual = []
    for S in STM:
        vi = S.idx()
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
    for S in STM:
        vi = S.idx()
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


#####################################################################################


class MixDist:
    def __init__(self, pd_ext: rv_frozen, stm: xr.DataArray):
        self.pd_nrm: ECDF = ECDF(stm)
        self.pd_ext: rv_frozen = pd_ext
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
        **kwargs,
    ):
        # Static attributes
        self.rng: np.random.Generator = np.random.default_rng()

        ds = kwargs.get("ds", None)
        mask = kwargs.get("mask", None)
        parent = kwargs.get("parent", None)
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
            self.area = self.parent.area
            self.thr_pct_mar = self.parent.thr_pct_mar
            self.thr_pct_com = self.parent.thr_pct_com
            self.mask = np.logical_and(mask, self.parent.mask)
            self.ds = self.get_root().ds.isel(event=self.mask)
            self.num_events: int = np.count_nonzero(self.mask)
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
            self.ndist = kwargs.get("ndist", None)
            self.occur_freq = kwargs.get("occur_freq", None)
            self.area = kwargs.get("area", None)
            self.dir_out = kwargs.get("dir_out", None)
            self.draw_fig = kwargs.get("draw_fig", False)
            self.gpe_method = kwargs.get("gpe_method", "MLE")

        self.tracks: xr.DataArray = self.ds.Tracks
        self.tm = self.ds[[v.key() for v in STM]].to_array()
        self.stm = self.ds[[f"STM_{v.key()}" for v in STM]].to_array()
        self.exp = self.ds[[f"EXP_{v.key()}" for v in STM]].to_array()
        self.latlon = np.array([self.ds.latitude, self.ds.longitude]).T
        self.thr_mar = np.percentile(self.stm, self.thr_pct_mar * 100, axis=1)
        self.is_e_mar: np.ndarray = self.stm.values > self.thr_mar[:, np.newaxis]
        self.gp, self.gp_params = _genpar_estimation(self.stm, self.thr_mar)
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
        self.thr_com: float = max(
            np.percentile(self.stm_g.max(axis=0), self.thr_pct_com * 100),
            self.thr_mar_in_com.max(),
        )
        self.is_e: np.ndarray = self.stm_g > self.thr_com
        self.pval, self.tval = _kendall_tau_mv(self.stm_g, self.exp, self.is_e)
        self.tree = KDTree(self.latlon)
        _, self.idx_pos_list = self.tree.query(
            [[16.150 - i * 0.05, -61.493] for i in range(4)]
        )
        if not isinstance(self.idx_pos_list, Iterable):
            self.idx_pos_list = [self.idx_pos_list]
        self.params_median, self.residual, self.params_uc = self.estimate_conmul()

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
        self.stm_node_idx = self.exp.argmax(axis=2)
        is_east = self.ds.longitude[self.stm_node_idx].values > -61.5
        is_west = np.logical_not(is_east)
        is_north = self.ds.latitude[self.stm_node_idx].values > 16.2
        is_south = np.logical_not(is_north)
        is_pos = self.tval[:, :, self.stm_node_idx] > 0
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
                mask = None
                is_pos = None
                raise (ValueError("idk"))
        self.draw("STM_Histogram_filtered", mask=mask, region_filter=region_filter)
        self.draw("STM_location", mask=mask)
        return mask, is_pos

    def estimate_conmul(self):
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
        params_median = np.median(params_uc, axis=1)
        print("Params_median:", params_median)
        residual = []
        for S in STM:
            vi = S.idx()
            var_name = S.name()
            _x = self.stm_g[vi, self.is_e[vi]]  # conditioning(extreme)
            _y = np.delete(self.stm_g[:, self.is_e[vi]], vi, axis=0)  # conditioned
            _a = params_median[vi, 0]
            _b = params_median[vi, 1]
            _z = (_y - _a * _x) / (_x**_b)
            residual.append(_z)
            # print(_z.flatten())
            for i, __z in enumerate(_z.squeeze()):
                if __z > 5:
                    print(f"{var_name}a,b,x,y", _a, _b, _x[i], _y[0, i])
            print(f"{var_name} min, max: {_z.min()},{_z.max()}")

        return params_median, residual, params_uc

    def kendallpval_cost(self, k, stm_norm, exp):
        _t, _p = kendalltau(stm_norm, exp**stm_norm**k)
        return -_p

    def sample_stm(self, N_sample=1000):
        # Sample from model
        vi_largest = self.stm_g.argmax(axis=0)
        is_me = np.empty((self.num_vars, self.num_events))

        for S in STM:
            vi = S.idx()
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
        for S in STM:
            vi = S.idx()
            sample_full[vi] = self.mix_dist[vi].ppf(sample_uni[vi])
        self.sample_full = sample_full
        self.sample_full_g = sample_full_g
        return sample_full

    def sample(self, size):
        # Sample STM from conmul model
        self.stm_sample = self.sample_stm(size)
        # Sample Exposure sets from events where STM is extreme in either variable
        _idx_evt = self.rng.choice(np.nonzero(self.is_e_any)[0], size=size)
        self.exp_sample = self.exp[:, _idx_evt, :]

        # factor
        self.tm_sample: np.ndarray = np.einsum(
            "ven,ve->ven", self.exp_sample, self.stm_sample
        )

    def sample_PWE(self, N_sample: int = 1000):
        tm_sample = np.zeros((len(self.idx_pos_list), self.num_vars, N_sample))
        tm_original = np.zeros((len(self.idx_pos_list), self.num_vars, self.num_events))
        print(tm_sample.shape, tm_original.shape)
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
        params_median = _estimate_conmul_params(stm_g_rep, _thr_com)
        residual = _calculate_residual(_tm_g, params_median, _thr_com)
        _tm_sample_g = _sample_stm_g(
            _tm_g, self.ndist, params_median, residual, _thr_com, N_sample
        )
        _tm_sample = np.zeros(_tm_sample_g.shape)
        for S in STM:
            vi = S.idx()
            _tm_sample[vi, :] = _mix_dist[vi].ppf(self.ndist.cdf(_tm_sample_g[vi]))

        # draw("Simulated_Conmul_vs_Back_Transformed")
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
            figsize=(8 * self.num_vars, 6),
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
            ax[0, 0].set_ylabel(GPPAR.XI.name())
            ax[0, vi].set_xlabel(f"Threshold{S.unit()}")
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
        N_subsample: int,
        N_year_pool: int,
    ):
        self.num_events_ss = round(N_year_pool * self.occur_freq)
        if N_subsample == 1:
            self.mask_bootstrap = np.full((1, self.get_root().num_events), True)
        else:
            self.mask_bootstrap = np.full(
                (N_subsample, self.get_root().num_events), False
            )
            for bi in range(N_subsample):
                # indices where mask is true
                _idx_cluster_mask = np.flatnonzero(self.mask)
                _idx_ss = rng.choice(
                    _idx_cluster_mask, size=self.num_events_ss, replace=False
                )
                self.mask_bootstrap[bi, _idx_ss] = True

            self.tm_MSTME_ss = np.zeros(
                (
                    N_subsample,
                    len(self.idx_pos_list),
                    self.num_vars,
                    self.N_sample,
                )
            )
            self.stm_MSTME_ss = np.zeros((N_subsample, self.num_vars, self.N_sample))
            self.tm_PWE_ss = np.zeros(
                (
                    N_subsample,
                    len(self.idx_pos_list),
                    self.num_vars,
                    self.N_sample,
                )
            )
            for bi in trange(N_subsample):
                _subcluster = MSTME(
                    mask=self.mask_bootstrap[bi],
                    parent=self,
                )
                _subcluster.sample(self.N_sample)
                _subcluster.sample_PWE(self.N_sample)
                self.tm_MSTME_ss[bi, :, :, :] = np.moveaxis(
                    _subcluster.tm_sample[:, :, self.idx_pos_list], 2, 0
                )
                self.tm_PWE_ss[bi, :, :, :] = _subcluster.tm_PWE
                self.stm_MSTME_ss[bi, :, :] = _subcluster.stm_sample
                del _subcluster

    def subsample_all(
        self,
        N_subsample: int,
        N_year_pool: int,
    ):
        pass

    def subsample_MSTME(self, N_subsample: int, N_year_pool: int):
        _num_events_ss = round(N_year_pool * self.occur_freq)
        _mask_bootstrap = np.full((N_subsample, self.get_root().num_events), False)
        for bi in range(N_subsample):
            # indices where mask is true
            _idx_cluster_mask = np.flatnonzero(self.mask)
            _idx_ss = rng.choice(_idx_cluster_mask, size=_num_events_ss, replace=False)
            _mask_bootstrap[bi, _idx_ss] = True

        tm_MSTME_ss = np.zeros(
            (
                N_subsample,
                len(self.idx_pos_list),
                self.num_vars,
                self.N_sample,
            )
        )
        stm_MSTME_ss = np.zeros((N_subsample, self.num_vars, self.N_sample))
        # tm_PWE_ss = np.zeros(
        #     (
        #         N_subsample,
        #         len(self.idx_pos_list),
        #         self.num_vars,
        #         self.N_sample,
        #     )
        # )
        for bi in trange(N_subsample):
            _subcluster = MSTME(
                mask=_mask_bootstrap[bi],
                parent=self,
            )
            _subcluster.sample(self.N_sample)
            _subcluster.sample_PWE(self.N_sample)
            tm_MSTME_ss[bi, :, :, :] = np.moveaxis(
                _subcluster.tm_sample[:, :, self.idx_pos_list], 2, 0
            )
            stm_MSTME_ss[bi, :, :] = _subcluster.stm_sample
            del _subcluster

    def draw(self, fig_name: str, **kwargs):
        """
        Genpar_Params
        Genpar_CDF
        """
        file_name = fig_name
        match fig_name:
            case "STM_Histogram_filtered":
                _mask = kwargs["mask"]
                fig, ax = plt.subplots(
                    2,
                    self.num_vars,
                    figsize=(8 * self.num_vars, 6 * 2),
                    facecolor="white",
                )
                stm_min = np.floor(self.stm.min(axis=1) / 5) * 5
                stm_max = np.ceil(self.stm.max(axis=1) / 5) * 5
                for S in STM:
                    vi = S.idx()
                    unit = S.unit()
                    var_name = S.name()
                    for i, b in enumerate([True, False]):
                        ax[i, vi].set_xlabel(f"{var_name[vi]}{unit[vi]}")
                        ax[i, vi].hist(
                            self.stm[vi, (_mask == b)],
                            bins=np.arange(stm_min[vi], stm_max[vi], 1),
                        )
                        ax[i, vi].set_title(
                            f'{"is" if b else "not"} {kwargs["region_filter"]}'
                        )

            case "STM_location":
                _mask = kwargs["mask"]
                fig, ax = plt.subplots(
                    1,
                    self.num_vars,
                    figsize=(8 * self.num_vars, 6),
                    facecolor="white",
                )
                for S in STM:
                    vi = S.idx()

                    ax[vi].scatter(
                        self.latlon[:, 1],
                        self.latlon[:, 0],
                        c="black",
                        s=2,
                    )
                    ax[vi].scatter(
                        self.latlon[self.stm_node_idx[vi, _mask], 1],
                        self.latlon[self.stm_node_idx[vi, _mask], 0],
                        c="red",
                        s=20,
                        alpha=0.1,
                    )
                    ax[vi].scatter(
                        self.latlon[self.stm_node_idx[vi, ~_mask], 1],
                        self.latlon[self.stm_node_idx[vi, ~_mask], 0],
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
                    facecolor="white",
                )
                for S in STM:
                    vi = S.idx()

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

            case "PWE_histogram_tm":
                fig, ax = plt.subplots(
                    1,
                    self.num_vars,
                    figsize=(8 * self.num_vars, 6),
                    facecolor="white",
                )

                ni = kwargs["idx_location"]
                for S in STM:
                    vi = S.idx()
                    var_name = S.name()
                    _ax: plt.Axes = ax[vi]
                    _ax.hist(self.tm[vi, :, ni], bins=20)
                    _ax.set_title(f"{var_name}")

            case "General_Map":
                fig, ax = plt.subplots(
                    1,
                    1,
                    figsize=(8, 6),
                    facecolor="white",
                )

                ax.scatter(
                    self.latlon[:, 1],
                    self.latlon[:, 0],
                    c="black",
                    s=5,
                )
                # idx = np.array(kwargs["idx_location"])
                idx = self.idx_pos_list
                for ni in idx:
                    ax.scatter(
                        self.latlon[ni, 1],
                        self.latlon[ni, 0],
                        s=20,
                    )

            case "Genpar_Params":
                fig, ax = plt.subplots(
                    len(list(GPPAR)),
                    self.num_vars,
                    figsize=(8 * self.num_vars, 6 * len(list(GPPAR))),
                    facecolor="white",
                )

                for S in STM:
                    vi = S.idx()
                    var_name = S.name()
                    ax[0, vi].set_title(var_name)
                    for par in GPPAR:
                        pi = par.idx()
                        par_name = par.name()
                        ax[pi, 0].set_ylabel(par_name)
                        ax[pi, vi].hist(self.gp_params[vi, :, pi])

            case "Genpar_CDF":
                fig, ax = plt.subplots(
                    1,
                    self.num_vars,
                    figsize=(8 * self.num_vars, 6),
                    facecolor="white",
                )

                N_gp = self.gp_params.shape[1]
                _res = 100
                for S in STM:
                    vi = S.idx()
                    var_name = S.name()
                    unit = S.unit()
                    _cdf_all = np.zeros((N_gp, _res))
                    _x = np.linspace(self.thr_mar[vi], self.stm[vi].max(), _res)
                    for i in range(N_gp):
                        _xp = self.gp_params[vi, i, 0]
                        _mp = self.gp_params[vi, i, 1]
                        _sp = self.gp_params[vi, i, 2]
                        _cdf_all[i, :] = genpareto(_xp, _mp, _sp).cdf(_x)

                    _y = self.gp[vi].cdf(_x)
                    _u95 = np.percentile(_cdf_all, 97.5, axis=0)
                    _l95 = np.percentile(_cdf_all, 2.5, axis=0)
                    ax[vi].plot(_x, _y, c="blue", lw=2, alpha=1)
                    ax[vi].fill_between(_x, _u95, _l95, alpha=0.5)
                    _ecdf = ECDF(self.stm[vi, self.is_e_mar[vi]])
                    _x = np.linspace(self.thr_mar[vi], self.stm[vi].max(), _res)
                    ax[vi].plot(_x, _ecdf(_x), lw=2, color="black")
                    ax[vi].set_xlabel(f"{var_name}[{unit}]")

            case "Original_vs_Normalized":
                fig, ax = plt.subplots(
                    1,
                    2,
                    figsize=(7, 3),
                    facecolor="white",
                )

                ax[0].scatter(self.stm[0], self.stm[1], s=5)
                ax[0].set_xlabel(f"{STM.H.name()}[{STM.H.unit()}]")
                ax[0].set_ylabel(f"{STM.U.name()}[{STM.U.unit()}]")
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

                for Si in STM:
                    vi = Si.idx()
                    var_name_i = Si.name()
                    for Sj in STM:
                        vj = Sj.idx()
                        var_name_j = Sj.name()
                        ax[vi, vj].set_xlabel("Longitude")
                        ax[vi, vj].set_ylabel("Latitude")
                        _c = [
                            "red" if p < 0.05 else "black" for p in self.pval[vi, vj, :]
                        ]
                        im = ax[vi, vj].scatter(
                            self.latlon[:, 1],
                            self.latlon[:, 0],
                            s=5,
                            c=_c,
                        )
                        ax[vi, vj].set_title(f"STM:{var_name_i} E:{var_name_j}")

            case "Kendall_Tau_all_var_tval":
                fig, ax = plt.subplots(
                    self.num_vars,
                    self.num_vars,
                    sharey=True,
                    figsize=(8 * self.num_vars, 6 * self.num_vars),
                    facecolor="white",
                    squeeze=False,
                )

                for Si in STM:
                    vi = Si.idx()
                    var_name_i = Si.name()
                    for Sj in STM:
                        vj = Sj.idx()
                        var_name_j = Sj.name()
                        ax[vi, vj].set_xlabel("Longitude")
                        ax[vi, vj].set_ylabel("Latitude")
                        im = ax[vi, vj].scatter(
                            self.latlon[:, 1],
                            self.latlon[:, 0],
                            s=5,
                            c=self.tval[vi, vj, :],
                            cmap="seismic",
                            vmax=np.abs(self.tval[vi]).max(),
                            vmin=-np.abs(self.tval[vi]).max(),
                        )
                        plt.colorbar(im, ax=ax[vi, vj])
                        ax[vi, vj].set_title(f"STM:{var_name_i} E:{var_name_j}")

            case "Kendall_Tau_marginal_pval":
                fig, ax = plt.subplots(
                    1,
                    self.num_vars,
                    sharey=True,
                    figsize=(8, 6 * self.num_vars),
                    facecolor="white",
                    squeeze=False,
                )

                for S in STM:
                    vi = S.idx()

                    ax[vi].set_xlabel("Longitude")
                    ax[vi].set_ylabel("Latitude")
                    _c = ["red" if p < 0.05 else "black" for p in self.pval[vi, vi, :]]
                    im = ax[vi].scatter(
                        self.latlon[:, 1],
                        self.latlon[:, 0],
                        s=5,
                        c=_c,
                    )
                    ax[vi].set_title(f"STM:{var_name_i} E:{var_name_j}")

            case "Kendall_Tau_marginal_tval":
                fig, ax = plt.subplots(
                    1,
                    self.num_vars,
                    sharey=True,
                    figsize=(8 * self.num_vars, 6 * self.num_vars),
                    facecolor="white",
                    squeeze=False,
                )

                for S in STM:
                    vi = S.idx()

                    ax[vi].set_xlabel("Longitude")
                    ax[vi].set_ylabel("Latitude")
                    im = ax[vi].scatter(
                        self.latlon[:, 1],
                        self.latlon[:, 0],
                        s=5,
                        c=self.tval[vi, vj, :],
                        cmap="seismic",
                        vmax=np.abs(self.tval[vi]).max(),
                        vmin=-np.abs(self.tval[vi]).max(),
                    )
                    ax[vi].set_title(f"STM:{var_name_i} E:{var_name_j}")

            case "Replacement":
                fig, ax = plt.subplots(
                    1,
                    1,
                    figsize=(8, 6),
                    facecolor="white",
                )

                ax.scatter(self.stm_g_rep[:, 0, :], self.stm_g_rep[:, 1, :], alpha=0.1)
                ax.scatter(self.stm_g[0], self.stm_g[1], color="blue")
                ax.set_xlabel(r"$\hat H_s$")
                ax.set_ylabel(r"$\hat U$")
                ax.set_xlim(-3, 15)
                ax.set_ylim(-3, 15)

            case "Conmul_Estimates":
                fig, ax = plt.subplots(
                    4,
                    self.num_vars,
                    figsize=(8 * self.num_vars, 6 * 4),
                    facecolor="white",
                )

                fig.tight_layout()
                ax[0, 0].set_ylabel("a")
                ax[1, 0].set_ylabel("b")
                ax[2, 0].set_ylabel("$\mu$")
                ax[3, 0].set_ylabel("$\sigma$")
                ax[3, 0].set_xlabel(STM.H.name())
                ax[3, 1].set_xlabel(STM.U.name())
                for S in STM:
                    vi = S.idx()

                    ax[0, vi].hist(self.params_uc[vi, :, 0])
                    ax[1, vi].hist(self.params_uc[vi, :, 1])
                    ax[2, vi].hist(self.params_uc[vi, :, 2])
                    ax[3, vi].hist(self.params_uc[vi, :, 3])

            case "ab_Estimates":
                fig, ax = plt.subplots(
                    1,
                    self.num_vars,
                    figsize=(8 * self.num_vars, 6),
                    facecolor="white",
                )

                fig.supxlabel("$a$")
                fig.supylabel("$b$")
                params_ml = np.zeros((4, self.num_vars))
                for S in STM:
                    vi = S.idx()
                    var_name = S.name()

                    ax[vi].scatter(
                        self.params_uc[vi, :, 0],
                        self.params_uc[vi, :, 1],
                        s=5,
                        label="Generated samples",
                    )
                    ax[vi].set_title(var_name)

            case "amu_Estimates":
                fig, ax = plt.subplots(
                    1,
                    self.num_vars,
                    figsize=(8 * self.num_vars, 6),
                    facecolor="white",
                )

                fig.supxlabel("$a$")
                fig.supylabel("$mu$")
                params_ml = np.zeros((4, self.num_vars))
                for S in STM:
                    vi = S.idx()
                    var_name = S.name()

                    ax[vi].scatter(
                        self.params_uc[vi, :, 0],
                        self.params_uc[vi, :, 2],
                        s=5,
                        label="Generated samples",
                    )
                    ax[vi].set_title(var_name)

            case "Residuals":
                fig, ax = plt.subplots(
                    1,
                    self.num_vars,
                    figsize=(8 * self.num_vars, 6),
                    facecolor="white",
                )

                # fig.tight_layout()
                for S in STM:
                    vi = S.idx()
                    var_name = S.name()
                    ax[vi].scatter(
                        self.ndist.cdf(self.stm_g[vi, self.is_e[vi]]),
                        self.residual[vi],
                        s=5,
                    )
                    ax[vi].set_xlabel(f"F({var_name})")
                ax[0].set_ylabel("$Z_{-j}$")

            case "Simulated_Conmul_vs_Back_Transformed":
                fig, ax = plt.subplots(
                    1,
                    self.num_vars,
                    figsize=(8 * self.num_vars, 6),
                    facecolor="white",
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

                ax[1].scatter(self.stm[0], self.stm[1], color="black", s=5)
                ax[1].scatter(sample_given_h[0], sample_given_h[1], color="orange", s=1)
                ax[1].scatter(sample_given_u[0], sample_given_u[1], color="teal", s=1)
                ax[1].set_xlabel(f"{STM.H.name()}[{STM.H.unit()}]")
                ax[1].set_ylabel(f"{STM.U.name()}[{STM.U.unit()}]")

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
                fig, axes = plt.subplots(
                    2,
                    2,
                    figsize=(8 * 2, 6 * 2),
                    facecolor="white",
                )

                return_period = kwargs["return_period"]
                file_name = file_name + f"_RP{return_period}"
                tm_sample = self.tm_sample
                tm_original = self.tm
                stm_min = [0, 0]
                stm_max = [25, 60]
                # stm_min = np.floor(tm_sample[:, :, self.idx_pos_list].min(axis=(1, 2)) / 5) * 5
                # stm_max = np.ceil(tm_sample[:, :, self.idx_pos_list].max(axis=(1, 2)) / 5) * 5
                #########################################################
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
                    ax.set_title(f"Location {i+1}")

            case "RV_PWE":
                fig, axes = plt.subplots(
                    2,
                    2,
                    figsize=(8 * 2, 6 * 2),
                    facecolor="white",
                )

                # tm_sample(#ofLoc(=4), num_vars, num_events)
                return_period = kwargs["return_period"]
                file_name = file_name + f"_RP{return_period}"
                tm_sample = self.tm_PWE
                tm_original = self.tm_original_PWE
                stm_min = [0, 0]
                stm_max = [25, 60]
                #########################################################
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
                    _ic_original[1, 0] = 0
                    _ic_original[0, -1] = 0
                    _ic_sample[1, 0] = 0
                    _ic_sample[0, -1] = 0
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
                    ax.set_title(f"Location {i+1}")

            case "RV_STM":
                stm_MSTME_ss = self.stm_MSTME_ss
                return_period = kwargs["return_period"]
                file_name = file_name + f"_RP{return_period}"
                stm_min = [0, 0]
                stm_max = [30, 80]
                N_subsample = stm_MSTME_ss.shape[0]
                # bi, vi, ei
                fig, ax = plt.subplots(
                    1,
                    1,
                    figsize=(8, 6),
                    facecolor="white",
                )
                fig.supxlabel(r"$H_s$[m]")
                fig.supylabel(r"$U$[m/s]")
                ax.set_xlim(stm_min[0], stm_max[0])
                ax.set_ylim(stm_min[1], stm_max[1])
                # Sample count over threshold
                _num_events_extreme = stm_MSTME_ss.shape[2]
                _exceedance_prob = 1 - self.thr_pct_com
                _count_sample = round(
                    _num_events_extreme
                    / (return_period * self.occur_freq * _exceedance_prob)
                )
                _num_events_original = self.num_events
                _count_original = round(
                    _num_events_original / (return_period * self.occur_freq)
                )

                # Bootstraps
                _ic_MSTME = []
                for bi in range(N_subsample):
                    _ic = _search_isocontour(stm_MSTME_ss[bi, :, :], _count_sample)
                    _ic[1, 0] = 0
                    _ic[0, -1] = 0
                    _ic_MSTME.append(_ic)

                # Original
                _ic_original = _search_isocontour(self.stm[:, :], _count_original)
                _ic_original[1, 0] = 0
                _ic_original[0, -1] = 0

                _ic_band_MSTME_u, _ic_band_MSTME_l, _ic_band_MSTME_m = _get_interp_band(
                    _ic_MSTME, scale=stm_max[1] / stm_max[0]
                )

                array = np.concatenate(
                    (_ic_band_MSTME_u, np.flip(_ic_band_MSTME_l, axis=1)), axis=1
                )
                ax.fill(array[0], array[1], alpha=0.5)

                ######################################
                ax.scatter(
                    self.stm[0, :],
                    self.stm[1, :],
                    s=10,
                    c="black",
                    label=f"Original",
                    marker="x",
                )
                ax.plot(
                    _ic_original[0],
                    _ic_original[1],
                    c="black",
                    lw=2,
                )

            case "RV_ALL":
                tm_original = np.moveaxis(
                    self.tm[:, :, self.idx_pos_list].to_numpy(), 2, 0
                )
                tm_MSTME_ss = self.tm_MSTME_ss
                tm_PWE_ss = self.tm_PWE_ss
                return_period = kwargs["return_period"]
                file_name = file_name + f"_RP{return_period}"

                # bi, ni, vi, ei
                assert tm_MSTME_ss.shape == tm_PWE_ss.shape
                stm_min = [0, 0]
                stm_max = [20, 70]
                N_subsample = tm_MSTME_ss.shape[0]
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
                    # Sample count over threshold
                    _num_events_extreme = tm_MSTME_ss.shape[3]
                    _exceedance_prob = 1 - self.thr_pct_com
                    _count_sample = round(
                        _num_events_extreme
                        / (return_period * self.occur_freq * _exceedance_prob)
                    )
                    _ic_original = []
                    _num_events_original = tm_original.shape[2]
                    _count_original = round(
                        _num_events_original / (return_period * self.occur_freq)
                    )

                    # Bootstraps
                    ic_MSTME = []
                    ic_PWE = []
                    for bi in range(N_subsample):
                        _ic_MSTME = _search_isocontour(
                            tm_MSTME_ss[bi, i, :, :], _count_sample
                        )
                        _ic_PWE = _search_isocontour(
                            tm_PWE_ss[bi, i, :, :], _count_sample
                        )
                        _ic_MSTME[1, 0] = 0
                        _ic_MSTME[0, -1] = 0
                        _ic_PWE[1, 0] = 0
                        _ic_PWE[0, -1] = 0
                        ic_MSTME.append(_ic_MSTME)
                        ic_PWE.append(_ic_PWE)
                    (
                        ic_band_MSTME_u,
                        ic_band_MSTME_l,
                        ic_band_MSTME_m,
                    ) = _get_interp_band(ic_MSTME, scale=stm_max[1] / stm_max[0])
                    ic_band_PWE_u, ic_band_PWE_l, ic_band_PWE_m = _get_interp_band(
                        ic_PWE, scale=stm_max[1] / stm_max[0]
                    )

                    _fill_MSTME = np.concatenate(
                        (ic_band_MSTME_u, np.flip(ic_band_MSTME_l, axis=1)), axis=1
                    )
                    _fill_PWE = np.concatenate(
                        (ic_band_PWE_u, np.flip(ic_band_PWE_l, axis=1)), axis=1
                    )
                    ax.fill(_fill_MSTME[0], _fill_MSTME[1], alpha=0.2)
                    ax.fill(_fill_PWE[0], _fill_PWE[1], alpha=0.2)

                    # Original
                    _ic_original = _search_isocontour(
                        tm_original[i, :, :], _count_original
                    )
                    _ic_original[1, 0] = 0
                    _ic_original[0, -1] = 0
                    ax.scatter(
                        tm_original[i, 0, :],
                        tm_original[i, 1, :],
                        s=10,
                        c="black",
                        label=f"Original",
                        marker="x",
                    )
                    ax.plot(
                        _ic_original[0],
                        _ic_original[1],
                        c="black",
                        lw=2,
                    )
                    ax.set_title(f"Location {i+1}")

            case "RV_MAP":
                grid_res = 10
                min_lat, min_lon = np.min(self.latlon, axis=0)
                max_lat, max_lon = np.max(self.latlon, axis=0)
                lat_list = np.linspace(min_lat, max_lat, grid_res)
                lon_list = np.linspace(min_lon, max_lon, grid_res)
                dist_list, pos_list = self.tree.query(
                    [[[lat, lon] for lat in lat_list] for lon in lon_list]
                )
                pos_list = pos_list.flatten()
                tm_MSTME_ss = self.tm_MSTME_ss
                tm_PWE_ss = self.tm_PWE_ss
                # bi, ni, vi, ei
                assert tm_MSTME_ss.shape == tm_PWE_ss.shape
                return_period = kwargs["return_period"]
                file_name = file_name + f"_RP{return_period}"

                stm_min = [0, 0]
                stm_max = [20, 70]
                N_subsample = tm_MSTME_ss.shape[0]
                #########################################################
                fig, axes = plt.subplots(
                    2,
                    2,
                    figsize=(8 * 2, 6 * 2),
                    facecolor="white",
                )
                ax.set_xlim(stm_min[0], stm_max[0])
                ax.set_ylim(stm_min[1], stm_max[1])
                fig.supxlabel(r"$H_s$[m]")
                fig.supylabel(r"$U$[m/s]")
                for ni in pos_list:
                    tm_original = np.moveaxis(self.tm[:, :, ni].to_numpy(), 2, 0)
                    # Sample count over threshold
                    _num_events_extreme = tm_MSTME_ss.shape[3]
                    _exceedance_prob = 1 - self.thr_pct_com
                    _count_sample = round(
                        _num_events_extreme
                        / (return_period * self.occur_freq * _exceedance_prob)
                    )
                    _ic_original = []
                    _num_events_original = tm_original.shape[2]
                    _count_original = round(
                        _num_events_original / (return_period * self.occur_freq)
                    )

                    # Bootstraps
                    ic_MSTME = []
                    ic_PWE = []
                    for bi in range(N_subsample):
                        _ic_MSTME = _search_isocontour(
                            tm_MSTME_ss[bi, i, :, :], _count_sample
                        )
                        _ic_PWE = _search_isocontour(
                            tm_PWE_ss[bi, i, :, :], _count_sample
                        )
                        _ic_MSTME[1, 0] = 0
                        _ic_MSTME[0, -1] = 0
                        _ic_PWE[1, 0] = 0
                        _ic_PWE[0, -1] = 0
                        ic_MSTME.append(_ic_MSTME)
                        ic_PWE.append(_ic_PWE)
                    (
                        ic_band_MSTME_u,
                        ic_band_MSTME_l,
                        ic_band_MSTME_m,
                    ) = _get_interp_band(ic_MSTME, scale=stm_max[1] / stm_max[0])
                    ic_band_PWE_u, ic_band_PWE_l, ic_band_PWE_m = _get_interp_band(
                        ic_PWE, scale=stm_max[1] / stm_max[0]
                    )

                    # Original
                    _ic_original = _search_isocontour(
                        tm_original[i, :, :], _count_original
                    )
                    _ic_original[1, 0] = 0
                    _ic_original[0, -1] = 0

            case "Equivalent_fetch":
                V_max_track = self.ds.V_max
                V_max_ww3 = self.ds.STM_UV_10m * G_F
                Vfm = self.ds.Vfm
                Radius = self.ds.Radius
                fetch_from_track = _calc_eq_fetch(V_max_track, Vfm, r=Radius)
                fetch_from_WW3 = G * (self.stm[0] / (0.0016 * V_max_ww3)) ** 2

                idx_in_range_ww3 = (
                    (V_max_ww3 > 20) & (V_max_ww3 < 60) & (Vfm > 0) & (Vfm < 12)
                )
                idx_in_range_track = (
                    (V_max_track > 20) & (V_max_track < 60) & (Vfm > 0) & (Vfm < 12)
                )
                fig, ax = plt.subplots(
                    1,
                    1,
                    figsize=(8, 6),
                    facecolor="white",
                )
                ax.set_ylabel("Equivalent fetch $x$[m]")
                ax.set_xlabel("$V_{max}$[m/s]")
                ax.scatter(
                    V_max_track[idx_in_range_track],
                    fetch_from_track[idx_in_range_track],
                    marker="o",
                    s=1,
                    label="from Young's definition",
                )
                ax.scatter(
                    V_max_ww3[idx_in_range_ww3],
                    fetch_from_WW3[idx_in_range_ww3],
                    marker="^",
                    s=1,
                    label="from JONSWAP relationship",
                )

                V_max_sample = self.stm_sample[1] * G_F
                idx_in_range_sample = (V_max_sample > 20) & (V_max_sample < 60)
                fetch_from_WW3_sample = (
                    G * (self.stm_sample[0] / (0.0016 * V_max_sample)) ** 2
                )
                ax.scatter(
                    V_max_sample[idx_in_range_sample],
                    fetch_from_WW3_sample[idx_in_range_sample],
                    marker="s",
                    s=1,
                    label="from JONSWAP relationship, MSTM-E sampled STM",
                )
                ax.legend()

            case _:
                raise (ValueError(f"No figure defined with the name {fig_name}"))

        if self.dir_out != None:
            plt.savefig(f"{self.dir_out}/{file_name}.pdf", bbox_inches="tight")
            plt.savefig(f"{self.dir_out}/{file_name}.png", bbox_inches="tight")
        if not self.draw_fig:
            plt.close()
