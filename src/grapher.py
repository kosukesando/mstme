from __future__ import annotations

from functools import partial
from pathlib import Path

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from cartopy.mpl.ticker import (
    LatitudeFormatter,
    LatitudeLocator,
    LongitudeFormatter,
    LongitudeLocator,
)
from pathos.multiprocessing import ProcessPool
from scipy.stats._continuous_distns import genpareto
from shapely.geometry import LineString, MultiLineString, MultiPoint, Point
from statsmodels.distributions.empirical_distribution import ECDF

import mstmeclass as mc
from mstmeclass import G_F, GPPAR, MSTME, STM, Area, G

pos_color = plt.rcParams["axes.prop_cycle"].by_key()["color"]
plt.style.use(Path(__file__).parent / "plot_style.txt")


# plt.style.use("ggplot")
def create_custom_ticks(vmin, vmax, tick):
    ticks = []
    assert vmin < vmax
    start = (vmin + tick) // tick * tick
    end = -(vmax // -tick) * tick
    return [vmin] + list(np.arange(start, end, tick)) + [vmax]


def custom_map(
    ax,
    area: Area,
    tick_interval=0.5,
):
    ax.add_feature(cartopy.feature.LAND, edgecolor="black")
    ax.coastlines()
    ax.yaxis.tick_right()
    ax.set_xticks(
        create_custom_ticks(area.min_lon, area.max_lon, tick_interval),
        crs=ccrs.PlateCarree(),
    )
    ax.set_yticks(
        create_custom_ticks(area.min_lat, area.max_lat, tick_interval),
        crs=ccrs.PlateCarree(),
    )
    lon_formatter = LongitudeFormatter(zero_direction_label=False)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.set_xlim(area.min_lon, area.max_lon)
    ax.set_ylim(area.min_lat, area.max_lat)
    return ax


def _search_isocontour(scatter, n):
    """
    scatter: shape(v,e)
    """
    if len(scatter.shape) != 2:
        raise ValueError(f"shape {scatter.shape} of input scatter is not 2-D")
    coords = []
    scatter = np.unique(scatter, axis=1)
    _num_events = scatter.shape[1]
    _xg, _yg = np.sort(scatter[0]), np.sort(scatter[1])
    _xi, _yi = _num_events - 1, 0
    # Search isocontour
    # Keep searching until the edge
    while _xi >= 0 and _yi <= _num_events - 1:
        # Check if you can go up, in which case you do
        _count_up = np.count_nonzero(
            np.logical_and(scatter[0] >= _xg[_xi], scatter[1] >= _yg[_yi + 1])
        )
        if _count_up == n:
            _yi += 1
            coords.append([_xg[_xi], _yg[_yi]])
        # If you can't, look directly to the left
        elif _count_up > n:
            raise ("something's wrong")
        else:
            _xi -= 1
            _dyi = 0
            # Keep moving up from there until you reach a viable point
            # # If you reach the 2nd column from left or the 2nd row from top, exit
            while _xi >= 0 and _yi + _dyi <= _num_events - 1:
                _count_left = np.count_nonzero(
                    np.logical_and(
                        scatter[0] >= _xg[_xi],
                        scatter[1] >= _yg[_yi + _dyi],
                    )
                )
                if _count_left == n:
                    _yi += _dyi
                    coords.append([_xg[_xi], _yg[_yi]])
                    break
                elif _count_left > n:
                    _dyi += 1
                elif _count_left < n:
                    break

    contour = np.array(coords).T
    contour[1, 0] = -10
    contour[0, -1] = -10
    return contour


def _search_isocontours(scatters, n):
    """
    scatter: shape(ss, v, e)
    """
    if len(scatters.shape) != 3:
        raise ValueError(f"shape {scatters.shape} of input scatter is not 3-D")
    pool = ProcessPool()
    worker_partial = partial(_search_isocontours_worker, n)
    results = pool.map(worker_partial, scatters)
    contours = []
    for contour in results:
        contours.append(contour)
    return contours


def _search_isocontours_worker(n, scatter):
    print(f"Input shape:{scatter.shape}")
    if scatter.shape[0] != 2:
        raise ValueError(f"Input scatter has shape {scatter.shape}")
    coords = []
    scatter = np.unique(scatter, axis=1)
    _num_events = scatter.shape[1]
    _xg, _yg = np.sort(scatter[0]), np.sort(scatter[1])
    _xi, _yi = _num_events - 1, 0
    # Search isocontour
    # Keep searching until the edge
    while _xi >= 0 and _yi <= _num_events - 2:
        _count = np.count_nonzero(
            np.logical_and(scatter[0] >= _xg[_xi], scatter[1] >= _yg[_yi])
        )
        # Check if you can go up, in which case you do
        _count_up = np.count_nonzero(
            np.logical_and(scatter[0] >= _xg[_xi], scatter[1] >= _yg[_yi + 1])
        )
        if _count_up > _count:
            raise ValueError("something's wrong")
        elif _count_up == n:
            _yi += 1
            coords.append([_xg[_xi], _yg[_yi]])
        # If you can't, look directly to the left
        else:
            _xi -= 1
            _dyi = 0
            # Keep moving up from there until you reach a viable point
            # # If you reach the 2nd column from left or the 2nd row from top, exit
            while _xi >= 0 and _yi + _dyi <= _num_events - 2:
                _count_left = np.count_nonzero(
                    np.logical_and(
                        scatter[0] >= _xg[_xi],
                        scatter[1] >= _yg[_yi + _dyi],
                    )
                )
                if _count_left == n:
                    _yi += _dyi
                    coords.append([_xg[_xi], _yg[_yi]])
                    break
                elif _count_left > n:
                    _dyi += 1
                elif _count_left < n:
                    break

    contour = np.array(coords).T
    contour[1, 0] = 0
    contour[0, -1] = 0
    return contour


def _get_interp_band(contours, scale, res=11):
    """
    Returns the upper band and lower band
    contours:list of contours, each contour is a list of (x,y) with differing length
    scale: scale factor of plot ylim/xlim
    upper:np.ndarray((2,res))
    lower:np.ndarray((2,res))
    """
    # Make contours into list of MultiLineString objects
    mls = MultiLineString([LineString(c.T) for c in contours])
    # output variable
    upper = np.empty((2, res))
    lower = np.empty((2, res))
    means = np.empty((2, res))

    pool = ProcessPool()
    rads = np.linspace(0, np.pi / 2, res, endpoint=True)
    slopes = np.tan(rads) * scale
    worker_partial = partial(_interp_band_worker, mls)
    results = pool.map(worker_partial, slopes)
    for i, (u, l, m) in enumerate(results):
        upper[:, i] = u
        lower[:, i] = l
        means[:, i] = m
    return upper, lower, means


def _interp_band_worker(mls, a):
    line = LineString([(0, 0), (100, a * 100)])
    intersections = line.intersection(mls)
    l_array = [point.distance(Point(0, 0)) for point in intersections.geoms]
    lu = np.percentile(l_array, 97.5)
    ll = np.percentile(l_array, 2.5)
    xu = lu * np.cos(np.arctan(a))
    yu = lu * np.sin(np.arctan(a))
    xl = ll * np.cos(np.arctan(a))
    yl = ll * np.sin(np.arctan(a))
    upper = [xu, yu]
    lower = [xl, yl]
    lm = np.mean(l_array)
    xm = lm * np.cos(np.arctan(a))
    ym = lm * np.sin(np.arctan(a))
    mean = [xm, ym]
    return upper, lower, mean


def _get_interp_band_diag(contours, scale):
    """ """
    a = scale
    l0 = LineString([(-100, a * -100), (100, a * 100)])
    l_array = []
    for i, ct in enumerate(contours):
        l1 = LineString(ct.T)
        p = l0.intersection(l1)
        l_array.append(p.distance(Point(0, 0)))
    return l_array


def _trunc_band(ic_band):
    ic_band_round = np.round(ic_band, 2)
    xmax, ymax = ic_band_round[0, 0], ic_band_round[1, -1]
    ic_band_trunc = []
    for i in range(1, ic_band_round.shape[1] - 1):
        if ic_band_round[0, i] != xmax and ic_band_round[1, i] != ymax:
            print(f"{ic_band_round[0,i]}!={xmax} and {ic_band_round[1,i]}!={ymax}")
            ic_band_trunc.append(ic_band_round.T[i])
    ic_band_trunc = np.array(ic_band_trunc).T
    return ic_band_trunc


class Grapher:
    def __init__(self, mstme: MSTME, **kwargs):
        self.mstme = mstme
        self.num = 10
        self.stm_min = [0, 0]
        self.stm_max = [30, 80]
        return

    def draw_all(self, fig_names: list, **kwargs):
        draw_fig = kwargs.get("draw_fig", self.mstme.draw_fig)
        dir_out = kwargs.get("dir_out", self.mstme.dir_out)
        return_period = kwargs.get("return_period")
        draw_partial = partial(
            self.draw, draw_fig=draw_fig, dir_out=dir_out, return_period=return_period
        )
        ProcessPool().map(draw_partial, fig_names)
        return

    def draw(self, fig_name: str, **kwargs):
        """
        Genpar_Params
        Genpar_CDF
        """
        mstme = self.mstme
        draw_fig = kwargs.get("draw_fig", mstme.draw_fig)
        dir_out = kwargs.get("dir_out", mstme.dir_out)
        file_name = fig_name

        match fig_name:
            case "STM_Histogram":
                fig, ax = plt.subplots(
                    1,
                    mstme.num_vars,
                    figsize=(4 * mstme.num_vars, 3),
                    facecolor="white",
                )
                for S in STM:
                    vi = S.idx()
                    unit = S.unit()
                    var_name = S.name()
                    ax[vi].set_xlabel(f"{var_name}[{unit}]")
                    ax[vi].hist(
                        mstme.stm[vi],
                        bins=np.linspace(self.stm_min[vi], self.stm_max[vi], 20),
                    )

            case "STM_Histogram_filtered":
                _mask = mstme.mask
                fig, ax = plt.subplots(
                    2,
                    mstme.num_vars,
                    figsize=(4 * mstme.num_vars, 3 * 2),
                    facecolor="white",
                )
                # stm_min = np.floor(mstme.stm.min(axis=1) / 5) * 5
                # stm_max = np.ceil(mstme.stm.max(axis=1) / 5) * 5
                for S in STM:
                    vi = S.idx()
                    unit = S.unit()
                    var_name = S.name()
                    for i, b in enumerate([True, False]):
                        ax[i, vi].set_xlabel(f"{var_name[vi]}{unit[vi]}")
                        ax[i, vi].hist(
                            mstme.stm[vi, (_mask == b)],
                            bins=np.arange(self.stm_min[vi], self.stm_max[vi], 1),
                        )
                        ax[i, vi].set_title(f'{"is" if b else "not"} {mstme.rf}')

            case "STM_location":
                _mask = mstme.mask
                fig, ax = plt.subplots(
                    1,
                    mstme.num_vars,
                    figsize=(4 * mstme.num_vars, 3),
                    facecolor="white",
                )
                for S in STM:
                    vi = S.idx()

                    ax[vi].scatter(
                        mstme.latlon[:, 1],
                        mstme.latlon[:, 0],
                        c="black",
                        s=2,
                    )
                    ax[vi].scatter(
                        mstme.latlon[mstme.stm_node_idx[vi, :], 1],
                        mstme.latlon[mstme.stm_node_idx[vi, :], 0],
                        c="red",
                        s=20,
                        alpha=0.1,
                    )

            case "Tracks_vs_STM":
                fig, ax = plt.subplots(
                    1,
                    mstme.num_vars,
                    subplot_kw={"projection": ccrs.PlateCarree()},
                    figsize=(4 * mstme.num_vars, 3),
                    facecolor="white",
                )
                for S in STM:
                    vi = S.idx()
                    ax[vi] = custom_map(ax[vi], mstme.area)
                    cmap = plt.get_cmap("viridis", 100)
                    for ei in range(mstme.num_events):
                        ax[vi].plot(
                            mstme.tracks[ei][:, 0],
                            mstme.tracks[ei][:, 1],
                            c=cmap(mstme.stm[vi, ei] / mstme.stm[vi].max()),
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
                            vmin=mstme.stm[vi].min(), vmax=mstme.stm[vi].max()
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
                    mstme.num_vars,
                    figsize=(4 * mstme.num_vars, 3),
                    facecolor="white",
                )

                ni = kwargs.get("idx_location")
                for S in STM:
                    vi = S.idx()
                    var_name = S.name()
                    _ax: plt.Axes = ax[vi]
                    _ax.hist(mstme.tm[vi, :, ni], bins=20)
                    _ax.set_title(f"{var_name}")

            case "General_Map":
                fig, ax = plt.subplots(
                    1,
                    1,
                    figsize=(4, 3),
                    facecolor="white",
                    subplot_kw={"projection": ccrs.PlateCarree()},
                )
                ax = custom_map(ax, mstme.area)

                ax.scatter(
                    mstme.latlon[:, 1],
                    mstme.latlon[:, 0],
                    c="black",
                    s=kwargs.get("node_size", 3),
                )
                idx = kwargs.get("pos_list", None)
                if idx is not None:
                    for i, ni in enumerate(idx):
                        ax.scatter(
                            mstme.latlon[ni, 1],
                            mstme.latlon[ni, 0],
                            marker="x",
                            s=20,
                            c="red",
                            label=f"Location #{i:d}",
                        )
                        text_pos = ax.transData.transform(
                            (mstme.latlon[ni, 1], mstme.latlon[ni, 0])
                        )
                        text_pos = ax.transAxes.inverted().transform(text_pos)
                        print(ax.get_xlim(), mstme.latlon[ni, 1], text_pos)
                        # ax.scatter(
                        #     text_pos[0], text_pos[1], c="blue", transform=ax.transAxes
                        # )
                        plt.text(
                            text_pos[0],
                            text_pos[1] - 0.1,
                            f"#{i+1}",
                            c="red",
                            fontfamily="sans-serif",
                            ha="center",
                            bbox=dict(facecolor="white"),
                            transform=ax.transAxes,
                        )
                # ax.legend()

            case "General_Map_2":
                from matplotlib.patches import Rectangle

                fig, ax = plt.subplots(
                    1,
                    2,
                    figsize=(4 * 3.2, 3),
                    facecolor="white",
                    subplot_kw={"projection": ccrs.PlateCarree()},
                )
                # Entire region of simulation
                ax[0] = custom_map(
                    ax[0],
                    Area(
                        min_lon=-84.70,
                        max_lon=-50.00,
                        min_lat=8.40,
                        max_lat=22.10,
                    ),
                    tick_interval=5,
                )
                ax[0].add_patch(
                    Rectangle(
                        (mstme.area.min_lon, mstme.area.min_lat),
                        (mstme.area.max_lon - mstme.area.min_lon),
                        (mstme.area.max_lat - mstme.area.min_lat),
                        edgecolor="red",
                        fill=False,
                    )
                )
                # Region of interest
                ax[1] = custom_map(ax[1], mstme.area)

                ax[1].scatter(
                    mstme.latlon[:, 1],
                    mstme.latlon[:, 0],
                    c="black",
                    s=kwargs.get("node_size", 3),
                )
                idx = kwargs.get("pos_list", None)
                if idx is not None:
                    for i, ni in enumerate(idx):
                        ax[1].scatter(
                            mstme.latlon[ni, 1],
                            mstme.latlon[ni, 0],
                            marker="x",
                            s=20,
                            c="red",
                            label=f"Location #{i:d}",
                        )
                        text_pos = ax[1].transData.transform(
                            (mstme.latlon[ni, 1], mstme.latlon[ni, 0])
                        )
                        text_pos = ax[1].transAxes.inverted().transform(text_pos)
                        plt.text(
                            text_pos[0],
                            text_pos[1] - 0.1,
                            f"#{i+1}",
                            c="red",
                            fontfamily="sans-serif",
                            ha="center",
                            bbox=dict(facecolor="white"),
                            transform=ax[1].transAxes,
                        )
                fig.tight_layout()
                # ax.legend()

            case "Genpar_Params":
                fig, ax = plt.subplots(
                    len(list(GPPAR)),
                    mstme.num_vars,
                    figsize=(4 * mstme.num_vars, 3 * len(list(GPPAR))),
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
                        ax[pi, vi].hist(mstme.gp_params[vi, :, pi])

            case "Genpar_CDF":
                fig, ax = plt.subplots(
                    1,
                    mstme.num_vars,
                    figsize=(4 * mstme.num_vars, 3),
                    facecolor="white",
                )

                N_gp = mstme.gp_params.shape[1]
                _res = 100
                for S in STM:
                    vi = S.idx()
                    var_name = S.name()
                    unit = S.unit()
                    _cdf_all = np.zeros((N_gp, _res))
                    _x = np.linspace(mstme.thr_mar[vi], mstme.stm[vi].max(), _res)
                    for i in range(N_gp):
                        _xp = mstme.gp_params[vi, i, 0]
                        _mp = mstme.gp_params[vi, i, 1]
                        _sp = mstme.gp_params[vi, i, 2]
                        _cdf_all[i, :] = genpareto(_xp, _mp, _sp).cdf(_x)

                    _y = mstme.gp[vi].cdf(_x)
                    _u95 = np.percentile(_cdf_all, 97.5, axis=0)
                    _l95 = np.percentile(_cdf_all, 2.5, axis=0)
                    ax[vi].plot(_x, _y, c="blue", lw=2, alpha=1, label="Bootstrap Mean")
                    ax[vi].fill_between(_x, _u95, _l95, alpha=0.5, label="95% CI")
                    _ecdf = ECDF(mstme.stm[vi, mstme.is_e_mar[vi]])
                    _x = np.linspace(mstme.thr_mar[vi], mstme.stm[vi].max(), _res)
                    ax[vi].plot(_x, _ecdf(_x), lw=2, color="black", label="Empirical")
                    ax[vi].set_xlabel(f"{var_name}[{unit}]")
                    ax[vi].legend()

            case "Original_vs_Normalized":
                fig, ax = plt.subplots(
                    1,
                    2,
                    figsize=(7, 3),
                    facecolor="white",
                )
                ax[0].set_aspect(1)
                ax[0].scatter(mstme.stm_g[0], mstme.stm_g[1], s=5)
                ax[0].set_xlabel(r"$\hat H_s$")
                ax[0].set_ylabel(r"$\hat U$")
                ax[0].set_xlim(-5, 15)
                ax[0].set_ylim(-5, 15)
                ax[0].set_xticks([-2 + 2 * i for i in range(6)])
                ax[0].set_yticks([-2 + 2 * i for i in range(6)])

                ax[1].scatter(mstme.stm[0], mstme.stm[1], s=5)
                ax[1].set_xlabel(f"{STM.H.name()}[{STM.H.unit()}]")
                ax[1].set_ylabel(f"{STM.U.name()}[{STM.U.unit()}]")
                ax[1].set_xlim(0, 20)
                ax[1].set_ylim(0, 60)

            case "Kendall_Tau_all_var_pval":
                ###
                # Logic code
                if not hasattr(mstme, "pval"):
                    mstme.calc_kendall_tau()
                elif mstme.pval is None:
                    mstme.calc_kendall_tau()
                ###
                fig, ax = plt.subplots(
                    mstme.num_vars,
                    mstme.num_vars,
                    sharey=True,
                    figsize=(4 * mstme.num_vars, 3 * mstme.num_vars),
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
                            "red" if p < 0.05 else "black"
                            for p in mstme.pval[vi, vj, :]
                        ]
                        im = ax[vi, vj].scatter(
                            mstme.latlon[:, 1],
                            mstme.latlon[:, 0],
                            s=5,
                            c=_c,
                        )
                        ax[vi, vj].set_title(f"STM:{var_name_i} E:{var_name_j}")

            case "Kendall_Tau_all_var_tval":
                ###
                # Logic code
                if not hasattr(mstme, "tval"):
                    mstme.calc_kendall_tau()
                elif mstme.tval is None:
                    mstme.calc_kendall_tau()
                ###
                fig, ax = plt.subplots(
                    mstme.num_vars,
                    mstme.num_vars,
                    sharey=True,
                    figsize=(4 * mstme.num_vars, 3 * mstme.num_vars),
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
                            mstme.latlon[:, 1],
                            mstme.latlon[:, 0],
                            s=5,
                            c=mstme.tval[vi, vj, :],
                            cmap="seismic",
                            vmax=np.abs(mstme.tval[vi]).max(),
                            vmin=-np.abs(mstme.tval[vi]).max(),
                        )
                        plt.colorbar(im, ax=ax[vi, vj])
                        ax[vi, vj].set_title(f"STM:{var_name_i} E:{var_name_j}")

            case "Kendall_Tau_marginal_pval":
                ###
                # Logic code
                if not hasattr(mstme, "pval"):
                    mstme.calc_kendall_tau()
                elif mstme.pval is None:
                    mstme.calc_kendall_tau()
                ###
                fig, ax = plt.subplots(
                    1,
                    mstme.num_vars,
                    sharey=True,
                    figsize=(4 * mstme.num_vars, 3),
                    facecolor="white",
                )

                for S in STM:
                    vi = S.idx()
                    var_name_i = S.name()
                    ax[vi].set_xlabel("Longitude")
                    ax[vi].set_ylabel("Latitude")
                    _c = ["red" if p < 0.05 else "black" for p in mstme.pval[vi, vi, :]]
                    im = ax[vi].scatter(
                        mstme.latlon[:, 1],
                        mstme.latlon[:, 0],
                        s=5,
                        c=_c,
                    )
                    ax[vi].set_title(f"{var_name_i}")

            case "Kendall_Tau_marginal_tval":
                ###
                # Logic code
                if not hasattr(mstme, "tval"):
                    mstme.calc_kendall_tau()
                elif mstme.tval is None:
                    mstme.calc_kendall_tau()
                ###
                fig, ax = plt.subplots(
                    1,
                    mstme.num_vars,
                    sharey=True,
                    figsize=(4 * mstme.num_vars, 3),
                    facecolor="white",
                )

                for S in STM:
                    vi = S.idx()
                    var_name_i = S.name()

                    ax[vi].set_xlabel("Longitude")
                    ax[vi].set_ylabel("Latitude")
                    im = ax[vi].scatter(
                        mstme.latlon[:, 1],
                        mstme.latlon[:, 0],
                        s=5,
                        c=mstme.tval[vi, vi, :],
                        cmap="seismic",
                        vmax=np.abs(mstme.tval[vi]).max(),
                        vmin=-np.abs(mstme.tval[vi]).max(),
                    )
                    ax[vi].set_title(f"{var_name_i}")

            case "Replacement":
                fig, ax = plt.subplots(
                    1,
                    1,
                    figsize=(4, 3),
                    facecolor="white",
                )

                ax.scatter(
                    mstme.stm_g_rep[:, 0, :],
                    mstme.stm_g_rep[:, 1, :],
                    label="Replacement",
                )
                ax.scatter(
                    mstme.stm_g[0],
                    mstme.stm_g[1],
                    color="black",
                    label="Original",
                )
                ax.set_xlabel(r"$\hat H_s$")
                ax.set_ylabel(r"$\hat U$")
                ax.set_xlim(-3, 15)
                ax.set_ylim(-3, 15)
                ax.legend(loc="upper left")

            case "Conmul_Estimates":
                fig, ax = plt.subplots(
                    4,
                    mstme.num_vars,
                    figsize=(4 * mstme.num_vars, 3 * 4),
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

                    ax[0, vi].hist(mstme.params_uc[vi, :, 0])
                    ax[1, vi].hist(mstme.params_uc[vi, :, 1])
                    ax[2, vi].hist(mstme.params_uc[vi, :, 2])
                    ax[3, vi].hist(mstme.params_uc[vi, :, 3])

            case "ab_Estimates":
                fig, ax = plt.subplots(
                    1,
                    mstme.num_vars,
                    figsize=(4 * mstme.num_vars, 3),
                    facecolor="white",
                )

                fig.supxlabel("$a$")
                fig.supylabel("$b$")
                params_ml = np.zeros((4, mstme.num_vars))
                for S in STM:
                    vi = S.idx()
                    var_name = S.name()
                    ax[vi].set_xlim(0, 1)
                    ax[vi].set_ylim(-1, 1)
                    ax[vi].scatter(
                        mstme.params_uc[vi, :, 0],
                        mstme.params_uc[vi, :, 1],
                        s=10,
                        label="Generated samples",
                    )
                    a_hat = mstme.params_median[vi, 0]
                    b_hat = mstme.params_median[vi, 1]
                    ax[vi].scatter(
                        a_hat,
                        b_hat,
                        s=40,
                        c="red",
                    )
                    ax[vi].text(
                        a_hat,
                        b_hat + 0.05,
                        f"$(\hat a, \hat b)$",
                        fontsize=30,
                        ha="center",
                        c="red",
                    )
                    ax[vi].set_title(var_name)

            case "amu_Estimates":
                fig, ax = plt.subplots(
                    1,
                    mstme.num_vars,
                    figsize=(4 * mstme.num_vars, 3),
                    facecolor="white",
                )

                fig.supxlabel("$a$")
                fig.supylabel("$\mu$")
                params_ml = np.zeros((4, mstme.num_vars))
                for S in STM:
                    vi = S.idx()
                    var_name = S.name()
                    ax[vi].set_xlim(0, 1)
                    ax[vi].set_ylim(-0.1, 2)
                    ax[vi].scatter(
                        mstme.params_uc[vi, :, 0],
                        mstme.params_uc[vi, :, 2],
                        s=5,
                        label="Generated samples",
                    )
                    ax[vi].set_title(var_name)

            case "a+mub_Estimates":
                fig, ax = plt.subplots(
                    1,
                    mstme.num_vars,
                    figsize=(4 * mstme.num_vars, 3),
                    facecolor="white",
                )

                fig.supxlabel("$a+\mu$")
                fig.supylabel("$b$")
                params_ml = np.zeros((4, mstme.num_vars))
                for S in STM:
                    vi = S.idx()
                    var_name = S.name()
                    ax[vi].set_xlim(0.5, 2)
                    ax[vi].set_ylim(-1, 1)
                    ax[vi].scatter(
                        mstme.params_uc[vi, :, 0] + mstme.params_uc[vi, :, 2],
                        mstme.params_uc[vi, :, 1],
                        s=5,
                        label="Generated samples",
                    )
                    ax[vi].set_title(var_name)

            case "Residuals":
                fig, ax = plt.subplots(
                    1,
                    mstme.num_vars,
                    figsize=(4 * mstme.num_vars, 3),
                    facecolor="white",
                )

                # fig.tight_layout()
                for S in STM:
                    vi = S.idx()
                    var_name = S.name()
                    ax[vi].scatter(
                        mstme.ndist.cdf(mstme.stm_g[vi, mstme.is_e[vi]]),
                        mstme.residual[vi],
                        s=5,
                    )
                    ax[vi].set_xlabel(f"$F^*$({var_name}$)$")
                ax[0].set_ylabel("$Z_{-j}$")

            case "Simulated_Conmul_vs_Back_Transformed":
                fig, ax = plt.subplots(
                    1,
                    mstme.num_vars,
                    figsize=(8 * mstme.num_vars, 6),
                    facecolor="white",
                )

                ax[0].set_aspect(1)
                a_h, b_h, mu_h, sg_h = mstme.params_median[0, :]
                a_u, b_u, mu_u, sg_u = mstme.params_median[1, :]
                # sample_given_h = []
                # sample_given_u = []
                # sample_given_hg = []
                # sample_given_ug = []
                # for i, vi in enumerate(mstme.vi_list):
                #     if vi == 0:
                #         sample_given_h.append(mstme.sample_full[:, i])
                #         sample_given_hg.append(mstme.sample_full_g[:, i])
                #     if vi == 1:
                #         sample_given_u.append(mstme.sample_full[:, i])
                #         sample_given_ug.append(mstme.sample_full_g[:, i])
                # sample_given_h = np.array(sample_given_h).T
                # sample_given_u = np.array(sample_given_u).T
                # sample_given_hg = np.array(sample_given_hg).T
                # sample_given_ug = np.array(sample_given_ug).T
                mask = mstme.vi_list == 0
                sample_given_h = mstme.sample_full[:, mask]
                sample_given_u = mstme.sample_full[:, ~mask]
                sample_given_hg = mstme.sample_full_g[:, mask]
                sample_given_ug = mstme.sample_full_g[:, ~mask]

                x_h = np.linspace(mstme.thr_com, 10, 100)
                y_h = x_h * a_h + (x_h**b_h) * mu_h
                ax[0].plot(
                    x_h, y_h, color="orange", label="$\hat{U}=a\hat{H}+\mu\hat{H}^b$"
                )

                y_u = np.linspace(mstme.thr_com, 10, 100)
                x_u = y_u * a_u + (y_u**b_u) * mu_u
                ax[0].plot(
                    x_u, y_u, color="teal", label="$\hat{H}=a\hat{U}+\mu\hat{U}^b$"
                )

                ax[0].scatter(
                    mstme.stm_g[0],
                    mstme.stm_g[1],
                    s=5,
                    color="black",
                    label="Original",
                )
                ax[0].axvline(mstme.thr_com, color="black")
                ax[0].axhline(mstme.thr_com, color="black")

                ax[0].set_xlabel(r"$\hat H_s$")
                ax[0].set_ylabel(r"$\hat U$")
                ax[0].set_xlim(-2, 10)
                ax[0].set_ylim(-2, 10)
                ax[0].scatter(
                    sample_given_hg[0],
                    sample_given_hg[1],
                    s=1,
                    color="orange",
                    label="Simulated $(\hat{U}|\hat{H}>\hat{\mu})$",
                )
                ax[0].scatter(
                    sample_given_ug[0],
                    sample_given_ug[1],
                    s=1,
                    color="teal",
                    label="Simulated $(\hat{H}|\hat{U}>\hat{\mu})$",
                )
                # ax[0].legend()

                ax[1].set_xlim(0, 25)
                ax[1].set_ylim(0, 60)
                ax[1].scatter(
                    mstme.stm[0],
                    mstme.stm[1],
                    color="black",
                    s=5,
                    label="Original",
                )
                ax[1].scatter(
                    sample_given_h[0],
                    sample_given_h[1],
                    color="orange",
                    s=1,
                    label="Simulated $(U|H>\mu_{H_s})$",
                )
                ax[1].scatter(
                    sample_given_u[0],
                    sample_given_u[1],
                    color="teal",
                    s=1,
                    label="Simulated $(H|U>\mu_{U_{10}})$",
                )
                ax[1].set_xlabel(f"{STM.H.name()}[{STM.H.unit()}]")
                ax[1].set_ylabel(f"{STM.U.name()}[{STM.U.unit()}]")

                # original
                return_period = 100

                _count_original = round(
                    mstme.num_events / (return_period * mstme.occur_freq)
                )
                _ic_original = _search_isocontour(mstme.stm, _count_original)

                # sample
                _num_events_sample = mstme.sample_full.shape[1]
                _exceedance_prob = 1 - mstme.thr_pct_com
                _count_sample = round(
                    _num_events_sample
                    / (return_period * mstme.occur_freq * _exceedance_prob)
                )
                _ic_sample = _search_isocontour(mstme.sample_full, _count_sample)

                # ax[1].plot(
                #     _ic_original[0],
                #     _ic_original[1],
                #     c="black",
                #     lw=2,
                #     label=f"Empirical {return_period}-yr RV",
                # )
                # ax[1].plot(
                #     _ic_sample[0],
                #     _ic_sample[1],
                #     c="red",
                #     lw=2,
                #     label=f"Simulated {return_period}-yr RV",
                # )
                # ax[1].legend()

            case "RV":
                fig, axes = plt.subplots(
                    2,
                    2,
                    figsize=(4 * mstme.num_vars, 3 * mstme.num_vars),
                    facecolor="white",
                )

                return_period = kwargs.get("return_period")
                file_name = file_name + f"_RP{return_period}"
                tm_sample = kwargs.get("tm_MSTME")  # (v,e,n)
                # tm_sample = mstme.tm_sample  # (v,e,n)
                tm_original = mstme.tm  # (v,e,n)
                # stm_min = np.floor(tm_sample[:, :, mstme.idx_pos_list].min(axis=(1, 2)) / 5) * 5
                # stm_max = np.ceil(tm_sample[:, :, mstme.idx_pos_list].max(axis=(1, 2)) / 5) * 5
                #########################################################
                fig.supxlabel(r"$H_s$[m]")
                fig.supylabel(r"$U$[m/s]")
                for i, ax in enumerate(axes.flatten()):
                    ax.set_xlim(self.stm_min[0], self.stm_max[0])
                    ax.set_ylim(self.stm_min[1], self.stm_max[1])
                    _linestyles = ["-", "--"]
                    _idx_pos = mstme.idx_pos_list[i]
                    # sample
                    _num_events_sample = tm_sample.shape[1]
                    _exceedance_prob = 1 - mstme.thr_pct_com
                    _count_sample = round(
                        _num_events_sample
                        / (return_period * mstme.occur_freq * _exceedance_prob)
                    )
                    _ic_sample = _search_isocontour(
                        tm_sample[:, :, _idx_pos], _count_sample
                    )

                    # original
                    _count_original = round(
                        mstme.num_events / (return_period * mstme.occur_freq)
                    )
                    _ic_original = _search_isocontour(
                        tm_original[:, :, _idx_pos], _count_original
                    )

                    ax.scatter(
                        tm_original[0, :, _idx_pos],
                        tm_original[1, :, _idx_pos],
                        s=10,
                        c="black",
                        label=f"Original temporal maxima",
                    )
                    ax.scatter(
                        tm_sample[0, :, _idx_pos],
                        tm_sample[1, :, _idx_pos],
                        s=2,
                        c=pos_color[i],
                        label=f"Simulated temporal maxima(MSTM-E)",
                    )
                    ax.plot(
                        _ic_original[0],
                        _ic_original[1],
                        c="black",
                        lw=2,
                        label=f"Empirical {return_period}-yr RV",
                    )
                    ax.plot(
                        _ic_sample[0],
                        _ic_sample[1],
                        c=pos_color[i],
                        lw=2,
                        label=f"Simulated {return_period}-yr RV(MSTM-E)",
                    )
                    ax.set_title(f"Location {i+1}")
                    ax.legend()

            case "RV_PWE":
                fig, axes = plt.subplots(
                    2,
                    2,
                    figsize=(4 * mstme.num_vars, 3 * mstme.num_vars),
                    facecolor="white",
                )

                # tm_sample(#ofLoc(=4), num_vars, num_events)
                return_period = kwargs.get("return_period")
                file_name = file_name + f"_RP{return_period}"
                tm_sample = mstme.tm_sample_PWE  # (v,e,n)
                tm_original = mstme.tm_original_PWE  # (v,e,n)
                #########################################################
                fig.supxlabel(r"$H_s$[m]")
                fig.supylabel(r"$U$[m/s]")
                for i, ax in enumerate(axes.flatten()):
                    ax.set_xlim(self.stm_min[0], self.stm_max[0])
                    ax.set_ylim(self.stm_min[1], self.stm_max[1])
                    _linestyles = ["-", "--"]
                    # sample
                    _num_events_sample = tm_sample.shape[1]
                    _exceedance_prob = 1 - mstme.thr_pct_com
                    _count_sample = round(
                        _num_events_sample
                        / (return_period * mstme.occur_freq * _exceedance_prob)
                    )
                    _ic_sample = _search_isocontour(tm_sample[:, :, i], _count_sample)

                    # original
                    _ic_original = []
                    _count_original = round(
                        mstme.num_events / (return_period * mstme.occur_freq)
                    )
                    _ic_original = _search_isocontour(
                        tm_original[:, :, i], _count_original
                    )
                    ax.scatter(
                        tm_original[0, :, i],
                        tm_original[1, :, i],
                        s=10,
                        c="black",
                        label=f"Original temporal maxima",
                    )
                    ax.scatter(
                        tm_sample[0, :, i],
                        tm_sample[1, :, i],
                        s=2,
                        c=pos_color[i],
                        label=f"Simulated temporal maxima(PWE)",
                    )
                    ax.plot(
                        _ic_original[0],
                        _ic_original[1],
                        c="black",
                        lw=2,
                        label=f"Empirical {return_period}-yr RV",
                    )
                    ax.plot(
                        _ic_sample[0],
                        _ic_sample[1],
                        c=pos_color[i],
                        lw=2,
                        label=f"Simulated {return_period}-yr RV(PWE)",
                    )
                    ax.set_title(f"Location {i+1}")
                    ax.legend()

            case "RV_STM":
                stm_MSTME_ss = kwargs.get("stm_MSTME_ss")
                return_period = kwargs.get("return_period")
                file_name = file_name + f"_RP{return_period}"
                N_subsample = stm_MSTME_ss.shape[0]
                # bi, vi, ei
                fig, ax = plt.subplots(
                    1,
                    1,
                    figsize=(4, 3),
                    facecolor="white",
                )
                ax.set_xlabel(r"$H_s$[m]")
                ax.set_ylabel(r"$U$[m/s]")
                ax.set_xlim(self.stm_min[0], self.stm_max[0])
                ax.set_ylim(self.stm_min[1], self.stm_max[1])
                # Sample count over threshold
                _num_events_sample = stm_MSTME_ss.shape[2]
                _exceedance_prob = 1 - mstme.thr_pct_com
                _count_sample = round(
                    _num_events_sample
                    / (return_period * mstme.occur_freq * _exceedance_prob)
                )
                _num_events_original = mstme.num_events
                _count_original = round(
                    _num_events_original / (return_period * mstme.occur_freq)
                )

                # Bootstraps
                _ic_MSTME = []
                for bi in range(N_subsample):
                    _ic = _search_isocontour(stm_MSTME_ss[bi, :, :], _count_sample)
                    _ic[1, 0] = 0
                    _ic[0, -1] = 0
                    _ic_MSTME.append(_ic)

                # Original
                _ic_original = _search_isocontour(mstme.stm[:, :], _count_original)

                (
                    _ic_band_MSTME_u,
                    _ic_band_MSTME_l,
                    _ic_band_MSTME_m,
                ) = _get_interp_band(_ic_MSTME, scale=self.stm_max[1] / self.stm_max[0])

                # array = np.concatenate(
                #     (_ic_band_MSTME_u, np.flip(_ic_band_MSTME_l, axis=1)), axis=1
                # )
                # ax.fill(array[0], array[1], alpha=0.5)
                ax.plot(
                    _ic_band_MSTME_u[0],
                    _ic_band_MSTME_u[1],
                    c=pos_color[0],
                    lw=2,
                    ls="--",
                )
                ax.plot(
                    _ic_band_MSTME_l[0],
                    _ic_band_MSTME_l[1],
                    c=pos_color[0],
                    lw=2,
                    ls="--",
                )
                ax.plot(
                    _ic_band_MSTME_m[0],
                    _ic_band_MSTME_m[1],
                    c=pos_color[0],
                    lw=3,
                    # ls="--",
                )

                ######################################
                ax.scatter(
                    mstme.stm[0, :],
                    mstme.stm[1, :],
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
                tm_original = mstme.tm[:, :, mstme.idx_pos_list]
                tm_MSTME_ss = mstme.tm_MSTME_ss
                tm_PWE_ss = mstme.tm_PWE_ss
                return_period = kwargs.get("return_period")
                file_name = file_name + f"_RP{return_period}"

                # bi, ni, vi, ei
                assert tm_MSTME_ss.shape == tm_PWE_ss.shape
                N_subsample = tm_MSTME_ss.shape[0]
                #########################################################
                fig, axes = plt.subplots(
                    2,
                    2,
                    figsize=(4 * mstme.num_vars, 3 * mstme.num_vars),
                    facecolor="white",
                )
                fig.supxlabel(r"$H_s$[m]")
                fig.supylabel(r"$U$[m/s]")
                for i, ax in enumerate(axes.flatten()):
                    ax.set_xlim(self.stm_min[0], self.stm_max[0])
                    ax.set_ylim(self.stm_min[1], self.stm_max[1])
                    # Sample count over threshold
                    _num_events_sample = tm_MSTME_ss.shape[2]
                    _exceedance_prob = 1 - mstme.thr_pct_com
                    _count_sample = round(
                        _num_events_sample
                        / (return_period * mstme.occur_freq * _exceedance_prob)
                    )
                    _ic_original = []
                    _num_events_original = tm_original.shape[1]
                    _count_original = round(
                        _num_events_original / (return_period * mstme.occur_freq)
                    )

                    # Bootstraps
                    ic_MSTME = []
                    ic_PWE = []
                    for bi in range(N_subsample):
                        _ic_MSTME = _search_isocontour(
                            tm_MSTME_ss[bi, :, :, i], _count_sample
                        )
                        _ic_PWE = _search_isocontour(
                            tm_PWE_ss[bi, :, :, i], _count_sample
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
                    ) = _get_interp_band(
                        ic_MSTME, scale=self.stm_max[1] / self.stm_max[0]
                    )
                    ic_band_PWE_u, ic_band_PWE_l, ic_band_PWE_m = _get_interp_band(
                        ic_PWE, scale=self.stm_max[1] / self.stm_max[0]
                    )

                    _fill_MSTME = np.concatenate(
                        (ic_band_MSTME_u, np.flip(ic_band_MSTME_l, axis=1)), axis=1
                    )
                    _fill_PWE = np.concatenate(
                        (ic_band_PWE_u, np.flip(ic_band_PWE_l, axis=1)), axis=1
                    )
                    ax.fill(
                        _fill_MSTME[0],
                        _fill_MSTME[1],
                        alpha=0.2,
                        label=f"MSTME {return_period}-yr RV 95%CI",
                    )
                    ax.fill(
                        _fill_PWE[0],
                        _fill_PWE[1],
                        alpha=0.2,
                        label=f"PWE {return_period}-yr RV 95%CI",
                    )

                    # Original
                    _ic_original = _search_isocontour(
                        tm_original[:, :, i], _count_original
                    )

                    ax.scatter(
                        tm_original[0, :, i],
                        tm_original[1, :, i],
                        s=10,
                        c="black",
                        label=f"Original temporal maxima",
                        marker="x",
                    )
                    ax.plot(
                        _ic_original[0],
                        _ic_original[1],
                        c="black",
                        lw=2,
                        label=f"Empirical {return_period}-yr RV",
                    )
                    ax.set_title(f"Location {i+1}")
                    if i == 0:
                        ax.legend()

            case "RV_MAP":
                grid_res = 10
                mstme = mstme
                area = mstme.area
                lat_list = np.linspace(area.min_lat, area.max_lat, grid_res)
                lon_list = np.linspace(area.min_lon, area.max_lon, grid_res)
                dist_list, pos_list = mstme.tree.query(
                    [[[lat, lon] for lat in lat_list] for lon in lon_list]
                )
                pos_list = pos_list.flatten()
                tm_MSTME_ss_norm = np.empty(mstme.tm_sample.shape)  # (ss,v,e,n)
                tm_PWE_ss_norm = np.empty(mstme.tm_PWE_ss.shape)
                tm_original = mstme.tm  # (v,e,n)

                for S in STM:
                    vi = S.idx()
                    tm_MSTME_ss_norm[:, vi, :, :] = mstme.ndist.ppf(
                        mstme.mix_dist[vi].cdf(mstme.tm_sample[:, vi, :, :])
                    )
                    tm_PWE_ss_norm[:, vi, :, :] = mstme.ndist.ppf(
                        mstme.mix_dist[vi].cdf(mstme.tm_PWE_ss[:, vi, :, :])
                    )
                # bi, ni, vi, ei
                assert tm_MSTME_ss_norm.shape == tm_PWE_ss_norm.shape
                return_period = kwargs.get("return_period")
                file_name = file_name + f"_RP{return_period}"

                N_subsample = tm_MSTME_ss_norm.shape[0]
                _num_events_sample = tm_MSTME_ss_norm.shape[2]
                _exceedance_prob = 1 - mstme.thr_pct_com

                l_array_MSTME = []
                l_array_PWE = []
                l_array_original = []
                for ni in pos_list:
                    # Sample count over threshold
                    _count_sample = round(
                        _num_events_sample
                        / (return_period * mstme.occur_freq * _exceedance_prob)
                    )
                    _ic_original = []
                    _num_events_original = tm_original.shape[1]
                    _count_original = round(
                        _num_events_original / (return_period * mstme.occur_freq)
                    )

                    # Bootstraps
                    ic_MSTME = []
                    # ic_PWE = []
                    for bi in range(N_subsample):
                        _ic_MSTME = _search_isocontour(
                            tm_MSTME_ss_norm[bi, :, :, ni], _count_sample
                        )
                        # _ic_PWE = _search_isocontour(
                        #     tm_PWE_ss_norm[bi, :, :, ni], _count_sample
                        # )
                        ic_MSTME.append(_ic_MSTME)
                        # # ic_PWE.append(_ic_PWE)
                    l_array_MSTME.append(_get_interp_band_diag(ic_MSTME))
                    # # l_array_PWE.append(_get_interp_band(ic_PWE))

                    # Original
                    _ic_original = _search_isocontour(
                        tm_original[:, :, ni], _count_original
                    )
                    l_array_original.append(_get_interp_band_diag(_ic_original))
                l_array_MSTME = np.array(l_array_MSTME)
                # # l_array_PWE = np.array(l_array_PWE)
                l_array_original = np.array(l_array_original)

                bias_MSTME = l_array_MSTME - l_array_original
                # # bias_PWE = l_array_PWE - l_array_original
                var_MSTME = np.var(l_array_MSTME, axis=1)
                # # var_PWE = np.var(l_array_PWE, axis=1)
                #########################################################
                fig, axes = plt.subplots(
                    2,
                    2,
                    figsize=(4 * mstme.num_vars, 3 * mstme.num_vars),
                    facecolor="white",
                )

                ax.set_xlim(area.min_lon, area.max_lon)
                ax.set_ylim(area.min_lat, area.max_lat)
                ax[0, 0].scatter(
                    mstme.latlon[pos_list, 1],
                    mstme.latlon[pos_list, 0],
                    c=bias_MSTME,
                )
                # ax[1, 0].scatter(
                #     mstme.latlon[pos_list, 1],
                #     mstme.latlon[pos_list, 0],
                #     c=var_MSTME,
                # )

            case "Equivalent_fetch":
                V_max_track = mstme.ds.V_max
                V_max_ww3 = mstme.ds.STM_UV_10m * G_F
                Vfm = mstme.ds.Vfm
                Radius = mstme.ds.Radius
                fetch_from_track = mc._calc_eq_fetch(V_max_track, Vfm, r=Radius)
                fetch_from_WW3 = G * (mstme.stm[0] / (0.0016 * V_max_ww3)) ** 2

                idx_in_range_ww3 = (
                    (V_max_ww3 > 20) & (V_max_ww3 < 60) & (Vfm > 0) & (Vfm < 12)
                )
                idx_in_range_track = (
                    (V_max_track > 20) & (V_max_track < 60) & (Vfm > 0) & (Vfm < 12)
                )
                fig, ax = plt.subplots(
                    1,
                    1,
                    figsize=(4, 3),
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

                V_max_sample = mstme.stm_sample[1] * G_F
                idx_in_range_sample = (V_max_sample > 20) & (V_max_sample < 60)
                fetch_from_WW3_sample = (
                    G * (mstme.stm_sample[0] / (0.0016 * V_max_sample)) ** 2
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

        if dir_out != None:
            plt.savefig(f"{dir_out}/{file_name}.pdf", bbox_inches="tight")
            plt.savefig(f"{dir_out}/{file_name}.png", bbox_inches="tight")
        if not draw_fig:
            plt.close()
