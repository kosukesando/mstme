# %% import libs
# init
import enum
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import argparse
from datetime import datetime
import importlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.spatial import KDTree
from statsmodels.distributions.empirical_distribution import ECDF
import sys
import xarray as xr
from tqdm import trange, tqdm
from sklearn.cluster import DBSCAN
from geopy.distance import geodesic
from datetime import datetime
import pickle
from shapely.geometry import LineString, Point

# Custom
import src.stme as stme
import src.threshold_search as threshold_search
import src.mstmeclass as mc
from src.mstmeclass import STM
from src.is_interactive import is_interactive

plt.style.use("plot_style.txt")
pos_color = plt.rcParams["axes.prop_cycle"].by_key()["color"]
rng = np.random.default_rng(seed=1000)

# %% define functions


def ccw(A, B, C):
    return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)


# Return true if line segments AB and CD intersect
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def get_interp_band(contours):
    res = 10
    upper = np.empty((2, res))
    lower = np.empty((2, res))
    for i, rad in enumerate(np.linspace(0, np.pi / 2, res, endpoint=True)):
        a = np.tan(rad)
        points = []
        for ci, c in enumerate(contours):
            if not np.isinf(a):
                p0 = Point(0, 0)
                p1 = Point(100, a * 100)
                for j in range(c.shape[1] - 1):
                    q0 = Point(c[0, j], c[1, j])
                    q1 = Point(c[0, j + 1], c[1, j + 1])
                    if intersect(p0, p1, q0, q1):
                        line1 = LineString([[p0.x, p0.y], [p1.x, p1.y]])
                        line2 = LineString([[q0.x, q0.y], [q1.x, q1.y]])
                        int_pt = line1.intersection(line2)
                        points.append(int_pt)
                        break
            else:
                continue
        l_array = [np.sqrt(p.x**2 + p.y**2) for p in points]
        lu = np.percentile(l_array, 90)
        ll = np.percentile(l_array, 10)
        xu = lu * np.cos(rad)
        yu = lu * np.sin(rad)
        xl = ll * np.cos(rad)
        yl = ll * np.sin(rad)
        upper[:, i] = [xu, yu]
        lower[:, i] = [xl, yl]
    return upper, lower


def plot_isocontour_stm(stm_original, stm_MSTME_ss, return_period, dir_out=None):
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
    _exceedance_prob = 1 - cluster.thr_pct_com
    _count_sample = round(
        _num_events_extreme / (return_period * cluster.occur_freq * _exceedance_prob)
    )
    _num_events_original = tm_original.shape[2]
    _count_original = round(_num_events_original / (return_period * cluster.occur_freq))

    # Bootstraps
    _ic_MSTME = []
    for bi in range(N_subsample):
        _ic = mc._search_isocontour(stm_MSTME_ss[bi, :, :], _count_sample)
        _ic[1, 0] = 0
        _ic[0, -1] = 0
        _ic_MSTME.append(_ic)

    # Original
    _ic_original = mc._search_isocontour(stm_original[:, :], _count_original)
    _ic_original[1, 0] = 0
    _ic_original[0, -1] = 0

    _ic_band_MSTME = get_interp_band(_ic_MSTME)

    # ax.plot(_ic_band_MSTME[0][0], _ic_band_MSTME[0][1])
    # ax.plot(_ic_band_MSTME[1][0], _ic_band_MSTME[1][1])
    array = np.concatenate(
        (_ic_band_MSTME[0], np.flip(_ic_band_MSTME[1], axis=1)), axis=1
    )
    ax.fill(array[0], array[1], alpha=0.5)

    ######################################
    ax.scatter(
        stm_original[0, :],
        stm_original[1, :],
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

    if dir_out != None:
        plt.savefig(f"{dir_out}/RV_STM_ss_rp{return_period}.pdf", bbox_inches="tight")
        plt.savefig(f"{dir_out}/RV_STM_ss_rp{return_period}.png", bbox_inches="tight")
    # if not draw_fig:
    #     plt.close()


def plot_isocontour_all(
    tm_original, tm_MSTME_ss, tm_PWE_ss, return_period, dir_out=None
):
    # bi, ni, vi, ei
    assert tm_MSTME_ss.shape == tm_PWE_ss.shape
    stm_min = [0, 0]
    stm_max = [20, 70]
    N_subsample = tm_MSTME_ss.shape[0]
    num_events = tm_MSTME_ss.shape[3]
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
        _exceedance_prob = 1 - cluster.thr_pct_com
        _count_sample = round(
            _num_events_extreme
            / (return_period * cluster.occur_freq * _exceedance_prob)
        )
        _ic_original = []
        _num_events_original = tm_original.shape[2]
        _count_original = round(
            _num_events_original / (return_period * cluster.occur_freq)
        )

        # Bootstraps
        ic_MSTME = []
        ic_PWE = []
        for bi in range(N_subsample):
            _ic_MSTME = mc._search_isocontour(tm_MSTME_ss[bi, i, :, :], _count_sample)
            _ic_PWE = mc._search_isocontour(tm_PWE_ss[bi, i, :, :], _count_sample)
            _ic_MSTME[1, 0] = 0
            _ic_MSTME[0, -1] = 0
            _ic_PWE[1, 0] = 0
            _ic_PWE[0, -1] = 0
            ic_MSTME.append(_ic_MSTME)
            ic_PWE.append(_ic_PWE)
        ic_band_MSTME = get_interp_band(ic_MSTME)
        ic_band_PWE = get_interp_band(ic_PWE)

        _fill_MSTME = np.concatenate(
            (ic_band_MSTME[0], np.flip(ic_band_MSTME[1], axis=1)), axis=1
        )
        _fill_PWE = np.concatenate(
            (ic_band_PWE[0], np.flip(ic_band_PWE[1], axis=1)), axis=1
        )
        ax.fill(_fill_MSTME[0], _fill_MSTME[1], alpha=0.2)
        ax.fill(_fill_PWE[0], _fill_PWE[1], alpha=0.2)

        # Original
        _ic_original = mc._search_isocontour(tm_original[i, :, :], _count_original)
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
        ax.set_title(f"Coord.{i+1}")
    if dir_out != None:
        plt.savefig(
            f"{dir_out}/RV_comparison_ss_rp{return_period}.pdf", bbox_inches="tight"
        )
        plt.savefig(
            f"{dir_out}/RV_comparison_ss_rp{return_period}.png", bbox_inches="tight"
        )


def calc_eq_fetch(vm, vf, r) -> float:
    a = -2.175e-3
    b = 1.506e-2
    c = -1.223e-1
    d = 2.190e-1
    e = 6.737e-1
    f = 7.980e-1
    # a = -2.175e-3
    # b = 1.506e-2
    # c = -1.223e-1
    # d = 2.190e-1
    # e = 6.737e-1
    # f = 7.980e-1
    return (
        1
        * (a * vm**2 + b * vm * vf + c * vf**2 + d * vm + e * vf + f)
        * (22.5e3 * np.log10(r) - 70.8e3)
    )


def calc_norm_fetch(vm, vf) -> float:
    a = -2.175e-3
    b = 1.506e-2
    c = -1.223e-1
    d = 2.190e-1
    e = 6.737e-1
    f = 7.980e-1
    # a = -2.175e-3
    # b = 1.506e-2
    # c = -1.223e-1
    # d = 2.190e-1
    # e = 6.737e-1
    # f = 7.980e-1
    return 1 * (a * vm**2 + b * vm * vf + c * vf**2 + d * vm + e * vf + f)


# # %%

# Vm = np.linspace(20, 60, 100)
# Vf = np.linspace(0, 12, 100)
# XX, YY = np.meshgrid(Vm, Vf)
# ZZ = calc_norm_fetch(XX, YY)
# fig, ax = plt.subplots(1, 1)
# cs = ax.contour(XX, YY, ZZ, colors="black", levels=np.arange(12))
# ax.clabel(cs)
# ax.set_xlabel("$V_{max}$[m/s]")
# ax.set_ylabel("$V_{fm}$[m/s]")

# %% Load dataset
# Load dataset
region = "guadeloupe"
depth = -100

match region:
    case "guadeloupe":
        dir_data = "./ww3_meteo_max/*.nc"
        dir_bathy = "./Bathy.nc"
        min_lon = -62.00
        min_lat = 15.80
        max_lon = -60.80
        max_lat = 16.60
        dir_tracks = "./tracks"
        occur_freq = 44 / (2021 - 1971 + 1)
    # the cyclones were selected as passing at distance of 200km from Guadeloupe.
    # According to IBTrACS, there were 44 storms of class 0~5 during 1971-2021
    case "caribbean":
        dir_data = "./ww3_meteo_max/*.nc"
        dir_bathy = "./Bathy.nc"
        min_lon = -65.00
        min_lat = 12.00
        max_lon = -58.00
        max_lat = 18.00
        dir_tracks = "./tracks"
        occur_freq = 44 / (2021 - 1971 + 1)
    # the cyclones were selected as passing at distance of 200km from Guadeloupe.
    # According to IBTrACS, there were 44 storms of class 0~5 during 1971-2021
    # case "reunion":
    #     dir_data = "./reunion_data/*.nc"
    #     min_lon = -180
    #     min_lat = -180
    #     max_lon = 180
    #     max_lat = 180
    #     occur_freq = 53 / (2021 - 1971 + 1)
    #     dir_tracks = None
    #     dir_bathy = None
    # Jeremy:
    # Acccording to the IBTrACS databse, there are 53 cyclones over the time period 1971-2021 within an area defined by radius of 400 km from the Island center.
    case _:
        min_lon = None
        min_lat = None
        max_lon = None
        max_lat = None
        dir_tracks = None
        occur_freq = None
        dir_bathy = None
        raise (ValueError(f"No region found with name {region}"))
ds_path = Path(f"./ds_filtered_{region}.txt")
ds_track_path = Path(f"./ds_track_{region}.txt")
if ds_path.exists():
    with open(ds_path, "rb") as f:
        ds_filtered: xr.Dataset = pickle.load(f)
else:
    ds_full = xr.open_mfdataset(
        dir_data,
        combine="nested",
        concat_dim="event",
        parallel=True,
        engine="netcdf4",
    )
    ds_bathy_full = xr.open_dataset(dir_bathy, engine="netcdf4")
    ds_filtered = xr.merge([ds_full, ds_bathy_full], compat="override").compute()
    ds_filtered = (
        ds_filtered.drop_dims(("single", "nele"))
        .where(lambda x: x.longitude >= min_lon, drop=True)
        .where(lambda x: x.longitude <= max_lon, drop=True)
        .where(lambda x: x.latitude >= min_lat, drop=True)
        .where(lambda x: x.latitude <= max_lat, drop=True)
        .where(lambda x: x.bathymetry < depth, drop=True)
        .compute()
    )
    _tracks_all = []
    Radius = []
    Timestamps = []
    Vfm = []
    U10_max = []
    XbyRd = []
    max_len = 0
    for ei, tp in enumerate(Path(dir_tracks).glob("*.txt")):
        _df = pd.read_csv(tp, delimiter="\t")
        # get time step
        tdelta = datetime.strptime(
            _df["time"].iloc[1], "%Y-%m-%d %H:%M:%S"
        ) - datetime.strptime(_df["time"].iloc[0], "%Y-%m-%d %H:%M:%S")
        dt = tdelta.total_seconds()

        _track = _df[["latitude", "longitude"]].to_numpy()
        _idx_U10_is_max: int = int(_df["U10_max"].idxmax())
        _prev = _track[_idx_U10_is_max - 1]
        _next = _track[_idx_U10_is_max + 1]
        _dist = geodesic(_prev, _next).m / 2
        _vfm = _dist / dt
        _radius = _df["Radius"].iloc[_idx_U10_is_max] * 1e3
        _u10_max = _df["U10_max"].iloc[_idx_U10_is_max]
        Vfm.append(_vfm)
        Radius.append(_radius)
        U10_max.append(_u10_max)
        Timestamps.append(_df["time"])
        XbyRd.append(calc_norm_fetch(_u10_max, _vfm))
        max_len = max(max_len, len(_track))
        _tracks_all.append(_track)
    for i, t in enumerate(_tracks_all):
        _tracks_all[i] = np.pad(
            _tracks_all[i],
            ((0, max_len), (0, 0)),
            mode="constant",
            constant_values=np.array([np.nan, np.nan]),
        )[:max_len]
    tracks_all = np.array(_tracks_all)
    Radius = np.array(Radius)
    U10_max = np.array(U10_max)
    Vfm = np.array(Vfm)

    # ds_filtered['tracks'] = xr.DataArray(tracks_all,dims=)
    Gf = 1.11
    ds_filtered["Radius"] = xr.DataArray(Radius, dims="event")
    ds_filtered["V_max"] = xr.DataArray(U10_max * Gf, dims="event")
    ds_filtered["Vfm"] = xr.DataArray(Vfm, dims="event")
    ds_filtered["Tracks"] = xr.DataArray(tracks_all, dims=("event", "time", "latlon"))
    for v in STM:
        ds_filtered[f"STM_{v.key()}"] = ds_filtered[v.key()].max(dim="node")
        ds_filtered[f"EXP_{v.key()}"] = (
            ds_filtered[v.key()] / ds_filtered[f"STM_{v.key()}"]
        )
    print(ds_filtered)

    # pickle Dataset
    with open(f"ds_filtered_{region}.txt", "wb") as fh:
        pickle.dump(ds_filtered, fh)
ds = ds_filtered
# # %%
stm = ds[[v.key() for v in STM]].max(dim="node").to_array()
# # fetch_from_track = calc_eq_fetch(U10_max, Vfm)
Gf = 1.11
V_max_track = ds.V_max
V_max_ww3 = ds.STM_UV_10m * Gf
Vfm = ds.Vfm
Radius = ds.Radius
fetch_from_track = calc_eq_fetch(V_max_track, Vfm, r=Radius)
fetch_from_WW3 = 9.8 * (stm[0] / (0.0016 * V_max_ww3)) ** 2

idx_in_range_ww3 = (V_max_ww3 > 20) & (V_max_ww3 < 60) & (Vfm > 0) & (Vfm < 12)
idx_in_range_track = (V_max_track > 20) & (V_max_track < 60) & (Vfm > 0) & (Vfm < 12)
# Vm = np.linspace(20, 60, 100)
# Vf = np.linspace(0, 12, 100)
# XX, YY = np.meshgrid(Vm, Vf)
# ZZ = calc_norm_fetch(XX, YY)
# fig, ax = plt.subplots(1, 1)
# cs = ax.contour(XX, YY, ZZ, colors="black", levels=np.arange(12))
# ax.clabel(cs)
# ax.set_xlabel("$V_{max}$[m/s]")
# ax.set_ylabel("$V_{fm}$[m/s]")
# im = ax.scatter(
#     V_max_ww3[idx_in_range_ww3],
#     Vfm[idx_in_range_ww3],
#     c=fetch_from_WW3[idx_in_range_ww3]
#     / (22.5e3 * np.log10(Radius[idx_in_range_ww3]) - 70.8e3),
# )
# plt.colorbar(im)
# ax.tricontour(stm[1],Vfm,fetch_from_WW3/(22.5e3 * np.log(Radius) - 70.8e3))
# %%
# fig, ax = plt.subplots(1, 1)
# ax.set_xlabel("$U_{10}$[m/s]")
# ax.set_ylabel("Equivalent fetch $x$[m]")
# ax.scatter(
#     V_max_ww3[idx_in_range_ww3],
#     fetch_from_WW3[idx_in_range_ww3],
#     label="from JONSWAP relationship",
# )
# ax.scatter(
#     V_max_track[idx_in_range_track],
#     fetch_from_track[idx_in_range_track],
#     label="from Young's definition",
# )
# ax.legend()
# %%
import openturns as ot

# _stm_pot = np.array(
#     [
#         15.61,
#         15.54,
#         12.80,
#         13.0,
#         17.81,
#         11.63,
#         11.04,
#         20.91,
#         11.66,
#         12.61,
#         12.015,
#         14.21,
#         13.25,
#         11.66,
#         15.37,
#         14.46,
#         14.88,
#         11.03,
#         20.91,
#         18.32,
#         12.17,
#         12.78,
#         12.63,
#         14.80,
#         17.63,
#         12.75,
#         13.15,
#         13.24,
#         13.37,
#         11.23,
#         12.61,
#         11.17,
#         11.80,
#         11.03,
#         12.19,
#         12.61,
#         12.61,
#         14.88,
#         12.63,
#         17.49,
#         13.96,
#         15.37,
#         11.36,
#         13.37,
#         12.03,
#         11.36,
#         15.16,
#         12.17,
#         16.19,
#         12.25,
#         18.32,
#         11.36,
#         15.71,
#         12.24,
#         13.94,
#         14.38,
#         13.46,
#         13.71,
#         11.79,
#         13.7,
#         11.55,
#         11.96,
#         11.05,
#         13.29,
#         11.55,
#         11.23,
#         11.66,
#         13.37,
#         12.61,
#         15.04,
#         13.24,
#         19.68,
#         13.94,
#         12.19,
#         13.24,
#         11.91,
#         11.01,
#         13.15,
#         18.02,
#         11.44,
#         12.36,
#         12.36,
#         13.46,
#         11.79,
#         13.71,
#         15.54,
#         12.12,
#         12.63,
#         11.44,
#         11.5,
#         12.54,
#         12.34,
#         15.42,
#         17.25,
#         11.29,
#         17.43,
#         11.03,
#         12.63,
#         11.25,
#         23.81,
#         14.88,
#         15.62,
#         13.34,
#         11.67,
#         12.03,
#         12.66,
#         13.25,
#         11.96,
#         12.54,
#         15.38,
#         11.92,
#         13.5,
#         11.25,
#         12.36,
#         16.74,
#         12.78,
#         11.25,
#         17.25,
#         13.25,
#         12.80,
#         20.36,
#         11.36,
#         11.25,
#         13.94,
#         11.17,
#         17.31,
#         11.08,
#         15.16,
#         12.61,
#         14.46,
#         21.52,
#         11.63,
#         13.29,
#         11.92,
#         11.66,
#         20.91,
#         15.19,
#         13.34,
#         12.34,
#         11.91,
#         13.0,
#         17.25,
#         13.24,
#     ]
# )
# _sample = ot.Sample(_stm_pot[:, np.newaxis])
distribution = ot.GeneralizedParetoFactory().build(_sample)
_sp, _xp, _mp = distribution.getParameter()
# %%
importlib.reload(mc)
# for n in [10]:
#     for cthr in [0.6, 0.7, 0.8]:
#         for mthr in [0.6, 0.7, 0.8]:
#             for rf in [
#                 "none",
#                 "h-east",
#                 # "h-west",
#             ]:
SAVE = True
RECALC = False
draw_fig = False
return_periods = [100, 200, 500]
N_subsample = 10
N_year_pool = 200

for cthr in [0.8]:
    for mthr in [0.8]:
        thr_pct_com = cthr
        thr_pct_mar = mthr
        dt_string = datetime.now().strftime("%Y-%m-%d-%H%M")
        dir_out = f"./output/{region}/GP{round(thr_pct_mar*100)}%_CM{round(thr_pct_com*100)}%/"
        path_out = Path(dir_out)
        if not path_out.exists():
            path_out.mkdir(parents=True, exist_ok=True)

        path_pickle_mstme = path_out.joinpath("mstme_pickle.txt")
        if path_pickle_mstme.exists() and not RECALC:
            with path_pickle_mstme.open("rb") as f:
                mstme = pickle.load(f)
        else:
            mstme = mc.MSTME(
                area=(min_lat, max_lat, min_lon, max_lon),
                occur_freq=occur_freq,
                ds=ds_filtered,
                thr_pct_mar=thr_pct_mar,
                thr_pct_com=thr_pct_com,
                # tracks=tracks_all,
                dir_out=dir_out,
                draw_fig=draw_fig,
                gpe_method="MLE",
            )
            with path_pickle_mstme.open("wb") as f:
                pickle.dump(mstme, f)

        mstme.draw("Tracks_vs_STM")
        # Loop over region clusters
        for rf in [
            # "none",
            "h-east",
            # "h-west",
        ]:
            path_out_cluster = Path(dir_out) / f"{N_subsample}subsamples_{rf}/"
            if not path_out_cluster.exists():
                path_out_cluster.mkdir(parents=True, exist_ok=True)
            dir_out_cluster = str(path_out_cluster)

            path_pickle_cluster = path_out_cluster.joinpath(f"cluster_pickle.txt")
            if path_pickle_cluster.exists() and not RECALC:
                with path_pickle_cluster.open("rb") as f:
                    cluster = pickle.load(f)
            else:
                cluster_mask, _ = mstme.get_region_filter(rf)
                cluster = mc.MSTME(
                    parent=mstme,
                    mask=cluster_mask,
                    dir_out=dir_out_cluster,
                )
                N_samples = 1000
                cluster.sample(N_samples)
                cluster.sample_PWE(N_samples)
                cluster.search_marginal(np.array([6, 35]), np.array([20, 55]))
                cluster.draw("Replacement")
                cluster.draw("Genpar_Params")
                cluster.draw("Genpar_CDF")
                cluster.draw("Original_vs_Normalized")
                cluster.draw("Kendall_Tau_all_var_pval")
                cluster.draw("Kendall_Tau_all_var_tval")
                cluster.draw("Conmul_Estimates")
                cluster.draw("ab_Estimates")
                cluster.draw("amu_Estimates")
                cluster.draw("Residuals")
                cluster.draw("Simulated_Conmul_vs_Back_Transformed")
                for rp in return_periods:
                    cluster.draw("RV", return_period=rp)
                    cluster.draw("RV_PWE", return_period=rp)
                with path_pickle_cluster.open("wb") as f:
                    pickle.dump(cluster, f)

            path_out_cluster_contours = path_out_cluster.joinpath("contour_objects.txt")
            if path_out_cluster_contours.exists() and not RECALC:
                with path_out_cluster_contours.open("rb") as f:
                    tm_MSTME_ss, tm_PWE_ss, stm_MSTME_ss = pickle.load(f)
            else:
                # setup subsampling
                num_events_ss = round(N_year_pool * cluster.occur_freq)
                if N_subsample == 1:
                    mask_bootstrap = np.full((1, cluster.get_root().num_events), True)
                else:
                    mask_bootstrap = np.full(
                        (N_subsample, cluster.get_root().num_events), False
                    )
                    for i in range(N_subsample):
                        # indices where mask is true
                        _idx_cluster_mask = np.flatnonzero(cluster.mask)
                        _idx_ss = rng.choice(
                            _idx_cluster_mask, size=num_events_ss, replace=False
                        )
                        mask_bootstrap[i, _idx_ss] = True

                    tm_original = np.moveaxis(
                        cluster.tm[:, :, cluster.idx_pos_list].to_numpy(), 2, 0
                    )
                    tm_MSTME_ss = np.zeros(
                        (
                            N_subsample,
                            len(cluster.idx_pos_list),
                            cluster.num_vars,
                            N_samples,
                        )
                    )
                    stm_MSTME_ss = np.zeros((N_subsample, cluster.num_vars, N_samples))
                    tm_PWE_ss = np.zeros(
                        (
                            N_subsample,
                            len(cluster.idx_pos_list),
                            cluster.num_vars,
                            N_samples,
                        )
                    )
                    for bi in trange(N_subsample):
                        _subcluster = mc.MSTME(
                            mask=mask_bootstrap[bi],
                            parent=cluster,
                            # dir_out=dir_out_cluster,
                        )
                        _subcluster.sample(N_samples)
                        _subcluster.sample_PWE(N_samples)
                        tm_MSTME_ss[bi, :, :, :] = np.moveaxis(
                            _subcluster.tm_sample[:, :, cluster.idx_pos_list], 2, 0
                        )
                        tm_PWE_ss[bi, :, :, :] = _subcluster.tm_PWE
                        stm_MSTME_ss[bi, :, :] = _subcluster.stm_sample
                        del _subcluster

                    with path_out_cluster_contours.open("wb") as f:
                        pickle.dump((tm_MSTME_ss, tm_PWE_ss, stm_MSTME_ss), f)

            for rp in return_periods:
                plot_isocontour_all(
                    tm_original,
                    tm_MSTME_ss,
                    tm_PWE_ss,
                    return_period=rp,
                    dir_out=dir_out,
                )
                plot_isocontour_stm(
                    cluster.stm,
                    stm_MSTME_ss,
                    return_period=rp,
                    dir_out=dir_out,
                )


# %%

V_max_ww3 = ds.STM_UV_10m * Gf
V_max_track = ds.V_max
fig, ax = plt.subplots(1, 1)
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

V_max_sample = cluster.stm_sample[1] * Gf
idx_in_range_sample = (V_max_sample > 20) & (V_max_sample < 60)
fetch_from_WW3_sample = 9.8 * (cluster.stm_sample[0] / (0.0016 * V_max_sample)) ** 2
ax.scatter(
    V_max_sample[idx_in_range_sample],
    fetch_from_WW3_sample[idx_in_range_sample],
    marker="s",
    s=1,
    label="from JONSWAP relationship, MSTM-E sampled STM",
)
ax.legend()

# %%
