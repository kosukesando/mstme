# %%
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

# Custom
import src.stme as stme
import src.threshold_search as threshold_search
import src.mstmeclass as mc
from src.is_interactive import is_interactive

plt.style.use("plot_style.txt")
pos_color = plt.rcParams["axes.prop_cycle"].by_key()["color"]
rng = np.random.default_rng(seed=1000)


class Region(enum.Enum):
    guadeloupe = "guadeloupe"
    caribbean = "caribbean"
    reunion = "reunion"


def _savefig(*args, **kwargs):
    plt.savefig(*args, **kwargs)
    plt.close(plt.gcf())


def _plot_isocontour_all(tm_original, tm_MSTME_bs, tm_PWE_bs, return_period):
    # bi, ni, vi, ei
    assert tm_MSTME_bs.shape == tm_PWE_bs.shape
    print(tm_original.shape, tm_MSTME_bs.shape, tm_PWE_bs)
    stm_min = [0, 0]
    stm_max = [20, 70]
    N_bootstrap = tm_MSTME_bs.shape[0]
    num_events = tm_MSTME_bs.shape[3]
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
        _num_events_extreme = tm_MSTME_bs.shape[3]
        _exceedance_prob = 1 - cluster_mstme.thr_pct_com
        _count_sample = round(
            _num_events_extreme
            / (return_period * cluster_mstme.occur_freq * _exceedance_prob)
        )
        _ic_original = []
        _num_events_original = tm_original.shape[2]
        _count_original = round(
            _num_events_original / (return_period * cluster_mstme.occur_freq)
        )

        # Bootstraps
        _ic_MSTME = []
        _ic_PWE = []
        for bi in range(N_bootstrap):
            _ic_MSTME = mc._search_isocontour(tm_MSTME_bs[bi, i, :, :], _count_sample)
            _ic_PWE = mc._search_isocontour(tm_PWE_bs[bi, i, :, :], _count_sample)
            _ic_MSTME[1, 0] = 0
            _ic_MSTME[0, -1] = 0
            _ic_PWE[1, 0] = 0
            _ic_PWE[0, -1] = 0
            ax.plot(
                _ic_MSTME[0],
                _ic_MSTME[1],
                # c=pos_color[i],
                c="blue",
                lw=5,
                alpha=0.2,
            )
            ax.plot(
                _ic_PWE[0],
                _ic_PWE[1],
                # c=pos_color[i],
                c="red",
                lw=5,
                alpha=0.2,
            )

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
        plt.savefig(f"{dir_out}/RV_comparison_bs.pdf", bbox_inches="tight")
        plt.savefig(f"{dir_out}/RV_comparison_bs.png", bbox_inches="tight")


def _plot_isocontour_stm(stm_original, stm_MSTME_bs, return_period):
    # bi, vi, ei
    stm_min = [0, 0]
    stm_max = [25, 60]
    N_bootstrap = stm_MSTME_bs.shape[0]
    #########################################################
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
    _num_events_extreme = stm_MSTME_bs.shape[2]
    _exceedance_prob = 1 - cluster_mstme.thr_pct_com
    _count_sample = round(
        _num_events_extreme
        / (return_period * cluster_mstme.occur_freq * _exceedance_prob)
    )
    _num_events_original = tm_original.shape[2]
    _count_original = round(
        _num_events_original / (return_period * cluster_mstme.occur_freq)
    )

    # Bootstraps
    _ic_MSTME = []
    for bi in range(N_bootstrap):
        _ic_MSTME = mc._search_isocontour(stm_MSTME_bs[bi, :, :], _count_sample)

        # ax.scatter(
        #     tm_MSTME_bs[bi,  0, :],
        #     tm_MSTME_bs[bi,  1, :],
        #     s=2,
        #     # c=pos_color[i],
        #     c="blue",
        #     label=f"Simulated",
        #     alpha=0.1,
        # )
        # ax.scatter(
        #     tm_PWE_bs[bi,  0, :],
        #     tm_PWE_bs[bi,  1, :],
        #     s=2,
        #     # c=pos_color[i],
        #     c="red",
        #     label=f"Simulated",
        #     alpha=0.1,
        # )
        _ic_MSTME[1, 0] = 0
        _ic_MSTME[0, -1] = 0
        ax.plot(
            _ic_MSTME[0],
            _ic_MSTME[1],
            # c=pos_color[i],
            c="blue",
            lw=5,
            alpha=0.2,
        )

    # Original
    _ic_original = mc._search_isocontour(stm_original[:, :], _count_original)
    _ic_original[1, 0] = 0
    _ic_original[0, -1] = 0
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
        plt.savefig(f"{dir_out}/RV_STM_bs.pdf", bbox_inches="tight")
        plt.savefig(f"{dir_out}/RV_STM_bs.png", bbox_inches="tight")
    # if not draw_fig:
    #     plt.close()

# %%
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
    case "reunion":
        dir_data = "./reunion_data/*.nc"
        min_lon = -180
        min_lat = -180
        max_lon = 180
        max_lat = 180
        occur_freq = 53 / (2021 - 1971 + 1)

        # Jeremy:
        # Acccording to the IBTrACS databse, there are 53 cyclones over the time period 1971-2021 within an area defined by radius of 400 km from the Island center.
    case _:
        raise (ValueError(f"No region found with name {region}"))

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
print(ds_filtered)
# Misc.
_tracks_all = []
for tp in Path(dir_tracks).glob("*.txt"):
    _arr = pd.read_csv(tp, delimiter="\t")[["longitude", "latitude"]].to_numpy()
    _tracks_all.append(_arr)
tracks_all = np.array(_tracks_all, dtype=object)
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
for n in [10]:
    for cthr in [0.8]:
        for mthr in [0.8]:
            for rf in [
                # "none",
                "h-east",
                # "h-west",
            ]:
                thr_pct_com = cthr
                thr_pct_mar = mthr
                N_bootstrap = n
                return_period = 100
                SAVE = True
                region_filter = rf
                exp_method = "original_exposure"    
                dt_string = datetime.now().strftime("%Y-%m-%d-%H%M")
                # dir_out = f"./output/{region}/{exp_method}/{thr_com}/{thr_mar[0]}m_{thr_mar[1]}mps_{dt_string}/"
                dir_out = f"./output/{region}/GP{round(thr_pct_mar*100)}%_CM{round(thr_pct_com*100)}%/{N_bootstrap}bootstrap_{region_filter}_{dt_string}/"
                path_out = Path(dir_out)
                if not path_out.exists():
                    path_out.mkdir(parents=True, exist_ok=True)
                draw_fig = False
                # Prepare boolean masks for events
                # importlib.reload(mc)

                mstme = mc.MSTME(
                    ds_filtered,
                    occur_freq,
                    area=[min_lon, max_lon, min_lat, max_lat],
                    thr_pct_mar=thr_pct_mar,
                    thr_pct_com=thr_pct_com,
                    tracks=tracks_all,
                    dir_out=dir_out,
                    draw_fig=draw_fig,
                    gpe_method="MLE",
                )
                mstme.draw("Tracks_vs_STM")
                # mstme.draw("Kendall_Tau_all_var_pval")
                # mstme.draw("Kendall_Tau_all_var_tval")

                cluster_mask, _ = mstme.get_region_filter(region_filter)
                cluster = mc.Cluster(cluster_mask, mstme)
                N_samples = 1000
                cluster.sample(N_samples)
                cluster.sample_PWE(N_samples)
                cluster.search_marginal(np.array([[6], [35]]), np.array([[20], [55]]))
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
                cluster.draw("RV", return_period=100)
                cluster.draw("RV_PWE", return_period=100)

                # setup boostrap
                # N_bootstrap = 2
                num_events_ss = round(200 * cluster.occur_freq)
                if N_bootstrap == 1:
                    mask_bootstrap = np.full((1, cluster.num_events), True)
                else:
                    mask_bootstrap = np.full((N_bootstrap, cluster.num_events), False)
                    for i in range(N_bootstrap):
                        _idx = rng.choice(
                            cluster.num_events, num_events_ss, replace=False
                        )
                        mask_bootstrap[i, _idx] = True
                # MSTME
                # importlib.reload(mc)
                cluster_mstme = mc.MSTME(
                    cluster.ds,
                    cluster.occur_freq,
                    area=cluster.parent.area,
                    thr_pct_mar=cluster.thr_pct_mar,
                    thr_pct_com=cluster.thr_pct_com,
                    dir_out=cluster.parent.dir_out,
                    draw_fig=cluster.parent.draw_fig,
                    tracks=cluster.tracks,
                    gpe_method=cluster.parent.gpe_method,
                )
                tm_original = np.moveaxis(
                    cluster.tm[:, :, cluster.idx_pos_list].to_numpy(), 2, 0
                )
                tm_MSTME_bs = np.zeros(
                    (
                        N_bootstrap,
                        len(cluster.idx_pos_list),
                        cluster.num_vars,
                        N_samples,
                    )
                )
                stm_MSTME_bs = np.zeros((N_bootstrap, cluster.num_vars, N_samples))
                tm_PWE_bs = np.zeros(
                    (
                        N_bootstrap,
                        len(cluster.idx_pos_list),
                        cluster.num_vars,
                        N_samples,
                    )
                )
                for bi in trange(N_bootstrap):
                    _subcluster = mc.Cluster(mask_bootstrap[bi], cluster_mstme)
                    _subcluster.sample(N_samples)
                    _subcluster.sample_PWE(N_samples)
                    tm_MSTME_bs[bi, :, :, :] = np.moveaxis(
                        _subcluster.tm_sample[:, :, cluster.idx_pos_list], 2, 0
                    )
                    tm_PWE_bs[bi, :, :, :] = _subcluster.tm_PWE
                    stm_MSTME_bs[bi, :, :] = _subcluster.stm_sample

                # tm_PWE_original = np.moveaxis(tm_original_bs[:, :, :, cluster_mstme.idx_pos_list], 3, 1)
                _plot_isocontour_all(
                    tm_original, tm_MSTME_bs, tm_PWE_bs, return_period=500
                )
                _plot_isocontour_stm(cluster.stm, stm_MSTME_bs, return_period=500)

# %%
