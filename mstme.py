# %% import libs
# init
import argparse
import importlib
import dill
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from geopy.distance import geodesic
from statsmodels.distributions.empirical_distribution import ECDF
from tqdm import trange

# Custom
import mstmeclass as mc
from mstmeclass import MSTME, STM, SIMSET, Area
import grapher

pos_color = plt.rcParams["axes.prop_cycle"].by_key()["color"]
rng = np.random.default_rng(seed=1000)

G = mc.G
G_F = mc.G_F


# %% Load dataset
# Load dataset
region = "guadeloupe"
depth = -100

match region:
    case "guadeloupe":
        dir_data = "./data/ww3_meteo_max/*.nc"
        dir_bathy = "./data/Bathy.nc"
        area = Area(
            min_lon=-62.00,
            min_lat=15.80,
            max_lon=-60.80,
            max_lat=16.60,
        )
        dir_tracks = "./data/tracks"
        occur_freq = 44 / (2021 - 1971 + 1)
    # the cyclones were selected as passing at distance of 200km from Guadeloupe.
    # According to IBTrACS, there were 44 storms of class 0~5 during 1971-2021
    case "caribbean":
        dir_data = "./data/ww3_meteo_max/*.nc"
        dir_bathy = "./data/Bathy.nc"
        area = Area(
            min_lon=-65.00,
            min_lat=12.00,
            max_lon=-58.00,
            max_lat=18.00,
        )
        dir_tracks = "./data/tracks"
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
        area = Area(
            min_lon=None,
            min_lat=None,
            max_lon=None,
            max_lat=None,
        )
        dir_tracks = None
        occur_freq = None
        dir_bathy = None
        raise (ValueError(f"No region found with name {region}"))

ds_path = Path(f"./data/ds_filtered_{region}.dill")
ds_track_path = Path(f"./data/ds_track_{region}.dill")
if ds_path.exists():
    with open(ds_path, "rb") as f:
        ds_filtered: xr.Dataset = dill.load(f)
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
        .where(lambda x: x.longitude >= area.min_lon, drop=True)
        .where(lambda x: x.longitude <= area.max_lon, drop=True)
        .where(lambda x: x.latitude >= area.min_lat, drop=True)
        .where(lambda x: x.latitude <= area.max_lat, drop=True)
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

    ds_filtered["Radius"] = xr.DataArray(Radius, dims="event")
    ds_filtered["V_max"] = xr.DataArray(U10_max * G_F, dims="event")
    ds_filtered["Vfm"] = xr.DataArray(Vfm, dims="event")
    ds_filtered["Tracks"] = xr.DataArray(tracks_all, dims=("event", "time", "latlon"))
    for v in STM:
        ds_filtered[f"STM_{v.key()}"] = ds_filtered[v.key()].max(dim="node")
        ds_filtered[f"EXP_{v.key()}"] = (
            ds_filtered[v.key()] / ds_filtered[f"STM_{v.key()}"]
        )
    print(ds_filtered)

    # dill Dataset
    with open(f"./data/ds_filtered_{region}.dill", "wb") as fh:
        dill.dump(ds_filtered, fh)
ds = ds_filtered
# %%
# importlib.reload(mc)
importlib.reload(grapher)
# for n in [10]:
#     for thr_pct_com in [0.6, 0.7, 0.8]:
#         for thr_pct_mar in [0.6, 0.7, 0.8]:
#             for rf in [
#                 "none",
#                 "h-east",
#                 # "h-west",
#             ]:
SAVE = True
RECALC = True
draw_fig = False
# return_periods = [100, 200, 500]
# N_subsample = 10
# N_year_pool = 200
return_periods = [100]
N_subsample = 10
N_year_pool = 200

print(f"SAVE:{SAVE},RECALCULATE:{RECALC},DRAW:{draw_fig},Subsample:{N_subsample}")

for thr_pct_mar in [0.8]:
    for thr_pct_com in [0.8]:
        # try:
        # Measure execution time
        start_mstme = time.time()

        # Output stuff
        dt_string = datetime.now().strftime("%Y-%m-%d-%H%M")
        dir_out = f"./output/{region}/DEBUG_GP{round(thr_pct_mar*100)}%_CM{round(thr_pct_com*100)}%/"
        # dir_out = f"./output/{region}/GP{round(thr_pct_mar*100)}%_CM{round(thr_pct_com*100)}%/"
        path_out = Path(dir_out)
        if not path_out.exists():
            path_out.mkdir(parents=True, exist_ok=True)

        # Pickle MSTME object for faster redraws
        path_dill_mstme = path_out.joinpath("mstme_dill.dill")
        if path_dill_mstme.exists() and not RECALC:
            with path_dill_mstme.open("rb") as f:
                mstme = dill.load(f)
        else:
            mstme = MSTME(
                area=area,
                occur_freq=occur_freq,
                ds=ds_filtered,
                thr_pct_mar=thr_pct_mar,
                thr_pct_com=thr_pct_com,
                # tracks=tracks_all,
                dir_out=dir_out,
                draw_fig=draw_fig,
                gpe_method="MLE",
            )
            with path_dill_mstme.open("wb") as f:
                dill.dump(mstme, f)

        # Draw plots
        grapher_mstme = grapher.Grapher(mstme)
        grapher_mstme.draw("Tracks_vs_STM")
        grapher_mstme.draw("General_Map")

        # Logging
        with path_out.joinpath(f"log.txt").open("w") as f:
            _mstme = mstme
            _output = ""
            _output += f"Marginal threshold:\t{_mstme.thr_pct_mar*100}%\n"
            for S in STM:
                _output += f"\t{_mstme.thr_mar[S.idx()]}[{S.unit()}]\n"
            _output += (
                f"Common threshold:\t{_mstme.thr_pct_com*100}%\t{_mstme.thr_com}\n"
            )
            f.write(_output)

        print(
            f"MSTME object calculations and plots for {thr_pct_com},{thr_pct_mar}, finished in {time.time()-start_mstme}"
        )
        print(mstme.thr_pct_mar)

        # Loop over region clusters
        for rf in [
            "none",
            # "h-east",
            # "h-west",
        ]:
            # Measure execution time
            start_cluster = time.time()

            # Output stuff
            path_out_cluster = Path(dir_out) / f"{N_subsample}subsamples_{rf}/"
            if not path_out_cluster.exists():
                path_out_cluster.mkdir(parents=True, exist_ok=True)
            dir_out_cluster = str(path_out_cluster)

            # Pickle MSTME object for faster redraws
            path_dill_cluster = path_out_cluster.joinpath(f"cluster_dill.txt")
            if path_dill_cluster.exists() and not RECALC:
                with path_dill_cluster.open("rb") as f:
                    cluster = dill.load(f)
            else:
                cluster_mask, _ = mstme.get_region_filter(rf)
                cluster = MSTME(
                    parent=mstme,
                    mask=cluster_mask,
                    dir_out=dir_out_cluster,
                    draw_fig=draw_fig,
                    rf=rf,
                )
                N_sample = 1000
                cluster.sample(N_sample)
                cluster.sample_PWE(N_sample)
                cluster.search_marginal(np.array([6, 35]), np.array([20, 55]))
                cluster.subsample(N_subsample, N_year_pool)
                with path_dill_cluster.open("wb") as f:
                    dill.dump(cluster, f)

            # Draw plots
            grapher_cluster = grapher.Grapher(cluster)
            grapher_cluster.draw("Replacement")
            grapher_cluster.draw("Genpar_Params")
            grapher_cluster.draw("Genpar_CDF")
            grapher_cluster.draw("Kendall_Tau_all_var_pval")
            grapher_cluster.draw("Kendall_Tau_all_var_tval")
            grapher_cluster.draw("Conmul_Estimates")
            grapher_cluster.draw("ab_Estimates")
            grapher_cluster.draw("amu_Estimates")
            grapher_cluster.draw("Residuals")
            grapher_cluster.draw("Simulated_Conmul_vs_Back_Transformed")
            grapher_cluster.draw("Equivalent_fetch")
            grapher_cluster.draw("STM_Histogram_filtered")
            grapher_cluster.draw("STM_location")

            for rp in return_periods:
                grapher_cluster.draw("RV", return_period=rp)
                grapher_cluster.draw("RV_PWE", return_period=rp)
                grapher_cluster.draw("RV_ALL", return_period=rp)
                grapher_cluster.draw("RV_STM", return_period=rp)

            # Logging
            with path_out_cluster.joinpath(f"log.dill").open("w") as f:
                _mstme = cluster
                _output = ""
                _output += f"Marginal threshold:\t{_mstme.thr_pct_mar*100}%\n"
                for S in STM:
                    _output += f"\t{_mstme.thr_mar[S.idx()]}[{S.unit()}]\n"
                _output += (
                    f"Common threshold:\t{_mstme.thr_pct_com*100}%\t{_mstme.thr_com}\n"
                )
                f.write(_output)

            print(
                f"Cluster object calculations and plots for {thr_pct_com},{thr_pct_mar},{rf} finished in {time.time()-start_cluster}"
            )
            print(cluster.thr_pct_mar)
        # except:
        #     print(f"Some error on cthr:{thr_pct_com}, mthr:{thr_pct_mar}, {rf}")
        #     continue
# %%
importlib.reload(grapher)
importlib.reload(mc)
grapher_cluster = grapher.Grapher(cluster)
grapher_mstme = grapher.Grapher(mstme)

# grapher_mstme.draw("General_Map", draw_fig=True, dir_out=None)

# grapher_cluster.draw("Replacement", draw_fig=True, dir_out=None)
# grapher_cluster.draw("Genpar_Params", draw_fig=True, dir_out=None)
# grapher_cluster.draw("Genpar_CDF", draw_fig=True, dir_out=None)
# grapher_cluster.draw("Kendall_Tau_all_var_pval", draw_fig=True, dir_out=None)
# grapher_cluster.draw("Kendall_Tau_all_var_tval", draw_fig=True, dir_out=None)
# grapher_cluster.draw("Conmul_Estimates", draw_fig=True, dir_out=None)
# grapher_cluster.draw("ab_Estimates", draw_fig=True, dir_out=None)
# grapher_cluster.draw("amu_Estimates", draw_fig=True, dir_out=None)
# grapher_cluster.draw("Residuals", draw_fig=True, dir_out=None)
# grapher_cluster.draw(
#     "Simulated_Conmul_vs_Back_Transformed", draw_fig=True, dir_out=None
# )
# grapher_cluster.draw("Equivalent_fetch", draw_fig=True, dir_out=None)
for rp in [50]:
    # grapher_cluster.draw("RV", return_period=rp, draw_fig=True, dir_out=None)
    # grapher_cluster.draw("RV_PWE", return_period=rp, draw_fig=True, dir_out=None)
    grapher_cluster.draw("RV_ALL", return_period=rp, draw_fig=True, dir_out=None)
    # grapher_cluster.draw("RV_STM", return_period=rp, draw_fig=True, dir_out=None)
# %%
