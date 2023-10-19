import argparse
import importlib
import os
import time
import warnings
from datetime import datetime
from pathlib import Path

import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from geopy.distance import geodesic
from statsmodels.distributions.empirical_distribution import ECDF
from tqdm import trange

import grapher

# Custom
import mstmeclass as mc
from mstmeclass import MSTME, SIMSET, STM, Area

os.environ["OPENBLAS_MAIN_FREE"] = "1"


def log(mstme: MSTME):
    with Path(mstme.dir_out).joinpath(f"log.txt").open("w") as f:
        _output = ""
        _output += f"Marginal threshold:\t{mstme.thr_pct_mar*100}%\n"
        for S in STM:
            vi = S.idx()
            _output += f"\t{mstme.thr_mar[vi]}[{S.unit()}]"
            _output += f"\tcount:{np.count_nonzero(mstme.is_e_mar[vi])}\n"
            _output += f"\tGP params(xi,mu,sigma):{mstme.gp[vi].args}\n"
        _output += f"Common threshold:\t{mstme.thr_pct_com*100}%\t{mstme.thr_com}\n"
        for S in STM:
            vi = S.idx()
            _output += f"\tcount:{np.count_nonzero(mstme.is_e[vi])}\n"
            _output += f"\tCM params(a,b,mu,sigma):{mstme.params_median[vi]}\n"
        f.write(_output)


def load_data(region, depth):
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
        case "guadeloupe-wide":
            dir_data = "./data/ww3_meteo_max/*.nc"
            dir_bathy = "./data/Bathy.nc"
            area = Area(
                min_lon=-62.50,
                min_lat=15.00,
                max_lon=-60.50,
                max_lat=17.00,
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
        case "whole":
            dir_data = "./data/ww3_meteo_max/*.nc"
            dir_bathy = "./data/Bathy.nc"
            area = Area(
                min_lon=-84.7,
                min_lat=8.40,
                max_lon=-50.00,
                max_lat=22.10,
            )
            dir_tracks = "./data/tracks"
            occur_freq = 44 / (2021 - 1971 + 1)
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
        G = mc.G
        G_F = mc.G_F
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
        ds_filtered["Tracks"] = xr.DataArray(
            tracks_all, dims=("event", "time", "latlon")
        )
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
    return ds_filtered, area, occur_freq


if __name__ == "__main__":
    pos_color = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    rng = np.random.default_rng(seed=1000)

    # Load dataset
    region = "guadeloupe-wide"
    depth = -100
    ds_filtered, area, occur_freq = load_data(region, depth)
    SAVE = True
    RECALC = False
    draw_fig = False
    # return_periods = [100, 200, 500]
    # N_subsample = 10
    # N_year_pool = 200
    return_periods = [100]
    N_subsample = 100
    N_year_pool = 200
    N_sample = 1000

    print(f"SAVE:{SAVE},RECALCULATE:{RECALC},DRAW:{draw_fig},Subsample:{N_subsample}")

    ##### Load region MSTME object
    start_mstme = time.time()
    # Output stuff
    dt_string = datetime.now().strftime("%Y-%m-%d-%H%M")
    dir_out_region = f"./output/{region}/"
    path_out_region = Path(dir_out_region)
    if not path_out_region.exists():
        path_out_region.mkdir(parents=True, exist_ok=True)

    # Pickle MSTME object for faster redraws
    path_dill_region = path_out_region.joinpath("mstme_region.dill")
    if path_dill_region.exists() and not RECALC:
        with path_dill_region.open("rb") as f:
            mstme_region = dill.load(f)
            print(f"MSTME Dill found for '{region}'!")
    else:
        print(f"No dill found for '{region}' or recalculating!")

        mstme_region = MSTME(
            area=area,
            occur_freq=occur_freq,
            ds=ds_filtered,
            thr_pct_mar=0.5,  # not used
            thr_pct_com=0.5,  # not used
            # tracks=tracks_all,
            dir_out=dir_out_region,
            draw_fig=draw_fig,
        )
        with path_dill_region.open("wb") as f:
            dill.dump(mstme_region, f)
    # #########################################
    # mstme_region.dir_out = dir_out_region
    # with path_dill_region.open("wb") as f:
    #     dill.dump(mstme_region, f)
    # #########################################
    grapher_region = grapher.Grapher(mstme_region)
    grapher_region.draw("General_Map")

    ##### Loop over clusters
    for rf in [
        # "none",
        # "h-east",
        "h-west",
    ]:
        start_mstme = time.time()
        # Output stuff
        dt_string = datetime.now().strftime("%Y-%m-%d-%H%M")
        path_out_cluster = path_out_region.joinpath(f"{rf}")
        if not path_out_cluster.exists():
            path_out_cluster.mkdir(parents=True, exist_ok=True)
        dir_out_cluster = str(path_out_cluster)

        # Pickle MSTME object for faster redraws
        path_dill_cluster = path_out_cluster.joinpath("mstme_cluster.dill")
        if path_dill_cluster.exists() and not RECALC:
            with path_dill_cluster.open("rb") as f:
                mstme_cluster = dill.load(f)
                print(f"Cluster Dill found for '{rf}'!")
        else:
            print(f"No dill found for '{rf}' or recalculating!")
            cluster_mask = mc.get_cluster_mask(mstme_region, rf)
            mstme_cluster = MSTME(
                parent=mstme_region,
                mask=cluster_mask,
                dir_out=dir_out_cluster,
                draw_fig=draw_fig,
                rf=rf,
            )
            with path_dill_cluster.open("wb") as f:
                dill.dump(mstme_cluster, f)
        # #########################################
        # mstme_cluster.dir_out = dir_out_cluster
        # with path_dill_cluster.open("wb") as f:
        #     dill.dump(mstme_cluster, f)
        # #########################################
        # Draw plots
        mstme_cluster.search_marginal([0, 0], [20, 55])
        grapher_cluster = grapher.Grapher(mstme_cluster)
        grapher_cluster.draw("Tracks_vs_STM")
        grapher_cluster.draw("STM_Histogram")
        grapher_cluster.draw("STM_location")

        log(mstme_cluster)

        print(
            f"MSTME object calculations and plots for {rf}, finished in {time.time()-start_mstme}"
        )

        ##### Loop over thresholds
        for thr_pct_mar in [
            0.60,
            # 0.65,
            # 0.70,
            # 0.75,
            # 0.8,
        ]:
            for thr_pct_com in [
                # 0.60,
                0.65,
                0.70,
                # 0.75,
                # 0.8,
                # 0.85,
                # 0.90,
                # 0.95,
            ]:
                # Measure execution time
                start_condition = time.time()

                # Output stuff
                path_out_condition = (
                    Path(dir_out_cluster)
                    / f"GP{round(thr_pct_mar*100)}%_CM{round(thr_pct_com*100)}%_{N_subsample}subsamples/"
                )
                if not path_out_condition.exists():
                    path_out_condition.mkdir(parents=True, exist_ok=True)
                dir_out_condition = str(path_out_condition)

                # Pickle MSTME object for faster redraws
                path_dill_condition = path_out_condition.joinpath(
                    f"mstme_condition.dill"
                )
                if path_dill_condition.exists() and not RECALC:
                    with path_dill_condition.open("rb") as f:
                        mstme_condition = dill.load(f)
                        print(
                            f"MSTME Dill found for GP{round(thr_pct_mar*100)}%_CM{round(thr_pct_com*100)}%!"
                        )
                else:
                    print(
                        f"MSTME dill not found or recalculating for GP{round(thr_pct_mar*100)}%_CM{round(thr_pct_com*100)}%!"
                    )
                    mstme_condition = MSTME(
                        parent=mstme_cluster,
                        mask=mstme_cluster.mask,
                        dir_out=dir_out_condition,
                        thr_pct_mar=thr_pct_mar,
                        thr_pct_com=thr_pct_com,
                        draw_fig=draw_fig,
                        rf=rf,
                    )
                    mstme_condition.sample(N_sample)
                    with path_dill_condition.open("wb") as f:
                        dill.dump(mstme_condition, f)
                # #########################################
                # mstme_condition.dir_out = dir_out_condition
                # with path_dill_condition.open("wb") as f:
                #     dill.dump(mstme_condition, f)
                # #########################################

                path_mstme_ss_dill = path_out_condition.joinpath(
                    f"mstme_ss_{N_subsample}_pool_{N_year_pool}.dill"
                )
                if path_mstme_ss_dill.exists():
                    with open(path_mstme_ss_dill, "rb") as f:
                        tm_MSTME_ss, stm_MSTME_ss = dill.load(f)
                    if tm_MSTME_ss.shape[0] != N_subsample:
                        warnings.warn(
                            f"Sample count of {tm_MSTME_ss.shape[0]} for mstme_ss_dill does not match the input:{N_subsample} (GP{round(thr_pct_mar*100)}%_CM{round(thr_pct_com*100)}%)"
                        )
                else:
                    try:
                        tm_MSTME_ss, stm_MSTME_ss = mc.subsample_MSTME(
                            mstme_condition, N_subsample, N_year_pool
                        )
                    except mc.SubsampleException as e:
                        print(e)
                        break
                    with path_mstme_ss_dill.open("wb") as f:
                        dill.dump((tm_MSTME_ss, stm_MSTME_ss), f)

                tm_MSTME, stm_MSTME, exp_MSTME = mc.sample_MSTME(
                    mstme_condition, N_sample
                )
                # Draw plots
                grapher_condition = grapher.Grapher(mstme_condition)
                # grapher_condition.draw_all(
                #     [
                #         "Replacement",
                #         "Genpar_Params",
                #         "Genpar_CDF",
                #         "Kendall_Tau_marginal_pval",
                #         "Kendall_Tau_marginal_tval",
                #         # "Kendall_Tau_all_var_pval",
                #         # "Kendall_Tau_all_var_tval",
                #         "Conmul_Estimates",
                #         "ab_Estimates",
                #         "amu_Estimates",
                #         "a+mub_Estimates",
                #         "Residuals",
                #         "Simulated_Conmul_vs_Back_Transformed",
                #         "Equivalent_fetch",
                #         "STM_Histogram",
                #         "STM_location",
                #     ]
                # )

                for rp in return_periods:
                    # grapher_condition.draw("RV", return_period=rp, tm_MSTME=tm_MSTME)
                    # grapher_condition.draw("RV_PWE", return_period=rp)
                    # grapher_condition.draw("RV_ALL", return_period=rp)
                    grapher_condition.draw(
                        "RV_STM", return_period=rp, stm_MSTME_ss=stm_MSTME_ss
                    )

                # Logging
                log(mstme_condition)

                print(
                    f"Cluster object calculations and plots for {thr_pct_com},{thr_pct_mar},{rf} finished in {time.time()-start_condition}"
                )
