# %%
# init
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
# Custom
import src.stme as stme
import src.threshold_search as threshold_search
from sklearn.cluster import DBSCAN

plt.style.use("plot_style.txt")
pos_color = plt.rcParams["axes.prop_cycle"].by_key()["color"]
rng = np.random.default_rng(seed=1000)


def is_interactive():
    ip = False
    if "ipykernel" in sys.modules:
        ip = True
    elif "IPython" in sys.modules:
        ip = True
    return ip


print("Interactive:", is_interactive())
if is_interactive():
    thr_mar = np.array([15, 45])
    thr_com = 2.0
    thr_pct = 0.25
    pct_com = 0.25
    depth = -100
    N_bootstrap = 1
    return_period = 100
    SAVE = False
    dir_out = None
    SEARCH_MTHR = False
    SEARCH_CTHR = False
    EXT_EXPOSURE = False
    region = 'guadeloupe'
    # region_filter = 'h-east'
    region_filter = 'none'
else:
    parser = argparse.ArgumentParser(description="Optional app description")

    parser.add_argument(
        "thr_com", type=float, help=""
    )
    parser.add_argument(
        "thr_pct", type=float, help=""
    )
    parser.add_argument(
        "-r", "--region", type=str, help="", required=True
    )
    parser.add_argument(
        "-f", "--filter", type=str, help="", required=False, default='none'
    )
    parser.add_argument(
        "--depth", type=float, help="", required=False, default=-100
    )
    parser.add_argument(
        "--nbootstrap", type=int, help="", required=False, default=1
    )
    parser.add_argument(
        "--rp", type=int, help="", required=False, default=100
    )
    parser.add_argument(
        "--search_mthr", type=bool, help="", required=False, default=False
    )
    parser.add_argument(
        "--search_cthr", type=bool, help="", required=False, default=False
    )
    parser.add_argument(
        "--extend_exposure", type=bool, help="", required=False, default=False
    )

    args = parser.parse_args()
    # thr_mar = np.array([args.thr_hs, args.thr_u10])
    thr_com = args.thr_com
    thr_pct = args.thr_pct
    # pct_com = 0.25
    depth = args.depth
    N_bootstrap = args.nbootstrap
    return_period = args.rp
    SAVE = True
    SEARCH_MTHR = args.search_mthr
    SEARCH_CTHR = args.search_cthr
    EXT_EXPOSURE = args.extend_exposure
    region = args.region
    region_filter = args.filter
    if EXT_EXPOSURE:
        exp_method = 'adjusted_exposure'
    else:
        exp_method = 'original_exposure'
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H%M")
    # dir_out = f"./output/{region}/{exp_method}/{thr_com}/{thr_mar[0]}m_{thr_mar[1]}mps_{dt_string}/"
    dir_out = f"./output/{region}/{exp_method}/{thr_com:.2f}/Threshold_{thr_pct*100}%_{N_bootstrap}bootstrap_{region_filter}_{dt_string}/"
    path_out = Path(dir_out)
    if not path_out.exists():
        path_out.mkdir(parents=True, exist_ok=True)
# %%
# Load dataset
if region == 'guadeloupe':
    ds_full = xr.open_mfdataset(
        "./ww3_meteo_max/*.nc", combine="nested", concat_dim="event", parallel=True
    )
    ds_bathy_full = xr.open_dataset("./Bathy.nc")
    # Guadeloupe
    min_lon = -62.00
    min_lat = 15.80
    max_lon = -60.80
    max_lat = 16.60
    # Carribean sea
    # min_lon = -65.00
    # min_lat = 12.00
    # max_lon = -58.00
    # max_lat = 18.00
    mask_lon = (ds_full.longitude >= min_lon) & (ds_full.longitude <= max_lon)
    mask_lat = (ds_full.latitude >= min_lat) & (ds_full.latitude <= max_lat)

    ds_all = (xr.merge([ds_full, ds_bathy_full], compat='override').drop_dims(("single", "nele"))
              .where(mask_lon & mask_lat, drop=True)
              .compute())
    # Misc.
    _tracks_all = []
    for tp in Path("./tracks").glob("*.txt"):
        _arr = pd.read_csv(tp, delimiter='\t')[
            ['longitude', 'latitude']].to_numpy()
        _tracks_all.append(_arr)
    tracks_all = np.array(_tracks_all, dtype=object)
    kval = xr.load_dataarray('ww3_meteo/other/kval.nc').values
    category = pd.read_csv("category.csv", header=None).to_numpy().squeeze()
    occur_freq = 44/(2021-1971+1)
    # the cyclones were selected as passing at distance of 200km from Guadeloupe.
    # According to IBTrACS, there were 44 storms of class 0~5 during 1971-2021
elif region == 'caribbean':
    ds_full = xr.open_mfdataset(
        "./ww3_meteo_max/*.nc", combine="nested", concat_dim="event", parallel=True
    )
    ds_bathy_full = xr.open_dataset("./Bathy.nc")
    # Guadeloupe
    # min_lon = -62.00
    # min_lat = 15.80
    # max_lon = -60.80
    # max_lat = 16.60
    # Carribean sea
    min_lon = -65.00
    min_lat = 12.00
    max_lon = -58.00
    max_lat = 18.00
    mask_lon = (ds_full.longitude >= min_lon) & (ds_full.longitude <= max_lon)
    mask_lat = (ds_full.latitude >= min_lat) & (ds_full.latitude <= max_lat)

    ds_all = (xr.merge([ds_full, ds_bathy_full], compat='override').drop_dims(("single", "nele"))
              .where(mask_lon & mask_lat, drop=True)
              .compute())
    # Misc.
    _tracks_all = []
    for tp in Path("./tracks").glob("*.txt"):
        _arr = pd.read_csv(tp, delimiter='\t')[
            ['longitude', 'latitude']].to_numpy()
        _tracks_all.append(_arr)
    tracks_all = np.array(_tracks_all, dtype=object)
    kval = xr.load_dataarray('ww3_meteo/other/kval.nc').values
    category = pd.read_csv("category.csv", header=None).to_numpy().squeeze()
elif region == 'reunion':
    ds_full = xr.open_mfdataset(
        "./reunion_data/*.nc", combine="nested", concat_dim="event", parallel=True
    )
    # Reunion
    min_lon = -180
    min_lat = -180
    max_lon = 180
    max_lat = 180
    mask_lon = (ds_full.longitude >= min_lon) & (ds_full.longitude <= max_lon)
    mask_lat = (ds_full.latitude >= min_lat) & (ds_full.latitude <= max_lat)
    ds_cropped = (
        ds_full.drop_dims(("single", "nele"))
        .where(mask_lon & mask_lat, drop=True)
        .compute()
    )
    occur_freq = 53/(2021-1971+1)
    # Jeremy:
    # Acccording to the IBTrACS databse, there are 53 cyclones over the time period 1971-2021 within an area defined by radius of 400 km from the Island center.


else:
    raise(ValueError(f"No region found with name {region}"))
# %%
# Prepare boolean masks for events
filter_node_depth = ds_all.bathymetry < depth
# filter_event_stm_loc = fitX.labels_.astype(bool)
filter_event_stm_loc = stme.get_region_filter(
    ds_all.isel({'node': filter_node_depth}), region_filter)

ds_filtered = ds_all.isel(
    {'node': filter_node_depth,
     'event': filter_event_stm_loc}
)
tracks = tracks_all[filter_event_stm_loc]
# Extract coordinates and variables as ndarrays
lonlat = np.array([ds_filtered.longitude, ds_filtered.latitude]).T
stm_all = ds_filtered[['hs', 'UV_10m']].max(dim="node").to_array()
exp_all = ds_filtered[['hs', 'UV_10m']].to_array() / stm_all
tm_all = ds_filtered[['hs', 'UV_10m']].to_array()

num_events_total = ds_filtered.event.size
num_nodes = ds_filtered.node.size
num_vars = 2
var_name = ["$H_s$", "$U$"]
var_name_g = ["$\hat H_s$", "$\hat U$"]
par_name = ["$\\xi$", "$\\mu$", "$\\sigma$"]
unit = ["[m]", "[m/s]"]
# setup boostrap
if N_bootstrap == 1:
    idx_bootstrap = [np.arange(0, num_events_total, 1)]
else:
    # idx_bootstrap = rng.choice(
    #     num_events_total, (N_bootstrap, num_events_total), replace=True)
    _idx_bootstrap = []
    for i in range(N_bootstrap):
        _idx = rng.choice(
            num_events_total, round(200*occur_freq), replace=False)
        _idx_bootstrap.append(_idx)
    idx_bootstrap = np.array(_idx_bootstrap)


# Guadeloupe city
tree = KDTree(lonlat)
_, idx_pos_list = tree.query([[-61.493, 16.150-i*0.05] for i in range(4)])
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.scatter(lonlat[:, 0], lonlat[:, 1], c='black')
for i, _idx_pos in enumerate(idx_pos_list):
    ax.scatter(
        lonlat[_idx_pos, 0], lonlat[_idx_pos, 1], s=50, color=pos_color[i]
    )
    ax.annotate(
        f"#{i + 1}",
        (
            lonlat[_idx_pos, 0] + (i % 2 - 0.65) * 2 * 0.2,
            lonlat[_idx_pos, 1] - 0.01,
        ),
        bbox=dict(facecolor="white", edgecolor=pos_color[i]),
    )
# %%
# MSTME
importlib.reload(stme)
bi = 0
tm_sample_bs = []
tm_original_bs = []
while bi < N_bootstrap:
    if N_bootstrap == 1:
        _dir_out = dir_out
    else:
        if bi == 0:
            _dir_out = dir_out
        else:
            _dir_out = None
    ds = ds_filtered.isel({'event': idx_bootstrap[bi]})

    tm_sample, exceedance_prob = stme.mstme(
        ds, occur_freq, bi, thr_pct=thr_pct, dir_out=dir_out, draw_fig=True)

    tm_sample_bs.append(tm_sample)
    tm_original_bs.append(tm_all[:, idx_bootstrap[bi], :])

    bi += 1

# %%
# Plot isocontours
tm_original_bs = np.array(tm_original_bs)
tm_sample_bs = np.array(tm_sample_bs)
importlib.reload(stme)
stme.plot_isocontour(tm_original_bs, tm_sample_bs, idx_pos_list, occur_freq,
                     exceedance_prob, return_period, dir_out=dir_out, draw_fig=True)
print("FINISHED")
# %%
if is_interactive() and N_bootstrap == 1:
    # %%
    # Top 9 over threshold
    for vi in range(num_vars):
        _idx_sorted = np.argsort(stm_all[vi].values)[::-1]
        _idx_extreme = np.nonzero(stm_all[vi].values > thr_mar[vi])
        _mask = _idx_sorted[np.in1d(_idx_sorted, _idx_extreme)]
        fig, ax = plt.subplots(3, 3, figsize=(8*3, 6*3))
        for i, _ax in enumerate(ax.ravel()):
            _ax.scatter(lonlat[:, 0], lonlat[:, 1],
                        c=exp_all[vi, _mask[i], :])
            _ax.set_title(
                f'{stm_all.values[vi,_mask[i]]:00.1f}{unit[vi]}')
            _track = tracks[_mask[i]]
            _ax.plot(_track[:, 0], _track[:, 1], c='black', lw=4)
            _ax.set_xlim(lonlat[:, 0].min(), lonlat[:, 0].max())
            _ax.set_ylim(lonlat[:, 1].min(), lonlat[:, 1].max())
        plt.savefig(f'./output/common/{region_filter}/top9_{var_name[vi]}.png',
                    facecolor='white', bbox_inches="tight")
    # %%
    # Bottom 9 over threshold
    for vi in range(num_vars):
        _idx_sorted = np.argsort(stm_all[vi].values)[::-1]
        _idx_extreme = np.nonzero(stm_all[vi].values > thr_mar[vi])
        # _mask = np.intersect1d(_idx_sorted,_idx_extreme)
        _mask = _idx_sorted[np.in1d(_idx_sorted, _idx_extreme)]
        fig, ax = plt.subplots(3, 3, figsize=(8*3, 6*3))
        for i, _ax in enumerate(ax.ravel()):
            _ax.scatter(lonlat[:, 0], lonlat[:, 1],
                        c=exp_all[vi, _mask[-i-1], :])
            _ax.set_title(
                f'{stm_all.values[vi,_mask[-i-1]]:00.1f}{unit[vi]}')
            _track = tracks[_mask[-i-1]]
            _ax.plot(_track[:, 0], _track[:, 1], c='black', lw=4)
            _ax.set_xlim(lonlat[:, 0].min(), lonlat[:, 0].max())
            _ax.set_ylim(lonlat[:, 1].min(), lonlat[:, 1].max())
        plt.savefig(f'./output/common/{region_filter}/bot9_{var_name[vi]}.png',
                    facecolor='white', bbox_inches="tight")

    # %%
    # Mean
    fig, ax = plt.subplots(2, num_vars, figsize=(9*num_vars, 6*2))
    for vi in range(num_vars):
        _idx_sorted = np.argsort(stm_all[vi].values)[::-1]
        _idx_extreme = np.nonzero(stm_all[vi].values > thr_mar[vi])
        _mask = _idx_sorted[np.in1d(_idx_sorted, _idx_extreme)]
        for i in range(2):
            if i == 0:
                mean_exp = exp_all[vi, _mask[:10], :].mean(axis=0)
                title = f'{var_name[vi]}, Top9'
            else:
                mean_exp = exp_all[vi, _mask[-10:], :].mean(axis=0)
                title = f'{var_name[vi]}, Bottom9'

            im = ax[vi, i].scatter(lonlat[:, 0], lonlat[:, 1],
                                   c=mean_exp, vmin=0.5, vmax=1)
            plt.colorbar(im, ax=ax[vi, i])
            ax[vi, i].set_title(title)
    plt.savefig(f'./output/common/{region_filter}/mean_exp1.png',
                facecolor='white', bbox_inches="tight")
    # %%
    # CDF
    fig, ax = plt.subplots(1, 2, figsize=(8*2, 6))
    inferno = plt.get_cmap('inferno')
    for vi in range(num_vars):
        _idx_sorted = np.argsort(stm_all[vi].values)[::-1]
        _idx_extreme = np.nonzero(stm_all[vi].values > thr_mar[vi])
        _mask = _idx_sorted[np.in1d(_idx_sorted, _idx_extreme)]
        ax[vi].set_xlabel(f'Exposure({var_name[vi]})')
        ax[vi].set_ylabel(f'CDF')
        # # all
        # for ei in range(num_events):
        #     _ecdf = ECDF(exp[vi, ei, :])
        #     _x = np.linspace(0, 1, 100)
        #     im = ax[vi].plot(_x, _ecdf(_x), c=plt.get_cmap(
        #         'viridis')(ei/num_events), alpha=0.3)
        # top 10
        for ei in _mask[:5]:
            _ecdf = ECDF(exp_all[vi, ei, :])
            _x = np.linspace(0, 1, 100)
            im = ax[vi].plot(_x, _ecdf(_x), c='red',
                             label=f'{stm_all.values[vi,ei]:.1f}{unit[vi]}')
        # bottom 10
        for ei in _mask[-5:]:
            _ecdf = ECDF(exp_all[vi, ei, :])
            _x = np.linspace(0, 1, 100)
            im = ax[vi].plot(_x, _ecdf(_x), c='black',
                             label=f'{stm_all.values[vi,ei]:.1f}{unit[vi]}')
        ax[vi].legend()
    plt.savefig(f'./output/common/{region_filter}/cdf_exp.png',
                facecolor='white', bbox_inches="tight")

    # %%
    fig, ax = plt.subplots(1, 2, figsize=(8*2, 6))
    _min = [0, 10]
    _max = [25, 70]
    for vi in range(num_vars):
        for ci in range(6):
            _stm = []
            for ei in range(num_events):
                if category[ei] == ci:
                    _stm.append(stm_all.values[vi, ei])
            ax[vi].hist(_stm, bins=np.linspace(
                _min[vi], _max[vi], 51), label=ci)
            ax[vi].legend()
    # %%
    fig, ax = plt.subplots()
    for ci in range(5):
        for ei in range(num_events):
            if category[ei] == ci+1:
                ax.plot(tracks[ei][:, 0], tracks[ei][:, 1], c=pos_color[ci])
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    # %%
    fig, ax = plt.subplots(1, 2, figsize=(8*2, 6))
    _min = [0, 10]
    _max = [25, 70]
    _nbins = [51, 61]
    for vi in range(num_vars):
        for ci in range(6):
            _stm = []
            for ei in range(num_events):
                if category[ei] <= ci:
                    # if category[ei] <= ci and category[ei] > 1:
                    _stm.append(stm_all.values[vi, ei])
            ax[vi].hist(_stm, bins=np.linspace(
                _min[vi], _max[vi], _nbins[vi]), label=ci, zorder=-ci)
            ax[vi].legend()

    # %%
    kval = xr.load_dataarray('ww3_meteo/other/kval.nc').values
    exp_ext = np.zeros(exp.shape)
    pval_ext = np.zeros((num_vars, num_nodes))
    tval_ext = np.zeros((num_vars, num_nodes))

    for vi in range(num_vars):
        _mask = stm[vi] > thr_mar[vi]
        for ni in range(num_nodes):
            exp_ext[vi, :, ni] = exp[vi, :, ni] **\
                ((stm[vi]/thr_mar[vi])**kval[vi, ni])
            tval_ext[vi, ni], pval_ext[vi, ni] = kendalltau(
                stm[vi, _mask], exp_ext[vi, _mask, ni])
    # %%
    # Normalized Exposure

    # def kendallpval(k, stm_norm, exp):
    #     _t, _p = kendalltau(stm_norm, exp**stm_norm**k)
    #     return -_p

    # %%
    exp_ext = np.zeros(exp.shape)
    kval = np.zeros((num_vars, num_nodes))
    pval_ext = np.zeros((num_vars, num_nodes))
    tval_ext = np.zeros((num_vars, num_nodes))
    # fig, ax = plt.subplots(1, num_vars, figsize=(16, 8))
    for ni in trange(num_nodes):
        for vi in range(num_vars):
            _mask = stm[vi] > thr_mar[vi]
            _p = 0
            for k0 in np.linspace(-1, 1, 3):
                _optres = minimize(kendallpval, k0, args=(
                    stm[vi, _mask]/thr_mar[vi], exp[vi, _mask, ni]), method='Powell')
                if _optres.fun < _p:
                    _k = _optres.x
                    _p = _optres.fun
                    if _optres.fun < -0.05:
                        break
            exp_ext[vi, :, ni] = exp[vi, :, ni] **\
                ((stm[vi]/thr_mar[vi])**kval[vi, ni])
            _t, _p = kendalltau(stm[vi]/thr_mar[vi], exp_ext[vi, :, ni])
            kval[vi, ni] = _k
            pval_ext[vi, ni] = _p
            tval_ext[vi, ni] = _t

    # ds_k = xr.Dataset(
    #     data_vars=dict(
    #         kval=(['variable', 'node'], kval)
    #     ),
    #     coords=dict(
    #         lon=(['node'], lonlat[:, 0]),
    #         lat=(['node'], lonlat[:, 1]),
    #     ),
    #     attrs=dict(description='k-val(Exp/STM^k) for Guadeloupe data')
    # )

    # ds_k.to_netcdf('ww3_meteo/other/kval.nc')
    # # %%
    # # pval
    # fig, ax = plt.subplots(1, num_vars, figsize=(8*2, 6))
    # for vi in range(num_vars):
    #     # ax[vi].scatter(lonlat[:,0],lonlat[:,1],facecolors='None',edgecolors='black',s=10,lw=0.1)
    #     ax[vi].scatter(lonlat[:, 0], lonlat[:, 1], c=[
    #         "red" if p < 0.05 else "black" for p in pval_ext[vi, :]], s=5)
    #     ax[vi].set_title(f'{var_name[vi]}')
    # %%
    # tau
    fig, ax = plt.subplots(1, num_vars, figsize=(8*2, 6))
    for vi in range(num_vars):
        im = ax[vi].scatter(lonlat[:, 0], lonlat[:, 1], c=tval_ext[vi],
                            cmap='seismic', vmax=0.01, vmin=-0.01, s=5)
        plt.colorbar(im, ax=ax[vi])
        ax[vi].set_title(f'{var_name[vi]}')

    # %%
    # pval
    fig, ax = plt.subplots(1, num_vars, figsize=(8*2, 6))
    for vi in range(num_vars):
        _c = ["red" if p < 0.05 else "black" for p in pval_ext[vi, :]]
        im = ax[vi].scatter(lonlat[:, 0], lonlat[:, 1], s=5, c=_c)
        # im = ax[vi].scatter(lonlat[:, 0], lonlat[:, 1], c=tval_ext[vi],
        #                     cmap='seismic', vmax=0.01, vmin=-0.01, s=5)
        # plt.colorbar(im, ax=ax[vi])
        ax[vi].set_title(f'{var_name[vi]}')
    # # %%
    # # kval
    # fig, ax = plt.subplots(1, num_vars, figsize=(8*2, 6))
    # fig.suptitle('kval')
    # for vi in range(num_vars):
    #     # ax[vi].scatter(lonlat[:,0],lonlat[:,1],facecolors='None',edgecolors='black',s=10,lw=0.1)
    #     im = ax[vi].scatter(
    #         lonlat[:, 0],
    #         lonlat[:, 1],
    #         c=kval[vi],
    #         cmap='seismic',
    #         vmax=np.abs(kval[vi]).max(),
    #         vmin=-np.abs(kval[vi]).max(),
    #         s=10)
    #     plt.colorbar(im, ax=ax[vi])
    #     ax[vi].set_title(f'{var_name[vi]}')

    # # %%
    # ds_k = xr.Dataset(
    #     data_vars=dict(
    #         kval=(['variable', 'node'], kval)
    #     ),
    #     coords=dict(
    #         lon=(['node'], lonlat[:, 0]),
    #         lat=(['node'], lonlat[:, 1]),
    #     ),
    #     attrs=dict(description='k-val(Exp/STM^k) for Guadeloupe data')
    # )
    # ds_k.to_netcdf(f'ww3_meteo/other/kval_{thr_com}_{thr_mar[0]}_{thr_mar[1]}.nc')

# %%
# stm_idx = exp_all.argmax(axis=2)
# stm_h_lon = lonlat[stm_idx[0], 0].T
# stm_h_lat = lonlat[stm_idx[0], 1].T
# stm_u_lon = lonlat[stm_idx[1], 0].T
# stm_u_lat = lonlat[stm_idx[1], 1].T
# X = np.array([stm_all[0], stm_all[1], stm_h_lon,
#              stm_u_lon, stm_h_lat, stm_u_lat]).T
# df = pd.DataFrame(X, columns=['stmh', 'stmu', 'lonh', 'lonu', 'lath', 'latu'])
# df = StandardScaler().fit_transform(df)
# fitX = KMeans(n_clusters=2).fit(df)
# fitX = DBSCAN(eps=0.3, min_samples=5).fit(df)


# %%
