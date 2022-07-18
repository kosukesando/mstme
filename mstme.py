# %%
# init
from ast import Str
from matplotlib import afm
import pandas as pd
import argparse
from datetime import datetime
import importlib
from unicodedata import numeric
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import genpareto
from scipy.stats import genextreme
from scipy.stats import laplace
from scipy.stats import kendalltau
from scipy.spatial import KDTree
from statsmodels.distributions.empirical_distribution import ECDF
from statsmodels.distributions.empirical_distribution import monotone_fn_inverter
from traitlets import Bool
import sys
import xarray as xr
# Custom
import stme
import threshold_search
from tqdm import trange, tqdm

plt.style.use("plot_style.txt")
pos_color = plt.rcParams["axes.prop_cycle"].by_key()["color"]
rng = np.random.default_rng()


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
    thr_gum = 2.0
    depth = -100
    SAVE = False
    dir_out = None
    SEARCH_MTHR = False
    SEARCH_CTHR = False
    EXT_EXPOSURE = False
    region = 'guadeloupe'
else:
    parser = argparse.ArgumentParser(description="Optional app description")

    parser.add_argument(
        "thr_gum", type=float, help=""
    )
    parser.add_argument(
        "thr_hs", type=float, help=""
    )
    parser.add_argument(
        "thr_u10", type=float, help=""
    )
    parser.add_argument(
        "-r", "--region", type=str, help="", required=True
    )
    parser.add_argument(
        "--depth", type=float, help="", required=False, default=-100
    )
    parser.add_argument(
        "--search_mthr", type=Bool, help="", required=False, default=False
    )
    parser.add_argument(
        "--search_cthr", type=Bool, help="", required=False, default=False
    )
    parser.add_argument(
        "--extend_exposure", type=Bool, help="", required=False, default=False
    )

    args = parser.parse_args()
    thr_mar = np.array([args.thr_hs, args.thr_u10])
    thr_gum = args.thr_gum
    depth = args.depth
    SAVE = True
    SEARCH_MTHR = args.search_mthr
    SEARCH_CTHR = args.search_cthr
    EXT_EXPOSURE = args.extend_exposure
    region = args.region
    if region not in ['reunion', 'guadeloupe']:
        raise(ValueError("region not in list"))
    if EXT_EXPOSURE:
        exp_method = 'adjusted_exposure'
    else:
        exp_method = 'original_exposure'
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H%M")
    dir_out = f"./output/{region}/{exp_method}/{thr_gum}/{thr_mar[0]}m_{thr_mar[1]}mps_{dt_string}/"
    path_out = Path(dir_out)
    if not path_out.exists():
        path_out.mkdir(parents=True, exist_ok=True)
# %%
# Load dataset
if region == 'guadeloupe':
    ds_full = xr.open_mfdataset(
        "./ww3_meteo_max/*.nc", combine="nested", concat_dim="event", parallel=True
    )
    kval = xr.load_dataarray('ww3_meteo/other/kval.nc').values
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
    category = pd.read_csv("category.csv", header=None).to_numpy().squeeze()

    mask_lon = (ds_full.longitude >= min_lon) & (ds_full.longitude <= max_lon)
    mask_lat = (ds_full.latitude >= min_lat) & (ds_full.latitude <= max_lat)
    ds_cropped = (
        ds_full.drop_dims(("single", "nele"))
        .where(mask_lon & mask_lat, drop=True)
        .compute()
    )
    # Load Bathymetry
    ds_bathy_full = xr.open_dataset("./Bathy.nc")
    ds_bathy = (
        ds_bathy_full.drop_dims(("single", "nele"))
        .where(mask_lon & mask_lat, drop=True)
        .compute()
    )
if region == 'reunion':
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

print(ds_cropped)


# %%
# working ds
ds_all = xr.merge([ds_cropped, ds_bathy])
is_deep = ds_all.bathymetry < depth
ds_all = ds_all.where(is_deep, drop=True)
tm_all = ds_all[['hs', 'UV_10m']].to_array()

num_events_total = ds_all.event.size
num_nodes = ds_all.node.size
num_vars = 2
# setup boostrap
N_bootstrap = 100
N_selevents = round(num_events_total*0.3)
idx_bootstrap = rng.choice(num_events_total, (N_bootstrap, N_selevents))
tm_sample = []

# %%
importlib.reload(stme)
for bi in trange(N_bootstrap):
    if N_bootstrap == 1:
        ds = ds_all
    else:
        ds = ds_all.isel({'event': idx_bootstrap[bi]})
    _rng = np.random.default_rng(seed=bi+1)
    num_events = ds.event.size
    var_name = ["$H_s$", "$U$"]
    var_name_g = ["$\hat H_s$", "$\hat U$"]
    par_name = ["$\\xi$", "$\\mu$", "$\\sigma$"]
    unit = ["[m]", "[m/s]"]
    # STM and Exposure
    tm = ds[['hs', 'UV_10m']].to_array()
    stm = ds[['hs', 'UV_10m']].max(dim="node").to_array()
    exp = ds[['hs', 'UV_10m']].to_array() / stm
    is_e_marginal = stm > thr_mar[:, np.newaxis]
    fig, ax = plt.subplots(1, num_vars, figsize=(8 * num_vars, 6))
    for vi in range(num_vars):
        ax[vi].hist(stm[vi], 20)
        ax[vi].set_xlabel(f"{var_name[vi]}{unit[vi]}")
        ax[vi].set_title("STM")
    lonlat = np.array([ds.longitude, ds.latitude]).T
    tree = KDTree(lonlat)

    exp_ext = np.zeros(exp.shape)
    for vi in range(num_vars):
        for ni in range(num_nodes):
            exp_ext[vi, :, ni] = exp[vi, :, ni] **\
                ((stm[vi]/thr_mar[vi])**kval[vi, ni])

    stm_idx = exp.argmax(axis=2)
    idx_east = lonlat[stm_idx, 0] > -61.5
    idx_west = lonlat[stm_idx, 0] < -61.5

    # Guadeloupe city
    _, idx_pos_list = tree.query([[-61.535, 16.200-i*0.05] for i in range(4)])
    # Generalized Pareto estimation
    if np.count_nonzero(stm > thr_mar) < 5:
        break
    gp = stme.genpar_estimation(
        stm, thr_mar, var_name, unit, par_name, dir_out=dir_out, draw_fig=True)
    # Threshold search
    if SEARCH_MTHR:
        importlib.reload(threshold_search)
        threshold_search.search_marginal(stm, [6, 35], [20, 55])
    # Laplace Transform
    importlib.reload(stme)
    stm_g, f_hat_cdf = stme.ndist_transform(
        stm, gp, var_name, unit, dir_out=dir_out, draw_fig=True)

    # ##################################################################################
    # def kendallpval(k, stm_norm, exp):
    #     _t, _p = kendalltau(stm_norm, exp**stm_norm**k)
    #     return -_p
    # if EXT_EXPOSURE:
    #     exp_ext = np.zeros(exp.shape)
    #     kval = np.zeros((num_vars, num_nodes))
    #     pval_ext = np.zeros((num_vars, num_nodes))
    #     tval_ext = np.zeros((num_vars, num_nodes))
    #     # fig, ax = plt.subplots(1, num_vars, figsize=(16, 8))
    #     for ni in trange(num_nodes):
    #         for vi in range(num_vars):
    #             _mask = stm[vi] > thr_mar[vi]
    #             _p = 0
    #             for k0 in np.linspace(-1, 1, 3):
    #                 _optres = minimize(kendallpval, k0, args=(
    #                     stm[vi, _mask]/thr_mar[vi], exp[vi, _mask, ni]), method='Powell')
    #                 if _optres.fun < _p:
    #                     _k = _optres.x
    #                     _p = _optres.fun
    #                     if _optres.fun < -0.05:
    #                         break
    #             exp_ext[vi, :, ni] = exp[vi, :, ni] **\
    #                 ((stm[vi]/thr_mar[vi])**kval[vi, ni])
    #             _t, _p = kendalltau(stm[vi]/thr_mar[vi], exp_ext[vi, :, ni])
    #             kval[vi, ni] = _k
    #             pval_ext[vi, ni] = _p
    #             tval_ext[vi, ni] = _t
    #     # pval
    #     fig, ax = plt.subplots(1, num_vars, figsize=(8*2, 6))
    #     for vi in range(num_vars):
    #         # ax[vi].scatter(lonlat[:,0],lonlat[:,1],facecolors='None',edgecolors='black',s=10,lw=0.1)
    #         ax[vi].scatter(lonlat[:, 0], lonlat[:, 1], c=[
    #             "red" if p < 0.05 else "black" for p in pval_ext[vi, :]], s=5)
    #         ax[vi].set_title(f'{var_name[vi]}')

    #     # tau
    #     fig, ax = plt.subplots(1, num_vars, figsize=(8*2, 6))
    #     for vi in range(num_vars):
    #         im = ax[vi].scatter(lonlat[:, 0], lonlat[:, 1], c=tval_ext[vi],
    #                             cmap='seismic', vmax=0.01, vmin=-0.01, s=5)
    #         plt.colorbar(im, ax=ax[vi])
    #         ax[vi].set_title(f'{var_name[vi]}')

    #     # kval
    #     fig, ax = plt.subplots(1, num_vars, figsize=(8*2, 6))
    #     fig.suptitle('kval')
    #     for vi in range(num_vars):
    #         # ax[vi].scatter(lonlat[:,0],lonlat[:,1],facecolors='None',edgecolors='black',s=10,lw=0.1)
    #         im = ax[vi].scatter(
    #             lonlat[:, 0],
    #             lonlat[:, 1],
    #             c=kval[vi],
    #             cmap='seismic',
    #             vmax=np.abs(kval[vi]).max(),
    #             vmin=-np.abs(kval[vi]).max(),
    #             s=10)
    #         plt.colorbar(im, ax=ax[vi])
    #         ax[vi].set_title(f'{var_name[vi]}')

    #     ds_k = xr.Dataset(
    #         data_vars=dict(
    #             kval=(['variable', 'node'], kval)
    #         ),
    #         coords=dict(
    #             lon=(['node'], lonlat[:, 0]),
    #             lat=(['node'], lonlat[:, 1]),
    #         ),
    #         attrs=dict(description='k-val(Exp/STM^k) for Guadeloupe data')
    #     )
    #     ds_k.to_netcdf(
    #         f'ww3_meteo/other/kval_{thr_gum}.nc')
    # else:
    #     print()
    ##################################################################################

    # Set boolean mask
    is_e = stm_g > thr_gum
    vi_largest = stm_g.argmax(axis=0)
    is_me = np.empty((num_vars, num_events))
    for vi in range(num_vars):
        is_me[vi] = np.logical_and(vi_largest == vi, is_e[vi])
    is_e_any = is_e.any(axis=0)
    exceedance_prob = np.count_nonzero(is_e_any) / num_events
    v_me_ratio = np.count_nonzero(is_me, axis=1) / np.count_nonzero(is_e_any)
    # Kendall's Tau for differing variables
    if EXT_EXPOSURE:
        stme.kendall_tau_mv(stm_g, exp_ext, is_e, var_name, lonlat,
                            dir_out=dir_out, draw_fig=True)
    else:
        stme.kendall_tau_mv(stm_g, exp, is_e, var_name, lonlat,
                            dir_out=dir_out, draw_fig=True)
    # Estimate conmul parameters
    params_median, residual = stme.estimate_conmul(
        stm_g, thr_gum, var_name, dir_out=dir_out, SEARCH=SEARCH_CTHR, draw_fig=True)
    # Sample from conmul model
    occur_prob = 1.04
    sample_full, ppf = stme.sample_stm(stm, stm_g, gp, f_hat_cdf, thr_gum, params_median,
                                       residual, occur_prob, var_name, unit, dir_out=dir_out, draw_fig=True)
    N_sample = sample_full.shape[1]
    # Exposure set sampling
    N = num_events * 1
    return_periods = [100]
    exp_sample = np.zeros((num_vars, N, num_nodes))
    stm_sample = np.zeros((num_vars, N))

    # choose random stm set from sample_full
    _idx_smp = _rng.choice(N_sample, size=N)
    stm_sample = sample_full[:, _idx_smp]
    # choose random exposure set from events where
    # STM is extreme in either variable
    _idx_evt = _rng.choice(np.nonzero(is_e_any)[0], size=N)
    if EXT_EXPOSURE:
        # transform exp_ext back to exposure
        exp_sample = np.zeros((num_vars, N, num_nodes))
        for vi in range(num_vars):
            for ni in range(num_nodes):
                exp_sample[vi, :, ni] = exp_ext[vi, _idx_evt, ni] **\
                    ((thr_mar[vi]/stm_sample[vi])**kval[vi, ni])
    else:
        exp_sample = exp[:, _idx_evt, :]

    # factor
    _tm_sample = np.einsum("ven,ve->ven", exp_sample, stm_sample)
    tm_sample.append(_tm_sample)

    # ##################################################################################################################
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(exp[0, :, 0], exp[1, :, 0], c='black', s=20)
    ax.scatter(exp_sample[0, :, 0], exp_sample[1, :, 0],
               edgecolors='red', s=30, facecolors='none')
    ax.set_xlabel(r"Exposure($H_s$)")
    ax.set_ylabel(r"Exposure($U$)")
    #########################################################
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(stm[0], stm[1], c='black', s=20)
    ax.scatter(stm_sample[0], stm_sample[1],
               edgecolors='red', s=30, facecolors='none')
    ax.set_xlabel(r"$H_s$")
    ax.set_ylabel(r"$U$")
    # ##################################################################################################################

    # output info
    if SAVE:
        with open(f"{dir_out}/info.txt", "w") as f:
            _sample_h = np.count_nonzero(stm[0] > thr_mar[0])
            _sample_u = np.count_nonzero(stm[1] > thr_mar[1])
            _sample_hg = np.count_nonzero(stm_g[0] > thr_gum)
            _sample_ug = np.count_nonzero(stm_g[1] > thr_gum)

            f.write(
                f"Marginal Threshold:\t{thr_mar[0]}[m], {thr_mar[1]}[m/s]\n")
            f.write(f"Marginal Sample Size:\t{_sample_h},{_sample_u}\n")
            f.write(f"Laplace Threshold: \t{thr_gum}\n")
            f.write(f"Laplace Sample Size:\t{_sample_hg},{_sample_ug}\n")
            f.write(
                f"Laplace Threshold in Marginal Scale:\t{ppf[0](np.exp(-np.exp(-thr_gum))):.2f}, {ppf[1](np.exp(-np.exp(-thr_gum))):.2f}"
            )

# %%
# Simple Contour Return Period


res = 100
_x = np.linspace(0, 18, res)
_y = np.linspace(0, 60, res)
_x_mg, _y_mg = np.meshgrid(_x, _y)

#########################################################
fig, axes = plt.subplots(2, 2, figsize=(8, 6), facecolor="white",)
# fig.suptitle(
#     f"{return_period}-yr return period (Saint-Denis) Laplace threshold = {thr_gum}",
#     y=0.90,
# )
fig.supxlabel(r"$H_s$[m]")
fig.supylabel(r"$U$[m/s]")

for i, ax in enumerate(axes.flatten()):
    _linestyles = ["-", "--"]
    _idx_pos = idx_pos_list[i]
    _z_mg_sample = np.zeros((N_bootstrap, res, res))
    _z_mg = np.zeros((res, res))

    for xi in range(res):
        for yi in range(res):
            _count = np.count_nonzero(
                np.logical_and(
                    tm_all[0, :, _idx_pos] > _x[xi], tm_all[1,
                                                            :, _idx_pos] > _y[yi],
                )
            )
            _z_mg[xi, yi] = _count

    for bi in range(N_bootstrap):
        for xi in range(res):
            for yi in range(res):
                _count_sample = np.count_nonzero(
                    np.logical_and(
                        tm_sample[bi][0, :, _idx_pos] > _x[xi],
                        tm_sample[bi][1, :, _idx_pos] > _y[yi],
                    )
                )
                _z_mg_sample[bi, xi, yi] = _count_sample

    for bi in range(N_bootstrap):
        ax.scatter(
            tm_sample[bi][0, :, _idx_pos],
            tm_sample[bi][1, :, _idx_pos],
            s=1,
            c=pos_color[i],
            label=f"Simulated",
            alpha=1.0,
        )
        _num_events_extreme = tm_sample[bi].shape[1]
        _levels_sample = [
            _num_events_extreme / (rp * occur_prob * exceedance_prob)
            for rp in return_periods
        ]
        ax.contour(
            _x_mg,
            _y_mg,
            _z_mg_sample[bi].T,
            levels=_levels_sample,
            linestyles=_linestyles,
            colors=pos_color[i],
            alpha=1.0
        )

    ax.scatter(
        tm_all[0, :, _idx_pos],
        tm_all[1, :, _idx_pos],
        s=5,
        c="black",
        label=f"Original",
        alpha=1.0,
    )
    _levels_original = [num_events /
                        (rp * occur_prob) for rp in return_periods]
    ax.contour(
        _x_mg,
        _y_mg,
        _z_mg.T,
        levels=_levels_original,
        linestyles=_linestyles,
        colors="black",
    )
    ax.set_title(f"Coord.{i+1}")
if SAVE:
    plt.savefig(f"{dir_out}/RV_(Saint-Denis).pdf", bbox_inches="tight")

# # print("FINISHED")
# # %%
# # kval = xr.load_dataarray('ww3_meteo/other/kval.nc').values
# exp_ext = np.zeros(exp.shape)
# pval_ext = np.zeros((num_vars, num_nodes))
# tval_ext = np.zeros((num_vars, num_nodes))

# for vi in range(num_vars):
#     _mask = stm[vi] > thr_mar[vi]
#     for ni in range(num_nodes):
#         exp_ext[vi, :, ni] = exp[vi, :, ni] **\
#             ((stm[vi]/thr_mar[vi])**kval[vi, ni])
#         tval_ext[vi, ni], pval_ext[vi, ni] = kendalltau(
#             stm[vi, _mask], exp_ext[vi, _mask, ni])
# # # %%
# # # Normalized Exposure


# # def kendallpval(k, stm_norm, exp):
# #     _t, _p = kendalltau(stm_norm, exp**stm_norm**k)
# #     return -_p


# # # %%
# # exp_ext = np.zeros(exp.shape)
# # kval = np.zeros((num_vars, num_nodes))
# # pval_ext = np.zeros((num_vars, num_nodes))
# # tval_ext = np.zeros((num_vars, num_nodes))
# # # fig, ax = plt.subplots(1, num_vars, figsize=(16, 8))
# # for ni in trange(num_nodes):
# #     for vi in range(num_vars):
# #         _mask = stm[vi] > thr_mar[vi]
# #         _p = 0
# #         for k0 in np.linspace(-1, 1, 3):
# #             _optres = minimize(kendallpval, k0, args=(
# #                 stm[vi, _mask]/thr_mar[vi], exp[vi, _mask, ni]), method='Powell')
# #             if _optres.fun < _p:
# #                 _k = _optres.x
# #                 _p = _optres.fun
# #                 if _optres.fun < -0.05:
# #                     break
# #         exp_ext[vi, :, ni] = exp[vi, :, ni] **\
# #             ((stm[vi]/thr_mar[vi])**kval[vi, ni])
# #         _t, _p = kendalltau(stm[vi]/thr_mar[vi], exp_ext[vi, :, ni])
# #         kval[vi, ni] = _k
# #         pval_ext[vi, ni] = _p
# #         tval_ext[vi, ni] = _t

# # ds_k = xr.Dataset(
# #     data_vars=dict(
# #         kval=(['variable', 'node'], kval)
# #     ),
# #     coords=dict(
# #         lon=(['node'], lonlat[:, 0]),
# #         lat=(['node'], lonlat[:, 1]),
# #     ),
# #     attrs=dict(description='k-val(Exp/STM^k) for Guadeloupe data')
# # )
# # ds_k.to_netcdf('ww3_meteo/other/kval.nc')
# # %%
# # pval
# fig, ax = plt.subplots(1, num_vars, figsize=(8*2, 6))
# for vi in range(num_vars):
#     # ax[vi].scatter(lonlat[:,0],lonlat[:,1],facecolors='None',edgecolors='black',s=10,lw=0.1)
#     ax[vi].scatter(lonlat[:, 0], lonlat[:, 1], c=[
#         "red" if p < 0.05 else "black" for p in pval_ext[vi, :]], s=5)
#     ax[vi].set_title(f'{var_name[vi]}')

# # tau
# fig, ax = plt.subplots(1, num_vars, figsize=(8*2, 6))
# for vi in range(num_vars):
#     im = ax[vi].scatter(lonlat[:, 0], lonlat[:, 1], c=tval_ext[vi],
#                         cmap='seismic', vmax=0.01, vmin=-0.01, s=5)
#     plt.colorbar(im, ax=ax[vi])
#     ax[vi].set_title(f'{var_name[vi]}')
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
# ds_k.to_netcdf(f'ww3_meteo/other/kval_{thr_gum}_{thr_mar[0]}_{thr_mar[1]}.nc')
# # %%
# tracks = []
# for tp in Path("tracks").glob("*.txt"):
#     _arr = pd.read_csv(tp, delimiter='\t')[
#         ['longitude', 'latitude']].to_numpy()
#     tracks.append(_arr)
# # %%
# top10 = np.argsort(stm.values, axis=1)
# fig, ax = plt.subplots(3, 3, figsize=(8*3, 6*3))
# for i, _ax in enumerate(ax.ravel()):
#     _ax.scatter(lonlat[:, 0], lonlat[:, 1], c=exp[0, top10[0, i+600], :])
#     _ax.set_title(f'{stm.values[0,top10[0,i+600]]:00.1f}m')
#     _track = tracks[top10[0, i+600]]
#     _ax.plot(_track[:, 0], _track[:, 1], c='black', lw=4)
#     _ax.set_xlim(lonlat[:, 0].min(), lonlat[:, 0].max())
#     _ax.set_ylim(lonlat[:, 1].min(), lonlat[:, 1].max())

# # %%
# mean_exp_h = exp[0, top10[0, 670:], :].mean(axis=0)
# mean_exp_u = exp[1, top10[1, 670:], :].mean(axis=0)
# plt.scatter(lonlat[:, 0], lonlat[:, 1], c=mean_exp_h)
# plt.colorbar()
# plt.title(f'Mean exposure(Hs) of bottom 30 cyclones')
# plt.show()
# plt.scatter(lonlat[:, 0], lonlat[:, 1], c=mean_exp_u)
# plt.colorbar()
# plt.title(f'Mean exposure(U) of bottom 30 cyclones')
# # %%
# fig, ax = plt.subplots(1, 2, figsize=(8*2, 6))
# inferno = plt.get_cmap('inferno')
# for vi in range(num_vars):
#     ax[vi].set_xlabel(f'Exposure({var_name[vi]})')
#     ax[vi].set_ylabel(f'CDF')
#     # all
#     for ei in range(num_events):
#         _ecdf = ECDF(exp[vi, ei, :])
#         _x = np.linspace(0, 1, 100)
#         im = ax[vi].plot(_x, _ecdf(_x), c=plt.get_cmap(
#             'viridis')(ei/num_events), alpha=0.3)
#     # # top 10
#     # for ei in top10[vi,-10:]:
#     #     _ecdf = ECDF(exp[vi,ei,:])
#     #     _x = np.linspace(0,1,100)
#     #     im = ax[vi].plot(_x, _ecdf(_x), c='red')
#     # # bottom 10
#     # for ei in top10[vi,:10]:
#     #     _ecdf = ECDF(exp[vi,ei,:])
#     #     _x = np.linspace(0,1,100)
#     #     im = ax[vi].plot(_x, _ecdf(_x), c='black')

# # %%
# fig, ax = plt.subplots(1, 2, figsize=(8*2, 6))
# _min = [0, 10]
# _max = [25, 70]
# for vi in range(num_vars):
#     for ci in range(6):
#         _stm = []
#         for ei in range(num_events):
#             if category[ei] == ci:
#                 _stm.append(stm.values[vi, ei])
#         ax[vi].hist(_stm, bins=np.linspace(_min[vi], _max[vi], 51), label=ci)
#         ax[vi].legend()
# # %%
# for ci in range(5):
#     for ei in range(num_events):
#         if category[ei] == ci+1:
#             plt.plot(tracks[ei][:, 0], tracks[ei][:, 1], c=pos_color[ci])
#     plt.show()

# # %%
# fig, ax = plt.subplots(1, 2, figsize=(8*2, 6))
# _min = [0, 10]
# _max = [25, 70]
# _nbins = [51, 61]
# for vi in range(num_vars):
#     for ci in range(6):
#         _stm = []
#         for ei in range(num_events):
#             # if category[ei] <= ci:
#             if category[ei] <= ci and category[ei] > 1:
#                 _stm.append(stm.values[vi, ei])
#         ax[vi].hist(_stm, bins=np.linspace(
#             _min[vi], _max[vi], _nbins[vi]), label=ci, zorder=-ci)
#         ax[vi].legend()
# # %%
