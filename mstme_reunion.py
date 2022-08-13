# %%
# init
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
import src.threshold_search as threshold_search

plt.style.use("plot_style.txt")
rng0 = np.random.default_rng(seed=0)


def is_interactive():
    ip = False
    if "ipykernel" in sys.modules:
        ip = True
    elif "IPython" in sys.modules:
        ip = True
    return ip


print("Interactive:", is_interactive())
if is_interactive():
    thr_mar = np.array([6, 20])
    thr_gum = 1.25
    depth = -100
    SAVE = False
    dir_out = None
    SEARCH_MTHR = False
    SEARCH_CTHR = False
    N_bootstrap = 1
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
        "--depth", type=float, help="", required=False, default=-100
    )
    parser.add_argument(
        "--nbootstrap", type=int, help="", required=False, default=1
    )
    parser.add_argument(
        "--search_mthr", type=Bool, help="", required=False, default=False
    )
    parser.add_argument(
        "--search_cthr", type=Bool, help="", required=False, default=False
    )

    args = parser.parse_args()
    thr_mar = np.array([args.thr_hs, args.thr_u10])
    thr_gum = args.thr_gum
    depth = args.depth
    N_bootstrap = args.nbootstrap
    SAVE = True
    SEARCH_MTHR = args.search_mthr
    SEARCH_CTHR = args.search_cthr

    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H%M")
    dir_out = f"./output/{thr_mar[0]}m_{thr_mar[1]}mps_{thr_gum}_{dt_string}_laplace_exceedance/"
    path_out = Path(dir_out)
    if not path_out.exists():
        path_out.mkdir()
# %%
# Load Dataset
ds_all = xr.open_mfdataset(
    "./summary_data/*.nc", combine="nested", concat_dim="event", parallel=True
).chunk("auto")
print(ds_all)
kval = xr.load_dataarray('reunion_data/other/kval.nc').values

num_events_total = ds_all.event.size
num_nodes = ds_all.node.size
num_vars = 2
# Latlon
__ds_lonlat = xr.open_dataset("./latlon.nc")
lonlat = np.array([__ds_lonlat.lon, __ds_lonlat.lat]).T
tree = KDTree(lonlat)

# Saint Denis
_, idx_pos_list_saint_denis = tree.query(
    [[55.450675, -20.882057 + 0.1 * i] for i in range(4)])
# %%
# setup boostrap
N_selevents = round(num_events_total*0.3)
idx_bootstrap = rng0.choice(
    num_events_total, (N_bootstrap, N_selevents), replace=False)
# tm_sample = np.zeros((N_bootstrap, num_vars, N_selevents, num_nodes))
tm_sample = []
tm_all = np.einsum(
    "ven,ve->ven", np.stack([ds_all.exp_h, ds_all.exp_u]), np.array([ds_all.stm_h, ds_all.stm_u]))

var_name = ["$H_s$", "$U$"]
var_name_g = ["$\hat H_s$", "$\hat U$"]
par_name = ["$\\xi$", "$\\mu$", "$\\sigma$"]
unit = ["[m]", "[m/s]"]
pos_color = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# %%
importlib.reload(stme)
for bi in range(N_bootstrap):
    print(bi, "----------------------------------------------------------")
    _rng = np.random.default_rng(seed=bi+1)
    ds = ds_all
    # ds = ds_all.isel({'event': idx_bootstrap[bi]})
    num_events = ds.event.size
    # STM and Exposure
    stm = np.array([ds.stm_h, ds.stm_u])
    exp = np.stack([ds.exp_h, ds.exp_u])
    tm = np.einsum("ven,ve->ven", exp, stm)
    is_e_marginal = stm > thr_mar[:, np.newaxis]
    fig, ax = plt.subplots(1, num_vars, figsize=(8 * num_vars, 6))
    for vi in range(num_vars):
        ax[vi].hist(stm[vi], 20)
        ax[vi].set_xlabel(f"{var_name[vi]}{unit[vi]}")
        ax[vi].set_title("STM")

    exp_ext = np.zeros(exp.shape)
    for vi in range(num_vars):
        for ni in range(num_nodes):
            exp_ext[vi, :, ni] = exp[vi, :, ni] **\
                ((stm[vi]/thr_mar[vi])**kval[vi, ni])
    # Guadeloupe city
    _, idx_pos_list = tree.query([[-61.535, 16.200-i*0.05] for i in range(4)])
    # Generalized Pareto estimation
    gp = stme.genpar_estimation(
        stm, thr_mar, var_name, unit, par_name, dir_out=dir_out, draw_fig=False)
    # Threshold search
    if SEARCH_MTHR:
        importlib.reload(threshold_search)
        threshold_search.search_marginal(stm, [6, 35], [20, 55])
    # Transform
    importlib.reload(stme)
    stm_g, f_hat_cdf = stme.ndist_transform(
        stm, gp, var_name, unit, dir_out=dir_out, draw_fig=False)

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
    stme.kendall_tau_mv(stm_g, exp_ext, is_e, var_name, lonlat,
                        dir_out=dir_out, draw_fig=False)
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

    # choose random exposure set from events where
    # STM is extreme in either variable
    _num_exp = np.count_nonzero(is_e_any)
    _idx_evt = _rng.choice(_num_exp, size=N)
    # _exp_e = exp[:, is_e_any, :]
    # exp_sample = _exp_e[:, _idx_evt, :]

    # choose random stm set from sample_full
    _idx_smp = _rng.choice(N_sample, size=N)
    stm_sample = sample_full[:, _idx_smp]

    # transform exp_ext back to exposure
    exp_sample = np.zeros((num_vars, N, num_nodes))
    for vi in range(num_vars):
        for ni in range(num_nodes):
            exp_sample[vi, :, ni] = exp_ext[vi, _idx_evt, ni] **\
                ((thr_mar[vi]/stm_sample[vi])**kval[vi, ni])

    # factor
    _tm_sample = np.einsum("ven,ve->ven", exp_sample, stm_sample)
    tm_sample[bi] = _tm_sample

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
                        tm_sample[bi, 0, :, _idx_pos] > _x[xi],
                        tm_sample[bi, 1, :, _idx_pos] > _y[yi],
                    )
                )
                _z_mg_sample[bi, xi, yi] = _count_sample

    for bi in range(N_bootstrap):
        ax.scatter(
            tm_sample[bi, 0, :, _idx_pos],
            tm_sample[bi, 1, :, _idx_pos],
            s=1,
            c=pos_color[i],
            label=f"Simulated",
            alpha=0.5,
        )
        _levels_sample = [
            tm_sample.shape[2] / (rp * occur_prob * exceedance_prob)
            for rp in return_periods
        ]
        ax.contour(
            _x_mg,
            _y_mg,
            _z_mg_sample[bi].T,
            levels=_levels_sample,
            linestyles=_linestyles,
            colors=pos_color[i],
            alpha=0.5,
        )

    ax.scatter(
        tm_all[0, :, _idx_pos],
        tm_all[1, :, _idx_pos],
        s=5,
        c="black",
        label=f"Original",
        alpha=1.0,
    )
    _levels_original = [tm_all.shape[1] /
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

print("FINISHED")

# %%
# Normalized Exposure


def kendallpval(k, stm_norm, exp):
    _t, _p = kendalltau(stm_norm, exp**stm_norm**k)
    return -_p


# %%
# kval = xr.load_dataarray('reunion_data/other/kval.nc').values
exp_ext = np.zeros(exp.shape)
pval_ext = np.zeros((num_vars, num_nodes))
tval_ext = np.zeros((num_vars, num_nodes))

for vi in range(num_vars):
    _mask = stm[vi] > thr_mar[vi]
    for ni in range(num_nodes):
        exp_ext[vi, :, ni] = exp[vi, :, ni] **\
            ((stm[vi]/thr_mar[vi])**kval[vi, ni])
        tval_ext[vi, ni], pval_ext[vi, ni] = kendalltau(
            stm[vi, _mask]/thr_mar[vi], exp_ext[vi, _mask, ni])
# %%
exp_ext = np.zeros(exp.shape)
kval = np.zeros((num_vars, num_nodes))
pval_ext = np.zeros((num_vars, num_nodes))
tval_ext = np.zeros((num_vars, num_nodes))
# fig, ax = plt.subplots(1, num_vars, figsize=(16, 8))
for vi in range(num_vars):
    _mask = stm[vi] > thr_mar[vi]
    for ni in range(num_nodes):
        _p = 0
        for k0 in np.linspace(-1, 1, 3):
            _optres = minimize(kendallpval, k0, args=(
                stm[vi, _mask]/thr_mar[vi], exp[vi, _mask, ni]), method='Powell')
            if _optres.fun < _p:
                _k = _optres.x
                _p = _optres.fun
                if _optres.fun < -0.05:
                    break
        exp_ext[vi, :, ni] = exp[vi, :, ni]/stm[vi]**_k
        _t, _p = kendalltau(stm[vi]/thr_mar[vi], exp_ext[vi, :, ni])
        kval[vi, ni] = _k
        pval_ext[vi, ni] = _p
        tval_ext[vi, ni] = _t
# %%
# pval
fig, ax = plt.subplots(1, num_vars, figsize=(8*2, 6))
for vi in range(num_vars):
    # ax[vi].scatter(lonlat[:,0],lonlat[:,1],facecolors='None',edgecolors='black',s=10,lw=0.1)
    ax[vi].scatter(lonlat[:, 0], lonlat[:, 1], c=[
        "red" if p < 0.05 else "black" for p in pval_ext[vi, :]], s=5)
    ax[vi].set_title(f'{var_name[vi]}')
# %%
# tau
fig, ax = plt.subplots(1, num_vars, figsize=(8*2, 6))
for vi in range(num_vars):
    im = ax[vi].scatter(
        lonlat[:, 0],
        lonlat[:, 1],
        c=tval_ext[vi],
        cmap='seismic',
        vmax=np.abs(tval_ext[vi]).max(),
        vmin=-np.abs(tval_ext[vi]).max(),
        s=5)
    plt.colorbar(im, ax=ax[vi])
    ax[vi].set_title(f'{var_name[vi]}')
# # %%
# # kval
# fig, ax = plt.subplots(1, num_vars, figsize=(8*2, 6))
# for vi in range(num_vars):
#     # ax[vi].scatter(lonlat[:,0],lonlat[:,1],facecolors='None',edgecolors='black',s=10,lw=0.1)
#     im = ax[vi].scatter(
#         lonlat[:, 0],
#         lonlat[:, 1],
#         c=kval[vi],
#         cmap='seismic',
#         vmax=np.abs(kval[vi]).max(),
#         vmin=-np.abs(kval[vi]).max(),
#         s=5)
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
#     attrs=dict(description='k-val(Exp/STM^k) for Reunion data')
# )
# # %%
# ds_k.to_netcdf('reunion_data/other/kval.nc')

# # %%

# %%
