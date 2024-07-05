# %%
# init
import importlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import stme
import xarray as xr
from scipy.optimize import minimize
from scipy.spatial import KDTree
from scipy.stats import genextreme, genpareto, kendalltau
from statsmodels.distributions.empirical_distribution import ECDF

plt.style.use("plot_style.txt")
rng = np.random.default_rng()
SAVE = True
depth = -100
thr_mar = np.array([15, 45])

dir_out = rf"./output/common"
path_out = Path(dir_out)
if not path_out.exists():
    path_out.mkdir()


# %%
# Load dataset
if not "ds_full" in globals():
    ds_full = xr.open_mfdataset(
        "./data/ww3_meteo_max/*.nc", combine="nested", concat_dim="event", parallel=True
    )
    print(ds_full)
    min_lon = -65.00
    min_lat = 12.00
    max_lon = -58.00
    max_lat = 18.00
    mask_lon = (ds_full.longitude >= min_lon) & (ds_full.longitude <= max_lon)
    mask_lat = (ds_full.latitude >= min_lat) & (ds_full.latitude <= max_lat)
    ds_cropped = (
        ds_full.drop_dims(("single", "nele"))
        .where(mask_lon & mask_lat, drop=True)
        .compute()
    )
    print(ds_cropped)

    # Load Bathymetry
    ds_bathy_full = xr.open_dataset("./data/Bathy.nc")
    ds_bathy = (
        ds_bathy_full.drop_dims(("single", "nele"))
        .where(mask_lon & mask_lat, drop=True)
        .compute()
    )
# %%
# working ds
ds = xr.merge([ds_cropped, ds_bathy])
is_deep = ds.bathymetry < depth
ds = ds.where(is_deep, drop=True)
num_events = ds.event.size
num_nodes = ds.node.size
num_vars = 2
var_name = [r"$H_s$", r"$U$"]
var_name_g = [r"$\tilde{H}$", r"$\tilde{U}$"]
unit = ["[m]", "[m/s]"]
pos_color = plt.rcParams["axes.prop_cycle"].by_key()["color"]
# %%
# STM and Exposure
stm = ds[["hs", "UV_10m"]].max(dim="node").to_array()
exp = ds[["hs", "UV_10m"]].to_array() / stm
is_e_marginal = stm > thr_mar[:, np.newaxis]
fig, ax = plt.subplots(1, num_vars, figsize=(8 * num_vars, 6))
for vi in range(num_vars):
    ax[vi].hist(stm[vi, is_e_marginal[vi]], 20)
    ax[vi].set_xlabel(f"{var_name[vi]}{unit[vi]}")
    ax[vi].set_title("STM")
lonlat = np.array([ds.longitude, ds.latitude]).T
tree = KDTree(lonlat)
# %%
# Guadeloupe city
_, idx_pos_list = tree.query([[-61.535, 16.200 - i * 0.05] for i in range(4)])

# %%
# Generalized Pareto estimation
N_gp = 100
genpar_params = np.zeros((num_vars, N_gp, 3))
gp = [None, None]

for vi in range(num_vars):
    _stm_bootstrap = rng.choice(stm[vi], size=(N_gp, num_events))
    for i in range(N_gp):
        _stm = _stm_bootstrap[i]
        _stm_pot = _stm[_stm > thr_mar[vi]]
        _xp, _mp, _sp = genpareto.fit(_stm_pot, floc=thr_mar[vi])
        genpar_params[vi, i, :] = [_xp, _mp, _sp]
    xp, mp, sp = np.mean(genpar_params[vi, :, :], axis=0)
    print(f"GENPAR{xp, mp, sp}")
    gp[vi] = genpareto(xp, mp, sp)
par_name = [r"$\xi$", r"$\mu$", r"$\sigma$"]

#########################################################
fig, ax = plt.subplots(
    len(par_name), num_vars, figsize=(8 * num_vars, 6 * len(par_name))
)

for vi in range(num_vars):
    ax[0, vi].set_title(var_name[vi])
    for pi, p in enumerate(par_name):
        ax[pi, 0].set_ylabel(par_name[pi])
        ax[pi, vi].hist(genpar_params[vi, :, pi])
if SAVE:
    plt.savefig(f"{dir_out}/Genpar_Params.pdf", bbox_inches="tight")
#########################################################

#########################################################
fig, ax = plt.subplots(1, num_vars, figsize=(8 * num_vars, 6))
fig.set_facecolor("white")
# ax.set_ylabel("CDF")

_res = 100
for vi in range(num_vars):
    _cdf_all = np.zeros((N_gp, _res))
    _x = np.linspace(thr_mar[vi], stm[vi].max(), _res)
    for i in range(N_gp):
        _xp = genpar_params[vi, i, 0]
        _mp = genpar_params[vi, i, 1]
        _sp = genpar_params[vi, i, 2]
        _cdf_all[i, :] = genpareto(_xp, _mp, _sp).cdf(_x)

    _y = gp[vi].cdf(_x)
    _u95 = np.percentile(_cdf_all, 97.5, axis=0)
    _l95 = np.percentile(_cdf_all, 2.5, axis=0)
    ax[vi].plot(_x, _y, c="blue", lw=2, alpha=1)
    ax[vi].fill_between(_x, _u95, _l95, alpha=0.5)
    _ecdf = ECDF(stm[vi, is_e_marginal[vi]])
    _x = np.linspace(thr_mar[vi], stm[vi].max(), _res)
    ax[vi].plot(_x, _ecdf(_x), lw=2, color="black")
    ax[vi].set_xlabel(f"{var_name[vi]}{unit[vi]}")
if SAVE:
    plt.savefig(f"{dir_out}/Genpar_CDF.pdf", bbox_inches="tight")
#########################################################

# %%
# Gumbel Transform
importlib.reload(stme)
stm_g = np.zeros(stm.shape)
f_hat_cdf = [None, None]
for vi in range(num_vars):
    f_hat_cdf[vi] = lambda x, idx=vi: stme._f_hat_cdf(ECDF(stm[idx]), gp[idx], x)
    _stm = stm[vi]
    stm_g[vi] = -np.log(-np.log(f_hat_cdf[vi](_stm)))
# print(stm_g[:,0].argmax(), stm_g[:,1].argmax())
print(stm_g[0].max(), stm_g[1].max())
print(stm_g[0].min(), stm_g[1].min())

#########################################################
fig, ax = plt.subplots(1, 2, figsize=(7, 3))

ax[0].scatter(stm[0], stm[1], s=5)
ax[0].set_xlabel(f"{var_name[0]}{unit[0]}")
ax[0].set_ylabel(f"{var_name[1]}{unit[1]}")
ax[0].set_xlim(0, 20)
ax[0].set_ylim(0, 60)


ax[1].set_aspect(1)
ax[1].scatter(stm_g[0], stm_g[1], s=5)
ax[1].set_xlabel(r"$\tilde{H}$")
ax[1].set_ylabel(r"$\tilde{U}$")
ax[1].set_xlim(-2, 8)
ax[1].set_ylim(-2, 8)

if SAVE:
    plt.savefig(f"{dir_out}/Original_vs_Gumbel.pdf", bbox_inches="tight")
#########################################################
# %%
# Set boolean mask
is_e = stm_g > thr_gum
vi_largest = stm_g.argmax(axis=0)
is_me = np.empty((num_vars, num_events))
for vi in range(num_vars):
    is_me[vi] = np.logical_and(vi_largest == vi, is_e[vi])
is_e_any = is_e.any(axis=0)
exceedance_prob = np.count_nonzero(is_e_any) / num_events
v_me_ratio = np.count_nonzero(is_me, axis=1) / np.count_nonzero(is_e_any)

# %%
# Kendall's Tau
tval = np.zeros((num_vars, num_nodes))
pval = np.zeros((num_vars, num_nodes))
for vi in range(num_vars):
    _stm = stm_g[vi]
    _exp = exp[vi, :, :]
    for i in range(num_nodes):
        _tval, _pval = kendalltau(_stm[is_e[vi]], _exp[is_e[vi], i])
        tval[vi, i] = _tval
        pval[vi, i] = _pval

#########################################################
fig, ax = plt.subplots(
    1, num_vars, sharey=True, figsize=(8 * num_vars, 6), facecolor="white"
)
for vi in range(num_vars):
    ax[vi].set_xlabel("Longitude")
    ax[vi].set_ylabel("Latitude")
    _c = ["red" if p < 0.05 else "black" for p in pval[vi]]
    im = ax[vi].scatter(lonlat[:, 0], lonlat[:, 1], s=5, c=_c)
    ax[vi].set_title(var_name[vi])
    for i, _idx_pos in enumerate(idx_pos_list):
        ax[vi].scatter(
            lonlat[_idx_pos, 1], lonlat[_idx_pos, 0], s=50, color=pos_color[i]
        )
        ax[vi].annotate(
            rf"#{i + 1}",
            (
                lonlat[_idx_pos, 1] + (i % 2 - 0.65) * 2 * 0.2,
                lonlat[_idx_pos, 0] - 0.01,
            ),
            bbox=dict(facecolor="white", edgecolor=pos_color[i]),
        )
if SAVE:
    plt.savefig(f"{dir_out}/Kendall_Tau_pval.pdf", bbox_inches="tight")

fig, ax = plt.subplots(
    1, num_vars, sharey=True, figsize=(8 * num_vars, 6), facecolor="white"
)
for vi in range(num_vars):
    ax[vi].set_xlabel("Longitude")
    ax[vi].set_ylabel("Latitude")
    im = ax[vi].scatter(
        lonlat[:, 0],
        lonlat[:, 1],
        s=5,
        c=tval[vi],
        cmap="seismic",
        vmax=np.abs(tval[vi]).max(),
        vmin=-np.abs(tval[vi]).max(),
    )
    plt.colorbar(im, ax=ax[vi])
    ax[vi].set_title(var_name[vi])
if SAVE:
    plt.savefig(f"{dir_out}/Kendall_Tau_tval.pdf", bbox_inches="tight")
#########################################################

# %%
# Kendall's Tau for differing variables
tval = np.zeros(((num_vars, num_vars, num_nodes)))
pval = np.zeros((num_vars, num_vars, num_nodes))
for vi in range(num_vars):
    for vj in range(num_vars):
        _stm = stm_g[vi]
        _exp = exp[vj, :, :]
        for ni in range(num_nodes):
            _tval, _pval = kendalltau(_stm[is_e[vi]], _exp[is_e[vi], ni])
            tval[vi, vj, ni] = _tval
            pval[vi, vj, ni] = _pval

# %%
#########################################################
fig, ax = plt.subplots(
    num_vars,
    num_vars,
    sharey=True,
    figsize=(8 * num_vars, 6 * num_vars),
    facecolor="white",
    squeeze=False,
)
# fig.supxlabel("Longitude")
# fig.supylabel("Latitude")

for vi in range(num_vars):
    for vj in range(num_vars):
        ax[vi, vj].set_xlabel("Longitude")
        ax[vi, vj].set_ylabel("Latitude")
        _c = ["red" if p < 0.05 else "black" for p in pval[vi, vj, :]]
        im = ax[vi, vj].scatter(lonlat[:, 0], lonlat[:, 1], s=5, c=_c)
        ax[vi, vj].set_title(f"STM:{var_name[vi]} E:{var_name[vj]}")
if SAVE:
    plt.savefig(f"{dir_out}/Kendall_Tau_all_var_pval.pdf", bbox_inches="tight")
#########################################################
#########################################################
fig, ax = plt.subplots(
    num_vars,
    num_vars,
    sharey=True,
    figsize=(8 * num_vars, 6 * num_vars),
    facecolor="white",
    squeeze=False,
)
# fig.supxlabel("Longitude")
# fig.supylabel("Latitude")

for vi in range(num_vars):
    for vj in range(num_vars):
        ax[vi, vj].set_xlabel("Longitude")
        ax[vi, vj].set_ylabel("Latitude")
        im = ax[vi, vj].scatter(
            lonlat[:, 0],
            lonlat[:, 1],
            s=5,
            c=tval[vi, vj, :],
            cmap="seismic",
            vmax=np.abs(tval[vi]).max(),
            vmin=-np.abs(tval[vi]).max(),
        )
        plt.colorbar(im, ax=ax[vi, vj])
        ax[vi, vj].set_title(f"STM:{var_name[vi]} E:{var_name[vj]}")
if SAVE:
    plt.savefig(f"{dir_out}/Kendall_Tau_all_var_tval.pdf", bbox_inches="tight")
#########################################################

##################################################################################################################
# %%
# Gumbel replacement
N_rep = 1000
stm_g_rep = np.zeros((num_vars, N_rep, num_events))
for i in range(N_rep):
    _idx = rng.choice(num_events, size=num_events)
    _stm = stm_g[:, _idx]
    for vi in range(num_vars):
        _gumbel_sample = genextreme.rvs(0, size=num_events)
        _gumbel_sample_sorted = np.sort(_gumbel_sample)
        _arg = np.argsort(_stm[vi])
        stm_g_rep[vi, i, _arg] = _gumbel_sample_sorted

#########################################################
fig, ax = plt.subplots(1, 1, figsize=(8, 6), facecolor="white")
ax.scatter(stm_g_rep[0, :, :], stm_g_rep[1, :, :], alpha=0.1)
ax.scatter(stm_g[0], stm_g[1], color="blue")
ax.set_xlabel(r"$\tilde{H}$")
ax.set_ylabel(r"$\tilde{U}$")
ax.set_xlim(-3, 15)
ax.set_ylim(-3, 15)
if SAVE:
    plt.savefig(f"{dir_out}/Gumbel_Replacement.pdf", bbox_inches="tight")
#########################################################

##################################################################################################################
# %%
# Estimate conditional model parameters
importlib.reload(stme)
lb = [0, None, -5, 0]
ub = [1, 1, 5, 5]
params_uc = np.zeros((num_vars, N_rep, 4))
for vi in range(num_vars):
    for i in range(N_rep):
        a0 = np.random.uniform(low=lb[0], high=ub[0])
        b0 = np.random.uniform(low=-1, high=ub[1])
        m0 = np.random.uniform(low=-1, high=1)
        s0 = np.random.uniform(low=lb[3], high=1)
        # m0 = np.random.uniform(low=lb[2], high=ub[2])
        # s0 = np.random.uniform(low=lb[3], high=ub[3])
        evt_mask = stm_g_rep[vi, i, :] > thr_gum
        var_mask = np.full((num_vars), True)
        var_mask[vi] = False

        def func(x):
            return stme.cost(x, stm_g_rep[:, i, evt_mask], vi)

        optres = minimize(
            func,
            np.array([a0, b0, m0, s0]),
            # method='trust-constr',
            bounds=((lb[0], ub[0]), (lb[1], ub[1]), (lb[2], ub[2]), (lb[3], ub[3])),
        )
        _param = optres.x
        params_uc[vi, i, :] = _param
params_mean = np.mean(params_uc, axis=1)

#########################################################
fig, ax = plt.subplots(4, num_vars, figsize=(8 * num_vars, 6 * 4))
fig.tight_layout()
ax[0, 0].set_ylabel("a")
ax[1, 0].set_ylabel("b")
ax[2, 0].set_ylabel(r"$\mu$")
ax[3, 0].set_ylabel(r"$\sigma$")

ax[3, 0].set_xlabel(var_name[0])
ax[3, 1].set_xlabel(var_name[1])

for vi in range(num_vars):
    ax[0, vi].hist(params_uc[vi, :, 0])
    ax[1, vi].hist(params_uc[vi, :, 1])
    ax[2, vi].hist(params_uc[vi, :, 2])
    ax[3, vi].hist(params_uc[vi, :, 3])
if SAVE:
    plt.savefig(f"{dir_out}/Conmul_Estimates.pdf", bbox_inches="tight")
#########################################################

# %% [markdown]
# ## Grid search gumbel threshold

# %%
N_THR = 10
thr_gum_list = np.linspace(0, 3, N_THR)

lb = [0, None, -5, 0]
ub = [1, 1, 5, 5]
N = stm_g_rep.shape[0]
params_search_uc = np.zeros((N_THR, N, 4, num_vars))
for ti in range(N_THR):
    _thr = thr_gum_list[ti]
    for i in range(N):
        for vi in range(num_vars):
            a0 = np.random.uniform(low=lb[0], high=ub[0])
            b0 = np.random.uniform(low=-1, high=ub[1])
            m0 = np.random.uniform(low=-1, high=1)
            s0 = np.random.uniform(low=lb[3], high=1)

            evt_mask = stm_g_rep[i, :, vi] > _thr
            var_mask = np.full((stm_g_rep[i].shape[1]), True)
            var_mask[vi] = False

            def func(x):
                return stme.cost(
                    x, stm_g_rep[i, evt_mask, vi], stm_g_rep[i, evt_mask, var_mask]
                )

            optres = minimize(
                func,
                np.array([a0, b0, m0, s0]),
                # method='trust-constr',
                bounds=((lb[0], ub[0]), (lb[1], ub[1]), (lb[2], ub[2]), (lb[3], ub[3])),
            )
            _param = optres.x
            params_search_uc[ti, i, :, vi] = _param
params_mean = np.mean(params_search_uc, axis=1)
params_u95 = np.percentile(params_search_uc, 97.5, axis=1)
params_l95 = np.percentile(params_search_uc, 2.5, axis=1)
params_u75 = np.percentile(params_search_uc, 75.0, axis=1)
params_l25 = np.percentile(params_search_uc, 25.0, axis=1)

# %%
fig, ax = plt.subplots(2, 2, sharex=True, figsize=(10, 6), constrained_layout=True)
# fig.tight_layout()
# fig.con
fig.supxlabel("Gumbel threshold")
fig.set_facecolor("white")
p_name = ["a", "b", r"$\mu$", r"$\sigma$"]
m_name = ["U|H", "H|U"]
for vi in range(num_vars):
    ax[0, vi].set_title(m_name[vi])
    # ax[0,vi].set_title(var_name[vi])
    for pi in range(2):
        # ax[pi,vi].plot(thr_gum_list, params_mean[:,pi,vi])
        ax[pi, vi].plot(thr_gum_list, params_mean[:, pi, vi])
        ax[pi, vi].fill_between(
            thr_gum_list,
            params_u95[:, pi, vi],
            params_l95[:, pi, vi],
            color="blue",
            alpha=0.1,
        )
        ax[pi, vi].fill_between(
            thr_gum_list,
            params_u75[:, pi, vi],
            params_l25[:, pi, vi],
            color="blue",
            alpha=0.1,
        )
        ax[pi, 0].set_ylabel(p_name[pi])
plt.savefig(f"{dir_out}/Conmul_param_vs_threshold.pdf", bbox_inches="tight")

# %%
num_samples = np.zeros((N_THR, N, num_vars))
for ti in range(N_THR):
    for i in range(N):
        for vi in range(num_vars):
            num_samples[ti, i, vi] = np.count_nonzero(
                stm_g_rep[i, :, vi] > thr_gum_list[ti]
            )

# %%
fig, ax = plt.subplots(
    1, 2, sharex=True, sharey=True, figsize=(8, 3), facecolor="white"
)
# fig.tight_layout()
fig.supxlabel("Gumbel threshold")
fig.supylabel("# of occurrences above threshold")

_med = np.percentile(num_samples, 50, axis=1)
_u95 = np.percentile(num_samples, 97.5, axis=1)
_l95 = np.percentile(num_samples, 2.5, axis=1)
for vi in range(num_vars):
    ax[vi].plot(thr_gum_list, _med[:, vi])
    ax[vi].fill_between(thr_gum_list, _u95[:, vi], _l95[:, vi], alpha=0.5)
    ax[vi].set_title(var_name[vi])

plt.savefig(f"{dir_out}/Conmul_sample_num.pdf", bbox_inches="tight")
