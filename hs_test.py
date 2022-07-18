# %%
# init
import numpy as np
from scipy.stats._continuous_distns import genpareto
from scipy.stats import kendalltau
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
import xarray as xr
import lwm
from scipy.spatial import KDTree
rng = np.random.default_rng()
plt.style.use("plot_style.txt")

# %%
# config
thr = 16
depth = -100
# %%
# Load STM & Exposure
if not 'ds_full' in globals():
    ds_full = xr.open_mfdataset(
        "./Hs_max/*.nc", combine="nested", concat_dim="event", parallel=True
    ).chunk("auto")
    print(ds_full)
    _min_lon = -65.00
    _min_lat = 12.00
    _max_lon = -58.00
    _max_lat = 18.00
    _mask_lon = (ds_full.longitude >= _min_lon) & (ds_full.longitude <= _max_lon)
    _mask_lat = (ds_full.latitude >= _min_lat) & (ds_full.latitude <= _max_lat)
    ds_cropped = (
        ds_full.drop_dims(("single", "nele"))
        .where(_mask_lon & _mask_lat, drop=True)
        .compute()
    )
    print(ds_cropped)
    # Load Bathymetry
    ds_bathy_full = xr.open_dataset("./Bathy.nc")
    ds_bathy = (
        ds_bathy_full.drop_dims(("single", "nele"))
        .where(_mask_lon & _mask_lat, drop=True)
        .compute()
    )
# %%
# working ds
ds = xr.merge([ds_cropped, ds_bathy])
is_deep = ds.bathymetry < depth
ds = ds.where(is_deep, drop=True)
num_events = ds.event.size
num_nodes = ds.node.size
lonlat = np.array([ds.longitude,ds.latitude]).T
tree = KDTree(lonlat)

# %%
# Bathymetry
fig, ax = plt.subplots(sharey=True, figsize=(4, 3), facecolor="white")

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
im = ax.scatter(ds.longitude, ds.latitude, s=1, c=ds.bathymetry)
fig.colorbar(im)

# %%
# STM and Exposure
stm = ds.hs_max.max(dim="node")
exp = ds.hs_max / stm
is_e_marginal = stm > thr
plt.hist(stm)
plt.xlabel("Hs[m]")
plt.title("STM")
# %%
# Generalized Pareto estimation
N_gp = 100
genpar_params = np.zeros((N_gp, 3))
_stm_bootstrap = rng.choice(stm, size=(N_gp, num_events))
for i in range(N_gp):
    _stm = _stm_bootstrap[i, :]
    _stm_pot = _stm[_stm > thr]
    _xp, _mp, _sp = genpareto.fit(_stm_pot, floc=thr)
    genpar_params[i, :,] = [_xp, _mp, _sp]
xp, mp, sp = np.mean(genpar_params[:, :], axis=0)
print(f"GENPAR{xp, mp, sp}")
gp = genpareto(xp, mp, sp)
par_name = ["$\\xi$", "$\\mu$", "$\\sigma$"]

#########################################################
fig, ax = plt.subplots(3, 1, figsize=(4, 9))

for pi, p in enumerate(par_name):
    ax[pi].set_ylabel(par_name[pi])
    ax[pi].hist(genpar_params[:, pi])
#########################################################

# %%
#########################################################
fig, ax = plt.subplots(figsize=(4, 3))
fig.set_facecolor("white")
ax.set_ylabel("CDF")

_res = 100
_cdf_all = np.zeros((N_gp, _res))
_x = np.linspace(thr, stm.max(), _res)
for i in range(N_gp):
    _xp = genpar_params[i, 0]
    _mp = genpar_params[i, 1]
    _sp = genpar_params[i, 2]
    _cdf_all[i, :] = genpareto(_xp, _mp, _sp).cdf(_x)

_y = gp.cdf(_x)
u95 = np.percentile(_cdf_all, 97.5, axis=0)
l95 = np.percentile(_cdf_all, 2.5, axis=0)
ax.plot(_x, _y, c="blue", lw=2, alpha=1)
ax.fill_between(_x, u95, l95, alpha=0.5)
_ecdf = ECDF(stm[stm > thr])
_x = np.linspace(thr, stm.max(), _res)
ax.plot(_x, _ecdf(_x), lw=2, color="black")
# ax.set_xlabel(f"{var_name}{unit}")

#########################################################


# %%
# Kendall's Tau
tau = np.zeros(num_nodes)
pval = np.zeros(num_nodes)
for i in range(num_nodes):
    _tau, _pval = kendalltau(stm[is_e_marginal], exp[is_e_marginal, i])
    tau[i] = _tau
    pval[i] = _pval

#########################################################
fig, ax = plt.subplots(sharey=True, figsize=(4, 3), facecolor="white")

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
_c = ["red" if p < 0.05 else "black" for p in pval]
ax.scatter(ds.longitude, ds.latitude, s=1, c=_c)
ax.set_title(f"Thr:{thr}, Depth<{depth}")
#########################################################
# %%
import importlib
import lwm

importlib.reload(lwm)
lwm.lwm_gpd(stm, [0.001], 15, [50, 100], 685)