# %%
from unicodedata import numeric
from matplotlib import projections
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.stats import kendalltau
import pandas as pd
from pathlib import Path
from geopy.distance import distance
from geographiclib.geodesic import Geodesic
plt.style.use("plot_style.txt")
rng = np.random.default_rng()
pos_color = plt.rcParams["axes.prop_cycle"].by_key()["color"]
# %%
thr_mar = np.array([15, 45])
thr_gum = 2.0
depth = -100
SAVE = False
SEARCH_MTHR = False
SEARCH_CTHR = False
# %%
# Load dataset
ds_stm_full = xr.open_mfdataset(
    "./ww3_meteo_max/*.nc", combine="nested", concat_dim="event", parallel=True
).chunk({'event': 1})
ds_aux_full = xr.open_mfdataset(
    "./ww3_meteo_aux/*.nc", combine="nested", concat_dim="event", parallel=True
).chunk({'event': 1})

min_lon = -65.00
min_lat = 12.00
max_lon = -58.00
max_lat = 18.00
mask_lon = (ds_stm_full.longitude >= min_lon) & (
    ds_stm_full.longitude <= max_lon)
mask_lat = (ds_stm_full.latitude >= min_lat) & (
    ds_stm_full.latitude <= max_lat)

ds_stm = (
    ds_stm_full.drop_dims(("single", "nele"))
    .where(mask_lon & mask_lat, drop=True)
    .compute()
)
ds_aux = (
    ds_aux_full
    .where(mask_lon & mask_lat, drop=True)
    .compute()
)
# %%
# Load Bathymetry
ds_bathy_full = xr.open_dataset("./Bathy.nc")
ds_bathy = (
    ds_bathy_full.drop_dims(("single", "nele"))
    .where(mask_lon & mask_lat, drop=True)
    .compute()
)
# %%
ds = xr.merge([ds_stm, ds_bathy])
is_deep = ds.bathymetry < depth
ds = ds.where(is_deep, drop=True)
ds_aux = ds_aux.where(is_deep, drop=True)
num_events = ds.event.size
num_nodes = ds.node.size
num_vars = 2
var_name = ["$H_s$", "$U$"]
var_name_g = ["$\hat H_s$", "$\hat U$"]
par_name = ["$\\xi$", "$\\mu$", "$\\sigma$"]
unit = ["[m]", "[m/s]"]
pos_color = plt.rcParams["axes.prop_cycle"].by_key()["color"]
# %%
# %%
# STM and Exposure
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
# %%
dir = np.deg2rad((ds_aux[['dp', 'Dir_10m']].to_array()+180) % 360-180)
# %%
tracks = []
for tp in Path("tracks").glob("*.txt"):
    _arr = pd.read_csv(tp, delimiter='\t')[
        ['longitude', 'latitude']].to_numpy()
    tracks.append(_arr)
    # tracks.append(_df[['longitude','latitude']].values())
# %%
# pointe-a-pitre
origin = [-61.537284, 16.240871]
approach_azi = np.zeros((num_events,))
for i, t in enumerate(tracks):
    plt.plot(t[:, 0], t[:, 1], lw=0.1, c='black', alpha=0.1)
    _inside = False
    _p = []
    _min_d = 99999999
    for p in t:
        _d = distance((origin[1], origin[0]), (p[1], p[0]))
        _min_d = min(_d, _min_d)
        if _d < 500 and not _inside:
            _p.append(p)
            _inside = True
        if _d > 500 and _inside:
            _p.append(p)
            _inside = False
            break
    if len(_p) == 1 and _inside:
        _p.append(t[-1])
        print(i, _p, _min_d)
    # print(_p)
    _azi = np.deg2rad(Geodesic.WGS84.Inverse(
        _p[0][1], _p[0][0], _p[1][1], _p[1][0])['azi1'])
    approach_azi[i] = _azi
# %%
fig, ax = plt.subplots()
ax.set_xlim(min_lon, max_lon)
ax.set_ylim(min_lat, max_lat)
ax.scatter(ds.longitude, ds.latitude, s=2, c='black')
for i, t in enumerate(tracks):
    ax.plot(t[:, 0], t[:, 1], lw=0.2, c='red')
# %%
fig, ax = plt.subplots(1, num_vars, subplot_kw={'projection': 'polar'})
for vi in range(num_vars):
    ax[vi].set_theta_direction(-1)
    ax[vi].set_theta_zero_location('N')
    ax[vi].scatter(approach_azi, stm[vi, :], alpha=0.2)
# %%
pos = 4177
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.set_theta_direction(-1)
ax.set_theta_zero_location('N')
ax.scatter(np.deg2rad(ds_aux['dp'][:, 0, pos]), stm[0, :])
# %%
_, idx_pos_list = tree.query(
    [[-61.5, 15.2], [-63.5, 17.6], [-64, 15], [-59, 15], [-64.5, 17]])

# %%
fig, ax = plt.subplots(1, len(idx_pos_list), figsize=(
    8, 6*len(idx_pos_list)), subplot_kw={'projection': 'polar'})
for i, pos in enumerate(idx_pos_list):
    ax[i].set_theta_direction(-1)
    ax[i].set_theta_zero_location('N')
    ax[i].scatter(dir[0, :, 0, pos], stm[0, :], alpha=0.2, s=3)
# %%
fig, ax = plt.subplots(len(idx_pos_list), num_vars, subplot_kw={
                       'projection': 'polar'}, figsize=(8*num_vars, 6*5))
for i, pos in enumerate(idx_pos_list):
    for vi in range(num_vars):
        _mask = dir[vi, :, vi, pos] > 0
        ax[i, vi].set_theta_direction(-1)
        ax[i, vi].set_theta_zero_location('N')
        ax[i, vi].set_ylim(stm[vi].min(), stm[vi].max())
        ax[i, vi].scatter(dir[vi, _mask, vi, pos], stm[vi, _mask], alpha=0.2)

# %%
fig, ax = plt.subplots(len(idx_pos_list), num_vars,
                       figsize=(8*2, 6*len(idx_pos_list)))
for i, pos in enumerate(idx_pos_list):
    for vi in range(num_vars):
        im = ax[i, vi].scatter(stm[vi, :], exp[vi, :, pos],
                               c=dir[vi, :, vi, pos], cmap='twilight', vmax=np.pi, vmin=-np.pi)
        plt.colorbar(im, ax=ax[i, vi])
# %%
fig, ax = plt.subplots(len(idx_pos_list), num_vars+2,
                       figsize=(8*(num_vars+1), 6*len(idx_pos_list)))
for i, pos in enumerate(idx_pos_list):
    # _mask = dir[0,:,0,pos] > 0
    ax[i, 0].scatter(ds.longitude, ds.latitude, s=5, c='black')
    ax[i, 0].scatter(ds.longitude[pos], ds.latitude[pos], s=50, c='red')
    for vi in range(num_vars):
        _mask = (dir[vi, :, vi, pos] > 0) & (stm[vi, :] > thr_mar[vi])
        im = ax[i, vi+1].scatter(stm[vi, _mask], exp[vi, _mask, pos],
                                 c=dir[vi, _mask, vi, pos], cmap='twilight', vmax=np.pi, vmin=-np.pi)
        plt.colorbar(im, ax=ax[i, vi+1])
        _t, _p = kendalltau(stm[vi, _mask], exp[vi, _mask, pos])
        ax[i, vi+1].set_title(f"Tau:{_t:.3f} pval:{_p:.3f}")
        ax[i, vi+1].set_xlabel(f"{var_name[vi]}")
        ax[i, vi+1].set_ylabel(f"Exposure")
        print(f"Tau:{_t:.3f} pval:{_p:.3f}")
# %%
tval = np.zeros((num_vars, num_nodes))
pval = np.zeros((num_vars, num_nodes))
for vi in range(num_vars):
    for i in range(num_nodes):
        _mask = (dir[vi, :, vi, i] > 0) & (stm[vi, :] > thr_mar[vi])
        _t, _p = kendalltau(stm[vi, _mask], exp[vi, _mask, i])
        tval[vi, i] = _t
        pval[vi, i] = _p
# %%
fig, ax = plt.subplots(1, num_vars, sharey=True,
                       figsize=(8*num_vars, 6), facecolor="white")
for vi in range(num_vars):
    ax[vi].set_xlabel("Longitude")
    ax[vi].set_ylabel("Latitude")
    _c = ["red" if p < 0.05 else "black" for p in pval[vi]]
    im = ax[vi].scatter(lonlat[:, 0], lonlat[:, 1], s=5, c=_c)
    ax[vi].set_title(var_name[vi])
# %%
fig, ax = plt.subplots(1, num_vars, sharey=True,
                       figsize=(8*num_vars, 6), facecolor="white")
for vi in range(num_vars):
    ax[vi].set_xlabel("Longitude")
    ax[vi].set_ylabel("Latitude")
    im = ax[vi].scatter(lonlat[:, 0], lonlat[:, 1], s=5,
                        c=tval[vi, :], cmap='seismic', vmax=0.2, vmin=-0.2)
    plt.colorbar(im, ax=ax[vi])
    ax[vi].set_title(var_name[vi])
# %%
for pi in range(num_nodes):
    _mask = dir[0, :, 0, pos] > 0
    _t, _p = kendalltau(stm[0, _mask], exp[0, _mask, pi])

# %%
