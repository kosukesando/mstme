# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("plot_style.txt")
rng = np.random.default_rng()
pos_color = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# %%
# Load dataset
ds_full = xr.open_mfdataset(
    "./ww3_meteo/*.nc", combine="nested", concat_dim="event", parallel=True
).chunk({'event':1})
# %%
for ei in range(685):
    with ds_full.isel({"event":ei}).drop_dims(["single","nele"]).dropna("time",how='all').compute() as ds_temp:
        stm_idx_h= ds_temp['hs'].argmax(dim=['time','node'])['time'].values
        stm_idx_u= ds_temp['UV_10m'].argmax(dim=['time','node'])['time'].values
        da_dh = ds_temp.isel({"time":stm_idx_h})
        da_du = ds_temp.isel({"time":stm_idx_u})
        ds_save = xr.concat([da_dh,da_du],dim="var")
        ds_save.to_netcdf(f'ww3_meteo_aux/{ei:03d}.nc')
# %%
