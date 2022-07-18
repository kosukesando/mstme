# %%
# init
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
rng = np.random.default_rng()

# %%
# Load dataset
for f in Path('./').glob("test*.nc"):
    _ds = xr.open_dataset(f)
    _tm = _ds[['hs', 'UV_10m']].to_array()
    _tm_max = _tm[0].max(axis=0)
    # plt.tricontourf(_ds.longitude,_ds.latitude,_tm[0].max(axis=0) )
    plt.scatter(_ds.longitude, _ds.latitude, c=[
                'red' if _tm_max[ni] == np.nan else 'black' for ni in range(86434)])
    plt.show()

# %%
mask = np.isnan(_tm).any(axis=1)

# %%
plt.scatter(_ds.longitude[mask[0]], _ds.latitude[mask[0]])
# %%
