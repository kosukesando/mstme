from __future__ import annotations

import time
from functools import partial
from pathlib import Path

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from cartopy.mpl.ticker import (
    LatitudeFormatter,
    LatitudeLocator,
    LongitudeFormatter,
    LongitudeLocator,
)
from numpy.random import random
from pathos.multiprocessing import ProcessPool
from scipy.stats._continuous_distns import genpareto
from shapely.geometry import LineString, MultiLineString, MultiPoint, Point
from statsmodels.distributions.empirical_distribution import ECDF

import mstmeclass as mc
from mstmeclass import G_F, GPPAR, MSTME, STM, Area, G

p = 0.3


def worker(x):
    # time.sleep(1)
    if x % 3 == 0:
        return None
    else:
        return x


if __name__ == "__main__":
    pool = ProcessPool()
    # for i in range(100):
    ##======================================================##
    num_done = 0
    results = []
    vals = []
    i = 0
    while num_done < 10:
        # make new mask
        if len(results) < pool.ncpus:
            print(i)
            results.append(pool.apipe(worker, i))
            i += 1
        # print(results)
        idx_got = []
        # for result in reversed(results):
        for j in range(len(results)):
            result = results[j]
            if num_done < 10:
                if result.ready():
                    val = result.get()
                    idx_got.append(j)
                    if val is not None:
                        vals.append(val)
                        num_done += 1
        for j in reversed(range(len(results))):
            if j in idx_got:
                results.pop(j)
    print(f"vals: {len(vals)}")
    for val in vals:
        print(f"\t{val}")
    # while num_done < 10:
    #     print(results)
    #     for result in results:
    #         if result.ready():
    #             val = result.get()
    #             vals.append(val)
    #     # make new mask
    #     try:
    #         results.append(pool.apipe(worker, num_done))
    #         num_done += 1
    #     except ValueError as e:
    #         print(e)

    # for val in vals:
    #     print(val)
