import xarray as xr
from pathlib import Path
import concurrent.futures as cf
import pickle

def rewrite(path,stm, i):
    ds = xr.open_dataset(path)
    _exp = []
    for vi,key in enumerate(['hs','UV_10m']):
        _exp.append(ds[[key]]/stm[vi,i].data)
    _ds = xr.combine_by_coords(_exp)
    _ds.to_netcdf(f'./exp_series/{i:03d}.nc')
    


def main():
    region = 'caribbean'
    dir_data = "./ww3_meteo_slim"
    with open(Path(f'./output/{region}/GP{80}%_CM{80}%/mstme_pickle.txt'),'rb') as f:
        mstme = pickle.load(f)
    stm = mstme.stm
    num_events = mstme.num_events

    with cf.ProcessPoolExecutor() as executor:
        for i in executor.map(rewrite, list(Path(dir_data).glob("*.nc")),stm, list(range(num_events))):
            continue


if __name__ == "__main__":
    main()
