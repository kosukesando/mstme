import xarray as xr
from pathlib import Path
import concurrent.futures as cf


def rewrite(path, i):
    print(path, i)
    ds = xr.open_dataset(path)
    ds_new = ds.drop_dims(("single", "nele", "nvertex")).drop_vars(('Dir_10m','dp','PRmsl'))
    ds_new.to_netcdf(f"./ww3_meteo_slim/{i:03d}.nc")
    return i


def main():
    paths = []
    indices = []
    for i, path in enumerate(Path(f"./ww3_meteo/").glob("*.nc")):
        if not Path(f"./ww3_meteo_slim/{i:03d}.nc").exists():
            paths.append(path)
            indices.append(i)
    with cf.ProcessPoolExecutor() as executor:
        for i in executor.map(rewrite, paths, indices):
            continue


if __name__ == "__main__":
    main()
