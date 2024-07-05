import concurrent.futures as cf
import time
from functools import partial
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pathos.multiprocessing import ProcessPool
from scipy.spatial.distance import directed_hausdorff
from scipy.stats import kendalltau
from statsmodels.distributions.empirical_distribution import ECDF

from mstme.mstmeclass import MSTME, SIMSET, STM


def calc_k(mat):
    assert mat.shape[0] == mat.shape[1]
    num_events = mat.shape[0]
    count = 0
    k = 0
    for i in range(num_events - 2):
        for j in range(i + 1, num_events - 1):
            count += 1
            if mat[i, j] < mat[i, j + 1]:
                k += 1
    return k / count


def calc_k2(mat):
    assert mat.shape[0] == mat.shape[1]
    num_events = mat.shape[0]
    k = 0
    mat_avg = np.average(mat, axis=0)

    for i in range(num_events - 1):
        if mat_avg[i] < mat_avg[i + 1]:
            k += 1
    return k / (num_events - 1)


def shuffle_mat(mat):
    assert mat.shape[0] == mat.shape[1]
    num_events = mat.shape[0]
    _idx_list = np.arange(num_events)
    np.random.shuffle(_idx_list)
    return mat[_idx_list, :][:, _idx_list]


def calc_dist_md(exp_array: np.ndarray):
    """
    exp_array: 3D-Array with shape(num_events, res, res)
    """
    num_events = exp_array.shape[0]
    d_mat = np.empty((num_events, num_events))
    for i in range(num_events):
        for j in range(i, num_events):
            _d = np.sum(np.abs(exp_array[i] - exp_array[j]))
            d_mat[i, j] = _d
            d_mat[j, i] = _d
    return d_mat


def calc_dist_hausdorff(exp_unique: list):
    """
    exp_unique: list of unique n-D vectors
    """
    num_events = exp_unique.shape[0]
    d_mat = np.empty((num_events, num_events))
    for i in range(num_events):
        for j in range(i, num_events):
            _d, _, _ = directed_hausdorff(exp_unique[i], exp_unique[j])
            d_mat[i, j] = _d
            d_mat[j, i] = _d
    return d_mat


def quantize(exp_series: list, res: int):
    _bins = np.linspace(0, 1, res, endpoint=False)
    # subtract 1 from digitize because 0.0~0.1 will return 1 and 0.9~1.0 will return res(and will cause indexerror)
    _exp_digitized = np.digitize(np.array(exp_series).T, _bins) - 1
    return _exp_digitized


def vecs2array(vecs, res: int):
    num_var = vecs.shape[1]
    shape = tuple([res] * num_var)
    array = np.zeros(shape)
    for vec in vecs:
        array[tuple(vec)] += 1
    return array


def quantize_exp(exp_series_ext, num_events, res, use_temporal):
    # Quantize
    exp_unique = []
    exp_array = np.zeros((num_events, res, res))

    for ei in range(num_events):
        h = exp_series_ext[ei]["hs"]
        u = exp_series_ext[ei]["UV_10m"]
        _exp_digitized = quantize([h, u], res)
        _exp_digitized_unique = np.unique(_exp_digitized, axis=0)
        exp_unique.append(_exp_digitized_unique)
        if use_temporal:
            exp_array[ei, :, :] = vecs2array(_exp_digitized, res)
        else:
            exp_array[ei, :, :] = vecs2array(_exp_digitized_unique, res)
    return exp_array, exp_unique


def get_mask(res, mask_type):
    match mask_type["type"]:
        case "none":
            mask = np.full((res, res), 1)
        case "circle":
            # circle mask
            mask = np.zeros((res, res))
            for i in range(res):
                for j in range(res):
                    mask[i, j] = np.sqrt(i**2 + j**2)
        case "square":
            thr = mask_type["threshold"]
            # square mask
            mask = np.zeros((res, res))
            for i in range(res):
                for j in range(res):
                    if 1 / res * i >= thr or 1 / res * j >= thr:
                        mask[i, j] = 1
    return mask


def calculate_stuff(
    stm_ext,
    exp_series_ext: list,
    num_events: int,
    res: int,
    dist_method: str,
    use_temporal: bool,
    mask_type: str,
    **kwargs,
):
    # Quantize
    exp_unique = []
    exp_array = np.zeros((num_events, res, res))

    for ei in range(num_events):
        h = exp_series_ext[ei]["hs"]
        u = exp_series_ext[ei]["UV_10m"]
        _exp_digitized = quantize([h, u], res)
        _exp_digitized_unique = np.unique(_exp_digitized, axis=0)
        exp_unique.append(_exp_digitized_unique)
        if use_temporal:
            exp_array[ei, :, :] = vecs2array(_exp_digitized, res)
        else:
            exp_array[ei, :, :] = vecs2array(_exp_digitized_unique, res)

        match mask_type:
            case "none":
                pass
            case "circle":
                # circle mask
                circ_mask = np.empty((res, res))
                for i in range(res):
                    for j in range(res):
                        circ_mask[i, j] = np.sqrt(i**2 + j**2)
                exp_array[ei, :, :] *= circ_mask
            case "square":
                thr = kwargs.get("threshold")
                # square mask
                mask = np.zeros((res, res))
                for i in range(res):
                    for j in range(res):
                        if 1 / res * i >= thr or 1 / res * j >= thr:
                            mask[i, j] = 1
                exp_array[ei, :, :] *= mask

    # distance matrix
    if dist_method == "md":
        d_mat = calc_dist_md(exp_array)
    elif dist_method == "hausdorff":
        d_mat = calc_dist_hausdorff(exp_unique)

    # null distribution of k
    count_beforehand = (num_events**2 - num_events) // 2 - (num_events - 1)
    k_null = []
    for ri in range(1000):
        d_mat_sorted_random = shuffle_mat(d_mat)
        k_null.append(calc_k(d_mat_sorted_random))

    # realization of k for this distribution of DM
    idx_list = stm_ext.argsort()
    d_mat_sorted = d_mat[idx_list, :][:, idx_list]
    k = calc_k(d_mat_sorted)

    return k, k_null, d_mat_sorted


def calculate_stuff2(
    stm_ext,
    exp_series_ext: list,
    num_events: int,
    res: int,
    dist_method: str,
    use_temporal: bool,
    mask_type: str,
    **kwargs,
):
    thr = kwargs.get("threshold", None)
    exp_array, exp_unique = quantize_exp(exp_series_ext, num_events, res, use_temporal)
    exp_array *= get_mask(res, {"type": mask_type, "threshold": thr})
    # distance matrix
    if dist_method == "md":
        d_mat = calc_dist_md(exp_array)
    elif dist_method == "hausdorff":
        d_mat = calc_dist_hausdorff(exp_unique)

    d_vec = np.mean(d_mat, axis=0)
    # Kendall's Tau
    tau, pval = kendalltau(stm_ext, d_vec)

    return tau, pval


def coerce_path(path: Path):
    if type(path) is not Path:
        path = Path(path)
        if not path.exists():
            if path.is_dir():
                path.mkdir()
            else:
                raise (ValueError(f"Input path string:{path} does not exist"))
    return path


def load_data_worker(path: Path | str, pos_list):
    path = coerce_path(path)
    # _ds = xr.open_dataset(path)
    _ds = xr.open_dataset(path).isel(node=pos_list).load()
    print("hoge")
    return _ds


def load_data(paths: list[Path], mask, pos_list) -> list[xr.Dataset]:
    # path = coerce_path(path)
    load_partial = partial(load_data_worker, pos_list=pos_list)
    num_events = np.count_nonzero(mask)
    ds_list = [[]] * num_events
    t0 = time.time()

    pool = ProcessPool()
    results = pool.imap(load_partial, paths)
    for ei, ds in zip(
        range(num_events),
        results,
    ):
        ds_list[ei] = ds
        print(ei)
    t1 = time.time()
    total = t1 - t0
    print(f"Finished loading datasets in {int(total//60):d}:{round(total%60):d}")
    return ds_list


# def load_data(path: Path | str, pos_list) -> list[xr.Dataset]:
#     path = coerce_path(path)
#     load_partial = partial(load_data_worker, pos_list=pos_list)
#     ds_list = [[]] * num_events
#     t0 = time.time()
#     with cf.ProcessPoolExecutor() as executor:
#         for ei, ds in zip(
#             range(num_events),
#             executor.map(load_partial, path.glob("*.nc")),
#         ):
#             ds_list[ei] = ds
#             print(ei)
#     t1 = time.time()
#     total = t1 - t0
#     print(f"Finished loading datasets in {int(total//60):d}:{round(total%60):d}")
#     return ds_list


class Grapher:
    def __init__(
        self,
        path_out: Path | str,
        mstme: MSTME,
        k_dict,
        region: str,
        dist_method: str,
        pos_list,
        num_vars: int,
        latlon,
    ) -> None:
        self.dist_method = dist_method
        self.region = region
        self.pos_list = pos_list
        self.num_vars = num_vars
        self.latlon = latlon
        self.path_out = coerce_path(path_out)
        self.mstme = mstme
        self.k_dict = k_dict
        if not self.path_out.exists():
            self.path_out.mkdir(parents=True, exist_ok=True)
        pass

    def plot(self, fig_name: str, **kwargs) -> None:
        match fig_name:
            case "1":
                for i, pi in enumerate(self.pos_list):
                    fig, ax = plt.subplots(
                        2,
                        self.num_vars,
                        figsize=(4 * self.num_vars, 4 * 2),
                        facecolor="white",
                    )
                    for S in STM:
                        vi = S.idx()
                        fig.suptitle(
                            rf"{self.dist_method} distance Location {pi}: ({self.mstme.latlon[pi,0]:.4f},{self.mstme.latlon[pi,1]:.4f})",
                            size=10,
                        )
                        ax[0, vi].set_title(f"STM:{S.name()}")
                        ax[0, vi].tick_params(
                            top=True, labeltop=True, bottom=False, labelbottom=False
                        )
                        ax[0, vi].imshow(self.k_dict["d_mat_sorted"][vi][i])
                        ax[0, vi].tick_params(
                            top=True, labeltop=True, bottom=False, labelbottom=False
                        )
                        ax[1, vi].hist(self.k_dict["k_null"][vi][i])
                        ax[1, vi].axvline(self.k_dict["k"][vi][i], c="red")
                        print(
                            rf"{self.k_dict['k'][vi][i]*100:.2f}% of SD increases as STM of H increases"
                        )
                    plt.savefig(
                        self.path_out / rf"{i:03d}_{pi}.pdf", bbox_inches="tight"
                    )
                    plt.savefig(
                        self.path_out / rf"{i:03d}_{pi}.png", bbox_inches="tight"
                    )
            case "2":
                fig, ax = plt.subplots(
                    1,
                    self.num_vars,
                    figsize=(8 * 2, 6),
                    subplot_kw={"projection": ccrs.PlateCarree()},
                )
                for S in STM:
                    vi = S.idx()
                    pval = []
                    for i, pi in enumerate(self.pos_list):
                        _mean = np.mean(self.k_dict["k_null"][vi][i])
                        _ecdf = ECDF(self.k_dict["k_null"][vi][i] - _mean)
                        _pval = 2 * (1 - _ecdf(np.abs(self.k_dict["k"][vi][i] - _mean)))
                        pval.append(_pval)
                    _c = ["red" if p < 0.05 else "black" for p in pval]
                    ax[vi].coastlines(lw=2)
                    ax[vi].scatter(
                        self.mstme.latlon[self.pos_list, 1],
                        self.mstme.latlon[self.pos_list, 0],
                        c=_c,
                    )
                    ax[vi].set_title(f"{S.name()}")
                    plt.savefig(self.path_out / rf"pval_map.png", bbox_inches="tight")
                    plt.savefig(self.path_out / rf"pval_map.pdf", bbox_inches="tight")


if __name__ == "__main__":
    import pickle
    from pathlib import Path

    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    import xarray as xr
    from scipy.spatial import KDTree
    from statsmodels.distributions.empirical_distribution import ECDF

    import mstme.mstmeclass as mc

    simset = SIMSET("caribbean", "none", -100)

    # get stm from previous result
    with open(
        Path(f"./output/{simset.region}/GP{80}%_CM{80}%/mstme_pickle.pickle"), "rb"
    ) as f:
        mstme = pickle.load(f)
    stm = mstme.stm
    num_events = mstme.num_events
    num_vars = mstme.num_vars
    exp_series = []
    time_max = 0
    for i in range(num_events):
        _ds = xr.open_dataset(f"./data/exp_series/{i:03d}.nc")
        exp_series.append(_ds)
        time_max = max(time_max, _ds.dims["time"])

    tree = KDTree(mstme.latlon)
    grid_res = 10
    lat_list = np.linspace(simset.min_lat, simset.max_lat, grid_res)
    lon_list = np.linspace(simset.min_lon, simset.max_lon, grid_res)
    dist_list, pos_list = tree.query(
        [[[lat, lon] for lat in lat_list] for lon in lon_list]
    )
    pos_list = pos_list.flatten()
    ds_exp_series = load_data(Path("./data/exp_series"), pos_list)

    res = 50
    di = 0
    dist_method = ["md", "hausdorff"]

    k_dict = {"k_null": [], "k": []}

    for vi in range(num_vars):
        k_arr = []
        k_null_arr = []
        d_mat_sorted_arr = []
        for i, ni in enumerate(pos_list):
            num_events_ext = np.count_nonzero(mstme.is_e[vi])
            exp_series_ext = [
                ds_exp_series[idx].isel(node=i) for idx in np.where(mstme.is_e[vi])[0]
            ]
            stm_ext = mstme.stm[vi, mstme.is_e[vi]]

            k, k_null, d_mat_sorted = calculate_stuff(
                stm_ext, exp_series_ext, num_events_ext, res, dist_method[di]
            )
            print(f"{k*100:.2f}% of SD increases as STM of H increases")
            k_arr.append(k)
            k_null_arr.append(k_null)
            d_mat_sorted_arr.append(d_mat_sorted)
        k_dict["k"].append(k_arr)
        k_dict["k_null"].append(k_null_arr)
        k_dict["d_mat_sorted"].append(d_mat_sorted_arr)

        path_out = Path(
            rf"./output/{simset.region}/GP{80}%_CM{80}%/dm/{simset.dist_method}/"
        )
    grapher = Grapher(
        Path(), simset.region, dist_method[di], pos_list, num_vars, mstme.latlon
    )
