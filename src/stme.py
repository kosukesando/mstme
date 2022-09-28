import numpy as np
from scipy.stats._continuous_distns import genpareto
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
from scipy.stats import laplace
from scipy.stats import genextreme
from scipy.stats import kendalltau
from scipy.optimize import minimize
import src.threshold_search as threshold_search
import xarray as xr
from tqdm import trange

rng = np.random.default_rng()
pos_color = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# ndist = genextreme(0)
ndist = laplace

var_name = ["$H_s$", "$U$"]
var_name_g = ["$\hat H_s$", "$\hat U$"]
par_name = ["$\\xi$", "$\\mu$", "$\\sigma$"]
unit = ["[m]", "[m/s]"]


def savefig(*args, **kwargs):
    plt.savefig(*args, **kwargs)
    plt.close(plt.gcf())


def get_region_filter(ds: xr.Dataset, region_filter: str = 'none', dir_out: str = None, draw_fig: bool = True) -> np.ndarray:
    global var_name, unit
    lonlat = np.array([ds.longitude, ds.latitude]).T
    stm_all = ds[['hs', 'UV_10m']].max(dim="node").to_array()
    exp_all = ds[['hs', 'UV_10m']].to_array() / stm_all
    mask = []
    num_events = ds.event.size
    num_vars = 2
    # Filter by STM location
    stm_idx = exp_all.argmax(axis=2)
    is_east = lonlat[stm_idx, 0] > -61.5
    is_west = np.logical_not(is_east)
    is_north = lonlat[stm_idx, 1] > 16.2
    is_south = np.logical_not(is_north)
    # Filter by Kendall's Tau
    # is_e = stm_all
    # kendall_tau_mv(stm_all, exp_all,)
    match region_filter:
        case 'h-east':
            mask = is_east[0]
        case 'h-west':
            mask = is_west[0]
        case 'u-east':
            mask = is_east[1]
        case 'u-west':
            mask = is_west[1]
        case 'h-north':
            mask = is_north[0]
        case 'h-south':
            mask = is_south[0]
        case 'u-north':
            mask = is_north[1]
        case 'u-south':
            mask = is_south[1]
        case 'none':
            mask = np.full((num_events,), True)
        # case 'h-tau-pos':
        #     mask = ()
        # case 'h-tau-neg':
        #     mask = ()
        # case 'u-tau-pos':
        #     mask = ()
        # case 'u-tau-neg':
        #     mask = ()
        case _:
            raise(ValueError('idk'))
    #############################################################################
    fig, ax = plt.subplots(2, num_vars, figsize=(8*num_vars, 6*2))
    stm_min = np.floor(stm_all.min(axis=1)/5)*5
    stm_max = np.ceil(stm_all.max(axis=1)/5)*5
    for vi in range(num_vars):
        for i, b in enumerate([True, False]):
            ax[i, vi].set_xlabel(f"{var_name[vi]}{unit[vi]}")
            ax[i, vi].hist(stm_all[vi, (mask == b)], bins=np.arange(
                stm_min[vi], stm_max[vi], 1))
            ax[i, vi].set_title(f'{"is" if b else "not"} {region_filter}')
    if dir_out != None:
        savefig(
            f"{dir_out}/Histogram_STM.pdf", bbox_inches="tight"
        )
    if not draw_fig:
        plt.close()
    #############################################################################
    return mask


def _f_hat_cdf(pd_nrm, pd_ext, X):
    X = np.asarray(X)
    scalar_input = False
    if X.ndim == 0:
        X = X[None]  # Makes x 1D
        scalar_input = True
    val = np.zeros(X.shape)
    mu = pd_ext.args[1]  # args -> ((shape, loc, scale),)
    for i, x in enumerate(X):
        if x > mu:
            val[i] = 1 - (1 - pd_nrm(mu)) * (1 - pd_ext.cdf(x))
        else:
            val[i] = pd_nrm(x)
    if scalar_input:
        return np.squeeze(val)
    return val


def _f_hat_ppf(pd_nrm, pd_ext, _stm, X_uni):
    _X_uni = np.asarray(X_uni)
    _scalar_input = False
    if _X_uni.ndim == 0:
        _X_uni = _X_uni[None]  # Makes x 1D
        _scalar_input = True
    _val = np.zeros(_X_uni.shape)
    _mu = pd_ext.args[1]  # args -> ((shape, loc, scale),)
    for i, x in enumerate(_X_uni):
        if x > pd_nrm(_mu):
            _val[i] = pd_ext.ppf(1 - (1-x)/(1 - pd_nrm(_mu)))
        else:
            _val[i] = np.quantile(_stm, x)
    if _scalar_input:
        return np.squeeze(_val)
    return _val


def ts_sample(N, pool_size, ts):
    from random import randint

    # check if event num in pool == event num in ts
    assert pool_size == sum(ts)
    # pool(477,2)
    samples = np.full((N, pool_size), False)
    for i in range(N):
        for j in range(len(ts)):
            k = sum(ts[0:j]) + randint(0, ts[j] - 1)
            samples[i, k] = True
    return samples


def cost(p: list, x: np.ndarray, y: np.ndarray) -> float:
    """
    cost(p,data,vi)->float
    p: parameter; [a,b,mu,sigma]
    data: ndarray with shape(num_vars, num_events)
    vi: Index of extreme variable
    minimize this.
    """
    q = 0
    a = p[0]
    b = p[1]
    mu = p[2]
    sg = p[3]
    if y.ndim < 2:
        y = np.expand_dims(y, axis=0)
    for vj in range(y.shape[0]):
        _qj = np.sum(
            np.log(sg * x ** b)
            + 0.5
            * ((y[vj] - (a * x + mu * x ** b)) / (sg * x ** b))
            ** 2
        )
        if np.isnan(_qj):
            print(a, b, mu, sg, x, y[vj])
            raise(ValueError('Qj is NaN'))
        q += _qj
    return q


def jacobian_custom(p, x, y) -> np.ndarray:
    a = p[0]
    b = p[1]
    mu = p[2]
    sg = p[3]
    da = np.sum(-(x**(1 - 2 * b)*(-a * x - mu * x ** b + y))/sg**2)
    db = np.sum((x**(-2 * b) * np.log(x) * (-a**2 * x ** 2 + a * x * (2 *
                y - mu * x ** b) + sg**3 * x**(3 * b) + mu * y * x ** b - y ** 2))/sg**2)
    dm = np.sum(-(x ** (-b) * (-a * x - mu * x**b + y))/sg**2)
    ds = np.sum(x**b - (x**(-2 * b) * (a * x + mu * x ** b - y)**2)/sg ** 3)
    return np.array([da, db, dm, ds])


def genpar_estimation(stm, thr_mar, var_name, unit, par_name, dir_out=None, draw_fig=True):
    """
    Bootstrap generalized pareto estimation for multivariates
    stm: ndarray(num_vars,num_events)
    thr_mar: array-like of n thresholds (n,)
    """
    global rng
    num_vars = stm.shape[0]
    num_events = stm.shape[1]
    is_e_marginal = stm > thr_mar[:, np.newaxis]
    if (np.count_nonzero(is_e_marginal, axis=1) == 0).any():
        raise(ValueError("No events above marginal threshold"))
    if len(thr_mar) != num_vars:
        raise ValueError(
            'Number of thresholds do not match number of variables')
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
        xp, mp, sp = np.median(genpar_params[vi, :, :], axis=0)
        print(f"GENPAR{xp, mp, sp}")
        gp[vi] = genpareto(xp, mp, sp)
    #########################################################
    fig, ax = plt.subplots(len(par_name), num_vars,
                           figsize=(8*num_vars, 6*len(par_name)))

    for vi in range(num_vars):
        ax[0, vi].set_title(var_name[vi])
        for pi, p in enumerate(par_name):
            ax[pi, 0].set_ylabel(par_name[pi])
            ax[pi, vi].hist(genpar_params[vi, :, pi])
    if dir_out != None:
        savefig(f"{dir_out}/Genpar_Params.pdf", bbox_inches="tight")
    if not draw_fig:
        plt.close()
    #########################################################

    #########################################################
    fig, ax = plt.subplots(1, num_vars, figsize=(8*num_vars, 6))
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
    if dir_out != None:
        savefig(f"{dir_out}/Genpar_CDF.pdf", bbox_inches="tight")
    if not draw_fig:
        plt.close()
    #########################################################
    return gp


def ndist_transform(stm, gp, var_name, unit, dir_out=None, draw_fig=True):
    global ndist
    num_vars = stm.shape[0]
    stm_g = np.zeros(stm.shape)
    f_hat_cdf = [None, None]
    _uniform = np.zeros(stm_g.shape)
    for vi in range(num_vars):
        f_hat_cdf[vi] = lambda x, idx=vi: _f_hat_cdf(
            ECDF(stm[idx]), gp[idx], x)
        _stm = stm[vi]
        _uniform[vi] = f_hat_cdf[vi](_stm)
        print(f'Uniform max {var_name[vi]}: {_uniform[vi].max()}')
        # stm_g[vi] = ndist.ppf(np.clip(_uniform[vi], None, 1-10**(-6)))
        stm_g[vi] = ndist.ppf(_uniform[vi])
    # fig, ax = plt.subplots()
    # ax.scatter(_uniform[0], _uniform[1])

    print("H_hat min, max:", stm_g[0].min(), stm_g[0].max())
    print("U_hat min, max:", stm_g[1].min(), stm_g[1].max())

    #########################################################
    fig, ax = plt.subplots(1, 2, figsize=(7, 3))

    ax[0].scatter(stm[0], stm[1], s=5)
    ax[0].set_xlabel(f"{var_name[0]}{unit[0]}")
    ax[0].set_ylabel(f"{var_name[1]}{unit[1]}")
    ax[0].set_xlim(0, 20)
    ax[0].set_ylim(0, 60)

    ax[1].set_aspect(1)
    ax[1].scatter(stm_g[0], stm_g[1], s=5)
    ax[1].set_xlabel(r"$\hat H_s$")
    ax[1].set_ylabel(r"$\hat U$")
    ax[1].set_xlim(-5, 15)
    ax[1].set_ylim(-5, 15)
    ax[1].set_xticks([-2+2*i for i in range(6)])
    ax[1].set_yticks([-2+2*i for i in range(6)])

    if dir_out != None:
        savefig(f"{dir_out}/Original_vs_Laplace.pdf",
                bbox_inches="tight")
    if not draw_fig:
        plt.close()
    #########################################################
    return stm_g, f_hat_cdf


def kendall_tau_mv(stm_g, exp, is_e, var_name, lonlat, dir_out=None, draw_fig=True):
    """
    stm_g: (num_vars, num_events)
    exp: (num_vars, num_events, num_nodes)
    is_e: (num_vars,)
    """
    global rng
    num_vars = stm_g.shape[0]
    num_nodes = exp.shape[2]

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
    if dir_out != None:
        savefig(f"{dir_out}/Kendall_Tau_all_var_pval.pdf",
                bbox_inches='tight')
    if not draw_fig:
        plt.close()
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
            im = ax[vi, vj].scatter(lonlat[:, 0], lonlat[:, 1], s=5, c=tval[vi, vj, :], cmap='seismic', vmax=np.abs(
                tval[vi]).max(), vmin=-np.abs(tval[vi]).max())
            plt.colorbar(im, ax=ax[vi, vj])
            ax[vi, vj].set_title(f"STM:{var_name[vi]} E:{var_name[vj]}")
    if dir_out != None:
        savefig(f"{dir_out}/Kendall_Tau_all_var_tval.pdf",
                bbox_inches="tight")
    if not draw_fig:
        plt.close()
    #########################################################
    return pval, tval


def estimate_conmul(stm_g, thr_gum, var_name, dir_out=None, SEARCH=False, draw_fig=True):
    global ndist
    num_vars = stm_g.shape[0]
    num_events = stm_g.shape[1]
    is_e = stm_g > thr_gum
    # Laplace replacement
    N_rep = 100
    stm_g_rep = np.zeros((N_rep, num_vars, num_events))
    for i in range(N_rep):
        _idx = rng.choice(num_events, size=num_events)
        _stm = stm_g[:, _idx]
        for vi in range(num_vars):
            _laplace_sample = ndist.rvs(size=num_events)
            _laplace_sample_sorted = np.sort(_laplace_sample)
            _arg = np.argsort(_stm[vi])
            stm_g_rep[i, vi, _arg] = _laplace_sample_sorted
    # Estimate conditional model parameters
    lb = [0, None, -5, 0.1]
    ub = [1, 1, 5, None]
    params_uc = np.zeros((num_vars, N_rep, 4))
    costs = np.zeros((num_vars, N_rep))
    for vi in range(num_vars):
        for i in range(N_rep):
            _stm = stm_g_rep[i]
            a0 = np.random.uniform(low=lb[0], high=ub[0])
            b0 = np.random.uniform(low=-1, high=ub[1])
            m0 = np.random.uniform(low=-1, high=1)
            # s0 = np.random.uniform(low=0.01, high=0.99)
            s0 = 1
            _p0 = np.array([a0, b0, m0, s0])
            if np.isnan(_p0).any():
                raise(ValueError("WTF"))
            evt_mask = np.logical_and(
                (_stm[vi, :] > thr_gum), (~np.isinf(_stm[vi, :])))
            x = _stm[vi, evt_mask]  # conditioning
            # conditioned
            y = np.delete(_stm[:, evt_mask], vi, axis=0)
            optres = minimize(
                cost,
                _p0,
                args=(x, y),
                jac=jacobian_custom,
                # hess=hessian_custom,
                method='L-BFGS-B',
                bounds=((lb[0], ub[0]), (lb[1], ub[1]),
                        (lb[2], ub[2]), (lb[3], ub[3])),
            )
            # if optres.success ==:
            # def func(x):
            #     return cost(
            #         x, stm_g_rep[:, i, evt_mask], vi
            #     )
            # optres = minimize(
            #     func,
            #     np.array([a0, b0, m0, s0]),
            #     # method='trust-constr',
            #     bounds=((lb[0], ub[0]), (lb[1], ub[1]),
            #             (lb[2], ub[2]), (lb[3], ub[3])),
            # )
            _param = optres.x
            _cost = optres.fun
            if np.isnan(_cost):
                print(_param)
                print(cost(_param, x, y))
                raise(ValueError("Cost is NaN"))
            params_uc[vi, i, :] = _param
            costs[vi, i] = _cost
    print(f'costs:{costs}')
    params_median = np.median(params_uc, axis=1)
    print("Params_median:", params_median)
    # Threshold search
    if SEARCH:
        threshold_search.search_conditional(stm_g_rep, 1.0, 3.0)
        # Calculating residuals
    residual = []
    print("Residuals")
    for vi in range(num_vars):
        _x = stm_g[vi, is_e[vi]]  # conditioning(extreme)
        _y = np.delete(stm_g[:, is_e[vi]], vi, axis=0)  # conditioned
        _a = params_median[vi, 0]
        _b = params_median[vi, 1]
        _z = (_y - _a * _x) / (_x ** _b)
        residual.append(_z)
        # print(_z.flatten())
        for i, __z in enumerate(_z.squeeze()):
            if __z > 5:
                print(f"{var_name[vi],}a,b,x,y", _a, _b, _x[i], _y[0, i])
        print(f'{var_name[vi]} min, max: {_z.min()},{_z.max()}')

    ##################################################################################################################
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), facecolor="white")
    ax.scatter(stm_g_rep[:, 0, :], stm_g_rep[:, 1, :], alpha=0.1)
    ax.scatter(stm_g[0], stm_g[1], color="blue")
    ax.set_xlabel(r"$\hat H_s$")
    ax.set_ylabel(r"$\hat U$")
    ax.set_xlim(-3, 15)
    ax.set_ylim(-3, 15)
    if dir_out != None:
        savefig(f"{dir_out}/Replacement.pdf", bbox_inches="tight")
    if not draw_fig:
        plt.close()
    #########################################################
    fig, ax = plt.subplots(4, num_vars, figsize=(8*num_vars, 6*4))
    fig.tight_layout()
    ax[0, 0].set_ylabel("a")
    ax[1, 0].set_ylabel("b")
    ax[2, 0].set_ylabel("$\mu$")
    ax[3, 0].set_ylabel("$\sigma$")
    ax[3, 0].set_xlabel(var_name[0])
    ax[3, 1].set_xlabel(var_name[1])
    for vi in range(num_vars):
        ax[0, vi].hist(params_uc[vi, :, 0])
        ax[1, vi].hist(params_uc[vi, :, 1])
        ax[2, vi].hist(params_uc[vi, :, 2])
        ax[3, vi].hist(params_uc[vi, :, 3])
    if dir_out != None:
        savefig(f"{dir_out}/Conmul_Estimates.pdf", bbox_inches="tight")
    if not draw_fig:
        plt.close()
    #########################################################
    fig, ax = plt.subplots(1, num_vars)
    for vi in range(num_vars):
        ax[vi].hist(costs[vi])
        ax[vi].set_title('cost')
    if not draw_fig:
        plt.close()
    #########################################################
    fig, ax = plt.subplots(1, num_vars, figsize=(
        8*num_vars, 6), facecolor="white")
    fig.supxlabel("$a$")
    fig.supylabel("$b$")
    params_ml = np.zeros((4, num_vars))
    for vi in range(num_vars):
        ax[vi].scatter(
            params_uc[vi, :, 0],
            params_uc[vi, :, 1],
            s=5,
            label="Generated samples",
        )
        ax[vi].set_title(var_name[vi])
    if dir_out != None:
        savefig(f"{dir_out}/ab_Estimates.pdf", bbox_inches="tight")
    if not draw_fig:
        plt.close()
    #########################################################
    fig, ax = plt.subplots(1, num_vars, figsize=(
        8*num_vars, 6), facecolor="white")
    fig.supxlabel("$a$")
    fig.supylabel("$mu$")
    params_ml = np.zeros((4, num_vars))
    for vi in range(num_vars):
        ax[vi].scatter(
            params_uc[vi, :, 0],
            params_uc[vi, :, 2],
            s=5,
            label="Generated samples",
        )
        ax[vi].set_title(var_name[vi])
    if not draw_fig:
        plt.close()
    #########################################################
    fig, ax = plt.subplots(1, num_vars, figsize=(
        8*num_vars, 6), facecolor="white")
    fig.supxlabel("$a$+$mu$")
    fig.supylabel("$b$")
    params_ml = np.zeros((4, num_vars))
    for vi in range(num_vars):
        ax[vi].scatter(
            params_uc[vi, :, 0]+params_uc[vi, :, 2],
            params_uc[vi, :, 1],
            s=5,
            label="Generated samples",
        )
        ax[vi].set_title(var_name[vi])
    if not draw_fig:
        plt.close()
    #########################################################
    fig, ax = plt.subplots(1, num_vars, figsize=(
        8*num_vars, 6), facecolor="white")
    # fig.tight_layout()
    for vi in range(num_vars):
        ax[vi].scatter(ndist.cdf(stm_g[vi, is_e[vi]]), residual[vi], s=5)
        ax[vi].set_xlabel(f"F({var_name[vi]})")
    ax[0].set_ylabel("$Z_{-j}$")
    if dir_out != None:
        savefig(f"{dir_out}/Residuals.pdf", bbox_inches="tight")
    if not draw_fig:
        plt.close()
    #########################################################
    return params_median, residual


def sample_stm(stm, stm_g, gp, f_hat_cdf, thr_gum, params_median, residual, occur_prob, var_name, unit, size=1000, dir_out=None, draw_fig=True):
    # Sample from model
    num_vars = stm.shape[0]
    num_events = stm.shape[1]
    N_sample = size

    is_e = stm_g > thr_gum
    vi_largest = stm_g.argmax(axis=0)
    is_me = np.empty((num_vars, num_events))
    for vi in range(num_vars):
        is_me[vi] = np.logical_and(vi_largest == vi, is_e[vi])
    is_e_any = is_e.any(axis=0)
    v_me_ratio = np.count_nonzero(is_me, axis=1) / np.count_nonzero(is_e_any)
    exceedance_prob = np.count_nonzero(is_e_any) / num_events

    thr_uni = ndist.cdf(thr_gum)
    std_gum = ndist.ppf(rng.uniform(thr_uni, 1, size=N_sample))
    vi_list = rng.choice(num_vars, size=N_sample, p=v_me_ratio)

    sample_full_g = np.zeros((num_vars, N_sample))
    for i, vi in enumerate(vi_list):
        _a = np.asarray(params_median[vi, 0])
        _b = np.asarray(params_median[vi, 1])
        while True:
            _z = rng.choice(residual[vi], axis=1)
            _y_given_x = std_gum[i] * _a + (std_gum[i] ** _b) * _z
            if (_y_given_x < std_gum[i]).all():
                _samples = np.insert(np.asarray(_y_given_x), vi, std_gum[i])
                sample_full_g[:, i] = _samples
                break

    # Transform back to original scale
    ppf = [None, None]
    sample_full = np.zeros(sample_full_g.shape)
    sample_uni = ndist.cdf(sample_full_g)
    for vi in range(num_vars):
        ppf[vi] = lambda x, i=vi: _f_hat_ppf(ECDF(stm[i]), gp[i], stm[i], x)
        sample_full[vi] = ppf[vi](sample_uni[vi])
    # _mask = sample_full[0, :] < 6
    # if np.count_nonzero(_mask) > 0:
    #     _stm_low, _ = ndist_transform(
    #         sample_full[:, _mask], gp, var_name, unit, None)
    #     fig, ax = plt.subplots()
    #     ax.scatter(stm_g[0], stm_g[1], c='black')
    #     ax.scatter(_stm_low[0], _stm_low[1], c='red')
    #     ax.set_xlim(-5, 15)
    #     ax.set_ylim(-5, 15)
    #     ax.set_aspect('equal')
    #     print(f"")
    #     raise(ValueError("Very low STM"))
    # _x = np.linspace(0,1,100)
    # fig,ax=plt.subplots(1,2)
    # for vi in range(num_vars):
    #     ax[vi].plot(_x,ppf[vi](_x))
    ##################################################################################################################
    fig, ax = plt.subplots(1, num_vars, figsize=(
        8*num_vars, 6), facecolor="white")

    ax[0].set_aspect(1)

    a_h, b_h, mu_h, sg_h = params_median[0, :]
    a_u, b_u, mu_u, sg_u = params_median[1, :]
    sample_given_h = []
    sample_given_u = []
    sample_given_hg = []
    sample_given_ug = []
    for i, vi in enumerate(vi_list):
        if vi == 0:
            sample_given_h.append(sample_full[:, i])
            sample_given_hg.append(sample_full_g[:, i])
        if vi == 1:
            sample_given_u.append(sample_full[:, i])
            sample_given_ug.append(sample_full_g[:, i])
    sample_given_h = np.array(sample_given_h).T
    sample_given_u = np.array(sample_given_u).T
    sample_given_hg = np.array(sample_given_hg).T
    sample_given_ug = np.array(sample_given_ug).T

    x_h = np.linspace(thr_gum, 10, 100)
    y_h = x_h * a_h + (x_h ** b_h) * mu_h
    ax[0].plot(x_h, y_h, color="orange", label="U|H")

    y_u = np.linspace(thr_gum, 10, 100)
    x_u = y_u * a_u + (y_u ** b_u) * mu_u
    ax[0].plot(x_u, y_u, color="teal", label="H|U")

    ax[0].scatter(stm_g[0], stm_g[1], s=5, color="black", label="original")
    ax[0].axvline(thr_gum, color="black")
    ax[0].axhline(thr_gum, color="black")

    ax[0].set_xlabel(r"$\hat H_s$")
    ax[0].set_ylabel(r"$\hat U$")
    ax[0].set_xlim(-2, 10)
    ax[0].set_ylim(-2, 10)
    ax[0].scatter(
        sample_given_hg[0], sample_given_hg[1], s=1, color="orange", label="U|H"
    )
    ax[0].scatter(
        sample_given_ug[0], sample_given_ug[1], s=1, color="teal", label="H|U"
    )
    print(sample_given_hg.max(), sample_given_ug.max())
    print(sample_given_hg.min(), sample_given_ug.min())

    ax[1].scatter(stm[0], stm[1], color="black", s=5)
    ax[1].scatter(sample_given_h[0], sample_given_h[1],
                  color="orange", s=1)
    ax[1].scatter(sample_given_u[0], sample_given_u[1], color="teal", s=1)
    ax[1].set_xlabel(f"{var_name[0]}{unit[0]}")
    ax[1].set_ylabel(f"{var_name[1]}{unit[1]}")

    res = 100
    _x = np.linspace(0, stm[0].max(), res)
    _y = np.linspace(0, stm[1].max(), res)
    _x_mg, _y_mg = np.meshgrid(_x, _y)
    _z_mg_sample = np.zeros((res, res))
    _z_mg = np.zeros((res, res))

    for xi in range(res):
        for yi in range(res):
            _count_sample = np.count_nonzero(
                np.logical_and(sample_full[0] >
                               _x[xi], sample_full[1] > _y[yi])
            )
            _count = np.count_nonzero(
                np.logical_and(stm[0] > _x[xi], stm[1] > _y[yi])
            )
            _z_mg_sample[xi, yi] = _count_sample
            _z_mg[xi, yi] = _count
    return_periods = [100]

    _levels_original = [num_events / (rp * occur_prob)
                        for rp in return_periods]
    _levels_sample = [
        N_sample / (rp * occur_prob * exceedance_prob) for rp in return_periods
    ]
    _linestyles = ["-", "--"]
    ax[1].contour(
        _x_mg,
        _y_mg,
        _z_mg.T,
        levels=_levels_original,
        linestyles=_linestyles,
        colors="black",
    )
    ax[1].contour(
        _x_mg,
        _y_mg,
        _z_mg_sample.T,
        levels=_levels_sample,
        linestyles=_linestyles,
        colors="red",
    )

    if dir_out != None:
        savefig(
            f"{dir_out}/Simulated_Conmul_vs_Back_Transformed.pdf", bbox_inches="tight"
        )
    if not draw_fig:
        plt.close()
    #########################################################
    return sample_full, ppf


def search_isocontour(scatter, n):
    coords = []
    scatter = np.unique(scatter, axis=1)
    _num_events = scatter.shape[1]
    _xg, _yg = np.sort(scatter[0]), np.sort(scatter[1])
    _idx_x, _idx_y = np.argsort(scatter[0]), np.argsort(scatter[1])
    _xi, _yi = _num_events-1, 0
    # Search isocontour
    while True:
        # Keep searching until the edge
        if _xi-1 < 0 or _yi+1 >= _num_events:
            break

        _x = _xg[_xi]
        _y = _yg[_yi]

        # Check if you can go up, in which case you do
        _count_up = np.count_nonzero(
            np.logical_and(scatter[0] >= _x, scatter[1] >= _yg[_yi+1])
        )
        if _count_up == n:
            _yi += 1
        # If you can't, look directly to the left
        else:
            _dxi = 1
            _success = False
            while not _success:
                _dyi = 0
                # Keep moving up from there until you reach a viable point
                while True:
                    _count_left = np.count_nonzero(
                        np.logical_and(scatter[0] >= _xg[_xi-_dxi],
                                       scatter[1] >= _yg[_yi+_dyi])
                    )
                    if _count_left == n:
                        _xi -= _dxi
                        _yi += _dyi
                        _success = True
                        break
                    elif _count_left > n:
                        _dyi += 1
                    else:
                        # No point found along the vertical transect
                        # So move to the left and search again
                        _success = False
                        _dxi += 1
                        break
        coords.append([_xg[_xi], _yg[_yi]])

    return np.array(coords).T


def plot_isocontour(tm_original_bs, tm_sample_bs, idx_pos_list, occur_freq, exceedance_prob, return_period=100, dir_out=None, draw_fig=True):
    # tm(N_bs,numvar,numevent,numnode)
    global pos_color
    N_bootstrap = len(tm_sample_bs)
    num_events = tm_original_bs.shape[2]
    stm_min = np.floor(
        tm_sample_bs[:, :, :, idx_pos_list].min(axis=(0, 2, 3))/5)*5
    stm_max = np.ceil(
        tm_sample_bs[:, :, :, idx_pos_list].max(axis=(0, 2, 3))/5)*5
    print(np.array(tm_sample_bs).min(axis=(0, 2, 3)),
          np.array(tm_sample_bs).max(axis=(0, 2, 3)))
    #########################################################
    fig, axes = plt.subplots(2, 2, figsize=(8*2, 6*2), facecolor="white",)
    # fig.suptitle(
    #     f"{return_period}-yr return period (Saint-Denis) Laplace threshold = {thr_gum}",
    #     y=0.90,
    # )
    fig.supxlabel(r"$H_s$[m]")
    fig.supylabel(r"$U$[m/s]")
    alpha = 1/min(N_bootstrap, 10)
    for i, ax in enumerate(axes.flatten()):
        ax.set_xlim(stm_min[0], stm_max[0])
        ax.set_ylim(stm_min[1], stm_max[1])
        _linestyles = ["-", "--"]
        _idx_pos = idx_pos_list[i]
        # sample
        _ic_sample = []
        for bi in range(N_bootstrap):
            _num_events_extreme = tm_sample_bs[bi].shape[1]
            _count_sample = round(_num_events_extreme /
                                  (return_period * occur_freq * exceedance_prob))
            _ic_sample.append(search_isocontour(
                tm_sample_bs[bi][:, :, _idx_pos], _count_sample))

        # original
        _ic_original = []
        _count_original = round(num_events / (return_period * occur_freq))
        for bi in range(N_bootstrap):
            _ic_original.append(search_isocontour(
                tm_original_bs[bi][:, :, _idx_pos], _count_original))

        for bi in range(N_bootstrap):
            ax.scatter(
                tm_sample_bs[bi][0, :, _idx_pos],
                tm_sample_bs[bi][1, :, _idx_pos],
                s=2,
                c=pos_color[i],
                label=f"Simulated",
                alpha=1.0/N_bootstrap,
            )
            ax.scatter(
                tm_original_bs[bi][0, :, _idx_pos],
                tm_original_bs[bi][1, :, _idx_pos],
                s=10,
                c="black",
                label=f"Original",
                alpha=1.0/N_bootstrap,
            )
            ax.plot(_ic_original[bi][0],
                    _ic_original[bi][1],
                    c='black',
                    alpha=alpha,
                    lw=2)
            ax.plot(_ic_sample[bi][0],
                    _ic_sample[bi][1],
                    c=pos_color[i],
                    alpha=alpha,
                    lw=2)
        ax.set_title(f"Coord.{i+1}")
    if dir_out != None:
        savefig(
            f"{dir_out}/RV_(Saint-Denis).pdf", bbox_inches="tight"
        )
    if not draw_fig:
        plt.close()


def kendallpval_cost(k, stm_norm, exp):
    _t, _p = kendalltau(stm_norm, exp**stm_norm**k)
    return -_p


def calculate_kval(exp: np.ndarray, stm_g: np.ndarray, thr_com: float):
    num_vars = exp.shape[0]
    num_nodes = exp.shape[2]
    exp_ext = np.zeros(exp.shape)
    kval = np.zeros((num_vars, num_nodes))
    pval = np.zeros((num_vars, num_nodes))
    tval = np.zeros((num_vars, num_nodes))
    # fig, ax = plt.subplots(1, num_vars, figsize=(16, 8))
    for ni in trange(num_nodes):
        for vi in range(num_vars):
            _mask = stm_g[vi] > thr_com
            k0 = 0
            _optres = minimize(
                kendallpval_cost,
                k0,
                args=(
                    stm_g[vi, _mask],
                    exp[vi, _mask, ni]
                ),
                # method='Powell'
            )
            _k = _optres.x
            _p = _optres.fun
            exp_ext[vi, :, ni] = exp[vi, :, ni] **\
                ((stm_g[vi])**kval[vi, ni])
            _t, _p = kendalltau(stm_g[vi, _mask], exp_ext[vi, _mask, ni])
            kval[vi, ni] = _k
            pval[vi, ni] = _p
            tval[vi, ni] = _t
    return kval, pval, tval, exp_ext


class MSTME:
    def __init__(self, ds: xr.Dataset, occur_freq: float, thr_pct: float = 0.25):
        self.ds: xr.Dataset = ds
        self.occur_freq: float = occur_freq
        self.thr_pct: float = thr_pct
        self.thr_mar: np.ndarray = None
        self.thr_com: float = None
        self.num_events: int = self.ds.event.size
        self.num_nodes: int = self.ds.node.size
        self.num_vars: int = 2
        self.stm: xr.DataArray = None
        self.exp: xr.DataArray = None
        self.var_name: list = ["$H_s$", "$U$"]
        self.var_name_g: list = ["$\hat H_s$", "$\hat U$"]
        self.par_name: list = ["$\\xi$", "$\\mu$", "$\\sigma$"]
        self.unit: list = ["[m]", "[m/s]"]
        self.is_e_marginal: np.ndarray = None
        self.gp: list = None

    def genpar_estimation(self, var_name, dir_out=None, draw_fig=True):
        """
        Bootstrap generalized pareto estimation for multivariates
        stm: ndarray(num_vars,num_events)
        thr_mar: array-like of n thresholds (n,)
        """
        global rng
        is_e_marginal = stm > thr_mar[:, np.newaxis]
        if (np.count_nonzero(is_e_marginal, axis=1) == 0).any():
            raise(ValueError("No events above marginal threshold"))
        if len(thr_mar) != num_vars:
            raise ValueError(
                'Number of thresholds do not match number of variables')
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
            xp, mp, sp = np.median(genpar_params[vi, :, :], axis=0)
            print(f"GENPAR{xp, mp, sp}")
            gp[vi] = genpareto(xp, mp, sp)
        #########################################################
        fig, ax = plt.subplots(len(par_name), num_vars,
                               figsize=(8*num_vars, 6*len(par_name)))

        for vi in range(num_vars):
            ax[0, vi].set_title(var_name[vi])
            for pi, p in enumerate(par_name):
                ax[pi, 0].set_ylabel(par_name[pi])
                ax[pi, vi].hist(genpar_params[vi, :, pi])
        if dir_out != None:
            savefig(f"{dir_out}/Genpar_Params.pdf", bbox_inches="tight")
        if not draw_fig:
            plt.close()
        #########################################################

        #########################################################
        fig, ax = plt.subplots(1, num_vars, figsize=(8*num_vars, 6))
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
        if dir_out != None:
            savefig(f"{dir_out}/Genpar_CDF.pdf", bbox_inches="tight")
        if not draw_fig:
            plt.close()
        #########################################################
        return gp

    def ndist_transform(self, dir_out=None, draw_fig=True):
        global ndist
        num_vars = stm.shape[0]
        stm_g = np.zeros(stm.shape)
        f_hat_cdf = [None, None]
        _uniform = np.zeros(stm_g.shape)
        for vi in range(num_vars):
            f_hat_cdf[vi] = lambda x, idx=vi: _f_hat_cdf(
                ECDF(stm[idx]), gp[idx], x)
            _stm = stm[vi]
            _uniform[vi] = f_hat_cdf[vi](_stm)
            print(f'Uniform max {var_name[vi]}: {_uniform[vi].max()}')
            # stm_g[vi] = ndist.ppf(np.clip(_uniform[vi], None, 1-10**(-6)))
            stm_g[vi] = ndist.ppf(_uniform[vi])
        # fig, ax = plt.subplots()
        # ax.scatter(_uniform[0], _uniform[1])

        print("H_hat min, max:", stm_g[0].min(), stm_g[0].max())
        print("U_hat min, max:", stm_g[1].min(), stm_g[1].max())

        #########################################################
        fig, ax = plt.subplots(1, 2, figsize=(7, 3))

        ax[0].scatter(stm[0], stm[1], s=5)
        ax[0].set_xlabel(f"{var_name[0]}{unit[0]}")
        ax[0].set_ylabel(f"{var_name[1]}{unit[1]}")
        ax[0].set_xlim(0, 20)
        ax[0].set_ylim(0, 60)

        ax[1].set_aspect(1)
        ax[1].scatter(stm_g[0], stm_g[1], s=5)
        ax[1].set_xlabel(r"$\hat H_s$")
        ax[1].set_ylabel(r"$\hat U$")
        ax[1].set_xlim(-5, 15)
        ax[1].set_ylim(-5, 15)
        ax[1].set_xticks([-2+2*i for i in range(6)])
        ax[1].set_yticks([-2+2*i for i in range(6)])

        if dir_out != None:
            savefig(f"{dir_out}/Original_vs_Laplace.pdf",
                    bbox_inches="tight")
        if not draw_fig:
            plt.close()
        #########################################################
        return stm_g, f_hat_cdf

    def kendall_tau_mv(self, dir_out=None, draw_fig=True):
        """
        stm_g: (num_vars, num_events)
        exp: (num_vars, num_events, num_nodes)
        is_e: (num_vars,)
        """
        global rng
        num_vars = stm_g.shape[0]
        num_nodes = exp.shape[2]

        tval = np.zeros(((num_vars, num_vars, num_nodes)))
        pval = np.zeros((num_vars, num_vars, num_nodes))
        for vi in range(num_vars):
            for vj in range(num_vars):
                _stm = stm_g[vi]
                _exp = exp[vj, :, :]
                for ni in range(num_nodes):
                    _tval, _pval = kendalltau(
                        _stm[is_e[vi]], _exp[is_e[vi], ni])
                    tval[vi, vj, ni] = _tval
                    pval[vi, vj, ni] = _pval

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
        if dir_out != None:
            savefig(f"{dir_out}/Kendall_Tau_all_var_pval.pdf",
                    bbox_inches="tight")
        if not draw_fig:
            plt.close()
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
                im = ax[vi, vj].scatter(lonlat[:, 0], lonlat[:, 1], s=5, c=tval[vi, vj, :], cmap='seismic', vmax=np.abs(
                    tval[vi]).max(), vmin=-np.abs(tval[vi]).max())
                plt.colorbar(im, ax=ax[vi, vj])
                ax[vi, vj].set_title(f"STM:{var_name[vi]} E:{var_name[vj]}")
        if dir_out != None:
            savefig(f"{dir_out}/Kendall_Tau_all_var_tval.pdf",
                    bbox_inches="tight")
        if not draw_fig:
            plt.close()
        #########################################################
        return pval, tval

    def estimate_conmul(self, dir_out=None, SEARCH=False, draw_fig=True):
        global ndist
        num_vars = stm_g.shape[0]
        num_events = stm_g.shape[1]
        is_e = stm_g > thr_gum
        # Laplace replacement
        N_rep = 100
        stm_g_rep = np.zeros((N_rep, num_vars, num_events))
        for i in range(N_rep):
            _idx = rng.choice(num_events, size=num_events)
            _stm = stm_g[:, _idx]
            for vi in range(num_vars):
                _laplace_sample = ndist.rvs(size=num_events)
                _laplace_sample_sorted = np.sort(_laplace_sample)
                _arg = np.argsort(_stm[vi])
                stm_g_rep[i, vi, _arg] = _laplace_sample_sorted
        # Estimate conditional model parameters
        lb = [0, None, -5, 0.1]
        ub = [1, 1, 5, None]
        params_uc = np.zeros((num_vars, N_rep, 4))
        costs = np.zeros((num_vars, N_rep))
        for vi in range(num_vars):
            for i in range(N_rep):
                _stm = stm_g_rep[i]
                a0 = np.random.uniform(low=lb[0], high=ub[0])
                b0 = np.random.uniform(low=-1, high=ub[1])
                m0 = np.random.uniform(low=-1, high=1)
                # s0 = np.random.uniform(low=0.01, high=0.99)
                s0 = 1
                _p0 = np.array([a0, b0, m0, s0])
                if np.isnan(_p0).any():
                    raise(ValueError("WTF"))
                evt_mask = np.logical_and(
                    (_stm[vi, :] > thr_gum), (~np.isinf(_stm[vi, :])))
                x = _stm[vi, evt_mask]  # conditioning
                # conditioned
                y = np.delete(_stm[:, evt_mask], vi, axis=0)
                optres = minimize(
                    cost,
                    _p0,
                    args=(x, y),
                    jac=jacobian_custom,
                    # hess=hessian_custom,
                    method='L-BFGS-B',
                    bounds=((lb[0], ub[0]), (lb[1], ub[1]),
                            (lb[2], ub[2]), (lb[3], ub[3])),
                )
                # if optres.success ==:
                # def func(self, x):
                #     return cost(
                #         x, stm_g_rep[:, i, evt_mask], vi
                #     )
                # optres = minimize(
                #     func,
                #     np.array([a0, b0, m0, s0]),
                #     # method='trust-constr',
                #     bounds=((lb[0], ub[0]), (lb[1], ub[1]),
                #             (lb[2], ub[2]), (lb[3], ub[3])),
                # )
                _param = optres.x
                _cost = optres.fun
                if np.isnan(_cost):
                    print(_param)
                    print(cost(_param, x, y))
                    raise(ValueError("Cost is NaN"))
                params_uc[vi, i, :] = _param
                costs[vi, i] = _cost
        print(f'costs:{costs}')
        params_median = np.median(params_uc, axis=1)
        print("Params_median:", params_median)
        # Threshold search
        if SEARCH:
            threshold_search.search_conditional(stm_g_rep, 1.0, 3.0)
            # Calculating residuals
        residual = []
        print("Residuals")
        for vi in range(num_vars):
            _x = stm_g[vi, is_e[vi]]  # conditioning(extreme)
            _y = np.delete(stm_g[:, is_e[vi]], vi, axis=0)  # conditioned
            _a = params_median[vi, 0]
            _b = params_median[vi, 1]
            _z = (_y - _a * _x) / (_x ** _b)
            residual.append(_z)
            # print(_z.flatten())
            for i, __z in enumerate(_z.squeeze()):
                if __z > 5:
                    print(f"{var_name[vi],}a,b,x,y", _a, _b, _x[i], _y[0, i])
            print(f'{var_name[vi]} min, max: {_z.min()},{_z.max()}')

        ##################################################################################################################
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), facecolor="white")
        ax.scatter(stm_g_rep[:, 0, :], stm_g_rep[:, 1, :], alpha=0.1)
        ax.scatter(stm_g[0], stm_g[1], color="blue")
        ax.set_xlabel(r"$\hat H_s$")
        ax.set_ylabel(r"$\hat U$")
        ax.set_xlim(-3, 15)
        ax.set_ylim(-3, 15)
        if dir_out != None:
            savefig(f"{dir_out}/Replacement.pdf", bbox_inches="tight")
        if not draw_fig:
            plt.close()
        #########################################################
        fig, ax = plt.subplots(4, num_vars, figsize=(8*num_vars, 6*4))
        fig.tight_layout()
        ax[0, 0].set_ylabel("a")
        ax[1, 0].set_ylabel("b")
        ax[2, 0].set_ylabel("$\mu$")
        ax[3, 0].set_ylabel("$\sigma$")
        ax[3, 0].set_xlabel(var_name[0])
        ax[3, 1].set_xlabel(var_name[1])
        for vi in range(num_vars):
            ax[0, vi].hist(params_uc[vi, :, 0])
            ax[1, vi].hist(params_uc[vi, :, 1])
            ax[2, vi].hist(params_uc[vi, :, 2])
            ax[3, vi].hist(params_uc[vi, :, 3])
        if dir_out != None:
            savefig(f"{dir_out}/Conmul_Estimates.pdf", bbox_inches="tight")
        if not draw_fig:
            plt.close()
        #########################################################
        fig, ax = plt.subplots(1, num_vars)
        for vi in range(num_vars):
            ax[vi].hist(costs[vi])
            ax[vi].set_title('cost')
        if not draw_fig:
            plt.close()
        #########################################################
        fig, ax = plt.subplots(1, num_vars, figsize=(
            8*num_vars, 6), facecolor="white")
        fig.supxlabel("$a$")
        fig.supylabel("$b$")
        params_ml = np.zeros((4, num_vars))
        for vi in range(num_vars):
            ax[vi].scatter(
                params_uc[vi, :, 0],
                params_uc[vi, :, 1],
                s=5,
                label="Generated samples",
            )
            ax[vi].set_title(var_name[vi])
        if dir_out != None:
            savefig(f"{dir_out}/ab_Estimates.pdf", bbox_inches="tight")
        if not draw_fig:
            plt.close()
        #########################################################
        fig, ax = plt.subplots(1, num_vars, figsize=(
            8*num_vars, 6), facecolor="white")
        fig.supxlabel("$a$")
        fig.supylabel("$mu$")
        params_ml = np.zeros((4, num_vars))
        for vi in range(num_vars):
            ax[vi].scatter(
                params_uc[vi, :, 0],
                params_uc[vi, :, 2],
                s=5,
                label="Generated samples",
            )
            ax[vi].set_title(var_name[vi])
        if not draw_fig:
            plt.close()
        #########################################################
        fig, ax = plt.subplots(1, num_vars, figsize=(
            8*num_vars, 6), facecolor="white")
        fig.supxlabel("$a$+$mu$")
        fig.supylabel("$b$")
        params_ml = np.zeros((4, num_vars))
        for vi in range(num_vars):
            ax[vi].scatter(
                params_uc[vi, :, 0]+params_uc[vi, :, 2],
                params_uc[vi, :, 1],
                s=5,
                label="Generated samples",
            )
            ax[vi].set_title(var_name[vi])
        if not draw_fig:
            plt.close()
        #########################################################
        fig, ax = plt.subplots(1, num_vars, figsize=(
            8*num_vars, 6), facecolor="white")
        # fig.tight_layout()
        for vi in range(num_vars):
            ax[vi].scatter(ndist.cdf(stm_g[vi, is_e[vi]]), residual[vi], s=5)
            ax[vi].set_xlabel(f"F({var_name[vi]})")
        ax[0].set_ylabel("$Z_{-j}$")
        if dir_out != None:
            savefig(f"{dir_out}/Residuals.pdf", bbox_inches="tight")
        if not draw_fig:
            plt.close()
        #########################################################
        return params_median, residual

    def sample_stm(self, params_median, residual, size=1000, dir_out=None, draw_fig=True):
        # Sample from model
        num_vars = stm.shape[0]
        num_events = stm.shape[1]
        N_sample = size

        is_e = stm_g > thr_gum
        vi_largest = stm_g.argmax(axis=0)
        is_me = np.empty((num_vars, num_events))
        for vi in range(num_vars):
            is_me[vi] = np.logical_and(vi_largest == vi, is_e[vi])
        is_e_any = is_e.any(axis=0)
        v_me_ratio = np.count_nonzero(
            is_me, axis=1) / np.count_nonzero(is_e_any)
        exceedance_prob = np.count_nonzero(is_e_any) / num_events

        thr_uni = ndist.cdf(thr_gum)
        std_gum = ndist.ppf(rng.uniform(thr_uni, 1, size=N_sample))
        vi_list = rng.choice(num_vars, size=N_sample, p=v_me_ratio)

        sample_full_g = np.zeros((num_vars, N_sample))
        for i, vi in enumerate(vi_list):
            _a = np.asarray(params_median[vi, 0])
            _b = np.asarray(params_median[vi, 1])
            while True:
                _z = rng.choice(residual[vi], axis=1)
                _y_given_x = std_gum[i] * _a + (std_gum[i] ** _b) * _z
                if (_y_given_x < std_gum[i]).all():
                    _samples = np.insert(np.asarray(
                        _y_given_x), vi, std_gum[i])
                    sample_full_g[:, i] = _samples
                    break

        # Transform back to original scale
        ppf = [None, None]
        sample_full = np.zeros(sample_full_g.shape)
        sample_uni = ndist.cdf(sample_full_g)
        for vi in range(num_vars):
            ppf[vi] = lambda x, i=vi: _f_hat_ppf(
                ECDF(stm[i]), gp[i], stm[i], x)
            sample_full[vi] = ppf[vi](sample_uni[vi])
        # _mask = sample_full[0, :] < 6
        # if np.count_nonzero(_mask) > 0:
        #     _stm_low, _ = ndist_transform(
        #         sample_full[:, _mask], gp, var_name, unit, None)
        #     fig, ax = plt.subplots()
        #     ax.scatter(stm_g[0], stm_g[1], c='black')
        #     ax.scatter(_stm_low[0], _stm_low[1], c='red')
        #     ax.set_xlim(-5, 15)
        #     ax.set_ylim(-5, 15)
        #     ax.set_aspect('equal')
        #     print(f"")
        #     raise(ValueError("Very low STM"))
        # _x = np.linspace(0,1,100)
        # fig,ax=plt.subplots(1,2)
        # for vi in range(num_vars):
        #     ax[vi].plot(_x,ppf[vi](_x))
        ##################################################################################################################
        fig, ax = plt.subplots(1, num_vars, figsize=(
            8*num_vars, 6), facecolor="white")

        ax[0].set_aspect(1)

        a_h, b_h, mu_h, sg_h = params_median[0, :]
        a_u, b_u, mu_u, sg_u = params_median[1, :]
        sample_given_h = []
        sample_given_u = []
        sample_given_hg = []
        sample_given_ug = []
        for i, vi in enumerate(vi_list):
            if vi == 0:
                sample_given_h.append(sample_full[:, i])
                sample_given_hg.append(sample_full_g[:, i])
            if vi == 1:
                sample_given_u.append(sample_full[:, i])
                sample_given_ug.append(sample_full_g[:, i])
        sample_given_h = np.array(sample_given_h).T
        sample_given_u = np.array(sample_given_u).T
        sample_given_hg = np.array(sample_given_hg).T
        sample_given_ug = np.array(sample_given_ug).T

        x_h = np.linspace(thr_gum, 10, 100)
        y_h = x_h * a_h + (x_h ** b_h) * mu_h
        ax[0].plot(x_h, y_h, color="orange", label="U|H")

        y_u = np.linspace(thr_gum, 10, 100)
        x_u = y_u * a_u + (y_u ** b_u) * mu_u
        ax[0].plot(x_u, y_u, color="teal", label="H|U")

        ax[0].scatter(stm_g[0], stm_g[1], s=5, color="black", label="original")
        ax[0].axvline(thr_gum, color="black")
        ax[0].axhline(thr_gum, color="black")

        ax[0].set_xlabel(r"$\hat H_s$")
        ax[0].set_ylabel(r"$\hat U$")
        ax[0].set_xlim(-2, 10)
        ax[0].set_ylim(-2, 10)
        ax[0].scatter(
            sample_given_hg[0], sample_given_hg[1], s=1, color="orange", label="U|H"
        )
        ax[0].scatter(
            sample_given_ug[0], sample_given_ug[1], s=1, color="teal", label="H|U"
        )
        print(sample_given_hg.max(), sample_given_ug.max())
        print(sample_given_hg.min(), sample_given_ug.min())

        ax[1].scatter(stm[0], stm[1], color="black", s=5)
        ax[1].scatter(sample_given_h[0], sample_given_h[1],
                      color="orange", s=1)
        ax[1].scatter(sample_given_u[0], sample_given_u[1], color="teal", s=1)
        ax[1].set_xlabel(f"{var_name[0]}{unit[0]}")
        ax[1].set_ylabel(f"{var_name[1]}{unit[1]}")

        res = 100
        _x = np.linspace(0, stm[0].max(), res)
        _y = np.linspace(0, stm[1].max(), res)
        _x_mg, _y_mg = np.meshgrid(_x, _y)
        _z_mg_sample = np.zeros((res, res))
        _z_mg = np.zeros((res, res))

        for xi in range(res):
            for yi in range(res):
                _count_sample = np.count_nonzero(
                    np.logical_and(sample_full[0] >
                                   _x[xi], sample_full[1] > _y[yi])
                )
                _count = np.count_nonzero(
                    np.logical_and(stm[0] > _x[xi], stm[1] > _y[yi])
                )
                _z_mg_sample[xi, yi] = _count_sample
                _z_mg[xi, yi] = _count
        return_periods = [100]

        _levels_original = [num_events / (rp * occur_prob)
                            for rp in return_periods]
        _levels_sample = [
            N_sample / (rp * occur_prob * exceedance_prob) for rp in return_periods
        ]
        _linestyles = ["-", "--"]
        ax[1].contour(
            _x_mg,
            _y_mg,
            _z_mg.T,
            levels=_levels_original,
            linestyles=_linestyles,
            colors="black",
        )
        ax[1].contour(
            _x_mg,
            _y_mg,
            _z_mg_sample.T,
            levels=_levels_sample,
            linestyles=_linestyles,
            colors="red",
        )

        if dir_out != None:
            savefig(
                f"{dir_out}/Simulated_Conmul_vs_Back_Transformed.pdf", bbox_inches="tight"
            )
        if not draw_fig:
            plt.close()
        #########################################################
        return sample_full, ppf

    def search_isocontour(self, scatter, n):
        coords = []
        scatter = np.unique(scatter, axis=1)
        _num_events = scatter.shape[1]
        _xg, _yg = np.sort(scatter[0]), np.sort(scatter[1])
        _idx_x, _idx_y = np.argsort(scatter[0]), np.argsort(scatter[1])
        _xi, _yi = _num_events-1, 0
        # Search isocontour
        while True:
            # Keep searching until the edge
            if _xi-1 < 0 or _yi+1 >= _num_events:
                break

            _x = _xg[_xi]
            _y = _yg[_yi]

            # Check if you can go up, in which case you do
            _count_up = np.count_nonzero(
                np.logical_and(scatter[0] >= _x, scatter[1] >= _yg[_yi+1])
            )
            if _count_up == n:
                _yi += 1
            # If you can't, look directly to the left
            else:
                _dxi = 1
                _success = False
                while not _success:
                    _dyi = 0
                    # Keep moving up from there until you reach a viable point
                    while True:
                        _count_left = np.count_nonzero(
                            np.logical_and(scatter[0] >= _xg[_xi-_dxi],
                                           scatter[1] >= _yg[_yi+_dyi])
                        )
                        if _count_left == n:
                            _xi -= _dxi
                            _yi += _dyi
                            _success = True
                            break
                        elif _count_left > n:
                            _dyi += 1
                        else:
                            # No point found along the vertical transect
                            # So move to the left and search again
                            _success = False
                            _dxi += 1
                            break
            coords.append([_xg[_xi], _yg[_yi]])

        return np.array(coords).T

    def plot_isocontour(self, tm_original_bs, tm_sample_bs, exceedance_prob, return_period=100, dir_out=None, draw_fig=True):
        # tm(N_bs,numvar,numevent,numnode)
        global pos_color
        N_bootstrap = len(tm_sample_bs)
        num_events = tm_original_bs.shape[2]
        stm_min = np.floor(
            tm_sample_bs[:, :, :, idx_pos_list].min(axis=(0, 2, 3))/5)*5
        stm_max = np.ceil(
            tm_sample_bs[:, :, :, idx_pos_list].max(axis=(0, 2, 3))/5)*5
        print(np.array(tm_sample_bs).min(axis=(0, 2, 3)),
              np.array(tm_sample_bs).max(axis=(0, 2, 3)))
        #########################################################
        fig, axes = plt.subplots(2, 2, figsize=(8*2, 6*2), facecolor="white",)
        # fig.suptitle(
        #     f"{return_period}-yr return period (Saint-Denis) Laplace threshold = {thr_gum}",
        #     y=0.90,
        # )
        fig.supxlabel(r"$H_s$[m]")
        fig.supylabel(r"$U$[m/s]")
        alpha = 1/min(N_bootstrap, 10)
        for i, ax in enumerate(axes.flatten()):
            ax.set_xlim(stm_min[0], stm_max[0])
            ax.set_ylim(stm_min[1], stm_max[1])
            _linestyles = ["-", "--"]
            _idx_pos = idx_pos_list[i]
            # sample
            _ic_sample = []
            for bi in range(N_bootstrap):
                _num_events_extreme = tm_sample_bs[bi].shape[1]
                _count_sample = round(_num_events_extreme /
                                      (return_period * occur_freq * exceedance_prob))
                _ic_sample.append(search_isocontour(
                    tm_sample_bs[bi][:, :, _idx_pos], _count_sample))

            # original
            _ic_original = []
            _count_original = round(num_events / (return_period * occur_freq))
            for bi in range(N_bootstrap):
                _ic_original.append(search_isocontour(
                    tm_original_bs[bi][:, :, _idx_pos], _count_original))

            for bi in range(N_bootstrap):
                ax.scatter(
                    tm_sample_bs[bi][0, :, _idx_pos],
                    tm_sample_bs[bi][1, :, _idx_pos],
                    s=2,
                    c=pos_color[i],
                    label=f"Simulated",
                    alpha=1.0/N_bootstrap,
                )
                ax.scatter(
                    tm_original_bs[bi][0, :, _idx_pos],
                    tm_original_bs[bi][1, :, _idx_pos],
                    s=10,
                    c="black",
                    label=f"Original",
                    alpha=1.0/N_bootstrap,
                )
                ax.plot(_ic_original[bi][0],
                        _ic_original[bi][1],
                        c='black',
                        alpha=alpha,
                        lw=2)
                ax.plot(_ic_sample[bi][0],
                        _ic_sample[bi][1],
                        c=pos_color[i],
                        alpha=alpha,
                        lw=2)
            ax.set_title(f"Coord.{i+1}")
        if dir_out != None:
            savefig(
                f"{dir_out}/RV_(Saint-Denis).pdf", bbox_inches="tight"
            )
        if not draw_fig:
            plt.close()

    def kendallpval_cost(self, k, stm_norm, exp):
        _t, _p = kendalltau(stm_norm, exp**stm_norm**k)
        return -_p

    def calculate_kval(self):
        exp_ext = np.zeros(exp.shape)
        kval = np.zeros((num_vars, num_nodes))
        pval = np.zeros((num_vars, num_nodes))
        tval = np.zeros((num_vars, num_nodes))
        # fig, ax = plt.subplots(1, num_vars, figsize=(16, 8))
        for ni in trange(num_nodes):
            for vi in range(num_vars):
                _mask = stm_g[vi] > thr_com
                k0 = 0
                _optres = minimize(
                    kendallpval_cost,
                    k0,
                    args=(
                        stm_g[vi, _mask],
                        exp[vi, _mask, ni]
                    ),
                    # method='Powell'
                )
                _k = _optres.x
                _p = _optres.fun
                exp_ext[vi, :, ni] = exp[vi, :, ni] **\
                    ((stm_g[vi])**kval[vi, ni])
                _t, _p = kendalltau(stm_g[vi, _mask], exp_ext[vi, _mask, ni])
                kval[vi, ni] = _k
                pval[vi, ni] = _p
                tval[vi, ni] = _t
        return kval, pval, tval, exp_ext

    def sample(self, seed=0, dir_out=None):
        _rng = np.random.default_rng(seed)
        # STM and Exposure
        tm = self.ds[['hs', 'UV_10m']].to_array()
        self.stm: xr.DataArray = self.ds[['hs', 'UV_10m']].max(
            dim="node").to_array()
        self.exp: xr.DataArray = self.ds[[
            'hs', 'UV_10m']].to_array() / self.stm
        lonlat = np.array([self.ds.longitude, self.ds.latitude]).T
        thr_mar = np.sort(self.stm, axis=1)[
            :, -round(self.num_events*self.thr_pct)]
        fig, ax = plt.subplots(
            1, self.num_vars, figsize=(8 * self.num_vars, 6))
        for vi in range(self.num_vars):
            ax[vi].hist(self.stm[vi], 20)
            ax[vi].set_xlabel(f"{var_name[vi]}{unit[vi]}")
            ax[vi].set_title("STM")

        # Generalized Pareto estimation
        gp = genpar_estimation(
            self.stm, thr_mar, var_name, unit, par_name, dir_out=dir_out, draw_fig=False)

        # Laplace Transform
        stm_g, f_hat_cdf = ndist_transform(
            self.stm, gp, var_name, unit, dir_out=dir_out, draw_fig=False)

        # Set common threshold
        thr_com = np.percentile(stm_g.min(axis=0), 100-self.thr_pct*100)
        print(thr_com, "Delete me")

        # Set boolean mask
        is_e = stm_g > thr_com
        vi_largest = stm_g.argmax(axis=0)
        is_me = np.empty((self.num_vars, self.num_events))
        for vi in range(self.num_vars):
            is_me[vi] = np.logical_and(vi_largest == vi, is_e[vi])
        is_e_any = is_e.any(axis=0)
        exceedance_prob = np.count_nonzero(is_e_any) / self.num_events
        v_me_ratio = np.count_nonzero(
            is_me, axis=1) / np.count_nonzero(is_e_any)
        # Kendall's Tau for differing variables
        # if EXT_EXPOSURE:
        #     pval, tval = kendall_tau_mv(stm_g, exp_ext, is_e, var_name, lonlat,
        #                                 dir_out=dir_out, draw_fig=False)
        # else:
        #     pval, tval = kendall_tau_mv(stm_g, exp, is_e, var_name, lonlat,
        #                                 dir_out=dir_out, draw_fig=False)
        pval, tval = kendall_tau_mv(stm_g, self.exp, is_e, var_name, lonlat,
                                    dir_out=dir_out, draw_fig=False)
        # Estimate conmul parameters
        params_median, residual = estimate_conmul(
            stm_g, thr_com, var_name, dir_out=dir_out, draw_fig=False)
        # Sample from conmul model
        sample_full, ppf = sample_stm(self.stm, stm_g, gp, f_hat_cdf, thr_com, params_median,
                                      residual, self.occur_freq, var_name, unit, dir_out=dir_out, draw_fig=False)
        N_sample = sample_full.shape[1]
        # Exposure set sampling
        N = self.num_events * 1
        exp_sample = np.zeros((self.num_vars, N, self.num_nodes))
        stm_sample = np.zeros((self.num_vars, N))

        # choose random stm set from sample_full
        _idx_smp = _rng.choice(N_sample, size=N)
        stm_sample = sample_full[:, _idx_smp]
        # choose random exposure set from events where
        # STM is extreme in either variable
        _idx_evt = _rng.choice(np.nonzero(is_e_any)[0], size=N)
        # if EXT_EXPOSURE:
        #     # transform exp_ext back to exposure
        #     exp_sample = np.zeros((self.num_vars, N, num_nodes))
        #     for vi in range(self.num_vars):
        #         for ni in range(num_nodes):
        #             exp_sample[vi, :, ni] = exp_ext[vi, _idx_evt, ni] **\
        #                 ((thr_mar[vi]/stm_sample[vi])**kval[vi, ni])
        # else:
        #     exp_sample = exp[:, _idx_evt, :]
        exp_sample = self.exp[:, _idx_evt, :]

        # factor
        _tm_sample = np.einsum("ven,ve->ven", exp_sample, stm_sample)


def mstme(ds: xr.Dataset, occur_freq: float, seed: int = 0, thr_pct: float = 0.25, dir_out: str = None, draw_fig: bool = True):
    _rng = np.random.default_rng(seed)
    num_events = ds.event.size
    num_nodes = ds.node.size
    num_vars = 2
    # STM and Exposure
    tm = ds[['hs', 'UV_10m']].to_array()
    stm = ds[['hs', 'UV_10m']].max(dim="node").to_array()
    exp = ds[['hs', 'UV_10m']].to_array() / stm
    lonlat = np.array([ds.longitude, ds.latitude]).T
    thr_mar = np.sort(stm, axis=1)[:, -round(num_events*thr_pct)]
    fig, ax = plt.subplots(1, num_vars, figsize=(8 * num_vars, 6))
    for vi in range(num_vars):
        ax[vi].hist(stm[vi], 20)
        ax[vi].set_xlabel(f"{var_name[vi]}{unit[vi]}")
        ax[vi].set_title("STM")

    # Generalized Pareto estimation
    gp = genpar_estimation(
        stm, thr_mar, var_name, unit, par_name, dir_out=dir_out, draw_fig=draw_fig)

    # Laplace Transform
    stm_g, f_hat_cdf = ndist_transform(
        stm, gp, var_name, unit, dir_out=dir_out, draw_fig=draw_fig)

    # Set common threshold
    thr_com = np.percentile(stm_g.max(axis=0), 100-thr_pct*100)
    print(thr_com, "Delete me")

    # Set boolean mask
    is_e = stm_g > thr_com
    vi_largest = stm_g.argmax(axis=0)
    is_me = np.empty((num_vars, num_events))
    for vi in range(num_vars):
        is_me[vi] = np.logical_and(vi_largest == vi, is_e[vi])
    is_e_any = is_e.any(axis=0)
    exceedance_prob = np.count_nonzero(is_e_any) / num_events
    v_me_ratio = np.count_nonzero(is_me, axis=1) / np.count_nonzero(is_e_any)
    # Kendall's Tau for differing variables
    # if EXT_EXPOSURE:
    #     pval, tval = kendall_tau_mv(stm_g, exp_ext, is_e, var_name, lonlat,
    #                                 dir_out=dir_out, draw_fig = draw_fig)
    # else:
    #     pval, tval = kendall_tau_mv(stm_g, exp, is_e, var_name, lonlat,
    #                                 dir_out=dir_out, draw_fig = draw_fig)
    pval, tval = kendall_tau_mv(stm_g, exp, is_e, var_name, lonlat,
                                dir_out=dir_out, draw_fig=draw_fig)
    # Estimate conmul parameters
    params_median, residual = estimate_conmul(
        stm_g, thr_com, var_name, dir_out=dir_out, draw_fig=draw_fig)
    # Sample from conmul model
    sample_full, ppf = sample_stm(stm, stm_g, gp, f_hat_cdf, thr_com, params_median,
                                  residual, occur_freq, var_name, unit, dir_out=dir_out, draw_fig=draw_fig)
    N_sample = sample_full.shape[1]
    # Exposure set sampling
    N = num_events * 1
    exp_sample = np.zeros((num_vars, N, num_nodes))
    stm_sample = np.zeros((num_vars, N))

    # choose random stm set from sample_full
    _idx_smp = _rng.choice(N_sample, size=N)
    stm_sample = sample_full[:, _idx_smp]
    # choose random exposure set from events where
    # STM is extreme in either variable
    _idx_evt = _rng.choice(np.nonzero(is_e_any)[0], size=N)
    # if EXT_EXPOSURE:
    #     # transform exp_ext back to exposure
    #     exp_sample = np.zeros((num_vars, N, num_nodes))
    #     for vi in range(num_vars):
    #         for ni in range(num_nodes):
    #             exp_sample[vi, :, ni] = exp_ext[vi, _idx_evt, ni] **\
    #                 ((thr_mar[vi]/stm_sample[vi])**kval[vi, ni])
    # else:
    #     exp_sample = exp[:, _idx_evt, :]
    exp_sample = exp[:, _idx_evt, :]

    # factor
    tm_sample = np.einsum("ven,ve->ven", exp_sample, stm_sample)
    # tm_sample_bs.append(tm_sample)
    # tm_original_bs.append(tm_all[:, idx_bootstrap[bi], :])

    # output info
    if dir_out != None:
        with open(f"{dir_out}/info_{seed:03d}.txt", "w") as f:
            _sample_h = np.count_nonzero(stm[0] > thr_mar[0])
            _sample_u = np.count_nonzero(stm[1] > thr_mar[1])
            _sample_hg = np.count_nonzero(stm_g[0] > thr_com)
            _sample_ug = np.count_nonzero(stm_g[1] > thr_com)

            f.write(
                f"Marginal Threshold:\t{thr_mar[0]}[m], {thr_mar[1]}[m/s]\n")
            f.write(f"Marginal Sample Size:\t{_sample_h},{_sample_u}\n")
            f.write(f"Laplace Threshold: \t{thr_com}\n")
            f.write(f"Laplace Sample Size:\t{_sample_hg},{_sample_ug}\n")
            f.write(
                f"Laplace Threshold in Marginal Scale:\t{ppf[0](np.exp(-np.exp(-thr_com))):.2f}, {ppf[1](np.exp(-np.exp(-thr_com))):.2f}"
            )
    return tm_sample, exceedance_prob
