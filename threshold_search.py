# %%
# init
from concurrent.futures import thread
import numpy as np
from scipy.optimize import minimize
from scipy.stats._continuous_distns import genpareto
import matplotlib.pyplot as plt
import stme

rng = np.random.default_rng()
plt.style.use("plot_style.txt")
depth = -100
dir_out = "./output/common"
var_name = ["$H_s$", "$U$"]
par_name = ["$\\xi$", "$\\mu$", "$\\sigma$"]

def search_marginal(stm, thr_start, thr_end, res=10, N_gp = 100):
    """
    stm: ndarray, shape(num_vars, num_events)
    thr_start: list
    thr_end: list
    """
    global rng, depth, dir_out, var_name, par_name
    num_vars = stm.shape[0]
    num_events = stm.shape[1]
    # Generalized Pareto estimation over threshold range
    N_gp = 100
    N_THR = res
    thr_list = np.linspace(thr_start, thr_end, N_THR).T
    genpar_params = np.zeros((N_THR, num_vars, N_gp, 3))
    num_samples = np.zeros((N_THR, num_vars, N_gp))
    for vi in range(num_vars):
        for ti, _thr in enumerate(thr_list[vi]):
            _stm_bootstrap = rng.choice(stm[vi], size=(N_gp, num_events))
            for i in range(N_gp):
                _stm = _stm_bootstrap[i]
                _stm_pot = _stm[_stm > _thr]
                _xp, _mp, _sp = genpareto.fit(_stm_pot, floc=_thr)
                genpar_params[ti, vi, i, :] = [_xp, _mp, _sp]
                num_samples[ti, vi, i] = np.count_nonzero(_stm_pot)
            # print(genpar_params[ti,vi].mean(axis=0))
    # Number of samples
    fig, ax = plt.subplots(
        1,
        num_vars,
        sharey=True,
        figsize=(8 * num_vars, 6),
        facecolor="white",
        squeeze=False,
    )
    fig.supylabel("# of occurrences above threshold")

    med = np.percentile(num_samples, 50, axis=2)
    u95 = np.percentile(num_samples, 97.5, axis=2)
    l95 = np.percentile(num_samples, 2.5, axis=2)
    for vi in range(num_vars):
        ax[0, vi].plot(thr_list[vi], med[:, vi])
        ax[0, vi].fill_between(thr_list[vi], u95[:, vi], l95[:, vi], alpha=0.5)
        ax[0, vi].set_title(var_name[vi])
        # ax[0,vi].axhline(20,color='red')
    ax[0, 0].set_xlabel("Threshold[m]")
    plt.savefig(f"{dir_out}/Marginal_sample_num.pdf", bbox_inches="tight")
    plt.savefig(f"{dir_out}/Marginal_sample_num.png", bbox_inches="tight")
    
    # Shape parameter
    fig, ax = plt.subplots(
        1,
        num_vars,
        sharey=True,
        figsize=(8 * num_vars, 6),
        facecolor="white",
        squeeze=False,
    )
    u95 = np.percentile(genpar_params, 97.5, axis=2)
    l95 = np.percentile(genpar_params, 2.5, axis=2)
    med = np.percentile(genpar_params, 50.0, axis=2)
    var_name = ["$H_s$", "$U$"]
    par_name = ["$\\xi$", "$\\mu$", "$\\sigma$"]
    for vi in range(num_vars):
        ax[0, vi].set_title(var_name[vi])
        ax[0, 0].set_ylabel(par_name[0])
        ax[0, vi].plot(thr_list[vi], med[:, vi, 0])
        ax[0, vi].fill_between(thr_list[vi], u95[:, vi, 0], l95[:, vi, 0], alpha=0.5)

    ax[0, 0].set_xlabel("Threshold[m]")
    plt.savefig(f"{dir_out}/Marginal_param_vs_threshold.pdf", bbox_inches="tight")
    plt.savefig(f"{dir_out}/Marginal_param_vs_threshold.png", bbox_inches="tight")

def search_conditional(stm_g_rep, thr_start, thr_end, N_THR=10):
    global rng, depth, dir_out, var_name, par_name

    thr_gum_list = np.linspace(thr_start,thr_end,N_THR)
    num_vars = stm_g_rep.shape[0]
    lb = [0, None, -5, 0]
    ub = [1, 1, 5, 5]
    N_rep = stm_g_rep.shape[1]
    params_search_uc = np.zeros((N_THR, N_rep, 4, num_vars))
    for ti in range(N_THR):
        _thr = thr_gum_list[ti]
        for i in range(N_rep):
            for vi in range(num_vars):
                a0 = np.random.uniform(low=lb[0], high=ub[0])
                b0 = np.random.uniform(low=-1, high=ub[1])
                m0 = np.random.uniform(low=-1, high=1)
                s0 = np.random.uniform(low=lb[3], high=1)

                evt_mask = stm_g_rep[vi, i, :] > _thr

                def func(x):
                    return stme.cost(
                        x, stm_g_rep[:, i, evt_mask], vi
                    )

                optres = minimize(
                    func,
                    np.array([a0, b0, m0, s0]),
                    # method='trust-constr',
                    bounds=((lb[0], ub[0]), (lb[1], ub[1]), (lb[2], ub[2]), (lb[3], ub[3])),
                )
                _param = optres.x
                params_search_uc[ti, i, :, vi] = _param
    params_median = np.median(params_search_uc, axis=1)
    params_u95 = np.percentile(params_search_uc,97.5,axis=1)
    params_l95 = np.percentile(params_search_uc, 2.5,axis=1)
    params_u75 = np.percentile(params_search_uc,75.0,axis=1)
    params_l25 = np.percentile(params_search_uc,25.0,axis=1)

    num_samples = np.zeros((N_THR, N_rep, num_vars))
    for ti in range(N_THR):
        for i in range(N_rep):
            for vi in range(num_vars):
                num_samples[ti,i,vi] = np.count_nonzero(stm_g_rep[vi, i, :] > thr_gum_list[ti])

    fig, ax = plt.subplots(1,2, sharex=True, sharey=True, figsize=(8,3),facecolor='white')
    # fig.tight_layout()
    fig.supxlabel("Gumbel threshold")
    fig.supylabel("# of occurrences above threshold")

    _med = np.percentile(num_samples,50,axis=1)
    _u95 = np.percentile(num_samples,97.5,axis=1)
    _l95 = np.percentile(num_samples, 2.5,axis=1)
    for vi in range(num_vars):
        ax[vi].plot(thr_gum_list, _med[:,vi])
        ax[vi].fill_between(thr_gum_list, _u95[:,vi], _l95[:,vi],alpha=0.5)
        ax[vi].set_title(var_name[vi])

    plt.savefig(f"{dir_out}/Conmul_sample_num.pdf", bbox_inches="tight")

    fig, ax = plt.subplots(2,2,sharex=True, figsize=(10,6),constrained_layout=True)
    fig.supxlabel("Gumbel threshold")
    fig.set_facecolor('white')
    p_name = ['a','b','$\mu$','$\sigma$']
    m_name = ['U|H', 'H|U']
    for vi in range(num_vars):
        ax[0,vi].set_title(m_name[vi])
        # ax[0,vi].set_title(var_name[vi])
        for pi in range(2):
            # ax[pi,vi].plot(thr_gum_list, params_mean[:,pi,vi])
            ax[pi,vi].plot(thr_gum_list, params_median[:,pi,vi])
            ax[pi,vi].fill_between(thr_gum_list,params_u95[:,pi,vi],params_l95[:,pi,vi],color='blue',alpha=0.1)
            ax[pi,vi].fill_between(thr_gum_list,params_u75[:,pi,vi],params_l25[:,pi,vi],color='blue',alpha=0.1)
            ax[pi,0].set_ylabel(p_name[pi])
    plt.savefig(f"{dir_out}/Conmul_param_a_b_vs_threshold.pdf", bbox_inches="tight")

    fig, ax = plt.subplots(2,2,sharex=True, figsize=(10,6),constrained_layout=True)
    fig.supxlabel("Gumbel threshold")
    fig.set_facecolor('white')
    p_name = ['a','b','$\mu$','$\sigma$']
    m_name = ['U|H', 'H|U']
    for vi in range(num_vars):
        ax[0,vi].set_title(m_name[vi])
        # ax[0,vi].set_title(var_name[vi])
        for pi in range(2,4):
            # ax[pi,vi].plot(thr_gum_list, params_mean[:,pi,vi])
            ax[pi-2,vi].plot(thr_gum_list, params_median[:,pi,vi])
            ax[pi-2,vi].fill_between(thr_gum_list,params_u95[:,pi,vi],params_l95[:,pi,vi],color='blue',alpha=0.1)
            ax[pi-2,vi].fill_between(thr_gum_list,params_u75[:,pi,vi],params_l25[:,pi,vi],color='blue',alpha=0.1)
            ax[pi-2,0].set_ylabel(p_name[pi])
    plt.savefig(f"{dir_out}/Conmul_param_mu_sigma_vs_threshold.pdf", bbox_inches="tight")