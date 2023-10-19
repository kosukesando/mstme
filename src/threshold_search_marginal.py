# %%
# init
import matplotlib.pyplot as plt
import numpy as np
import openturns as ot
from scipy.stats._continuous_distns import genpareto

rng = np.random.default_rng()
plt.style.use("plot_style.txt")
depth = -100
dir_out = "./output/common"


def search_marginal(stm, thr_start, thr_end):
    num_vars = stm.shape[0]
    num_events = stm.shape[1]
    # Generalized Pareto estimation over threshold range
    N_gp = 100
    N_THR = 10
    thr_list = np.linspace(thr_start, thr_end, N_THR)
    genpar_params = np.zeros((N_THR, num_vars, N_gp, 3))
    num_samples = np.zeros((N_THR, num_vars, N_gp))
    for ti, _thr in enumerate(thr_list):
        for vi in range(num_vars):
            _stm_bootstrap = rng.choice(stm, size=(N_gp, num_events))
            for i in range(N_gp):
                _stm = _stm_bootstrap[i]
                _stm_pot = _stm[_stm > _thr[vi]]
                _xp, _mp, _sp = genpareto.fit(_stm_pot, floc=_thr[vi])
                genpar_params[ti, vi, i, :] = [_xp, _mp, _sp]
                num_samples[ti, vi, i] = np.count_nonzero(_stm_pot)
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
        ax[0, vi].plot(thr_list[vi], med[:, 0, vi])
        ax[0, vi].fill_between(thr_list[vi], u95[:, 0, vi], l95[:, 0, vi], alpha=0.5)

    ax[0, 0].set_xlabel("Threshold[m]")
    plt.savefig(f"{dir_out}/Marginal_param_vs_threshold.pdf", bbox_inches="tight")
    plt.savefig(f"{dir_out}/Marginal_param_vs_threshold.png", bbox_inches="tight")
