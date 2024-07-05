fig, ax = plt.subplots(1, 2, figsize=(4 * 2, 3), facecolor="white")
for S in STM:
    vi = S.idx()
    stm = mstme_condition.stm[vi, mstme_condition.is_e_mar[vi]]
    x = np.linspace(mstme_condition.thr_mar[vi], stm.max(), 1000)
    ax[vi].plot(
        x,
        1
        - genpareto(
            c=genpar_params_u95[vi][0],
            loc=genpar_params_u95[vi][1],
            scale=genpar_params_u95[vi][2],
        ).cdf(x),
        lw=2,
        linestyle="--",
        c=pos_color[0],
        label="Subsample 95th percentile",
    )
    ax[vi].plot(
        x,
        1
        - genpareto(
            c=genpar_params_l95[vi][0],
            loc=genpar_params_l95[vi][1],
            scale=genpar_params_l95[vi][2],
        ).cdf(x),
        lw=2,
        linestyle="--",
        c=pos_color[0],
    )
    ax[vi].plot(
        x, 1 - mstme_condition.gp[vi].cdf(x), lw=2, c=pos_color[0], label="All events"
    )
    ecdf = ECDF(stm)
    ax[vi].scatter(
        stm, 1 - ecdf(stm), c="k", marker="D", s=3, zorder=10, label="Sample"
    )
    ax[vi].grid(which="major")
    # ax[vi].grid(which='minor')
    ax[vi].set_xlabel(f"{S.name()}")
    ax[vi].set_yscale("log")
    ax[vi].set_ylim(bottom=1e-3, top=1.5)
ax[0].set_ylabel("$1-CDF$")
# ax[0].legend()

plt.savefig(dir_out / f"GPD_subsample.png", bbox_inches="tight")
plt.savefig(dir_out / f"GPD_subsample.pdf", bbox_inches="tight")
