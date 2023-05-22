from __future__ import annotations
import cartopy.crs as ccrs
import cartopy
import cartopy.feature as cfeature
import numpy as np
from scipy.stats._continuous_distns import genpareto
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
from pathlib import Path
import mstmeclass as mc
from mstmeclass import MSTME, STM, GPPAR, G_F, G, Area
from cartopy.mpl.ticker import (
    LongitudeFormatter,
    LatitudeFormatter,
    LatitudeLocator,
    LongitudeLocator,
)

pos_color = plt.rcParams["axes.prop_cycle"].by_key()["color"]
print(Path("plot_style.txt").exists())
plt.style.use(Path("plot_style.txt"))


# plt.style.use("ggplot")
def create_custom_ticks(vmin, vmax, tick):
    ticks = []
    assert vmin < vmax
    start = (vmin + tick) // tick * tick
    end = -(vmax // -tick) * tick
    return [vmin] + list(np.arange(start, end, tick)) + [vmax]


def custom_map(
    ax,
    area: Area,
):
    ax.add_feature(cartopy.feature.LAND, edgecolor="black")
    ax.coastlines()
    ax.yaxis.tick_right()
    ax.set_xticks(
        create_custom_ticks(area.min_lon, area.max_lon, 0.5), crs=ccrs.PlateCarree()
    )
    ax.set_yticks(
        create_custom_ticks(area.min_lat, area.max_lat, 0.5), crs=ccrs.PlateCarree()
    )
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    return ax


class Grapher:
    def __init__(self, mstme: MSTME, **kwargs):
        self.mstme = mstme
        self.num = 10
        self.stm_min = [0, 0]
        self.stm_max = [30, 80]
        return

    def draw(self, fig_name: str, **kwargs):
        """
        Genpar_Params
        Genpar_CDF
        """
        draw_fig = kwargs.get("draw_fig", self.mstme.draw_fig)
        dir_out = kwargs.get("dir_out", self.mstme.dir_out)
        file_name = fig_name
        match fig_name:
            case "STM_Histogram":
                fig, ax = plt.subplots(
                    1,
                    self.mstme.num_vars,
                    figsize=(8 * self.mstme.num_vars, 6),
                    facecolor="white",
                )
                for S in STM:
                    vi = S.idx()
                    unit = S.unit()
                    var_name = S.name()
                    ax[vi].set_xlabel(f"{var_name}[{unit}]")
                    ax[vi].hist(
                        self.mstme.stm[vi],
                        bins=np.linspace(self.stm_min[vi], self.stm_max[vi], 20),
                    )

            case "STM_Histogram_filtered":
                _mask = self.mstme.mask
                fig, ax = plt.subplots(
                    2,
                    self.mstme.num_vars,
                    figsize=(8 * self.mstme.num_vars, 6 * 2),
                    facecolor="white",
                )
                # stm_min = np.floor(self.mstme.stm.min(axis=1) / 5) * 5
                # stm_max = np.ceil(self.mstme.stm.max(axis=1) / 5) * 5
                for S in STM:
                    vi = S.idx()
                    unit = S.unit()
                    var_name = S.name()
                    for i, b in enumerate([True, False]):
                        ax[i, vi].set_xlabel(f"{var_name[vi]}{unit[vi]}")
                        ax[i, vi].hist(
                            self.mstme.stm[vi, (_mask == b)],
                            bins=np.arange(self.stm_min[vi], self.stm_max[vi], 1),
                        )
                        ax[i, vi].set_title(f'{"is" if b else "not"} {self.mstme.rf}')

            case "STM_location":
                _mask = self.mstme.mask
                fig, ax = plt.subplots(
                    1,
                    self.mstme.num_vars,
                    figsize=(8 * self.mstme.num_vars, 6),
                    facecolor="white",
                )
                for S in STM:
                    vi = S.idx()

                    ax[vi].scatter(
                        self.mstme.latlon[:, 1],
                        self.mstme.latlon[:, 0],
                        c="black",
                        s=2,
                    )
                    ax[vi].scatter(
                        self.mstme.latlon[self.mstme.stm_node_idx[vi, _mask], 1],
                        self.mstme.latlon[self.mstme.stm_node_idx[vi, _mask], 0],
                        c="red",
                        s=20,
                        alpha=0.1,
                    )
                    ax[vi].scatter(
                        self.mstme.latlon[self.mstme.stm_node_idx[vi, ~_mask], 1],
                        self.mstme.latlon[self.mstme.stm_node_idx[vi, ~_mask], 0],
                        c="blue",
                        s=20,
                        alpha=0.1,
                    )

            case "Tracks_vs_STM":
                fig, ax = plt.subplots(
                    1,
                    self.mstme.num_vars,
                    subplot_kw={"projection": ccrs.PlateCarree()},
                    figsize=(8 * self.mstme.num_vars, 6),
                    facecolor="white",
                )
                for S in STM:
                    vi = S.idx()
                    ax[vi] = custom_map(ax[vi], self.mstme.area)
                    cmap = plt.get_cmap("viridis", 100)
                    for ei in range(self.mstme.num_events):
                        ax[vi].plot(
                            self.mstme.tracks[ei][:, 0],
                            self.mstme.tracks[ei][:, 1],
                            c=cmap(self.mstme.stm[vi, ei] / self.mstme.stm[vi].max()),
                            lw=10,
                            alpha=0.4,
                        )
                    ax[vi].coastlines(lw=5)
                    cax = fig.add_axes(
                        [
                            ax[vi].get_position().x1 + 0.01,
                            ax[vi].get_position().y0,
                            0.02,
                            ax[vi].get_position().height,
                        ]
                    )
                    sm = plt.cm.ScalarMappable(
                        cmap=cmap,
                        norm=plt.Normalize(
                            vmin=self.mstme.stm[vi].min(), vmax=self.mstme.stm[vi].max()
                        ),
                    )
                    plt.colorbar(sm, cax=cax)
                    gl = ax[vi].gridlines(draw_labels=True)
                    gl.top_labels = False
                    gl.right_labels = False
                    gl.xlines = False
                    gl.ylines = False
            # case "Mean_residual_life":
            #     # Mean residual life plot
            #     fig, ax = plt.subplots(
            #         1,
            #         self.num_vars,
            #         sharey=True,
            #         figsize=(8 * self.num_vars, 6),
            #         facecolor="white",
            #         squeeze=False,
            #     )
            #     # [N_THR, vi, 3]
            #     for S in STM:
            #         vi = S.idx()
            #         var_name = S.name()
            #         u = np.linspace(0, self.stm_max[vi], 25)
            #         me = []
            #         std = []
            #         for thr in u:
            #             excess = self.mstme.stm[vi, self.stm[vi] > thr]
            #             mean_excess = excess.mean().values.item() - thr
            #             _std = excess.std().values.item()
            #             me.append(mean_excess)
            #             std.append(_std)
            #         me=np.array(me)
            #         std=np.array(std)
            #         ax[0, vi].set_title(var_name)
            #         ax[0, vi].set_ylim(-1, 1)
            #         ax[0, 0].set_ylabel("Mean Excess")
            #         ax[0, vi].set_xlabel(f"Threshold[{S.unit()}]")
            #         ax[0, vi].plot(u, me)
            #         ax[0, vi].fill_between(
            #             u,
            #             me+std*2,
            #             me-std*2,
            #             alpha=0.5,
            #         )

            case "PWE_histogram_tm":
                fig, ax = plt.subplots(
                    1,
                    self.mstme.num_vars,
                    figsize=(8 * self.mstme.num_vars, 6),
                    facecolor="white",
                )

                ni = kwargs["idx_location"]
                for S in STM:
                    vi = S.idx()
                    var_name = S.name()
                    _ax: plt.Axes = ax[vi]
                    _ax.hist(self.mstme.tm[vi, :, ni], bins=20)
                    _ax.set_title(f"{var_name}")

            case "General_Map":
                fig, ax = plt.subplots(
                    1,
                    1,
                    figsize=(8, 6),
                    facecolor="white",
                    subplot_kw={"projection": ccrs.PlateCarree()},
                )
                ax = custom_map(ax, self.mstme.area)

                ax.scatter(
                    self.mstme.latlon[:, 1],
                    self.mstme.latlon[:, 0],
                    c="black",
                    s=3,
                )
                # idx = np.array(kwargs["idx_location"])
                idx = self.mstme.idx_pos_list
                for i, ni in enumerate(idx):
                    ax.scatter(
                        self.mstme.latlon[ni, 1],
                        self.mstme.latlon[ni, 0],
                        s=20,
                        c=pos_color[i],
                        label=f"Location #{i:d}",
                    )
                ax.legend()

            case "Genpar_Params":
                fig, ax = plt.subplots(
                    len(list(GPPAR)),
                    self.mstme.num_vars,
                    figsize=(8 * self.mstme.num_vars, 6 * len(list(GPPAR))),
                    facecolor="white",
                )

                for S in STM:
                    vi = S.idx()
                    var_name = S.name()
                    ax[0, vi].set_title(var_name)
                    for par in GPPAR:
                        pi = par.idx()
                        par_name = par.name()
                        ax[pi, 0].set_ylabel(par_name)
                        ax[pi, vi].hist(self.mstme.gp_params[vi, :, pi])

            case "Genpar_CDF":
                fig, ax = plt.subplots(
                    1,
                    self.mstme.num_vars,
                    figsize=(8 * self.mstme.num_vars, 6),
                    facecolor="white",
                )

                N_gp = self.mstme.gp_params.shape[1]
                _res = 100
                for S in STM:
                    vi = S.idx()
                    var_name = S.name()
                    unit = S.unit()
                    _cdf_all = np.zeros((N_gp, _res))
                    _x = np.linspace(
                        self.mstme.thr_mar[vi], self.mstme.stm[vi].max(), _res
                    )
                    for i in range(N_gp):
                        _xp = self.mstme.gp_params[vi, i, 0]
                        _mp = self.mstme.gp_params[vi, i, 1]
                        _sp = self.mstme.gp_params[vi, i, 2]
                        _cdf_all[i, :] = genpareto(_xp, _mp, _sp).cdf(_x)

                    _y = self.mstme.gp[vi].cdf(_x)
                    _u95 = np.percentile(_cdf_all, 97.5, axis=0)
                    _l95 = np.percentile(_cdf_all, 2.5, axis=0)
                    ax[vi].plot(_x, _y, c="blue", lw=2, alpha=1, label="Bootstrap Mean")
                    ax[vi].fill_between(_x, _u95, _l95, alpha=0.5, label="95% CI")
                    _ecdf = ECDF(self.mstme.stm[vi, self.mstme.is_e_mar[vi]])
                    _x = np.linspace(
                        self.mstme.thr_mar[vi], self.mstme.stm[vi].max(), _res
                    )
                    ax[vi].plot(_x, _ecdf(_x), lw=2, color="black", label="Empirical")
                    ax[vi].set_xlabel(f"{var_name}[{unit}]")
                    ax[vi].legend()

            case "Original_vs_Normalized":
                fig, ax = plt.subplots(
                    1,
                    2,
                    figsize=(7, 3),
                    facecolor="white",
                )
                ax[0].set_aspect(1)
                ax[0].scatter(self.mstme.stm_g[0], self.mstme.stm_g[1], s=5)
                ax[0].set_xlabel(r"$\hat H_s$")
                ax[0].set_ylabel(r"$\hat U$")
                ax[0].set_xlim(-5, 15)
                ax[0].set_ylim(-5, 15)
                ax[0].set_xticks([-2 + 2 * i for i in range(6)])
                ax[0].set_yticks([-2 + 2 * i for i in range(6)])

                ax[1].scatter(self.mstme.stm[0], self.mstme.stm[1], s=5)
                ax[1].set_xlabel(f"{STM.H.name()}[{STM.H.unit()}]")
                ax[1].set_ylabel(f"{STM.U.name()}[{STM.U.unit()}]")
                ax[1].set_xlim(0, 20)
                ax[1].set_ylim(0, 60)

            case "Kendall_Tau_all_var_pval":
                ###
                # Logic code
                if not hasattr(self.mstme, "pval"):
                    self.mstme.calc_kendall_tau()
                elif self.mstme.pval is None:
                    self.mstme.calc_kendall_tau()
                ###
                fig, ax = plt.subplots(
                    self.mstme.num_vars,
                    self.mstme.num_vars,
                    sharey=True,
                    figsize=(8 * self.mstme.num_vars, 6 * self.mstme.num_vars),
                    facecolor="white",
                    squeeze=False,
                )

                for Si in STM:
                    vi = Si.idx()
                    var_name_i = Si.name()
                    for Sj in STM:
                        vj = Sj.idx()
                        var_name_j = Sj.name()
                        ax[vi, vj].set_xlabel("Longitude")
                        ax[vi, vj].set_ylabel("Latitude")
                        _c = [
                            "red" if p < 0.05 else "black"
                            for p in self.mstme.pval[vi, vj, :]
                        ]
                        im = ax[vi, vj].scatter(
                            self.mstme.latlon[:, 1],
                            self.mstme.latlon[:, 0],
                            s=5,
                            c=_c,
                        )
                        ax[vi, vj].set_title(f"STM:{var_name_i} E:{var_name_j}")

            case "Kendall_Tau_all_var_tval":
                ###
                # Logic code
                if not hasattr(self.mstme, "tval"):
                    self.mstme.calc_kendall_tau()
                elif self.mstme.tval is None:
                    self.mstme.calc_kendall_tau()
                ###
                fig, ax = plt.subplots(
                    self.mstme.num_vars,
                    self.mstme.num_vars,
                    sharey=True,
                    figsize=(8 * self.mstme.num_vars, 6 * self.mstme.num_vars),
                    facecolor="white",
                    squeeze=False,
                )

                for Si in STM:
                    vi = Si.idx()
                    var_name_i = Si.name()
                    for Sj in STM:
                        vj = Sj.idx()
                        var_name_j = Sj.name()
                        ax[vi, vj].set_xlabel("Longitude")
                        ax[vi, vj].set_ylabel("Latitude")
                        im = ax[vi, vj].scatter(
                            self.mstme.latlon[:, 1],
                            self.mstme.latlon[:, 0],
                            s=5,
                            c=self.mstme.tval[vi, vj, :],
                            cmap="seismic",
                            vmax=np.abs(self.mstme.tval[vi]).max(),
                            vmin=-np.abs(self.mstme.tval[vi]).max(),
                        )
                        plt.colorbar(im, ax=ax[vi, vj])
                        ax[vi, vj].set_title(f"STM:{var_name_i} E:{var_name_j}")

            case "Kendall_Tau_marginal_pval":
                ###
                # Logic code
                if not hasattr(self.mstme, "pval"):
                    self.mstme.calc_kendall_tau()
                elif self.mstme.pval is None:
                    self.mstme.calc_kendall_tau()
                ###
                fig, ax = plt.subplots(
                    1,
                    self.mstme.num_vars,
                    sharey=True,
                    figsize=(8, 6 * self.mstme.num_vars),
                    facecolor="white",
                    squeeze=False,
                )

                for S in STM:
                    vi = S.idx()

                    ax[vi].set_xlabel("Longitude")
                    ax[vi].set_ylabel("Latitude")
                    _c = [
                        "red" if p < 0.05 else "black"
                        for p in self.mstme.pval[vi, vi, :]
                    ]
                    im = ax[vi].scatter(
                        self.mstme.latlon[:, 1],
                        self.mstme.latlon[:, 0],
                        s=5,
                        c=_c,
                    )
                    ax[vi].set_title(f"STM:{var_name_i} E:{var_name_j}")

            case "Kendall_Tau_marginal_tval":
                ###
                # Logic code
                if not hasattr(self.mstme, "tval"):
                    self.mstme.calc_kendall_tau()
                elif self.mstme.tval is None:
                    self.mstme.calc_kendall_tau()
                ###
                fig, ax = plt.subplots(
                    1,
                    self.mstme.num_vars,
                    sharey=True,
                    figsize=(8 * self.mstme.num_vars, 6 * self.mstme.num_vars),
                    facecolor="white",
                    squeeze=False,
                )

                for S in STM:
                    vi = S.idx()

                    ax[vi].set_xlabel("Longitude")
                    ax[vi].set_ylabel("Latitude")
                    im = ax[vi].scatter(
                        self.mstme.latlon[:, 1],
                        self.mstme.latlon[:, 0],
                        s=5,
                        c=self.mstme.tval[vi, vj, :],
                        cmap="seismic",
                        vmax=np.abs(self.mstme.tval[vi]).max(),
                        vmin=-np.abs(self.mstme.tval[vi]).max(),
                    )
                    ax[vi].set_title(f"STM:{var_name_i} E:{var_name_j}")

            case "Replacement":
                fig, ax = plt.subplots(
                    1,
                    1,
                    figsize=(8, 6),
                    facecolor="white",
                )

                ax.scatter(
                    self.mstme.stm_g_rep[:, 0, :],
                    self.mstme.stm_g_rep[:, 1, :],
                    label="Replacement",
                )
                ax.scatter(
                    self.mstme.stm_g[0],
                    self.mstme.stm_g[1],
                    color="black",
                    label="Original",
                )
                ax.set_xlabel(r"$\hat H_s$")
                ax.set_ylabel(r"$\hat U$")
                ax.set_xlim(-3, 15)
                ax.set_ylim(-3, 15)
                ax.legend(loc="upper left")

            case "Conmul_Estimates":
                fig, ax = plt.subplots(
                    4,
                    self.mstme.num_vars,
                    figsize=(8 * self.mstme.num_vars, 6 * 4),
                    facecolor="white",
                )

                fig.tight_layout()
                ax[0, 0].set_ylabel("a")
                ax[1, 0].set_ylabel("b")
                ax[2, 0].set_ylabel("$\mu$")
                ax[3, 0].set_ylabel("$\sigma$")
                ax[3, 0].set_xlabel(STM.H.name())
                ax[3, 1].set_xlabel(STM.U.name())
                for S in STM:
                    vi = S.idx()

                    ax[0, vi].hist(self.mstme.params_uc[vi, :, 0])
                    ax[1, vi].hist(self.mstme.params_uc[vi, :, 1])
                    ax[2, vi].hist(self.mstme.params_uc[vi, :, 2])
                    ax[3, vi].hist(self.mstme.params_uc[vi, :, 3])

            case "ab_Estimates":
                fig, ax = plt.subplots(
                    1,
                    self.mstme.num_vars,
                    figsize=(8 * self.mstme.num_vars, 6),
                    facecolor="white",
                )

                fig.supxlabel("$a$")
                fig.supylabel("$b$")
                params_ml = np.zeros((4, self.mstme.num_vars))
                for S in STM:
                    vi = S.idx()
                    var_name = S.name()
                    ax[vi].set_xlim(0, 1)
                    ax[vi].set_ylim(-1, 1)
                    ax[vi].scatter(
                        self.mstme.params_uc[vi, :, 0],
                        self.mstme.params_uc[vi, :, 1],
                        s=10,
                        label="Generated samples",
                    )
                    a_hat = self.mstme.params_median[vi, 0]
                    b_hat = self.mstme.params_median[vi, 1]
                    ax[vi].scatter(
                        a_hat,
                        b_hat,
                        s=40,
                        c="red",
                    )
                    ax[vi].text(
                        a_hat,
                        b_hat + 0.05,
                        f"$(\hat a, \hat b)$",
                        fontsize=30,
                        ha="center",
                        c="red",
                    )
                    ax[vi].set_title(var_name)

            case "amu_Estimates":
                fig, ax = plt.subplots(
                    1,
                    self.mstme.num_vars,
                    figsize=(8 * self.mstme.num_vars, 6),
                    facecolor="white",
                )

                fig.supxlabel("$a$")
                fig.supylabel("$\mu$")
                params_ml = np.zeros((4, self.mstme.num_vars))
                for S in STM:
                    vi = S.idx()
                    var_name = S.name()
                    ax[vi].set_xlim(0, 1)
                    ax[vi].set_ylim(-0.1, 2)
                    ax[vi].scatter(
                        self.mstme.params_uc[vi, :, 0],
                        self.mstme.params_uc[vi, :, 2],
                        s=5,
                        label="Generated samples",
                    )
                    ax[vi].set_title(var_name)
            case "a+mub_Estimates":
                fig, ax = plt.subplots(
                    1,
                    self.mstme.num_vars,
                    figsize=(8 * self.mstme.num_vars, 6),
                    facecolor="white",
                )

                fig.supxlabel("$a+\mu$")
                fig.supylabel("$b$")
                params_ml = np.zeros((4, self.mstme.num_vars))
                for S in STM:
                    vi = S.idx()
                    var_name = S.name()
                    ax[vi].set_xlim(0.5, 2)
                    ax[vi].set_ylim(-1, 1)
                    ax[vi].scatter(
                        self.mstme.params_uc[vi, :, 0] + self.mstme.params_uc[vi, :, 2],
                        self.mstme.params_uc[vi, :, 1],
                        s=5,
                        label="Generated samples",
                    )
                    ax[vi].set_title(var_name)

            case "Residuals":
                fig, ax = plt.subplots(
                    1,
                    self.mstme.num_vars,
                    figsize=(8 * self.mstme.num_vars, 6),
                    facecolor="white",
                )

                # fig.tight_layout()
                for S in STM:
                    vi = S.idx()
                    var_name = S.name()
                    ax[vi].scatter(
                        self.mstme.ndist.cdf(self.mstme.stm_g[vi, self.mstme.is_e[vi]]),
                        self.mstme.residual[vi],
                        s=5,
                    )
                    ax[vi].set_xlabel(f"$F^*$({var_name}$)$")
                ax[0].set_ylabel("$Z_{-j}$")

            case "Simulated_Conmul_vs_Back_Transformed":
                fig, ax = plt.subplots(
                    1,
                    self.mstme.num_vars,
                    figsize=(8 * self.mstme.num_vars, 6),
                    facecolor="white",
                )

                ax[0].set_aspect(1)
                a_h, b_h, mu_h, sg_h = self.mstme.params_median[0, :]
                a_u, b_u, mu_u, sg_u = self.mstme.params_median[1, :]
                sample_given_h = []
                sample_given_u = []
                sample_given_hg = []
                sample_given_ug = []
                for i, vi in enumerate(self.mstme.vi_list):
                    if vi == 0:
                        sample_given_h.append(self.mstme.sample_full[:, i])
                        sample_given_hg.append(self.mstme.sample_full_g[:, i])
                    if vi == 1:
                        sample_given_u.append(self.mstme.sample_full[:, i])
                        sample_given_ug.append(self.mstme.sample_full_g[:, i])
                sample_given_h = np.array(sample_given_h).T
                sample_given_u = np.array(sample_given_u).T
                sample_given_hg = np.array(sample_given_hg).T
                sample_given_ug = np.array(sample_given_ug).T

                x_h = np.linspace(self.mstme.thr_com, 10, 100)
                y_h = x_h * a_h + (x_h**b_h) * mu_h
                ax[0].plot(
                    x_h, y_h, color="orange", label="$\hat{U}=a\hat{H}+\mu\hat{H}^b$"
                )

                y_u = np.linspace(self.mstme.thr_com, 10, 100)
                x_u = y_u * a_u + (y_u**b_u) * mu_u
                ax[0].plot(
                    x_u, y_u, color="teal", label="$\hat{H}=a\hat{U}+\mu\hat{U}^b$"
                )

                ax[0].scatter(
                    self.mstme.stm_g[0],
                    self.mstme.stm_g[1],
                    s=5,
                    color="black",
                    label="Original",
                )
                ax[0].axvline(self.mstme.thr_com, color="black")
                ax[0].axhline(self.mstme.thr_com, color="black")

                ax[0].set_xlabel(r"$\hat H_s$")
                ax[0].set_ylabel(r"$\hat U$")
                ax[0].set_xlim(-2, 10)
                ax[0].set_ylim(-2, 10)
                ax[0].scatter(
                    sample_given_hg[0],
                    sample_given_hg[1],
                    s=1,
                    color="orange",
                    label="Simulated $(\hat{U}|\hat{H}>\hat{\mu})$",
                )
                ax[0].scatter(
                    sample_given_ug[0],
                    sample_given_ug[1],
                    s=1,
                    color="teal",
                    label="Simulated $(\hat{H}|\hat{U}>\hat{\mu})$",
                )
                ax[0].legend()

                ax[1].set_xlim(0, 25)
                ax[1].set_ylim(0, 60)
                ax[1].scatter(
                    self.mstme.stm[0],
                    self.mstme.stm[1],
                    color="black",
                    s=5,
                    label="Original",
                )
                ax[1].scatter(
                    sample_given_h[0],
                    sample_given_h[1],
                    color="orange",
                    s=1,
                    label="Simulated $(U|H>\mu_{H_s})$",
                )
                ax[1].scatter(
                    sample_given_u[0],
                    sample_given_u[1],
                    color="teal",
                    s=1,
                    label="Simulated $(H|U>\mu_{U_{10}})$",
                )
                ax[1].set_xlabel(f"{STM.H.name()}[{STM.H.unit()}]")
                ax[1].set_ylabel(f"{STM.U.name()}[{STM.U.unit()}]")

                # original
                return_period = 100

                _count_original = round(
                    self.mstme.num_events / (return_period * self.mstme.occur_freq)
                )
                _ic_original = mc._search_isocontour(self.mstme.stm, _count_original)

                # sample
                _num_events_sample = self.mstme.sample_full.shape[1]
                _exceedance_prob = 1 - self.mstme.thr_pct_com
                _count_sample = round(
                    _num_events_sample
                    / (return_period * self.mstme.occur_freq * _exceedance_prob)
                )
                _ic_sample = mc._search_isocontour(
                    self.mstme.sample_full, _count_sample
                )

                ax[1].plot(
                    _ic_original[0],
                    _ic_original[1],
                    c="black",
                    lw=2,
                    label=f"Empirical {return_period}-yr RV",
                )
                ax[1].plot(
                    _ic_sample[0],
                    _ic_sample[1],
                    c="red",
                    lw=2,
                    label=f"Simulated {return_period}-yr RV",
                )
                ax[1].legend()

            case "RV":
                fig, axes = plt.subplots(
                    2,
                    2,
                    figsize=(8 * 2, 6 * 2),
                    facecolor="white",
                )

                return_period = kwargs["return_period"]
                file_name = file_name + f"_RP{return_period}"
                tm_sample = self.mstme.tm_sample  # (v,e,n)
                tm_original = self.mstme.tm  # (v,e,n)
                # stm_min = np.floor(tm_sample[:, :, self.mstme.idx_pos_list].min(axis=(1, 2)) / 5) * 5
                # stm_max = np.ceil(tm_sample[:, :, self.mstme.idx_pos_list].max(axis=(1, 2)) / 5) * 5
                #########################################################
                fig.supxlabel(r"$H_s$[m]")
                fig.supylabel(r"$U$[m/s]")
                for i, ax in enumerate(axes.flatten()):
                    ax.set_xlim(self.stm_min[0], self.stm_max[0])
                    ax.set_ylim(self.stm_min[1], self.stm_max[1])
                    _linestyles = ["-", "--"]
                    _idx_pos = self.mstme.idx_pos_list[i]
                    # sample
                    _num_events_sample = tm_sample.shape[1]
                    _exceedance_prob = 1 - self.mstme.thr_pct_com
                    _count_sample = round(
                        _num_events_sample
                        / (return_period * self.mstme.occur_freq * _exceedance_prob)
                    )
                    _ic_sample = mc._search_isocontour(
                        tm_sample[:, :, _idx_pos], _count_sample
                    )

                    # original
                    _count_original = round(
                        self.mstme.num_events / (return_period * self.mstme.occur_freq)
                    )
                    _ic_original = mc._search_isocontour(
                        tm_original[:, :, _idx_pos], _count_original
                    )

                    ax.scatter(
                        tm_original[0, :, _idx_pos],
                        tm_original[1, :, _idx_pos],
                        s=10,
                        c="black",
                        label=f"Original temporal maxima",
                    )
                    ax.scatter(
                        tm_sample[0, :, _idx_pos],
                        tm_sample[1, :, _idx_pos],
                        s=2,
                        c=pos_color[i],
                        label=f"Simulated temporal maxima(MSTM-E)",
                    )
                    ax.plot(
                        _ic_original[0],
                        _ic_original[1],
                        c="black",
                        lw=2,
                        label=f"Empirical {return_period}-yr RV",
                    )
                    ax.plot(
                        _ic_sample[0],
                        _ic_sample[1],
                        c=pos_color[i],
                        lw=2,
                        label=f"Simulated {return_period}-yr RV(MSTM-E)",
                    )
                    ax.set_title(f"Location {i+1}")
                    ax.legend()

            case "RV_PWE":
                fig, axes = plt.subplots(
                    2,
                    2,
                    figsize=(8 * 2, 6 * 2),
                    facecolor="white",
                )

                # tm_sample(#ofLoc(=4), num_vars, num_events)
                return_period = kwargs["return_period"]
                file_name = file_name + f"_RP{return_period}"
                tm_sample = self.mstme.tm_sample_PWE  # (v,e,n)
                tm_original = self.mstme.tm_original_PWE  # (v,e,n)
                #########################################################
                fig.supxlabel(r"$H_s$[m]")
                fig.supylabel(r"$U$[m/s]")
                for i, ax in enumerate(axes.flatten()):
                    ax.set_xlim(self.stm_min[0], self.stm_max[0])
                    ax.set_ylim(self.stm_min[1], self.stm_max[1])
                    _linestyles = ["-", "--"]
                    # sample
                    _num_events_sample = tm_sample.shape[1]
                    _exceedance_prob = 1 - self.mstme.thr_pct_com
                    _count_sample = round(
                        _num_events_sample
                        / (return_period * self.mstme.occur_freq * _exceedance_prob)
                    )
                    _ic_sample = mc._search_isocontour(
                        tm_sample[:, :, i], _count_sample
                    )

                    # original
                    _ic_original = []
                    _count_original = round(
                        self.mstme.num_events / (return_period * self.mstme.occur_freq)
                    )
                    _ic_original = mc._search_isocontour(
                        tm_original[:, :, i], _count_original
                    )
                    ax.scatter(
                        tm_original[0, :, i],
                        tm_original[1, :, i],
                        s=10,
                        c="black",
                        label=f"Original temporal maxima",
                    )
                    ax.scatter(
                        tm_sample[0, :, i],
                        tm_sample[1, :, i],
                        s=2,
                        c=pos_color[i],
                        label=f"Simulated temporal maxima(PWE)",
                    )
                    ax.plot(
                        _ic_original[0],
                        _ic_original[1],
                        c="black",
                        lw=2,
                        label=f"Empirical {return_period}-yr RV",
                    )
                    ax.plot(
                        _ic_sample[0],
                        _ic_sample[1],
                        c=pos_color[i],
                        lw=2,
                        label=f"Simulated {return_period}-yr RV(PWE)",
                    )
                    ax.set_title(f"Location {i+1}")
                    ax.legend()

            case "RV_STM":
                stm_MSTME_ss = self.mstme.stm_MSTME_ss
                return_period = kwargs["return_period"]
                file_name = file_name + f"_RP{return_period}"
                N_subsample = stm_MSTME_ss.shape[0]
                # bi, vi, ei
                fig, ax = plt.subplots(
                    1,
                    1,
                    figsize=(8, 6),
                    facecolor="white",
                )
                fig.supxlabel(r"$H_s$[m]")
                fig.supylabel(r"$U$[m/s]")
                ax.set_xlim(self.stm_min[0], self.stm_max[0])
                ax.set_ylim(self.stm_min[1], self.stm_max[1])
                # Sample count over threshold
                _num_events_sample = stm_MSTME_ss.shape[2]
                _exceedance_prob = 1 - self.mstme.thr_pct_com
                _count_sample = round(
                    _num_events_sample
                    / (return_period * self.mstme.occur_freq * _exceedance_prob)
                )
                _num_events_original = self.mstme.num_events
                _count_original = round(
                    _num_events_original / (return_period * self.mstme.occur_freq)
                )

                # Bootstraps
                _ic_MSTME = []
                for bi in range(N_subsample):
                    _ic = mc._search_isocontour(stm_MSTME_ss[bi, :, :], _count_sample)
                    _ic[1, 0] = 0
                    _ic[0, -1] = 0
                    _ic_MSTME.append(_ic)

                # Original
                _ic_original = mc._search_isocontour(
                    self.mstme.stm[:, :], _count_original
                )

                (
                    _ic_band_MSTME_u,
                    _ic_band_MSTME_l,
                    _ic_band_MSTME_m,
                ) = mc._get_interp_band(
                    _ic_MSTME, scale=self.stm_max[1] / self.stm_max[0]
                )

                array = np.concatenate(
                    (_ic_band_MSTME_u, np.flip(_ic_band_MSTME_l, axis=1)), axis=1
                )
                ax.fill(array[0], array[1], alpha=0.5)

                ######################################
                ax.scatter(
                    self.mstme.stm[0, :],
                    self.mstme.stm[1, :],
                    s=10,
                    c="black",
                    label=f"Original",
                    marker="x",
                )
                ax.plot(
                    _ic_original[0],
                    _ic_original[1],
                    c="black",
                    lw=2,
                )

            case "RV_ALL":
                tm_original = np.moveaxis(
                    self.mstme.tm[:, :, self.mstme.idx_pos_list].to_numpy(), 2, 0
                )
                tm_MSTME_ss = self.mstme.tm_MSTME_ss
                tm_PWE_ss = self.mstme.tm_PWE_ss
                return_period = kwargs["return_period"]
                file_name = file_name + f"_RP{return_period}"

                # bi, ni, vi, ei
                assert tm_MSTME_ss.shape == tm_PWE_ss.shape
                N_subsample = tm_MSTME_ss.shape[0]
                #########################################################
                fig, axes = plt.subplots(
                    2,
                    2,
                    figsize=(8 * 2, 6 * 2),
                    facecolor="white",
                )
                fig.supxlabel(r"$H_s$[m]")
                fig.supylabel(r"$U$[m/s]")
                for i, ax in enumerate(axes.flatten()):
                    ax.set_xlim(self.stm_min[0], self.stm_max[0])
                    ax.set_ylim(self.stm_min[1], self.stm_max[1])
                    # Sample count over threshold
                    _num_events_sample = tm_MSTME_ss.shape[2]
                    _exceedance_prob = 1 - self.mstme.thr_pct_com
                    _count_sample = round(
                        _num_events_sample
                        / (return_period * self.mstme.occur_freq * _exceedance_prob)
                    )
                    _ic_original = []
                    _num_events_original = tm_original.shape[2]
                    _count_original = round(
                        _num_events_original / (return_period * self.mstme.occur_freq)
                    )

                    # Bootstraps
                    ic_MSTME = []
                    ic_PWE = []
                    for bi in range(N_subsample):
                        _ic_MSTME = mc._search_isocontour(
                            tm_MSTME_ss[bi, :, :, i], _count_sample
                        )
                        _ic_PWE = mc._search_isocontour(
                            tm_PWE_ss[bi, :, :, i], _count_sample
                        )
                        _ic_MSTME[1, 0] = 0
                        _ic_MSTME[0, -1] = 0
                        _ic_PWE[1, 0] = 0
                        _ic_PWE[0, -1] = 0
                        ic_MSTME.append(_ic_MSTME)
                        ic_PWE.append(_ic_PWE)
                    (
                        ic_band_MSTME_u,
                        ic_band_MSTME_l,
                        ic_band_MSTME_m,
                    ) = mc._get_interp_band(
                        ic_MSTME, scale=self.stm_max[1] / self.stm_max[0]
                    )
                    ic_band_PWE_u, ic_band_PWE_l, ic_band_PWE_m = mc._get_interp_band(
                        ic_PWE, scale=self.stm_max[1] / self.stm_max[0]
                    )

                    _fill_MSTME = np.concatenate(
                        (ic_band_MSTME_u, np.flip(ic_band_MSTME_l, axis=1)), axis=1
                    )
                    _fill_PWE = np.concatenate(
                        (ic_band_PWE_u, np.flip(ic_band_PWE_l, axis=1)), axis=1
                    )
                    ax.fill(
                        _fill_MSTME[0],
                        _fill_MSTME[1],
                        alpha=0.2,
                        label=f"MSTME {return_period}-yr RV 95%CI",
                    )
                    ax.fill(
                        _fill_PWE[0],
                        _fill_PWE[1],
                        alpha=0.2,
                        label=f"PWE {return_period}-yr RV 95%CI",
                    )

                    # Original
                    _ic_original = mc._search_isocontour(
                        tm_original[i, :, :], _count_original
                    )

                    ax.scatter(
                        tm_original[i, 0, :],
                        tm_original[i, 1, :],
                        s=10,
                        c="black",
                        label=f"Original temporal maxima",
                        marker="x",
                    )
                    ax.plot(
                        _ic_original[0],
                        _ic_original[1],
                        c="black",
                        lw=2,
                        label=f"Empirical {return_period}-yr RV",
                    )
                    ax.set_title(f"Location {i+1}")
                    if i == 0:
                        ax.legend()

            case "RV_MAP":
                grid_res = 10
                min_lat, min_lon = np.min(self.mstme.latlon, axis=0)
                max_lat, max_lon = np.max(self.mstme.latlon, axis=0)
                lat_list = np.linspace(min_lat, max_lat, grid_res)
                lon_list = np.linspace(min_lon, max_lon, grid_res)
                dist_list, pos_list = self.mstme.tree.query(
                    [[[lat, lon] for lat in lat_list] for lon in lon_list]
                )
                pos_list = pos_list.flatten()
                tm_MSTME_ss = self.mstme.tm_MSTME_ss
                tm_PWE_ss = self.mstme.tm_PWE_ss
                # bi, ni, vi, ei
                assert tm_MSTME_ss.shape == tm_PWE_ss.shape
                return_period = kwargs["return_period"]
                file_name = file_name + f"_RP{return_period}"

                N_subsample = tm_MSTME_ss.shape[0]
                #########################################################
                fig, axes = plt.subplots(
                    2,
                    2,
                    figsize=(8 * 2, 6 * 2),
                    facecolor="white",
                )
                ax.set_xlim(self.stm_min[0], self.stm_max[0])
                ax.set_ylim(self.stm_min[1], self.stm_max[1])
                fig.supxlabel(r"$H_s$[m]")
                fig.supylabel(r"$U$[m/s]")
                for ni in pos_list:
                    tm_original = np.moveaxis(self.mstme.tm[:, :, ni].to_numpy(), 2, 0)
                    # Sample count over threshold
                    _num_events_sample = tm_MSTME_ss.shape[2]
                    _exceedance_prob = 1 - self.mstme.thr_pct_com
                    _count_sample = round(
                        _num_events_sample
                        / (return_period * self.mstme.occur_freq * _exceedance_prob)
                    )
                    _ic_original = []
                    _num_events_original = tm_original.shape[2]
                    _count_original = round(
                        _num_events_original / (return_period * self.mstme.occur_freq)
                    )

                    # Bootstraps
                    ic_MSTME = []
                    ic_PWE = []
                    for bi in range(N_subsample):
                        _ic_MSTME = mc._search_isocontour(
                            tm_MSTME_ss[bi, :, :, i], _count_sample
                        )
                        _ic_PWE = mc._search_isocontour(
                            tm_PWE_ss[bi, :, :, i], _count_sample
                        )
                        _ic_MSTME[1, 0] = 0
                        _ic_MSTME[0, -1] = 0
                        _ic_PWE[1, 0] = 0
                        _ic_PWE[0, -1] = 0
                        ic_MSTME.append(_ic_MSTME)
                        ic_PWE.append(_ic_PWE)
                    (
                        ic_band_MSTME_u,
                        ic_band_MSTME_l,
                        ic_band_MSTME_m,
                    ) = mc._get_interp_band(
                        ic_MSTME, scale=self.stm_max[1] / self.stm_max[0]
                    )
                    ic_band_PWE_u, ic_band_PWE_l, ic_band_PWE_m = mc._get_interp_band(
                        ic_PWE, scale=self.stm_max[1] / self.stm_max[0]
                    )

                    # Original
                    _ic_original = mc._search_isocontour(
                        tm_original[i, :, :], _count_original
                    )

            case "Equivalent_fetch":
                V_max_track = self.mstme.ds.V_max
                V_max_ww3 = self.mstme.ds.STM_UV_10m * G_F
                Vfm = self.mstme.ds.Vfm
                Radius = self.mstme.ds.Radius
                fetch_from_track = mc._calc_eq_fetch(V_max_track, Vfm, r=Radius)
                fetch_from_WW3 = G * (self.mstme.stm[0] / (0.0016 * V_max_ww3)) ** 2

                idx_in_range_ww3 = (
                    (V_max_ww3 > 20) & (V_max_ww3 < 60) & (Vfm > 0) & (Vfm < 12)
                )
                idx_in_range_track = (
                    (V_max_track > 20) & (V_max_track < 60) & (Vfm > 0) & (Vfm < 12)
                )
                fig, ax = plt.subplots(
                    1,
                    1,
                    figsize=(8, 6),
                    facecolor="white",
                )
                ax.set_ylabel("Equivalent fetch $x$[m]")
                ax.set_xlabel("$V_{max}$[m/s]")
                ax.scatter(
                    V_max_track[idx_in_range_track],
                    fetch_from_track[idx_in_range_track],
                    marker="o",
                    s=1,
                    label="from Young's definition",
                )
                ax.scatter(
                    V_max_ww3[idx_in_range_ww3],
                    fetch_from_WW3[idx_in_range_ww3],
                    marker="^",
                    s=1,
                    label="from JONSWAP relationship",
                )

                V_max_sample = self.mstme.stm_sample[1] * G_F
                idx_in_range_sample = (V_max_sample > 20) & (V_max_sample < 60)
                fetch_from_WW3_sample = (
                    G * (self.mstme.stm_sample[0] / (0.0016 * V_max_sample)) ** 2
                )
                ax.scatter(
                    V_max_sample[idx_in_range_sample],
                    fetch_from_WW3_sample[idx_in_range_sample],
                    marker="s",
                    s=1,
                    label="from JONSWAP relationship, MSTM-E sampled STM",
                )
                ax.legend()

            case _:
                raise (ValueError(f"No figure defined with the name {fig_name}"))

        if dir_out != None:
            plt.savefig(f"{dir_out}/{file_name}.pdf", bbox_inches="tight")
            plt.savefig(f"{dir_out}/{file_name}.png", bbox_inches="tight")
        if not draw_fig:
            plt.close()
