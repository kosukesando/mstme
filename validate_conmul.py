# %%
# init
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
from scipy.stats import kendalltau
from scipy.stats import norm
from scipy.spatial import KDTree
from statsmodels.distributions.empirical_distribution import ECDF
from statsmodels.distributions.empirical_distribution import monotone_fn_inverter
from traitlets import Bool
import sys
import xarray as xr
# Custom
import stme
import src.threshold_search as threshold_search

plt.style.use("plot_style.txt")
rng = np.random.default_rng()

# %%
thr_gum = 2.0
num_events = 500
num_vars = 2
# %%
# Generate pseudo data
param_true = [0.5, 0.5, 0.5, 1]
_x = genextreme.rvs(0, size=num_events)
_z = norm.rvs(loc=param_true[2], scale=param_true[3], size=_x.shape[0])
# _x = _x[_x>thr_gum]
_y = _x*param_true[0]+(param_true[2]*_x**param_true[1])*_z
is_e = _x > thr_gum
stm_g = np.array([_x, _y])
plt.scatter(_x, _y)

# %%


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

    # plt.scatter(x,y)
    if y.ndim < 2:
        y = np.expand_dims(y, axis=0)
    for vj in range(y.shape[0]):
        q += sum(
            np.log(sg * x ** b)
            + 0.5
            * ((y[vj] - (a * x + mu * x ** b)) / (sg * x ** b))
            ** 2
        )
    return q
# def cost(p, data, vi):
#     """
#     cost(p,data,vi)->float
#     p: parameter; [a,b,mu,sigma]
#     data: ndarray with shape(num_vars, num_events)
#     vi: Index of extreme variable
#     minimize this.
#     """
#     q = 0
#     a = p[0]
#     b = p[1]
#     mu = p[2]
#     sg = p[3]

#     x = np.asarray(data[vi])  # conditioning
#     y = np.asarray(np.delete(data, vi, axis=0))  # conditioned
#     # plt.scatter(x,y)
#     if y.ndim < 2:
#         y = np.expand_dims(y, axis=0)
#     for vj in range(y.shape[0]):
#         q += sum(
#             np.log(sg * x ** b)
#             + 0.5
#             * ((y[vj] - (a * x + mu * x ** b)) / (sg * x ** b))
#             ** 2
#         )
#     # for vi in range(y.shape[0]):
#     #     q += sum(
#     #         np.log(sg * x ** b)
#     #         + 0.5
#     #         * ((y[vi] - (a * x + mu * x ** b)) / (sg * x ** b))
#     #         ** 2
#     #     )
#     return q


def jacobian_custom(p, x, y):
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
    # return np.array([0., 0., 0., 0.])


# def hessian_custom(p, x, y):
#     a = p[0]
#     b = p[1]
#     mu = p[2]
#     sg = p[3]
#     # e00 = np.sum((x ** (2 - 2*b))/sg ** 2)
#     # e01 = np.sum((mu * x ** (1 - b) * np.log(x))/sg ** 2 +
#     #              (2 * x ** (1 - 2*b) * (y - a*x - mu*x ** b)*np.log(x))/sg ** 2)
#     # e02 = np.sum((x ** (1 - b))/sg ** 2)
#     # e03 = np.sum((2 * x ** (1 - 2*b)*(y - a*x - mu*x ** b))/sg ** 3)
#     # # e11 =
#     # print(np.sum((x ** (2 - 2*b))/sg ** 2))
#     arr = np.array(
#         [
#             [
#                 np.sum((x ** (2 - 2*b))/sg ** 2),
#                 np.sum((mu * x ** (1 - b) * np.log(x))/sg ** 2 +
#                        (2 * x ** (1 - 2*b) * (y - a*x - mu*x ** b)*np.log(x))/sg ** 2),
#                 np.sum((x ** (1 - b))/sg ** 2),
#                 np.sum((2 * x ** (1 - 2*b)*(y - a*x - mu*x ** b))/sg ** 3)
#             ],

#             [
#                 np.sum((mu * x ** (1 - b) * np.log(x))/sg ** 2 +
#                        (2 * x ** (1 - 2*b) *
#                         (y - a*x - mu*x ** b)*np.log(x))/sg ** 2),
#                 np.sum(sg * x ** b * np.log(x) ** 2 + (2 * (y - a*x - mu*x ** b) ** 2 * np.log(x) ** 2)/(x ** (2 * b) * sg ** 2) + (
#                     3 * mu * (y - a*x - mu*x ** b) * np.log(x) ** 2)/(x ** b * sg ** 2) + (mu ** 2 * np.log(x) ** 2)/sg ** 2),
#                 np.sum((mu * np.log(x))/sg ** 2 +
#                        ((y - a*x - mu*x ** b) * np.log(x)) /
#                        (x ** b * sg ** 2)),
#                 np.sum(x ** b * np.log(x) + (2 * mu * (y - a*x - mu*x ** b) * np.log(x))/(x ** b * sg ** 3) + (
#                     2 * (y - a*x - mu*x ** b) ** 2 * np.log(x))/(x ** (2 * b) * sg ** 3))
#             ],

#             [
#                 np.sum((x ** (1 - b))/sg ** 2),
#                 np.sum((mu * np.log(x))/sg ** 2 +
#                        ((y - a*x - mu*x ** b) * np.log(x))/(x ** b * sg ** 2)),
#                 np.sum(1 / sg ** 2),
#                 np.sum((2 * (y - a*x - mu*x ** b))/(x ** b * sg ** 3))
#             ],

#             [
#                 np.sum((2 * x ** (1 - 2*b)*(y - a*x - mu*x ** b))/sg ** 3),
#                 np.sum(x ** b * np.log(x) + (2 * mu * (y - a*x - mu*x ** b) * np.log(x))/(x ** b * sg ** 3) + (
#                     2 * (y - a*x - mu*x ** b) ** 2 * np.log(x))/(x ** (2 * b) * sg ** 3)),
#                 np.sum((2 * (y - a*x - mu*x ** b))/(x ** b * sg ** 3)),
#                 np.sum((3 * (y - a*x - mu*x ** b) ** 2) /
#                        (x ** (2 * b) * sg ** 4))
#             ]
#         ]
#     )
#     return arr


##################################################################################################################
# %%
# Estimate conditional model parameters
# importlib.reload(stme)
# methods = ['Nelder-Mead', 'L-BFGS-B', 'Powell', 'trust-constr']
methods = ['trust-constr']
# methods = ['dogleg']
# methods = ['Newton-CG']
for method in methods:
    N_bs = 100
    params_uc = np.zeros((N_bs, 4))
    costs = np.zeros((N_bs,))
    for bi in range(N_bs):
        for vi in range(1):
            _stm = rng.choice(
                stm_g, size=stm_g.shape[1], replace=True, axis=1)
            # _stm = stm_g
            lb = [0, None, -5, 0]
            ub = [1, 1, 5, 5]
            a0 = np.random.uniform(low=lb[0], high=ub[0])
            b0 = np.random.uniform(low=-1, high=ub[1])
            m0 = np.random.uniform(low=-1, high=1)
            s0 = np.random.uniform(low=lb[3], high=1)
            evt_mask = _stm[0] > thr_gum
            x = np.asarray(_stm[0, evt_mask])  # conditioning
            # conditioned
            y = np.asarray(np.delete(_stm[:, evt_mask], 0, axis=0)).squeeze()

            optres = minimize(
                cost,
                np.array([a0, b0, m0, s0]),
                args=(x, y),
                jac=jacobian_custom,
                # hess=hessian_custom,
                method=method,
                bounds=((lb[0], ub[0]), (lb[1], ub[1]),
                        (lb[2], ub[2]), (lb[3], ub[3])),
            )
            _param = optres.x
            params_uc[bi, :] = _param
            costs[bi] = cost(_param, x, y)
    params_median = np.median(params_uc, axis=1)
    #########################################################
    # fig, ax = plt.subplots(4, 1, figsize=(8, 6*4))
    # fig.tight_layout()
    # ax[0].set_ylabel("a")
    # ax[1].set_ylabel("b")
    # ax[2].set_ylabel("$\mu$")
    # ax[3].set_ylabel("$\sigma$")

    # # ax[3, 0].set_xlabel(var_name[0])
    # # ax[3, 1].set_xlabel(var_name[1])

    # ax[0].hist(params_uc[:, 0])
    # ax[1].hist(params_uc[:, 1])
    # ax[2].hist(params_uc[:, 2])
    # ax[3].hist(params_uc[:, 3])

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    fig.supxlabel("$a$")
    fig.supylabel("$b$")
    params_ml = np.zeros((4, num_vars))
    ax.set_title(method)
    ax.set_xlim(0, 1)
    ax.set_ylim(None, 1)
    im = ax.scatter(
        params_uc[:, 0],
        params_uc[:, 1],
        s=50*costs/costs.max(),
        c=-costs/costs.max(),
        label="Generated samples",
    )
    plt.colorbar(im)
    plt.savefig(
        f'./output/common/opttest_params_method={method}_{num_events}events_{thr_gum}_{np.count_nonzero(evt_mask)}above_jacob.png', bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    ax.set_title(method)
    ax.hist(costs)
    plt.savefig(
        f'./output/common/opttest_costs_method={method}_{num_events}events_{thr_gum}_{np.count_nonzero(evt_mask)}above_jacob.png', bbox_inches="tight")

# %%
