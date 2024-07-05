import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import genpareto

import mstme.conmul as conmul
import mstme.marginal as marginal
from mstme.marginal import MixDist

NUM_VARS = 2
NUM_SAMPLES = 1000
THRESHOLD_COMMON = 1.5


def main():
    gpds = []
    samples = np.empty((NUM_VARS, NUM_SAMPLES))
    samples_laplace = np.empty((NUM_VARS, NUM_SAMPLES))
    mix_dists = []
    for vi in range(NUM_VARS):
        samples[vi, :] = genpareto.rvs(c=0.5, size=NUM_SAMPLES)
        _gp, _ = marginal.genpar_estimation(samples[vi], 2.0)
        gpds.append(_gp)
        _mix_dist = MixDist(_gp, samples[vi])
        mix_dists.append(_mix_dist)
        samples_laplace[vi, :] = marginal.transform_to_laplace(_mix_dist, samples[vi])

    cme_estimator = conmul.ConmulExtremeEstimator(samples_laplace)
    params_uc = cme_estimator.estimate(THRESHOLD_COMMON)
    params_uc_array = np.array(params_uc)
    params_mean = np.mean(params_uc_array, axis=0)
    print(params_mean)
    # plt.scatter(params_uc_array[:, 0, 0], params_uc_array[:, 0, 1])
    # plt.show()

    cme_model = conmul.ConmulExtremeModel(
        samples_laplace, THRESHOLD_COMMON, params_mean
    )
    (
        samples_conmul_extreme_laplace,
        samples_conmul_nonextreme_laplace,
    ) = cme_model.sample_with_body(1000)

    # plt.scatter(samples_laplace[0], samples_laplace[1], c="k")
    # plt.scatter(samples_conmul_extreme[0], samples_conmul_extreme[1])
    # plt.scatter(samples_conmul_nonextreme[0], samples_conmul_nonextreme[1])
    # plt.show()

    samples_conmul_extreme = np.empty(samples_conmul_extreme_laplace.shape)
    samples_conmul_nonextreme = np.empty(samples_conmul_nonextreme_laplace.shape)
    for vi in range(NUM_VARS):
        samples_conmul_extreme[vi] = marginal.transform_from_laplace(
            mix_dists[vi], samples_conmul_extreme_laplace[vi]
        )
        samples_conmul_nonextreme[vi] = marginal.transform_from_laplace(
            mix_dists[vi], samples_conmul_nonextreme_laplace[vi]
        )

    plt.scatter(samples[0], samples[1], c="k")
    plt.scatter(samples_conmul_extreme[0], samples_conmul_extreme[1])
    plt.scatter(samples_conmul_nonextreme[0], samples_conmul_nonextreme[1])
    plt.show()
    return


if __name__ == "__main__":
    main()
