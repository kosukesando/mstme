from pathlib import Path

import matplotlib.pyplot as plt
import scienceplots

import mstme.mstme as mstme
from mstme.mstme import STM, Area, SimulationConfig

plt.style.use("science")
sim_config = SimulationConfig(
    Path("./data/"),
    Area(-50, 10, -45, 15),
    0.47,
    [
        STM("hs", r"$H_s$", r"$\tilde{H}$", "m"),
        STM("UV_10m", r"$U_{10}$", r"$\tilde{U}$", "m/s"),
    ],
)

cm = 1 / 2.54
fig, ax = plt.subplots(1, 2, figsize=(14 * cm, 5 * cm), facecolor="white")
for vi, s in enumerate(sim_config.stm):
    print(vi, s)
    ax[vi].set_title(s.name)
plt.show()
