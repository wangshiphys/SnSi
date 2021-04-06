import matplotlib.pyplot as plt
import numpy as np

from utilities import Mu

site_num = 12
data_name_temp = "data/dos/DOS_Model1_t0={t0:.3f}_t1={t1:.3f}_U={U:.3f}.npz"

ids = [
    {"t0": -1.0, "t1": -0.5, "U": 0.0},
    {"t0": -1.0, "t1": -0.5, "U": 3.0},
    {"t0": -1.0, "t1": -0.5, "U": 4.0},
    {"t0": -1.0, "t1": -0.5, "U": 5.0},
    {"t0": -1.0, "t1": -0.5, "U": 6.0},
]

lines = []
labels = []
step = 0.5
baselines = [0.0, step, 2 * step, 3 * step, 4 * step]
fig, ax = plt.subplots()
for index, id in enumerate(ids):
    with np.load(data_name_temp.format(**id)) as ld:
        dos = ld["dos"]
        omegas = ld["omegas"]

    avg_dos = np.mean(dos, axis=1)
    total_dos = np.sum(dos, axis=1)
    mu_h = Mu(total_dos, omegas, site_num, 2*site_num, reverse=True)
    mu_p = Mu(total_dos, omegas, site_num, 2*site_num, reverse=False)
    mu = (mu_p + mu_h) / 2

    line, = ax.plot(omegas - mu, avg_dos + baselines[index], lw=3.0)
    lines.append(line)
    labels.append("U={U:.1f}".format(**id))
ax.axvline(0, ls="dashed", lw=2.0, color="black", zorder=0)
ax.legend(lines[::-1], labels[::-1], loc="lower right", fontsize=16)

ax.set_xlim(-7.5, 6.5)
ax.set_ylim(-0.1, 2.4)
ax.set_yticks(baselines)
ax.set_yticklabels([""] * len(baselines))
ax.set_xticks([-6, -4, -2, 0, 2, 4, 6])
ax.set_xticklabels(["-6", "-4", "-2", "0", "2", "4", "6"], fontsize=20)
ax.set_xlabel(r"$\omega/t_0$", fontsize=20)
ax.set_ylabel("DOS (arb. units)", fontsize=20)
ax.tick_params(axis="y", left=False)
ax.grid(axis="y", ls="dashed", lw=1.5, color="gray")

# top = 0.99,
# bottom = 0.15,
# left = 0.092,
# right = 0.977,
# hspace = 0.2,
# wspace = 0.2

plt.show()
fig.savefig("fig/CPTForPhase2AndPhase3.pdf", transparent=True)
plt.close("all")
