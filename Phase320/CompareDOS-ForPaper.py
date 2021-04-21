import matplotlib.pyplot as plt
import numpy as np

from utilities import Mu

site_num = 20
dos_data_name_temp = "data/dos/Phase320_Model1_t0={t0:.3f}" \
                     "_t1={t1:.3f}_U={U:.3f}_NotScaled.npz"

ids = [
    {"t0":-0.5, "t1": -1.0, "U": 0.0},
    {"t0":-0.5, "t1": -1.0, "U": 2.0},
    {"t0":-0.5, "t1": -1.0, "U": 3.0},
    {"t0":-0.5, "t1": -1.0, "U": 4.0},
    {"t0":-0.5, "t1": -1.0, "U": 8.0},
]

step = 1.5
baselines = [0.0, step, 2 * step, 3 * step, 4 * step - 0.7, 4 * step]
fig, ax = plt.subplots()
for index, id in enumerate(ids):
    dos_data_name = dos_data_name_temp.format(**id)
    with np.load(dos_data_name) as ld:
        dos = ld["dos"]
        omegas = ld["omegas"]

    avg_dos = np.mean(dos, axis=1)
    total_dos = np.sum(dos, axis=1)
    mu_p = Mu(
        total_dos, omegas,
        occupied_num=site_num, total_num=2*site_num, reverse=False
    )
    mu_h = Mu(
        total_dos, omegas,
        occupied_num=site_num, total_num=2*site_num, reverse=True
    )
    mu = (mu_p + mu_h) / 2

    ax.plot(omegas - mu, avg_dos + baselines[index], lw=3.0)

with np.load("../TriangleNNHubbard/Triangle34_U=8.00.npz") as ld:
    dos = ld["dos"]
    omegas = ld["omegas"]
avg_dos = np.mean(dos, axis=1)
total_dos = np.sum(dos, axis=1)
mu_p = Mu(total_dos, omegas, occupied_num=12, total_num=24, reverse=False)
mu_h = Mu(total_dos, omegas, occupied_num=12, total_num=24, reverse=True)
mu = (mu_p + mu_h) / 2
ax.plot(omegas - mu + 0.6, avg_dos + baselines[-1], lw=3.0, color="tab:cyan")

ax.axvline(0, ls="dashed", lw=2.0, color="black", zorder=0)
ax.text(8.5, baselines[0], "U=0.0", ha="right", va="bottom", fontsize=20)
ax.text(8.5, baselines[1], "U=2.0", ha="right", va="bottom", fontsize=20)
ax.text(8.5, baselines[2], "U=3.0", ha="right", va="bottom", fontsize=20)
ax.text(8.5, baselines[3], "U=4.0", ha="right", va="bottom", fontsize=20)
ax.text(8.5, baselines[4] + 0.10, "U=8.0", ha="right", va="bottom", fontsize=20)
ax.text(
    8.5, baselines[5] + 0.05, "TL,U=8.0", ha="right", va="bottom", fontsize=20
)

ax.set_xlim(-8.5, 8.5)
ax.set_ylim(-0.1, 6.7)
ax.set_yticks(baselines)
ax.set_yticklabels([""]*len(baselines))
xticks = [-8, -6, -4, -2, 0, 2, 4, 6, 8]
xticklabels = ["{0}".format(xtick) for xtick in xticks]
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, fontsize=20)
ax.set_xlabel(r"$\omega/t_1$", fontsize=20)
ax.set_ylabel("DOS (arb. units)", fontsize=20)
ax.tick_params(axis="y", left=False)
ax.grid(axis="y", ls="dashed", lw=1.5, color="gray")

fig.set_size_inches(4.80, 4.67)
fig.text(0.02, 0.98, "(b)", ha="left", va="top", fontsize=25)
plt.tight_layout()
plt.show()
fig.savefig("fig/CPTForPhase1.pdf", transparent=True)
plt.close("all")
