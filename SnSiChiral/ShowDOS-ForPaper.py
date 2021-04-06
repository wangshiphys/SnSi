import matplotlib.pyplot as plt
import numpy as np

from utilities import Mu

site_num = 12
data_name = "data/dos/DOS_Model1_t0={t0:.3f}_t1={t1:.3f}_U={U:.3f}.npz".format(
    t0=-1.0, t1=-0.5, U=6.0
)
with np.load(data_name) as ld:
    dos = ld["dos"]
    omegas = ld["omegas"]

total_dos = np.sum(dos, axis=1)
mu_h = Mu(total_dos, omegas, site_num, 2*site_num, reverse=True)
mu_p = Mu(total_dos, omegas, site_num, 2*site_num, reverse=False)
mu = (mu_p + mu_h) / 2

step = 0.5
fig, axes = plt.subplots(2, 2, sharex="all")
for index, site in enumerate([0, 1, 2]):
    axes[0, 0].plot(omegas, dos[:, 2 * site] + index * step, lw=5.0)
    axes[0, 0].text(
        mu + 0.15, index * step, "{0: >2d}".format(site),
        ha="left", va="bottom", fontsize=20
    )
for index, site in enumerate([3, 6, 9]):
    axes[0, 1].plot(omegas, dos[:, 2 * site] + index * step, lw=5.0)
    axes[0, 1].text(
        mu + 0.15, index * step, "{0: >2d}".format(site),
        ha="left", va="bottom", fontsize=20
    )
for index, site in enumerate([4, 7, 10]):
    axes[1, 0].plot(omegas, dos[:, 2 * site] + index * step, lw=5.0)
    axes[1, 0].text(
        mu + 0.15, index * step, "{0: >2d}".format(site),
        ha="left", va="bottom", fontsize=20
    )
for index, site in enumerate([5, 8, 11]):
    axes[1, 1].plot(omegas, dos[:, 2 * site] + index * step, lw=5.0)
    axes[1, 1].text(
        mu + 0.15, index * step, "{0: >2d}".format(site),
        ha="left", va="bottom", fontsize=20
    )

for i in range(2):
    for j in range(2):
        axes[i, j].set_xlim(-3.5, 9.5)
        axes[i, j].axvline(mu, ls="dashed", lw=3.0, color="black", zorder=0)
        if i == 1:
            xticks = [-2, 0, 2, 4, 6, 8]
            xticklabels = ["{0}".format(xtick) for xtick in xticks]
            axes[i, j].set_xticks(xticks)
            axes[i, j].set_xticklabels(xticklabels, fontsize=20)
            axes[i, j].set_xlabel(r"$\omega/t_1$", fontsize=20)

        yticks = [0, step, 2 * step]
        axes[i, j].set_yticks(yticks)
        axes[i, j].set_yticklabels([""] * len(yticks))
        axes[i, j].tick_params(axis="y", left=False)
        axes[i, j].grid(axis="y", ls="dashed", lw=2.0, color="gray")
        if j == 0:
            axes[i, j].set_ylabel("DOS (arb. units)", fontsize=20)

# top = 0.995,
# bottom = 0.08,
# left = 0.03,
# right = 0.995,
# hspace = 0.02,
# wspace = 0.01

plt.get_current_fig_manager().window.showMaximized()
plt.show()
fig.savefig("fig/CPTForPhase2AndPhase3LDOS.pdf", transparent=True)
plt.close("all")
