import matplotlib.pyplot as plt
import numpy as np

from utilities import Mu

site_num = 20
dos_data_name = "data/dos/Phase320_Model1_t0={t0:.3f}_t1={t1:.3f}_" \
                "U={U:.3f}_NotScaled.npz".format(t0=-0.5, t1=-1.0, U=6.0)

with np.load(dos_data_name) as ld:
    dos = ld["dos"]
    omegas = ld["omegas"]

total_dos = np.sum(dos, axis=1)
mu_h = Mu(
    total_dos, omegas,
    occupied_num=site_num, total_num=2*site_num, reverse=True
)
mu_p = Mu(
    total_dos, omegas,
    occupied_num=site_num, total_num=2*site_num, reverse=False
)
mu = (mu_p + mu_h) / 2

step0 = 3.0
step1 = 0.5
fig, axes = plt.subplots(2, 2, sharex="all")
for index, site in enumerate([0, 10]):
    axes[0, 0].plot(omegas, dos[:, site] + index * step0, lw=3.0)
    axes[0, 0].text(
        mu + 0.15, index * step0, "{0: >2d}".format(site),
        ha="left", va="bottom", fontsize=20
    )
for index, site in enumerate([1, 4, 7, 12, 15, 18]):
    axes[0, 1].plot(omegas, dos[:, site] + index * step1, lw=3.0)
    axes[0, 1].text(
        mu + 0.15, index * step1, "{0: >2d}".format(site),
        ha="left", va="bottom", fontsize=20
    )
for index, site in enumerate([2, 5, 8, 13, 16, 19]):
    axes[1, 0].plot(omegas, dos[:, site] + index * step1, lw=3.0)
    axes[1, 0].text(
        mu + 0.15, index * step1, "{0: >2d}".format(site),
        ha="left", va="bottom", fontsize=20
    )
for index, site in enumerate([3, 6, 9, 11, 14, 17]):
    axes[1, 1].plot(omegas, dos[:, site] + index * step1, lw=3.0)
    axes[1, 1].text(
        mu + 0.15, index * step1, "{0: >2d}".format(site),
        ha="left", va="bottom", fontsize=20
    )

for i in range(2):
    for j in range(2):
        axes[i, j].axvline(mu, ls="dashed", lw=2.0, color="black", zorder=0)

        axes[i, j].set_xlim(-4, 10)
        if i == 1:
            xticks = [-4, -2, 0, 2, 4, 6, 8, 10]
            xticklabels = ["{0}".format(xtick) for xtick in xticks]
            axes[i, j].set_xticks(xticks)
            axes[i, j].set_xticklabels(xticklabels, fontsize=20)
            axes[i, j].set_xlabel(r"$\omega/t_1$", fontsize=20)

        if (i, j) == (0, 0):
            yticks = [0, step0]
        else:
            yticks = [0, step1, 2 * step1, 3 * step1, 4 * step1, 5 * step1]
        axes[i, j].set_yticks(yticks)
        axes[i, j].set_yticklabels([""] * len(yticks))
        axes[i, j].tick_params(axis="y", left=False)
        axes[i, j].grid(axis="y", ls="dashed", lw=1.5, color="gray")
        if j == 0:
            axes[i, j].set_ylabel("DOS (arb. units)", fontsize=20)

# top = 0.99,
# bottom = 0.08,
# left = 0.03,
# right = 0.99,
# hspace = 0.02,
# wspace = 0.04
plt.get_current_fig_manager().window.showMaximized()
plt.tight_layout()
plt.show()
fig.savefig("CPTForPhase1LDOS.pdf", transparent=True)
plt.close("all")
