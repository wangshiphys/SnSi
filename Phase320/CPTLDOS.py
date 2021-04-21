import matplotlib.pyplot as plt
import numpy as np

from utilities import Mu

site_num = 20
dos_data_name = "data/dos/Phase320_Model1_t0={t0:.3f}_t1={t1:.3f}_" \
                "U={U:.3f}_NotScaled.npz".format(t0=-0.50, t1=-1.0, U=8.0)

with np.load(dos_data_name) as ld:
    dos = ld["dos"]
    omegas = ld["omegas"]

total_dos = np.sum(dos, axis=1)
mu_h = Mu(
    total_dos, omegas,
    occupied_num=site_num, total_num=2 * site_num, reverse=True
)
mu_p = Mu(
    total_dos, omegas,
    occupied_num=site_num, total_num=2 * site_num, reverse=False
)
mu = (mu_p + mu_h) / 2

interval = 0.75
fig, ax = plt.subplots()
ax.plot(omegas, dos[:, 0] + 0 * interval, lw=3.0, zorder=2)
ax.plot(omegas, dos[:, 1] + 1 * interval, lw=3.0, zorder=4)
ax.plot(omegas, dos[:, 2] + 2 * interval, lw=3.0, zorder=3)
ax.plot(omegas, dos[:, 3] + 3 * interval, lw=3.0, zorder=4)
ax.axvline(mu, ls="dashed", lw=2.0, color="black", zorder=0)
ax.text(mu + 0.1, 0 * interval, "0", ha="left", va="bottom", fontsize=20)
ax.text(mu + 0.1, 1 * interval, "1", ha="left", va="bottom", fontsize=20)
ax.text(mu + 0.1, 2 * interval, "2", ha="left", va="bottom", fontsize=20)
ax.text(mu + 0.1, 3 * interval, "3", ha="left", va="bottom", fontsize=20)

ax.set_xlim(-4.5, 12.0)
ax.set_ylim(-0.1, 3.10)

xticks = [-4, -2, 0, 2, 4, 6, 8, 10]
xticklabels = ["{0}".format(xtick) for xtick in xticks]
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, fontsize=20)

yticks = [0.0, 1 * interval, 2 * interval, 3 * interval]
ax.set_yticks(yticks)
ax.set_yticklabels([""] * len(yticks))

ax.set_xlabel(r"$\omega/t_1$", fontsize=20)
ax.set_ylabel("DOS (arb. units)", fontsize=20)
ax.tick_params(axis="y", left=False)
ax.grid(axis="y", ls="dashed", lw=1.5, color="gray", zorder=0)

fig.set_size_inches(4.80, 4.67)
fig.text(0.02, 0.98, "(d)", ha="left", va="top", fontsize=25)
plt.show()
fig.savefig("fig/CPTForPhase1LDOS.pdf", transparent=True)
plt.close("all")
