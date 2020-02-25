import matplotlib.pyplot as plt
import numpy as np
from HamiltonianPy import Lattice, KPath

from DataBase import POINTS, VECTORS
from TightBinding import TightBinding, TypicalSolver


cluster = Lattice(POINTS, VECTORS)
M = cluster.bs[0] / 2
K = np.dot(np.array([2, 1]), cluster.bs) / 3
kpoints, indices = KPath([np.array([0.0, 0.0]), K, M])

t0 = 0.0
t1 = 1.0
params_identity = "t0={0:.4f}_t1={1:.4f}".format(t0, t1)
GKMGEs = TightBinding(kpoints, cluster, return_vectors=False, t0=t0,
                      t1=t1)
GE, mu, avg_particle_nums, omegas, projected_dos = TypicalSolver(
    cluster, numk=100, gamma=0.02, t0=t0, t1=t1
)

info = [
    "GE = {0}".format(GE), "Mu = {0}".format(mu),
    "Total particle number: {0:.6f}".format(np.sum(avg_particle_nums))
]
for i, num in enumerate(avg_particle_nums):
    info.append(
        "Particle number on {0:2d}th site: {1:.6f}".format(i, num))
with open("INFO_" + params_identity + ".txt", "w") as fp:
    fp.write("\n".join(info))

fig_EB, ax_EB = plt.subplots()
ax_EB.plot(GKMGEs, lw=2)
ax_EB.axhline(mu, ls="dashed", color="red", lw=1)
ax_EB.set_xticks(indices)
ax_EB.set_xticklabels([r"$\Gamma$", "K", "M", r"$\Gamma$"])
ax_EB.set_title(r"EB with $t_0$={0:.2f}, $t_1$={1:.2f}".format(t0, t1))
ax_EB.grid(True, ls="dashed", color="gray")
fig_EB.set_size_inches(9.6, 4.27)
fig_EB.savefig("EB_" + params_identity + ".png", dpi=300)

fig_DOS, axes_DOS = plt.subplots(2, 7, sharey="all")
for index in range(13):
    row, col = divmod(index, 7)
    axes_DOS[row, col].plot(projected_dos[:, index], omegas, lw=2)
    axes_DOS[row, col].axhline(mu, ls="dashed", color="red", lw=1)
    axes_DOS[row, col].set_title("index={0}".format(index))
    axes_DOS[row, col].grid(ls="dashed", color="gray")

total_dos = np.sum(projected_dos, axis=1)
axes_DOS[1, 6].plot(total_dos / 13, omegas, lw=2)
axes_DOS[1, 6].axhline(mu, ls="dashed", color="red", lw=1)
axes_DOS[1, 6].set_title("total")
axes_DOS[1, 6].grid(True, ls="dashed", color="gray")
fig_DOS.set_size_inches(9.6, 4.27)
fig_DOS.savefig("DOS_" + params_identity + ".png", dpi=300)
plt.close("all")
