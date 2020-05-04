import matplotlib.pyplot as plt
import numpy as np
from HamiltonianPy import Lattice

from database import POINTS, VECTORS

cluster_points = np.append(POINTS, [[0.0, 1/np.sqrt(3)]], axis=0)
cluster = Lattice(cluster_points, VECTORS)
intra_bonds_1st, inter_bonds_1st = cluster.bonds(nth=1)
intra_bonds_2nd, inter_bonds_2nd = cluster.bonds(nth=2)

fig, ax = plt.subplots()
for point in cluster_points:
    cluster_index = cluster.getIndex(point, fold=True)
    if cluster_index in tuple(range(12)):
        color = "black"
        marker_size = 12
    else:
        color = "tab:red"
        marker_size = 15
    ax.plot(
        point[0], point[1], marker="o",
        ms=marker_size, color=color, zorder=1
    )
    ax.text(
        point[0], point[1] - 0.2, str(cluster_index),
        ha="center", va="top", fontsize="x-large", zorder=2
    )
for bond in intra_bonds_1st:
    p0, p1 = bond.endpoints
    ax.plot(
        [p0[0], p1[0]], [p0[1], p1[1]],
        ls="dashed", lw=3, color="tab:red", zorder=0
    )
for bond in intra_bonds_2nd:
    p0, p1 = bond.endpoints
    ax.plot(
        [p0[0], p1[0]], [p0[1], p1[1]],
        lw=3, color="tab:blue", zorder=0
    )

ax.set_axis_off()
ax.set_aspect("equal")
fig.set_size_inches(4.7, 4.2)
plt.show()
fig.savefig("Model.png", dpi=300, transparent=True)
plt.close("all")
