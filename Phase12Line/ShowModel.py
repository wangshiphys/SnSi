import matplotlib.pyplot as plt
import numpy as np
from HamiltonianPy import Lattice

from database import POINTS, VECTORS

cluster_points = np.append(POINTS, [[0.0, 0.0]], axis=0)
cluster = Lattice(cluster_points, VECTORS)
intra_bonds_1st, inter_bonds_1st = cluster.bonds(nth=1)
intra_bonds_3rd, inter_bonds_3rd = cluster.bonds(nth=3)

fig, ax = plt.subplots()
for point in cluster_points:
    cluster_index = cluster.getIndex(point, fold=True)
    if cluster_index == 12:
        color = "tab:red"
        marker_size = 25
    else:
        color = "black"
        marker_size = 20
    ax.plot(
        point[0], point[1], marker="o",
        ms=marker_size, color=color, zorder=2
    )
    ax.text(
        point[0], point[1] - 0.2, str(cluster_index),
        ha="center", va="top", fontsize="xx-large", zorder=3
    )
for bond in intra_bonds_1st:
    p0, p1 = bond.endpoints
    ax.plot(
        [p0[0], p1[0]], [p0[1], p1[1]],
        ls="dashed", lw=3, color="tab:red", zorder=1
    )
for bond in intra_bonds_3rd:
    p0, p1 = bond.endpoints
    ax.plot(
        [p0[0], p1[0]], [p0[1], p1[1]],
        ls="solid", lw=6, color="tab:blue", zorder=0
    )

ax.set_axis_off()
ax.set_aspect("equal")
plt.get_current_fig_manager().window.showMaximized()
plt.tight_layout()
plt.show()
plt.close("all")
