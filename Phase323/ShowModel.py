"""
Demonstrate the model Hamiltonian.
"""


import matplotlib.pyplot as plt
import numpy as np
from HamiltonianPy import Lattice

from database import ALL_POINTS, TRANSLATION_VECTORS

CELL = Lattice(ALL_POINTS, TRANSLATION_VECTORS)
_ids = [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [1, 1]]
POINTS_COLLECTION = np.concatenate(
    [np.dot([i, j], TRANSLATION_VECTORS) + ALL_POINTS for i, j in _ids]
)
INTRA_HOPPING_INDICES = [
    [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9],
    [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 1],

    [10, 11], [10, 12], [10, 13], [10, 14],
    [10, 15], [10, 16], [10, 17], [10, 18], [10, 19],
    [11, 12], [12, 13], [13, 14], [14, 15],
    [15, 16], [16, 17], [17, 18], [18, 19], [19, 11],

    [20, 21], [21, 22], [22, 20],

    [6, 11], [15, 20],
]
INTER_HOPPING_INDICES = [
    [3, 40], [9, 83], [14, 101], [17, 49],
    [12, 45], [18, 90], [21, 110], [22, 58],
    [1, 136], [4, 45], [7, 89], [20, 99], [21, 139], [22, 50],
]
ALL_HOPPING_INDICES = INTRA_HOPPING_INDICES + INTER_HOPPING_INDICES

fig, ax = plt.subplots(num="ModelDefinition")
for index, point in enumerate(POINTS_COLLECTION):
    cell_index = CELL.getIndex(point, fold=True)
    if cell_index in (0, 10):
        color = "tab:red"
        marker_size = 18
    elif cell_index in (20, 21, 22):
        color = "tab:green"
        marker_size = 15
    else:
        color = "tab:blue"
        marker_size = 12

    ax.plot(
        point[0], point[1],
        ls="", marker="o", color=color, ms=marker_size, zorder=1,
    )
    ax.text(
        point[0], point[1],
        # str(index),
        str(cell_index),
        ha="center", va="center", fontsize="medium", zorder=2, clip_on=True,
    )

for ls, bonds in [
    ["solid", INTRA_HOPPING_INDICES], ["dashed", INTER_HOPPING_INDICES]
]:
    for ij in bonds:
        bond = POINTS_COLLECTION[ij]
        ax.plot(bond[:, 0], bond[:, 1], ls=ls, lw=3.0, zorder=0)

ax.set_axis_off()
ax.set_aspect("equal")

plt.get_current_fig_manager().window.showMaximized()
plt.tight_layout()
plt.show()
plt.close("all")
