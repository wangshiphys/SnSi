"""
Demonstrate the model Hamiltonian.
"""


import matplotlib.pyplot as plt
import numpy as np
from HamiltonianPy import Lattice

from database import ALL_POINTS, TRANSLATION_VECTORS

model = "Model1"
CELL = Lattice(ALL_POINTS[0:20], TRANSLATION_VECTORS)
_ids = [
    [0, 0], [-1, 0], [1, 0], [0, -1], [0, 1],
    # [-1, -1], [-1, 1], [1, -1], [1, 1],
]
POINTS_COLLECTION = np.concatenate(
    [np.dot([i, j], TRANSLATION_VECTORS) + ALL_POINTS[0:20] for i, j in _ids]
)
INTRA_HOPPING_INDICES = [
    [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9],
    [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 1],
    [10, 11], [10, 12], [10, 13], [10, 14],
    [10, 15], [10, 16], [10, 17], [10, 18], [10, 19],
    [11, 12], [12, 13], [13, 14], [14, 15],
    [15, 16], [16, 17], [17, 18], [18, 19], [19, 11],
    [6, 11],
]
INTER_HOPPING_INDICES = [[3, 37], [9, 74], [14, 89], [17, 43]]
if model == "Model2":
    INTRA_HOPPING_INDICES += [[6, 12], [6, 19], [5, 11], [7, 11]]
    INTER_HOPPING_INDICES += [
        [3, 36], [3, 38], [2, 37], [4, 37],
        [9, 73], [9, 75], [1, 74], [8, 74],
        [14, 81], [14, 88], [13, 89], [15, 89],
        [17, 42], [17, 44], [16, 43], [18, 43],
    ]
ALL_HOPPING_INDICES = INTRA_HOPPING_INDICES + INTER_HOPPING_INDICES

fig, ax = plt.subplots(num=model)
for index, point in enumerate(POINTS_COLLECTION):
    cell_index = CELL.getIndex(point, fold=True)
    if cell_index in (0, 10):
        color = "tab:red"
        marker_size = 20
    else:
        color = "tab:green"
        marker_size = 16

    ax.plot(
        point[0], point[1],
        ls="", marker="o", color=color, ms=marker_size, zorder=1,
    )
    ax.text(
        point[0], point[1],
        # str(index),
        str(cell_index),
        ha="center", va="center", fontsize="large", zorder=2, clip_on=True,
    )

for ij in INTRA_HOPPING_INDICES:
    bond = POINTS_COLLECTION[ij]
    ax.plot(
        bond[:, 0], bond[:, 1],
        # color="black",
        ls="solid", lw=3.0, zorder=0
    )
for ij in INTER_HOPPING_INDICES:
    bond = POINTS_COLLECTION[ij]
    ax.plot(
        bond[:, 0], bond[:, 1],
        # color="tab:gray",
        ls="dashed", lw=3.0, zorder=0
    )

ax.set_axis_off()
ax.set_aspect("equal")
plt.get_current_fig_manager().window.showMaximized()
plt.tight_layout()
plt.show()
plt.close("all")
