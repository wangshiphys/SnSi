"""
Demonstrate the model Hamiltonian.
"""


from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from database import POINTS, VECTORS

model = "Model1"
_ids = [[0, 0], [1, -1], [0, -1], [-1, 0], [-1, 1], [0, 1], [1, 0]]
POINTS_COLLECTION = np.concatenate(
    [np.dot([i, j], VECTORS) + POINTS for i, j in _ids]
)
INTER_HOPPING_INDICES = [
    [5, 83], [5, 82], [6, 82], [6, 81], [6, 63], [7, 63],
    [7, 71], [8, 71], [8, 53], [8, 52], [9, 52], [9, 51],
    [5, 20], [4, 20], [4, 21], [3, 21], [3, 30], [3, 31],
    [11, 31], [11, 32], [11, 41], [10, 41], [10, 42], [9, 42],
]
if model == "Model1":
    INTRA_HOPPING_INDICES0 = [[0, 4], [4, 5], [1, 7], [7, 8], [2, 10], [10, 11]]
    INTRA_HOPPING_INDICES1 = [
        [0, 1], [1, 2], [2, 0], [0, 3], [0, 10], [0, 11],
        [1, 4], [1, 5], [1, 6], [2, 7], [2, 8], [2, 9],
        [3, 4], [3, 11], [6, 5], [6, 7], [9, 8], [9, 10],
    ]
else:
    INTRA_HOPPING_INDICES0 = [
        [0, 4], [4, 5], [5, 6],
        [1, 7], [7, 8], [8, 9],
        [2, 10], [10, 11], [11, 3],
    ]
    INTRA_HOPPING_INDICES1 = [
        [0, 1], [1, 2], [2, 0],
        [0, 3], [0, 10], [0, 11],
        [1, 4], [1, 5], [1, 6],
        [2, 7], [2, 8], [2, 9],
        [3, 4], [6, 7], [9, 10],
    ]


lw = 5
ms = 25
fontsize = 18
colors = plt.get_cmap("tab10")(range(7))
fig, ax = plt.subplots(num=model)
for index, point in enumerate(POINTS_COLLECTION):
    cell_index = index % 12
    color = colors[index//12]
    ax.plot(point[0], point[1], marker="o", ms=ms, color=color, zorder=1)
    ax.text(
        point[0], point[1],
        # str(index),
        str(cell_index),
        ha="center", va="center", fontsize=fontsize, zorder=2
    )

for ij in INTRA_HOPPING_INDICES0:
    bond = POINTS_COLLECTION[ij]
    line0, = ax.plot(
        bond[:, 0], bond[:, 1], ls="solid", lw=lw, color="tab:red", zorder=0
    )
for ij in INTRA_HOPPING_INDICES1:
    bond = POINTS_COLLECTION[ij]
    line1, = ax.plot(
        bond[:, 0], bond[:, 1],
        ls="dashed", lw=lw/2, color="tab:green", zorder=0
    )
for ij in INTER_HOPPING_INDICES:
    bond = POINTS_COLLECTION[ij]
    line2, = ax.plot(
        bond[:, 0], bond[:, 1], ls="dotted", lw=lw/2, color="gray", zorder=0
    )
ax.legend(
    [line0, line1, line2], ["$t_0$", "$t_1$", "$t_1$"],
    bbox_to_anchor=(0.70, 0.80), bbox_transform=ax.transAxes,
    loc="lower left", fontsize=30, labelspacing=0.15, borderpad=0.15
)

ax.set_axis_off()
ax.set_aspect("equal")
fig.set_size_inches(9.36, 9.36)
plt.tight_layout()
plt.show()
Path("fig/Model/").mkdir(exist_ok=True, parents=True)
fig.savefig("fig/Model/" + model + ".pdf", transparent=True)
plt.close("all")
