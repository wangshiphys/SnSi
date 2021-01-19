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
fontsize = 15
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
    ax.plot(
        bond[:, 0], bond[:, 1], ls="solid", lw=lw, color="tab:red", zorder=0
    )
for ij in INTRA_HOPPING_INDICES1:
    bond = POINTS_COLLECTION[ij]
    ax.plot(
        bond[:, 0], bond[:, 1],
        ls="dashed", lw=lw/2, color="tab:green", zorder=0
    )
for ij in INTER_HOPPING_INDICES:
    bond = POINTS_COLLECTION[ij]
    ax.plot(
        bond[:, 0], bond[:, 1], ls="dotted", lw=lw/2, color="gray", zorder=0
    )

x0, y0 = POINTS_COLLECTION[2]
x1, y1 = POINTS_COLLECTION[10]
ax.text(
    (x0 + x1) / 2, (y0 + y1) / 2, "t$_0$", color="tab:red",
    va="bottom", ha="center", fontsize=fontsize+6
)
x0, y0 = POINTS_COLLECTION[0]
x1, y1 = POINTS_COLLECTION[1]
ax.text(
    (x0 + x1) / 2, (y0 + y1) / 2, "t$_1$", color="tab:green",
    va="bottom", ha="center", fontsize=fontsize+6
)
x0, y0 = POINTS_COLLECTION[6]
x1, y1 = POINTS_COLLECTION[81]
ax.text(
    (x0 + x1) / 2, (y0 + y1) / 2, "t$_1$", color="gray",
    va="bottom", ha="center", fontsize=fontsize+6
)

ax.set_axis_off()
ax.set_aspect("equal")
fig.set_size_inches(8.7, 9.26)
plt.tight_layout()
plt.show()
Path("fig/Model/").mkdir(exist_ok=True, parents=True)
# fig.savefig("fig/Model/" + model + ".png", transparent=True)
plt.close("all")
