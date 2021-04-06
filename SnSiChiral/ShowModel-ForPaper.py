"""
Demonstrate the model Hamiltonian.
"""


import matplotlib.pyplot as plt
import numpy as np

from database import POINTS, VECTORS

POINTS_COLLECTION = np.concatenate(
    [np.dot([i, j], VECTORS) + POINTS for i, j in [[0, 0], [-1, 1], [0, 1]]]
)
INTER_HOPPING_INDICES = [
    [6, 27], [7, 27], [7, 35], [8, 35],
    [8, 17], [8, 16], [9, 16], [9, 15],
    [17, 35], [17, 34], [18, 34], [18, 33],
]
INTRA_HOPPING_INDICES0 = [[0, 4], [4, 5], [1, 7], [7, 8], [2, 10], [10, 11]]
INTRA_HOPPING_INDICES1 = [
    [0, 1], [1, 2], [2, 0], [0, 3], [0, 10], [0, 11],
    [1, 4], [1, 5], [1, 6], [2, 7], [2, 8], [2, 9],
    [3, 4], [3, 11], [6, 5], [6, 7], [9, 8], [9, 10],
]

ms = 25
fontsize = 18
fig, ax = plt.subplots()
colors = plt.get_cmap("tab10")(range(10))
for index, point in enumerate(POINTS_COLLECTION):
    cell_index = index % 12
    color = colors[index // 12]
    ax.plot(point[0], point[1], marker="o", ms=ms, color=color, zorder=1)
    ax.text(
        point[0], point[1],
        # str(index),
        str(cell_index),
        ha="center", va="center", fontsize=fontsize, zorder=2
    )

for i, j in INTRA_HOPPING_INDICES0:
    for shift in [0, 12, 24]:
        bond = POINTS_COLLECTION[[i+shift, j+shift]]
        line0, = ax.plot(
            bond[:, 0], bond[:, 1],
            ls="solid", lw=6.0, color="tab:red", zorder=0
        )
for i, j in INTRA_HOPPING_INDICES1:
    for shift in [0, 12, 24]:
        bond = POINTS_COLLECTION[[i+shift, j+shift]]
        line1, = ax.plot(
            bond[:, 0], bond[:, 1],
            ls="dashed", lw=3.0, color="tab:green", zorder=0
        )
for ij in INTER_HOPPING_INDICES:
    bond = POINTS_COLLECTION[ij]
    line2, = ax.plot(
        bond[:, 0], bond[:, 1], ls="dotted", lw=3.0, color="gray", zorder=0
    )
ax.legend(
    [line0, line1, line2], ["$t_0$", "$t_1$", "$t_1$"],
    bbox_to_anchor=(0.60, 0.73), bbox_transform=ax.transAxes,
    loc="lower left", fontsize=30,
)

ax.set_axis_off()
ax.set_aspect("equal")
fig.set_size_inches(9.36, 9.36)
plt.tight_layout()

# top = 1.0,
# bottom = 0.0,
# left = 0.0,
# right = 1.0,
# hspace = 0.2,
# wspace = 0.2

plt.show()
fig.savefig("fig/Model.pdf", transparent=True)
plt.close("all")
