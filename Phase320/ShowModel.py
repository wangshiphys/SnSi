"""
Demonstrate the model Hamiltonian.
"""


import matplotlib.pyplot as plt
import numpy as np
from HamiltonianPy import Lattice

from database import ALL_POINTS, TRANSLATION_VECTORS


def Rotation2D(theta, *, deg=False):
    """
    Rotation about the axis perpendicular to the plane by theta angle.

    Parameters
    ----------
    theta : float
        The rotation angle.
    deg : bool, optional, keyword-only
        Whether the given `theta` is in degree or radian.
        Default: False.

    Returns
    -------
    R : (2, 2) orthogonal matrix
        The corresponding transformation matrix.
    """

    theta = (theta * np.pi / 180) if deg else theta
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    return np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])


rotation_angle = -np.arctan2(ALL_POINTS[6, 1], ALL_POINTS[6, 0])
rotation_matrix = Rotation2D(rotation_angle).T
all_points = np.dot(ALL_POINTS, rotation_matrix)
translation_vectors = np.dot(TRANSLATION_VECTORS, rotation_matrix)
cell = Lattice(all_points[0:20], translation_vectors)
points_collection = np.concatenate(
    [
        np.dot([i, j], translation_vectors) + all_points[0:20]
        for i, j in [[0, 0], [0, -1], [1, -1], [1, 0]]
    ]
)
intra_hopping_indices0 = [
    [0, 1], [0, 2], [0, 3], [0, 4],
    [0, 5], [0, 6], [0, 7], [0, 8], [0, 9],
    [10, 11], [10, 12], [10, 13], [10, 14],
    [10, 15], [10, 16], [10, 17], [10, 18], [10, 19],
    [30, 31], [30, 32], [30, 33], [30, 34],
    [30, 35], [30, 36], [30, 37], [30, 38], [30, 39],
    [40, 41], [40, 42], [40, 43], [40, 44],
    [40, 45], [40, 46], [40, 47], [40, 48], [40, 49],
    [50, 51], [50, 52], [50, 53], [50, 54],
    [50, 55], [50, 56], [50, 57], [50, 58], [50, 59],
    [60, 61], [60, 62], [60, 63], [60, 64],
    [60, 65], [60, 66], [60, 67], [60, 68], [60, 69],
]
intra_hopping_indices1 = [
    [6, 11], [46, 51],
    [1, 2], [2, 3], [3, 4], [4, 5],
    [5, 6], [6, 7], [7, 8], [8, 9], [9, 1],
    [11, 12], [12, 13], [13, 14], [14, 15],
    [15, 16], [16, 17], [17, 18], [18, 19], [19, 11],
    [31, 32], [32, 33], [33, 34], [34, 35],
    [35, 36], [36, 37], [37, 38], [38, 39], [39, 31],
    [41, 42], [42, 43], [43, 44], [44, 45],
    [45, 46], [46, 47], [47, 48], [48, 49], [49, 41],
    [51, 52], [52, 53], [53, 54], [54, 55],
    [55, 56], [56, 57], [57, 58], [58, 59], [59, 51],
    [61, 62], [62, 63], [63, 64], [64, 65],
    [65, 66], [66, 67], [67, 68], [68, 69], [69, 61],
]
inter_hopping_indices = [[9, 34], [37, 43], [54, 69], [17, 63]]

fig, ax = plt.subplots()
for index, point in enumerate(points_collection):
    cell_index = cell.getIndex(point, fold=True)
    if cell_index in (0, 10):
        color = "tab:red"
        marker_size = 20
    else:
        color = "tab:blue"
        marker_size = 16

    ax.plot(
        point[0], point[1],
        ls="", marker="o", color=color, ms=marker_size, zorder=1,
    )
    ax.text(
        point[0], point[1],
        # str(index),
        str(cell_index),
        ha="center", va="center", color="black",
        fontsize=12, zorder=2, clip_on=True,
    )

for ij in intra_hopping_indices0:
    bond = points_collection[ij]
    line0, = ax.plot(
        bond[:, 0], bond[:, 1], color="tab:red",
        ls="solid", lw=2.0, zorder=0
    )
for ij in intra_hopping_indices1:
    bond = points_collection[ij]
    line1, = ax.plot(
        bond[:, 0], bond[:, 1], color="tab:blue",
        ls="solid", lw=3.0, zorder=0
    )
for ij in inter_hopping_indices:
    bond = points_collection[ij]
    line2, = ax.plot(
        bond[:, 0], bond[:, 1], color="tab:blue",
        ls="dotted", lw=3.0, zorder=0
    )
ax.legend(
    [line0, line1, line2], ["$t_0$", "$t_1$", "$t_1$"],
    loc="center", fontsize=20,
)

ax.set_xlim(-4.5, 9.0)
ax.set_ylim(-2.0, 10.0)
ax.set_axis_off()
ax.set_aspect("equal")
fig.set_size_inches(4.80, 4.67)
fig.text(0.02, 0.98, "(a)", ha="left", va="top", fontsize=25)
plt.tight_layout()
plt.show()
fig.savefig("fig/Model.pdf", transparent=True)
plt.close("all")
