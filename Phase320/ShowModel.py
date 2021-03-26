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


model = "Model1"

# rotation_angle = 0
rotation_angle = -np.arctan2(ALL_POINTS[6, 1], ALL_POINTS[6, 0])
rotation_matrix = Rotation2D(rotation_angle).T
all_points = np.dot(ALL_POINTS, rotation_matrix)
translation_vectors = np.dot(TRANSLATION_VECTORS, rotation_matrix)
cell = Lattice(all_points[0:20], translation_vectors)
points_collection = np.concatenate(
    [
        np.dot([i, j], translation_vectors) + all_points[0:20]
        for i, j in [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]]
    ]
)

intra_hopping_indices = [
    [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9],
    [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 1],
    [10, 11], [10, 12], [10, 13], [10, 14],
    [10, 15], [10, 16], [10, 17], [10, 18], [10, 19],
    [11, 12], [12, 13], [13, 14], [14, 15],
    [15, 16], [16, 17], [17, 18], [18, 19], [19, 11],
    [6, 11],
]
inter_hopping_indices = [[3, 37], [9, 74], [14, 89], [17, 43]]
if model == "Model2":
    intra_hopping_indices += [[6, 12], [6, 19], [5, 11], [7, 11]]
    inter_hopping_indices += [
        [3, 36], [3, 38], [2, 37], [4, 37],
        [9, 73], [9, 75], [1, 74], [8, 74],
        [14, 81], [14, 88], [13, 89], [15, 89],
        [17, 42], [17, 44], [16, 43], [18, 43],
    ]
all_hopping_indices = intra_hopping_indices + inter_hopping_indices

fig, ax = plt.subplots(num=model)
for index, point in enumerate(points_collection):
    cell_index = cell.getIndex(point, fold=True)
    if cell_index in (0, 10):
        color = "tab:red"
        marker_size = 25
    else:
        color = "tab:green"
        marker_size = 20

    ax.plot(
        point[0], point[1],
        ls="", marker="o", color=color, ms=marker_size, zorder=1,
    )
    ax.text(
        point[0], point[1],
        str(index),
        # str(cell_index),
        ha="center", va="center", fontsize="x-large", zorder=2, clip_on=True,
    )

for ij in intra_hopping_indices:
    bond = points_collection[ij]
    ax.plot(
        bond[:, 0], bond[:, 1],
        # color="black",
        ls="solid", lw=4.0, zorder=0
    )
for ij in inter_hopping_indices:
    bond = points_collection[ij]
    ax.plot(
        bond[:, 0], bond[:, 1],
        # color="tab:gray",
        ls="dashed", lw=4.0, zorder=0
    )

ax.set_axis_off()
ax.set_aspect("equal")
plt.get_current_fig_manager().window.showMaximized()
plt.tight_layout()
plt.show()
plt.close("all")
