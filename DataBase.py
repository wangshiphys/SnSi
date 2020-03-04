"""
Commonly used data of this project.
"""


__all__ = [
    "POINTS",
    "VECTORS",
    "NNPAIRS",
    "POINTS13",
    "POINTS14",
]


from itertools import combinations
import numpy as np


SQRT3 = np.sqrt(3)

VECTORS = np.array([[6.0, -2 * SQRT3], [6.0, 2 * SQRT3]], dtype=np.float64)
POINTS = np.array(
    [
        [-1.0, 2 * SQRT3], [1.0, 2 * SQRT3],
        [-2.0, SQRT3], [0.0, SQRT3], [2.0, SQRT3],
        [-3.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [3.0, 0.0],
        [-2.0, -SQRT3], [0.0, -SQRT3], [2.0, -SQRT3],
    ], dtype=np.float64
)
NNPAIRS = (
    (0, 1), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11),
    (8, 11), (4, 7), (7, 10), (1, 3), (3, 6), (6, 9), (0, 2), (2, 5),
    (5, 9), (2, 6), (6, 10), (0, 3), (3, 7), (7, 11), (1, 4), (4, 8),
)

POINTS13 = {}
for i, j in NNPAIRS:
    key = "({0},{1})".format(i, j)
    mid = (POINTS[[i]] + POINTS[[j]]) / 2
    POINTS13[key] = np.append(POINTS, mid, axis=0)

POINTS14 = {}
for (i, j), (k, l) in combinations(NNPAIRS, 2):
    key = "({0},{1},{2},{3})".format(i, j, k, l)
    mids = (POINTS[[i, k]] + POINTS[[j, l]]) / 2
    POINTS14[key] = np.append(POINTS, mids, axis=0)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    def update(key, ax, which):
        ax.cla()
        if which == 13:
            points = POINTS13[key]
            points_k = points[:-1]
            points_r = points[-1:]
        else:
            points = POINTS14[key]
            points_k = points[:-2]
            points_r = points[-2:]
        ax.plot(points_k[:, 0], points_k[:, 1], ls="", marker="o", color="k")
        ax.plot(points_r[:, 0], points_r[:, 1], ls="", marker="o", color="r")
        ax.set_title(key)
        ax.set_axis_off()

    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.set_aspect("equal")
    ani = FuncAnimation(
        fig, update, fargs=(ax, 13),
        frames=POINTS13, interval=1000, repeat=False,
    )
    plt.show()
    plt.close("all")

    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.set_aspect("equal")
    ani = FuncAnimation(
        fig, update, fargs=(ax, 14),
        frames=POINTS14, interval=1000, repeat=False,
    )
    plt.show()
    plt.close("all")
