"""
Commonly used data in this sub-package.
"""


__all__ = ["ALL_POINTS", "TRANSLATION_VECTORS"]


import matplotlib.pyplot as plt
import numpy as np


SQRT3 = np.sqrt(3)
ALL_POINTS = np.array(
    [
        [0.0, 0.0],
        [-1.5, SQRT3 / 6],
        [-1.0, -SQRT3 / 3],
        [-0.5, -5 * SQRT3 / 6],
        [0.5, -5 * SQRT3 / 6],
        [1.0, -SQRT3 / 3],
        [1.5, SQRT3 / 6],
        [1.0, 2 * SQRT3 / 3],
        [0.0, 2 * SQRT3 / 3],
        [-1.0, 2 * SQRT3 / 3],
        [4.5, SQRT3 / 2],
        [3.0, SQRT3 / 3],
        [3.5, -SQRT3 / 6],
        [4.5, -SQRT3 / 6],
        [5.5, -SQRT3 / 6],
        [6.0, SQRT3 / 3],
        [5.5, 5 * SQRT3 / 6],
        [5.0, 4 * SQRT3 / 3],
        [4.0, 4 * SQRT3 / 3],
        [3.5, 5 * SQRT3 / 6],
        [8.0, SQRT3 / 3],
        [10.5, 5 * SQRT3 / 6],
        [8.5, 11 * SQRT3 / 6],
    ], dtype=np.float64
)
TRANSLATION_VECTORS = vectors = np.array(
    [[6.0, 3 * SQRT3], [7.5, -1.5 * SQRT3]], dtype=np.float64
)
ALL_POINTS.setflags(write=False)
TRANSLATION_VECTORS.setflags(write=False)


if __name__ == "__main__":
    fig, ax = plt.subplots()
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            points = np.dot([i, j], TRANSLATION_VECTORS) + ALL_POINTS
            ax.plot(
                points[:, 0], points[:, 1],
                ls="", marker="o", markersize=5, zorder=1
            )

    ax.set_axis_off()
    ax.set_aspect("equal")
    plt.get_current_fig_manager().window.showMaximized()
    plt.show()
    plt.close("all")
