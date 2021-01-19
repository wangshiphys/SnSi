"""
Commonly used data in this sub-package.
"""


__all__ = ["POINTS", "VECTORS"]

import matplotlib.pyplot as plt
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
POINTS.setflags(write=False)
VECTORS.setflags(write=False)


if __name__ == "__main__":
    fig, ax = plt.subplots()
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            points = np.dot([i, j], VECTORS) + POINTS
            ax.plot(points[:, 0], points[:, 1], ls="", marker="o", ms=20)
    ax.set_axis_off()
    ax.set_aspect("equal")

    plt.get_current_fig_manager().window.showMaximized()
    plt.tight_layout()
    plt.show()
    plt.close("all")
