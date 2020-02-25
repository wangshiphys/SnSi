"""
Commonly used data of this project.
"""


__all__ = [
    "POINTS", "VECTORS",
]


import numpy as np


SQRT3 = np.sqrt(3)

POINTS = np.array(
    [
        [0.0, 0.0],
        [-1.0, 2 * SQRT3], [1.0, 2 * SQRT3],
        [-2.0, SQRT3], [0.0, SQRT3], [2.0, SQRT3],
        [-3.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [3.0, 0.0],
        [-2.0, -SQRT3], [0.0, -SQRT3], [2.0, -SQRT3],
    ], dtype=np.float64
)
VECTORS = np.array([[6.0, -2 * SQRT3], [6.0, 2 * SQRT3]], dtype=np.float64)
