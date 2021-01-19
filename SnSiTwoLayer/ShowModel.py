import matplotlib.pyplot as plt
import numpy as np

SQRT3 = np.sqrt(3)
POINTS = np.array(
    [
        [0.0, 0.0], [1.0, SQRT3], [2.0, 0.0], [3.0, SQRT3],
        [4.0, 0.0], [5.0, SQRT3], [6.0, 0.0], [7.0, SQRT3],
        [1.0, 1 / SQRT3], [5.0, 1 / SQRT3],
    ], dtype=np.float64
)
VECTORS = np.array([[8.0, 0.0], [2.0, 2 * SQRT3]], dtype=np.float64)


fig, ax = plt.subplots()
for dR in [np.array([0, 0]), np.array([2, 2 * SQRT3])]:
    points = dR + POINTS
    lines = [
        (0, 6), (1, 7), (0, 1), (2, 3), (4, 5), (6, 7), (1, 2), (3, 4), (5, 6)
    ]
    for i, j in lines:
        x0, y0 = points[i]
        x1, y1 = points[j]
        line_t1, = ax.plot([x0, x1], [y0, y1], color="black", zorder=0)
    lines = [(0, 8), (1, 8), (2, 8), (4, 9), (5, 9), (6, 9)]
    for i, j in lines:
        x0, y0 = points[i]
        x1, y1 = points[j]
        line_t0, = ax.plot(
            [x0, x1], [y0, y1], ls="dashed", color="tab:red", zorder=1
        )
    lines = [(8, 9)]
    for i, j in lines:
        x0, y0 = points[i]
        x1, y1 = points[j]
        line_t2, = ax.plot(
            [x0, x1], [y0, y1], ls="solid", color="tab:purple", zorder=3
        )

    for index in range(8):
        x, y = points[index]
        point_U0, = ax.plot(
            x, y, ls="", marker="o", ms=12,
            color="black", zorder=2, clip_on=False
        )
        ax.text(
            x, y - 0.1, "{0}".format(index),
            ha="center", va="top", fontsize="xx-large",
        )

    for index in (8, 9):
        x, y = points[index]
        point_U1, = ax.plot(
            x, y, ls="", marker="o", ms=15,
            color="tab:red", zorder=4, clip_on=False
        )
        ax.text(
            x, y - 0.1, "{0}".format(index),
            ha="center", va="top", fontsize="xx-large",
        )

ax.legend(
    [point_U0, point_U1, line_t0, line_t1, line_t2],
    ["$U_0$", "$U_1$", "$t_0$", "$t_1$", "$t_2$"], loc=0,
)
ax.set_axis_off()
ax.set_aspect("equal")
plt.show()
# fig.savefig("ShowModel.png", dpi=500)
plt.close("all")
