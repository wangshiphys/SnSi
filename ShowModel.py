"""
Demonstrate the model used in this project.
"""


import matplotlib.pyplot as plt

from DataBase import POINTS, POINTS13, POINTS14, NNPAIRS


LW = 4
MS = 15

site_num = 13
where0 = (6, 7)

# site_num = 14
# where0 = (6, 7)
# where1 = (7, 8)

# site_num = 14
# where0 = (6, 7)
# where1 = (3, 6)

fig, ax = plt.subplots(num="ShowModel")
ax.set_axis_off()
ax.set_aspect("equal")

# Plot the original triangular lattice
for i, j in NNPAIRS:
    line_t1, = ax.plot(
        POINTS[[i, j], 0], POINTS[[i, j], 1],
        ls="solid", lw=LW, color="tab:blue", zorder=0
    )
point_U1, = ax.plot(
    POINTS[:, 0], POINTS[:, 1],
    ls="", marker="o", color="black", ms=MS, zorder=2
)
for index, point in enumerate(POINTS):
    ax.text(
        point[0], point[1] - 0.2, str(index),
        ha="center", va="top", color="black", fontsize="xx-large",
    )

# Plot the the extra point(s).
if site_num == 13:
    i, j = where0
    points = POINTS13["({0},{1})".format(i, j)]

    line_t0, = ax.plot(
        POINTS[[i, j], 0], POINTS[[i, j], 1],
        ls="dashed", lw=LW / 2, color="tab:red", zorder=1
    )
    point_U0, = ax.plot(
        points[-1:, 0], points[-1:, 1],
        ls="", marker="o", color="tab:red", ms=MS, zorder=2
    )
    ax.text(
        points[-1, 0], points[-1, 1] - 0.2, str(index + 1),
        ha="center", va="top", color="black", fontsize="xx-large",
    )
    fig_name = "ShowModel_num={0}_where=({1},{2}).png".format(
        site_num, *where0
    )
else:
    i, j = where0
    k, l = where1
    points = POINTS14["({0},{1},{2},{3})".format(i, j, k, l)]

    line_t0, = ax.plot(
        POINTS[[i, j], 0], POINTS[[i, j], 1],
        ls="dashed", lw=LW / 2, color="tab:red", zorder=1
    )
    line_t0, = ax.plot(
        POINTS[[k, l], 0], POINTS[[k, l], 1],
        ls="dashed", lw=LW / 2, color="tab:red", zorder=1
    )
    point_U0, = ax.plot(
        points[-2:, 0], points[-2:, 1],
        ls="", marker="o", color="tab:red", ms=MS, zorder=2
    )
    ax.text(
        points[-2, 0], points[-2, 1] - 0.2, str(index + 1),
        ha="center", va="top", color="black", fontsize="xx-large",
    )
    ax.text(
        points[-1, 0], points[-1, 1] - 0.2, str(index + 2),
        ha="center", va="top", color="black", fontsize="xx-large",
    )
    fig_name = "ShowModel_num={0}_where=({1},{2})_where1=({3},{4}).png".format(
        site_num, *where0, *where1
    )

ax.legend(
    [point_U0, point_U1, line_t0, line_t1],
    ["$U_0$", "$U_1$", "$t_0$", "$t_1$"],
    loc=0, markerscale=0.8,
)
fig.set_size_inches(4.6, 4.2)
plt.show()
# fig.savefig(fig_name, dpi=200)
plt.close("all")
