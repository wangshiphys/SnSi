"""
Demonstrate the model used in this project.
"""


import matplotlib.pyplot as plt

from DataBase import POINTS


LW = 4
MS = 15
fig, ax = plt.subplots(num="ShowModel")

x0, y0 = POINTS[7]
x1, y1 = POINTS[8]
line_t0, = ax.plot(
    [x0, x1], [y0, y1], ls="dashed", lw=LW/2, color="tab:red", zorder=1
)

lines_t1 = [
    (1, 2), (3, 5), (6, 9), (10, 12),
    (1, 6), (2, 10), (5, 11), (9, 12),
    (2, 9), (1, 12), (3, 11), (6, 10),
]
for i, j in lines_t1:
    x0, y0 = POINTS[i]
    x1, y1 = POINTS[j]
    line_t1, = ax.plot(
        [x0, x1], [y0, y1], ls="solid", lw=LW, color="tab:blue", zorder=0
    )

point_U0, = ax.plot(
    POINTS[0, 0], POINTS[0, 1],
    ls="", marker="o", color="tab:red", ms=MS, zorder=2
)
point_U1, = ax.plot(
    POINTS[1:, 0], POINTS[1:, 1],
    ls="", marker="o", color="black", ms=MS, zorder=2
)

ax.legend(
    [point_U0, point_U1, line_t0, line_t1],
    ["$U_0$", "$U_1$", "$t_0$", "$t_1$"],
    loc=0, markerscale=0.8,
)

for i in range(13):
    color = "tab:red" if i == 0 else "black"
    ax.text(
        POINTS[i, 0], POINTS[i, 1] - 0.2, str(i),
        ha="center", va="top", color=color, fontsize="xx-large",
    )

for i, j in [(0, 7), (0, 8)]:
    x, y = (POINTS[i] + POINTS[j]) / 2
    ax.text(
        x, y + 0.05, "$t_0$",
        ha="center", va="bottom", color="tab:red", fontsize="xx-large",
    )
x, y = (POINTS[4] + POINTS[7]) / 2
ax.text(
    x, y, "$t_1$",
    ha="right", va="bottom", color="tab:blue", fontsize="xx-large",
)

ax.text(
    POINTS[0, 0], POINTS[0, 1] + 0.1, "$U_0$",
    ha="center", va="bottom", color="tab:red", fontsize="xx-large",
)
x, y = POINTS[4]
ax.annotate(
    "$U_1$", xy=(x, y + 0.1), xytext=(x, y + 0.5),
    ha="center", va="bottom", color="tab:blue", fontsize="xx-large",
    arrowprops={"width": 1.0, "headwidth": 4.0, "color": "tab:blue"},
)

ax.set_aspect("equal")
ax.set_axis_off()
try:
    plt.get_current_fig_manager().window.showMaximized()
except Exception:
    fig.set_size_inches(10.4, 9.18)
plt.show()
plt.close("all")
