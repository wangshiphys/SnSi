import matplotlib.pyplot as plt
import numpy as np

from utilities import Mu

site_num = 13
data_name_temp = "data/dos/amplitude={amplitude:.2f}/run={run}/"
data_name_temp += "Phase12Center_t=-1.00_U={U:.2f}.npz"

ids = [
    {"amplitude": 0.00, "run": 1, "U": 0.0},
    {"amplitude": 0.00, "run": 1, "U": 0.5},
    {"amplitude": 0.00, "run": 1, "U": 1.0},
    {"amplitude": 0.00, "run": 1, "U": 1.5},
    {"amplitude": 0.00, "run": 1, "U": 2.0},
    {"amplitude": 0.00, "run": 1, "U": 2.5},
    {"amplitude": 0.00, "run": 1, "U": 3.0},
    {"amplitude": 0.00, "run": 1, "U": 3.5},
    {"amplitude": 0.00, "run": 1, "U": 4.0},
    {"amplitude": 0.00, "run": 1, "U": 4.5},
    {"amplitude": 0.00, "run": 1, "U": 5.0},
    {"amplitude": 0.00, "run": 1, "U": 5.5},
    {"amplitude": 0.00, "run": 1, "U": 6.0},
    {"amplitude": 0.00, "run": 1, "U": 6.5},
    {"amplitude": 0.00, "run": 1, "U": 7.0},
    {"amplitude": 0.00, "run": 1, "U": 7.5},
]

# ids = [
#     {"amplitude": 0.00, "run": 1, "U": 6.0},
#     {"amplitude": 0.05, "run": 1, "U": 6.0},
#     {"amplitude": 0.10, "run": 1, "U": 6.0},
#     {"amplitude": 0.20, "run": 1, "U": 6.0},
#     {"amplitude": 0.30, "run": 1, "U": 6.0},
#     {"amplitude": 0.40, "run": 1, "U": 6.0},
#     {"amplitude": 0.50, "run": 1, "U": 6.0},
# ]

# ids = [
#     {"amplitude": 0.50, "run": 1, "U": 6.0},
#     {"amplitude": 0.50, "run": 2, "U": 6.0},
#     {"amplitude": 0.50, "run": 3, "U": 6.0},
#     {"amplitude": 0.50, "run": 4, "U": 6.0},
#     {"amplitude": 0.50, "run": 5, "U": 6.0},
# ]

lines = []
labels = []
yticks = []
baseline = 0.0
interval = 0.5
fig, ax = plt.subplots()
for id in ids:
    with np.load(data_name_temp.format(**id)) as ld:
        dos = ld["dos"]
        omegas = ld["omegas"]

    avg_dos = np.mean(dos, axis=1)
    total_dos = np.sum(dos, axis=1)
    mu_h = Mu(
        total_dos, omegas,
        occupied_num=site_num, total_num=2*site_num, reverse=True
    )
    mu_p = Mu(
        total_dos, omegas,
        occupied_num=site_num, total_num=2*site_num, reverse=False
    )
    mu = (mu_p + mu_h) / 2

    line, = ax.plot(omegas - mu, avg_dos + baseline, lw=2)
    lines.append(line)
    yticks.append(baseline)
    labels.append("amplitude={amplitude:.2f},U={U:.1f}".format(**id))
    baseline += interval
ax.set_yticks(yticks)
ax.grid(axis="both", ls="dashed", color="gray")
ax.legend(lines[::-1], labels[::-1], loc="lower right", fontsize="xx-large")

plt.get_current_fig_manager().window.showMaximized()
plt.tight_layout()
plt.show()
plt.close("all")
