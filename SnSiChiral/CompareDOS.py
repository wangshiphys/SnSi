import matplotlib.pyplot as plt
import numpy as np

from utilities import Mu

site_num = 12
data_name_temp = "data/dos/DOS_{model}_t0={t0:.3f}_t1={t1:.3f}_U={U:.3f}.npz"

ids = [
    {"model": "Model1", "t0": -1.00, "t1": -0.5, "U": 0.00},
    {"model": "Model1", "t0": -1.00, "t1": -0.5, "U": 0.50},
    {"model": "Model1", "t0": -1.00, "t1": -0.5, "U": 1.00},
    {"model": "Model1", "t0": -1.00, "t1": -0.5, "U": 1.50},
    {"model": "Model1", "t0": -1.00, "t1": -0.5, "U": 2.00},
    {"model": "Model1", "t0": -1.00, "t1": -0.5, "U": 2.50},
    {"model": "Model1", "t0": -1.00, "t1": -0.5, "U": 3.00},
    {"model": "Model1", "t0": -1.00, "t1": -0.5, "U": 3.50},
    {"model": "Model1", "t0": -1.00, "t1": -0.5, "U": 4.00},
    {"model": "Model1", "t0": -1.00, "t1": -0.5, "U": 4.50},
    {"model": "Model1", "t0": -1.00, "t1": -0.5, "U": 5.00},
    {"model": "Model1", "t0": -1.00, "t1": -0.5, "U": 5.50},
    {"model": "Model1", "t0": -1.00, "t1": -0.5, "U": 6.00},
    {"model": "Model1", "t0": -1.00, "t1": -0.5, "U": 6.50},
    {"model": "Model1", "t0": -1.00, "t1": -0.5, "U": 7.00},
    {"model": "Model1", "t0": -1.00, "t1": -0.5, "U": 7.50},
    {"model": "Model1", "t0": -1.00, "t1": -0.5, "U": 8.00},
    {"model": "Model1", "t0": -1.00, "t1": -0.5, "U": 8.50},
    {"model": "Model1", "t0": -1.00, "t1": -0.5, "U": 9.00},
    {"model": "Model1", "t0": -1.00, "t1": -0.5, "U": 9.50},
]

ids = [
    {"model": "Model1", "t0": -1.00, "t1":  0.00, "U": 5.00},
    {"model": "Model1", "t0": -1.00, "t1": -0.10, "U": 5.00},
    {"model": "Model1", "t0": -1.00, "t1": -0.20, "U": 5.00},
    {"model": "Model1", "t0": -1.00, "t1": -0.30, "U": 5.00},
    {"model": "Model1", "t0": -1.00, "t1": -0.40, "U": 5.00},
    {"model": "Model1", "t0": -1.00, "t1": -0.50, "U": 5.00},
    {"model": "Model1", "t0": -1.00, "t1": -0.60, "U": 5.00},
    {"model": "Model1", "t0": -1.00, "t1": -0.70, "U": 5.00},
    {"model": "Model1", "t0": -1.00, "t1": -0.80, "U": 5.00},
    {"model": "Model1", "t0": -1.00, "t1": -0.90, "U": 5.00},
    {"model": "Model1", "t0": -1.00, "t1": -1.00, "U": 5.00},
]

lines = []
labels = []
yticks = []
baseline = 0.0
interval = 0.3
fig, ax = plt.subplots()
for id in ids:
    with np.load(data_name_temp.format(**id)) as ld:
        dos = ld["dos"]
        omegas = ld["omegas"]
    avg_dos = np.mean(dos, axis=1)
    total_dos = np.sum(dos, axis=1)
    mu_h = Mu(total_dos, omegas, site_num, 2*site_num, reverse=True)
    mu_p = Mu(total_dos, omegas, site_num, 2*site_num, reverse=False)
    mu = (mu_p + mu_h) / 2

    line, = ax.plot(omegas - mu, avg_dos + baseline, lw=2)
    lines.append(line)
    yticks.append(baseline)
    labels.append("t0={t0:.2f},t1={t1:.2f},U={U:.2f}".format(**id))
    baseline += interval
ax.set_yticks(yticks)
ax.grid(axis="both", ls="dashed", color="gray")
ax.legend(lines[::-1], labels[::-1], loc="lower left", fontsize=15)
ax.set_title("t0={t0:.2f},U={U:.2f}".format(**ids[0]), fontsize=15)

plt.get_current_fig_manager().window.showMaximized()
plt.show()
plt.close("all")
