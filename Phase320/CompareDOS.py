import matplotlib.pyplot as plt
import numpy as np

from utilities import Mu

site_num = 20
dos_data_path = "data/dos/"
dos_data_name_temp = "Phase320_{model}_t0={t0:.3f}_t1={t1:.3f}_U={U:.3f}.npz"

ids = [
    {"model": "Model1", "t1":-1.00, "t0": -0.00, "U": 6.00},
    {"model": "Model1", "t1":-1.00, "t0": -0.10, "U": 6.00},
    {"model": "Model1", "t1":-1.00, "t0": -0.20, "U": 6.00},
    {"model": "Model1", "t1":-1.00, "t0": -0.30, "U": 6.00},
    {"model": "Model1", "t1":-1.00, "t0": -0.40, "U": 6.00},
    {"model": "Model1", "t1":-1.00, "t0": -0.50, "U": 6.00},
    {"model": "Model1", "t1":-1.00, "t0": -0.60, "U": 6.00},
    {"model": "Model1", "t1":-1.00, "t0": -0.70, "U": 6.00},
    {"model": "Model1", "t1":-1.00, "t0": -0.80, "U": 6.00},
    {"model": "Model1", "t1":-1.00, "t0": -0.90, "U": 6.00},
    {"model": "Model1", "t1":-1.00, "t0": -1.00, "U": 6.00},
]

lines = []
labels = []
yticks = []
baseline = 0.0
interval = 0.8

fig, ax = plt.subplots()
for id in ids:
    dos_data_name = dos_data_path + dos_data_name_temp.format(**id)
    try:
        with np.load(dos_data_name) as ld:
            dos = ld["dos"]
            omegas = ld["omegas"]
    except Exception:
        continue

    domega = omegas[1] - omegas[0]
    avg_dos = np.mean(dos, axis=1)
    total_dos = np.sum(dos, axis=1)
    mu_p = Mu(
        total_dos, omegas,
        occupied_num=site_num, total_num=2*site_num, reverse=False
    )
    mu_h = Mu(
        total_dos, omegas,
        occupied_num=site_num, total_num=2*site_num, reverse=True
    )
    mu = (mu_p + mu_h) / 2

    line, = ax.plot(omegas - mu, avg_dos + baseline, lw=2)
    lines.append(line)
    yticks.append(baseline)
    labels.append("t0={t0:.2f}".format(**id))
    baseline += interval
ax.set_yticks(yticks)
ax.grid(axis="y", ls="dashed", color="gray")
ax.axvline(0, ls="dashed", color="gray", lw=2.0, zorder=0)
ax.legend(lines[::-1], labels[::-1], loc="lower right", fontsize=8)
ax.set_xlabel(r"$\omega$", fontsize=8)
ax.set_ylabel(r"DOS (arb. units)", fontsize=8)
ax.tick_params(axis="both", labelsize=8)

plt.get_current_fig_manager().window.showMaximized()
plt.show()
plt.close("all")
