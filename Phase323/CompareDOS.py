import matplotlib.pyplot as plt
import numpy as np

from utilities import Mu

site_num = 23
dos_data_name_temp = "data/dos/Phase323_t=-1.00_U={0:.2f}.npz"

lines = []
labels = []
yticks = []
baseline = 0.0
interval = 0.5
Us = np.arange(0.0, 10, 0.5)

fig, ax = plt.subplots()
for U in Us:
    try:
        with np.load(dos_data_name_temp.format(U)) as ld:
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
    labels.append("U={0:.1f}".format(U))
    baseline += interval
ax.set_yticks(yticks)
ax.grid(axis="both", ls="dashed", color="gray")
ax.legend(lines[::-1], labels[::-1], loc="lower right", fontsize="xx-large")

plt.get_current_fig_manager().window.showMaximized()
plt.show()
plt.close("all")
