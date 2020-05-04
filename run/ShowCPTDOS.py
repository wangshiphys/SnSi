import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.optimize import brentq

DOS_PATH = "../data/DOS/"
DOS_NAME_TEMP = "(26,13)_{t0:.1f}_{t1:.1f}_{U0:.1f}_{U1:.1f}_VCA_DOS.dat"


def _MuCore(mu, dos, omegas, occupied_num=13, total_num=26, reverse=False):
    delta_omega = omegas[1] - omegas[0]
    if reverse:
        num = total_num - occupied_num
        indices = omegas > mu
    else:
        num = occupied_num
        indices = omegas < mu
    return np.sum(dos[indices]) * delta_omega - num


def Mu(dos, omegas, occupied_num=13, total_num=26, reverse=False):
    args = (dos, omegas, occupied_num, total_num, reverse)
    return brentq(_MuCore, a=omegas[0], b=omegas[-1], args=args)

t0 = -3.0
t1 = -1.0
U0 = U1 = 10.0
fig_name_temp = "DOS_t0={0:.2f}_t1={1:.2f}_U0={2:.2f}_U1={3:.2f}.png"
fig_name = "../fig/DOS/" + fig_name_temp.format(t0, t1, U0, U1)
file_name = DOS_PATH + DOS_NAME_TEMP.format(t0=t0, t1=t1, U0=U0, U1=U1)
data = np.loadtxt(file_name)

omegas = data[:, 0]
domega = omegas[1] - omegas[0]
projected_dos = data[:, 1:]
total_dos = np.sum(projected_dos, axis=1)
mu_p = Mu(total_dos, omegas, reverse=False)
mu_h = Mu(total_dos, omegas, reverse=True)
print("mu_p = {0}".format(mu_p))
print("mu_h = {0}".format(mu_h))
mu = (mu_p + mu_h) / 2

fig, axes = plt.subplots(2, 7, sharey="all", num=fig_name)
for index in range(13):
    row, col = divmod(index, 7)
    dos_down = projected_dos[:, 2 * index]
    dos_up = projected_dos[:, 2 * index + 1]
    axes[row, col].plot(dos_down, omegas, lw=2, color="k")
    axes[row, col].plot(dos_up, omegas, lw=1, color="red", ls="dashed")
    axes[row, col].axhline(mu, lw=1, color="green", ls="dashed")
    axes[row, col].set_title("index={0}".format(index))
axes[1, 6].plot(total_dos, omegas, lw=2, color="k")
axes[1, 6].axhline(mu, lw=1, color="green", ls="dashed")
axes[1, 6].set_title("total")
plt.get_current_fig_manager().window.showMaximized()
plt.show()
fig.savefig(fig_name, dpi=300)
plt.close("all")
