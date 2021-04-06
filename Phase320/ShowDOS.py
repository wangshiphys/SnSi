import matplotlib.pyplot as plt
import numpy as np

from utilities import Mu

site_num = 20
dos_data_name_temp = "data/dos/Phase320_Model1_t0={t0:.3f}" \
                     "_t1={t1:.3f}_U={U:.3f}_NotScaled.npz"

params = {"t0": -0.5, "t1": -1.0, "U": 6.0}
dos_data_name = dos_data_name_temp.format(**params)
with np.load(dos_data_name) as ld:
    dos = ld["dos"]
    omegas = ld["omegas"]
domega = omegas[1] - omegas[0]

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
particle_num = np.sum(total_dos[omegas < mu]) * domega
local_particle_num = np.sum(dos[omegas < mu], axis=0) * domega

print("t0 = {t0:.1f}, t1 = {t1:.1f}, U = {U:.1f}".format(**params))
print("Sum of DoS: {0:.8f}".format(np.sum(dos) * domega))
print("mu_p = {0:.8f}, mu_h = {1:.8f}, mu = {2:.8f}".format(mu_p, mu_h, mu))
print("Particle number: {0:.8f}".format(particle_num))
for index in range(site_num):
    num = local_particle_num[index] + local_particle_num[index+site_num]
    print("Particle number on {0:0>2d}th site: {1:8f}".format(index, num))

fig_dos0, axes_dos0 = plt.subplots(2, 5, sharex="all", sharey="all")
fig_dos1, axes_dos1 = plt.subplots(2, 5, sharex="all", sharey="all")
for index in range(site_num):
    if index < 10:
        ax = axes_dos0[divmod(index, 5)]
    else:
        ax = axes_dos1[divmod(index - 10, 5)]
    ax.plot(omegas, dos[:, index], lw=2.0, color="black")
    ax.plot(omegas, dos[:, index+site_num], lw=1.0, color="red", ls="dashed")
    ax.set_xlim(omegas[0], omegas[-1])
    ax.set_title("Index={0:0>2d}".format(index))
    ax.grid(axis="both", ls="dashed", color="gray")
    ax.axvline(mu, ls="dashed", color="orange", lw=1.0)

fig_dos_avg, ax_dos_avg = plt.subplots()
ax_dos_avg.plot(omegas, np.mean(dos, axis=1), lw=2.0)
ax_dos_avg.set_title("Averaged DoS")
ax_dos_avg.set_xlim(omegas[0], omegas[-1])
ax_dos_avg.grid(axis="both", ls="dashed", color="gray")
ax_dos_avg.axvline(mu, ls="dashed", color="orange", lw=1.0)

plt.show()
plt.close("all")
