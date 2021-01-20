import matplotlib.pyplot as plt
import numpy as np

from utilities import Mu

site_num = 12
data_name = "data/DOS_{model}_t0={t0:.3f}_t1={t1:.3f}_U={U:.3f}.npz".format(
    model="Model1", t0=-1.00, t1=-1.00, U=0.00
)
with np.load(data_name) as ld:
    dos = ld["dos"]
    omegas = ld["omegas"]
domega = omegas[1] - omegas[0]

avg_dos = np.mean(dos, axis=1)
total_dos = np.sum(dos, axis=1)
mu_h = Mu(total_dos, omegas, site_num, 2*site_num, reverse=True)
mu_p = Mu(total_dos, omegas, site_num, 2*site_num, reverse=False)
mu = (mu_p + mu_h) / 2
particle_num = np.sum(total_dos[omegas < mu]) * domega
local_particle_num = np.sum(dos[omegas < mu], axis=0) * domega

print(data_name)
print("Sum of DoS: {0:.8f}".format(np.sum(dos) * domega))
print("mu_p = {0:.8f}, mu_h = {1:.8f}, mu = {2:.8f}".format(mu_p, mu_h, mu))
print("Particle number: {0:.8f}".format(particle_num))
for index in range(site_num):
    num = local_particle_num[2*index] + local_particle_num[2*index+1]
    print("Particle number on {0:0>2d}th site: {1:8f}".format(index, num))

fig, axes = plt.subplots(2, 6, sharex="all", sharey="all")
for index in range(site_num):
    ax = axes[divmod(index, 6)]
    ax.plot(omegas, dos[:, 2*index], lw=2, color="black")
    ax.plot(omegas, dos[:, 2*index+1], lw=1, color="red", ls="dashed")
    ax.set_xlim(omegas[0], omegas[-1])
    ax.grid(axis="both", ls="dashed", color="gray")
    ax.axvline(mu, ls="dashed", color="orange", lw=1)
    ax.set_title("Site={0:0>2d}".format(index), fontsize="xx-large")

fig_dos_avg, ax_dos_avg = plt.subplots(num="Avg_DOS")
ax_dos_avg.plot(omegas, avg_dos, lw=4)
ax_dos_avg.set_title("Averaged DoS")
ax_dos_avg.set_xlim(omegas[0], omegas[-1])
ax_dos_avg.grid(axis="both", ls="dashed", color="gray")
ax_dos_avg.axvline(mu, ls="dashed", color="orange", lw=1)

plt.show()
plt.close("all")
