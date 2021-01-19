import matplotlib.pyplot as plt
import numpy as np

from utilities import Mu

U = 6.0
run = 1
site_num = 13
amplitude = 0.0
data_path = "data/dos/amplitude={0:.2f}/run={1}/".format(amplitude, run)
data_name = "Phase12Line_t=-1.00_U={0:.2f}.npz".format(U)

with np.load(data_path + data_name) as ld:
    dos = ld["dos"]
    omegas = ld["omegas"]
domega = omegas[1] - omegas[0]

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
particle_num = np.sum(total_dos[omegas < mu]) * domega
local_particle_num = np.sum(dos[omegas < mu], axis=0) * domega

print("U = {0:.1f}".format(U))
print("Sum of DoS: {0:.8f}".format(np.sum(dos) * domega))
print("mu_p = {0:.8f}, mu_h = {1:.8f}, mu = {2:.8f}".format(mu_p, mu_h, mu))
print("Particle number: {0:.8f}".format(particle_num))
for index in range(site_num):
    num = local_particle_num[index] + local_particle_num[index+site_num]
    print("Particle number on {0:0>2d}th site: {1:8f}".format(index, num))

for index in range(site_num):
    fig, ax = plt.subplots(num="Index={0:0>2d}".format(index))
    ax.plot(omegas, dos[:, index], lw=2, color="black")
    ax.plot(omegas, dos[:, index+site_num], lw=1, color="red", ls="dashed")
    ax.set_xlim(omegas[0], omegas[-1])
    ax.grid(axis="both", ls="dashed", color="gray")
    ax.axvline(mu, ls="dashed", color="orange", lw=1)
    ax.set_title("Index={0:0>2d}".format(index), fontsize="xx-large")

fig_dos_avg, ax_dos_avg = plt.subplots(num="Avg_DOS")
ax_dos_avg.plot(omegas, avg_dos, lw=2)
ax_dos_avg.set_title("Averaged DoS")
ax_dos_avg.set_xlim(omegas[0], omegas[-1])
ax_dos_avg.grid(axis="both", ls="dashed", color="gray")
ax_dos_avg.axvline(mu, ls="dashed", color="orange", lw=1)

plt.show()
plt.close("all")
