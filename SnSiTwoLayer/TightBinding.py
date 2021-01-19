import matplotlib.pyplot as plt
import numpy as np
from HamiltonianPy import KPath, Lattice


SQRT3 = np.sqrt(3)
POINTS = np.array(
    [[0.0, 0.0], [1.0, SQRT3], [2.0, 0.0], [3.0, SQRT3], [1.0, 1.0 / SQRT3]],
    dtype=np.float64
)
VECTORS = np.array([[4.0, 0.0], [2.0, 2 * SQRT3]], dtype=np.float64)


t0 = 0.0
t1 = -1.00
t2 = -0.25
id = "t0={0:.2f}_t1={1:.2f}_t2={2:.2f}".format(t0, t1, t2)
cluster = Lattice(POINTS, VECTORS)
intra_bonds_1st, inter_bonds_1st = cluster.bonds(nth=1)
intra_bonds_2nd, inter_bonds_2nd = cluster.bonds(nth=2)
intra_bonds_6th, inter_bonds_6th = cluster.bonds(nth=6)

HTerms = []
for bond in intra_bonds_1st + inter_bonds_1st:
    p0, p1 = bond.endpoints
    p0_eqv, dR0 = cluster.decompose(p0)
    p1_eqv, dR1 = cluster.decompose(p1)
    index0 = cluster.getIndex(p0_eqv)
    index1 = cluster.getIndex(p1_eqv)
    HTerms.append((index0, index1, t0, dR0 - dR1))

for bond in intra_bonds_2nd + inter_bonds_2nd:
    p0, p1 = bond.endpoints
    p0_eqv, dR0 = cluster.decompose(p0)
    p1_eqv, dR1 = cluster.decompose(p1)
    index0 = cluster.getIndex(p0_eqv)
    index1 = cluster.getIndex(p1_eqv)
    HTerms.append((index0, index1, t1, dR0 - dR1))

for bond in intra_bonds_6th + inter_bonds_6th:
    p0, p1 = bond.endpoints
    p0_eqv, dR0 = cluster.decompose(p0)
    p1_eqv, dR1 = cluster.decompose(p1)
    index0 = cluster.getIndex(p0_eqv)
    index1 = cluster.getIndex(p1_eqv)
    if index0 == 4 and index1 == 4:
        HTerms.append((index0, index1, t2, dR0 - dR1))

site_num = cluster.point_num
bs = cluster.bs
Gamma = np.array([0.0, 0.0], dtype=np.float64)
M = bs[0] / 2
K = (2 * bs[0] + bs[1]) / 3
kpoints, indices = KPath([Gamma, K, M])

HMs = np.zeros((kpoints.shape[0], site_num, site_num), dtype=np.complex128)
for i, j, coeff, dR in HTerms:
    HMs[:, i, j] += coeff * np.exp(1j * np.matmul(kpoints, dR))
HMs += np.transpose(HMs, (0, 2, 1)).conj()

values = np.linalg.eigvalsh(HMs)
fig, ax = plt.subplots()
ax.plot(values)
ax.set_xlim(0, len(kpoints))
ax.set_xticks(indices)
ax.set_xticklabels([r"$\Gamma$", "$K$", "$M$", r"$\Gamma$"])
ax.set_title(id)
ax.grid(True, axis="x")
plt.show()
# fig.savefig("TBEB_" + id + ".png", dpi=300)
plt.close("all")
