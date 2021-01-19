"""
Solve the tight-binding model Hamiltonian.
"""

import matplotlib.pyplot as plt
import numpy as np
from HamiltonianPy import Lattice, KPath

from database import POINTS, VECTORS
from utilities import Lorentzian

DEFAULT_MODEL_PARAMETERS = {"t0": -1.0, "t1": -1.0}


def TightBinding(kpoints, model="Model1", return_vectors=True, **model_params):
    actual_model_params = dict(DEFAULT_MODEL_PARAMETERS)
    actual_model_params.update(model_params)
    t0 = actual_model_params["t0"]
    t1 = actual_model_params["t1"]

    cell = Lattice(points=POINTS, vectors=VECTORS)
    ids = [[0, 0], [1, -1], [0, -1], [-1, 0], [-1, 1], [0, 1], [1, 0]]
    points_collection = np.concatenate(
        [np.dot([i, j], VECTORS) + POINTS for i, j in ids]
    )
    inter_hopping_indices = [
        [5, 83], [5, 82], [6, 82], [6, 81], [6, 63], [7, 63],
        [7, 71], [8, 71], [8, 53], [8, 52], [9, 52], [9, 51],
    ]
    if model == "Model1":
        intra_hopping_indices0 = [
            [0, 4], [4, 5], [1, 7], [7, 8], [2, 10], [10, 11]
        ]
        intra_hopping_indices1 = [
            [0, 1], [1, 2], [2, 0],
            [0, 3], [0, 10], [0, 11],
            [1, 4], [1, 5], [1, 6], [2, 7], [2, 8], [2, 9],
            [3, 4], [3, 11], [6, 5], [6, 7], [9, 8], [9, 10],
        ]
    else:
        intra_hopping_indices0 = [
            [0, 4], [4, 5], [5, 6],
            [1, 7], [7, 8], [8, 9],
            [2, 10], [10, 11], [11, 3],
        ]
        intra_hopping_indices1 = [
            [0, 1], [1, 2], [2, 0],
            [0, 3], [0, 10], [0, 11],
            [1, 4], [1, 5], [1, 6],
            [2, 7], [2, 8], [2, 9],
            [3, 4], [6, 7], [9, 10],
        ]

    terms = []
    for coeff, hopping_indices in [
        (t0, intra_hopping_indices0),
        (t1, intra_hopping_indices1), (t1, inter_hopping_indices),
    ]:
        for ij in hopping_indices:
            p0, p1 = points_collection[ij]
            p0_eqv, dR0 = cell.decompose(p0)
            p1_eqv, dR1 = cell.decompose(p1)
            index0 = cell.getIndex(p0_eqv, fold=False)
            index1 = cell.getIndex(p1_eqv, fold=False)
            terms.append((index0, index1, coeff, dR0 - dR1))

    site_num = cell.point_num
    kpoint_num = kpoints.shape[0]
    HMs = np.zeros((kpoint_num, site_num, site_num), dtype=np.complex128)
    for i, j, coeff, dR in terms:
        HMs[:, i, j] += coeff * np.exp(1j * np.matmul(kpoints, dR))
    HMs += np.transpose(HMs, (0, 2, 1)).conj()

    if return_vectors:
        return np.linalg.eigh(HMs)
    else:
        return np.linalg.eigvalsh(HMs)


def TypicalSolver(
        cell, model="Model1", nump=None, numk=100, gamma=0.01, **model_params
):
    if nump is None:
        nump = cell.point_num

    total_particle_num = numk * numk * nump
    if total_particle_num != int(total_particle_num):
        raise ValueError("Total number of particle must be integer!")

    ratio = np.linspace(0, 1, numk, endpoint=False)
    ratio_mesh = np.stack(
        np.meshgrid(ratio, ratio, indexing="ij"), axis=-1
    ).reshape((-1, 2))
    kpoints = np.matmul(ratio_mesh, cell.bs)
    del ratio, ratio_mesh

    BZMeshEs, BZMeshVectors = TightBinding(kpoints, model, **model_params)
    BZMeshEs = BZMeshEs.reshape((-1,))
    BZMeshProbs = np.transpose(
        (BZMeshVectors * BZMeshVectors.conj()).real, axes=(0, 2, 1)
    ).reshape((-1, BZMeshVectors.shape[1]))
    del BZMeshVectors

    kth = int(total_particle_num) // 2
    partition_indices = np.argpartition(BZMeshEs, kth=[kth - 1, kth])
    GE = 2 * np.sum(BZMeshEs[partition_indices[0:kth]])
    avg_particle_nums = 2 * np.sum(
        BZMeshProbs[partition_indices[0:kth]], axis=0
    )
    if total_particle_num % 2:
        mu = BZMeshEs[partition_indices[kth]]
        avg_particle_nums += BZMeshProbs[partition_indices[kth]]
        GE += mu
    else:
        index0, index1 = partition_indices[kth - 1:kth + 1]
        mu = (BZMeshEs[index0] + BZMeshEs[index1]) / 2
    GE /= (numk * numk)
    avg_particle_nums /= (numk * numk)
    del partition_indices

    E_min = np.min(BZMeshEs)
    E_max = np.max(BZMeshEs)
    extra = 0.1 * (E_max - E_min)
    omegas = np.arange(E_min - extra, E_max + extra, 0.01)
    projected_dos = np.array(
        [
            np.dot(
                Lorentzian(x=omega, x0=BZMeshEs, gamma=gamma), BZMeshProbs
            ) for omega in omegas
        ]
    ) / (numk * numk)
    return GE, mu, avg_particle_nums, omegas, projected_dos


if __name__ == "__main__":
    cell = Lattice(points=POINTS, vectors=VECTORS)
    M = cell.bs[0] / 2
    K = np.dot(np.array([2, 1]), cell.bs) / 3
    kpoints, indices = KPath([np.array([0.0, 0.0]), K, M])

    t0 = -1.00
    t1 = -1.00
    model = "Model1"
    GKMGEs = TightBinding(kpoints, model, return_vectors=False, t0=t0, t1=t1)
    GE, mu, avg_particle_nums, omegas, projected_dos = TypicalSolver(
        cell, model, numk=200, gamma=0.05, t0=t0, t1=t1
    )

    info = [
        "GE = {0}".format(GE), "Mu = {0}".format(mu),
        "Total particle number: {0:.6f}".format(np.sum(avg_particle_nums))
    ]
    for i, num in enumerate(avg_particle_nums):
        info.append("Particle number on {0:2d}th site: {1:.6f}".format(i, num))
    print("\n".join(info))

    fig_EB, ax_EB = plt.subplots()
    ax_EB.plot(GKMGEs, lw=4)
    ax_EB.set_xlim(0, kpoints.shape[0] - 1)
    ax_EB.axhline(mu, ls="dashed", color="tab:red", lw=2)
    ax_EB.set_xticks(indices)
    ax_EB.set_xticklabels([r"$\Gamma$", "K", "M", r"$\Gamma$"])
    ax_EB.tick_params(axis="both", labelsize="xx-large")
    ax_EB.grid(ls="dashed", lw=1.5, color="gray", axis="both")

    fig_dos, axes_dos = plt.subplots(2, 6, sharex="all", sharey="all")
    for index in range(12):
        ax = axes_dos[divmod(index, 6)]
        ax.plot(omegas, projected_dos[:, index], lw=4.0)
        ax.axvline(mu, ls="dashed", color="tab:red", lw=2.0)
        ax.set_title("index={0}".format(index))

    fig_dos_avg, axes_dos_avg = plt.subplots()
    dos_avg = np.mean(projected_dos, axis=1)
    axes_dos_avg.plot(omegas, dos_avg, lw=4.0)
    axes_dos_avg.axvline(mu, ls="dashed", color="tab:red", lw=2.0)
    axes_dos_avg.set_title("Averaged DoS")

    plt.show()
    plt.close("all")
