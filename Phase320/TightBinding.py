"""
Solve the tight-binding model Hamiltonian.
"""

import matplotlib.pyplot as plt
import numpy as np
from HamiltonianPy import Lattice, KPath

from database import ALL_POINTS, TRANSLATION_VECTORS
from utilities import Lorentzian

DEFAULT_MODEL_PARAMETERS = {"t0": -1.0, "t1": -1.0, "mu0": 0.0, "mu1": 0.0}


def TightBinding(kpoints, model="Model1", return_vectors=True, **model_params):
    actual_model_params = dict(DEFAULT_MODEL_PARAMETERS)
    actual_model_params.update(model_params)
    t0 = actual_model_params["t0"]
    t1 = actual_model_params["t1"]
    mu0 = actual_model_params["mu0"] / 2
    mu1 = actual_model_params["mu1"] / 2

    ids = ([0, 0], [-1, 0], [1, 0], [0, -1], [0, 1])
    cell = Lattice(ALL_POINTS[0:20], TRANSLATION_VECTORS)
    points_collection = np.concatenate(
        [np.dot(ij, TRANSLATION_VECTORS) + ALL_POINTS[0:20] for ij in ids]
    )
    intra_hopping_indices0 = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [0, 5], [0, 6], [0, 7], [0, 8], [0, 9],
        [10, 11], [10, 12], [10, 13], [10, 14],
        [10, 15], [10, 16], [10, 17], [10, 18], [10, 19],
    ]
    intra_hopping_indices1 = [
        [1, 2], [2, 3], [3, 4], [4, 5],
        [5, 6], [6, 7], [7, 8], [8, 9], [9, 1],
        [11, 12], [12, 13], [13, 14], [14, 15],
        [15, 16], [16, 17], [17, 18], [18, 19], [19, 11],
        [6, 11]
    ]
    inter_hopping_indices = [[3, 37], [9, 74]]
    if model == "Model2":
        intra_hopping_indices1 += [[6, 12], [6, 19], [5, 11], [7, 11]]
        inter_hopping_indices += [
            [3, 36], [3, 38], [2, 37], [4, 37],
            [9, 73], [9, 75], [1, 74], [8, 74],
        ]

    terms = []
    zero_dr = np.array([0.0, 0.0], dtype=np.float64)
    for point in cell.points:
        index = cell.getIndex(point, fold=False)
        mu = mu0 if index in (0, 10) else mu1
        terms.append((2 * index, 2 * index, mu, zero_dr))
        terms.append((2 * index + 1, 2 * index + 1, mu, zero_dr))

    for t, ijs in [(t0, intra_hopping_indices0), (t1, intra_hopping_indices1)]:
        for ij in ijs:
            p0, p1 = points_collection[ij]
            coeff = t / np.dot(p1 - p0, p1 - p0)
            index0 = cell.getIndex(p0, fold=False)
            index1 = cell.getIndex(p1, fold=False)
            terms.append((2 * index0, 2 * index1, coeff, zero_dr))
            terms.append((2 * index0 + 1, 2 * index1 + 1, coeff, zero_dr))
    for ij in inter_hopping_indices:
        p0, p1 = points_collection[ij]
        coeff = t1 / np.dot(p1 - p0, p1 - p0)
        p0_eqv, dR0 = cell.decompose(p0)
        p1_eqv, dR1 = cell.decompose(p1)
        index0 = cell.getIndex(p0_eqv, fold=False)
        index1 = cell.getIndex(p1_eqv, fold=False)
        terms.append((2 * index0, 2 * index1, coeff, dR1 - dR0))
        terms.append((2 * index0 + 1, 2 * index1 + 1, coeff, dR1 - dR0))

    # msg = "({0:2d},{1:2d}), t={2:.8f}, dR=({3:.8f}, {4:.8f})"
    # print("Hamiltonian Terms:")
    # for i, j, coeff, dR in terms:
    #     print(msg.format(i, j, coeff, dR[0], dR[1]))

    point_num = cell.point_num
    shape = (kpoints.shape[0], 2 * point_num, 2 * point_num)
    HMs = np.zeros(shape, dtype=np.complex128)
    for i, j, coeff, dR in terms:
        HMs[:, i, j] += coeff * np.exp(1j * np.matmul(kpoints, dR))
    HMs += np.transpose(HMs, (0, 2, 1)).conj()

    if return_vectors:
        return np.linalg.eigh(HMs)
    else:
        return np.linalg.eigvalsh(HMs)


def TypicalSolver(
        cell, model="Model1", enum=None, numk=100, gamma=0.01, **model_params
):
    if enum is None:
        enum = cell.point_num

    total_particle_num = numk * numk * enum
    if total_particle_num != int(total_particle_num):
        raise ValueError("The total number of particle must be integer!")

    ratio = np.linspace(0, 1, numk, endpoint=False)
    ratio_mesh = np.stack(
        np.meshgrid(ratio, ratio, indexing="ij"), axis=-1
    ).reshape((-1, 2))
    kpoints = np.matmul(ratio_mesh, cell.bs)
    kpoint_num = kpoints.shape[0]
    del ratio, ratio_mesh

    BZMeshEs, BZMeshVectors = TightBinding(kpoints, model, **model_params)
    BZMeshEs = BZMeshEs.reshape((-1,))
    BZMeshProbs = np.transpose(
        (BZMeshVectors * BZMeshVectors.conj()).real, axes=(0, 2, 1)
    ).reshape((-1, BZMeshVectors.shape[1]))
    del BZMeshVectors

    kth = int(total_particle_num)
    partition_indices = np.argpartition(BZMeshEs, kth=[kth - 1, kth])
    GE = np.sum(BZMeshEs[partition_indices[0:kth]]) / kpoint_num
    avg_particle_nums = np.sum(
        BZMeshProbs[partition_indices[0:kth]], axis=0
    ) / kpoint_num

    index0, index1 = partition_indices[[kth - 1, kth]]
    EF = (BZMeshEs[index0] + BZMeshEs[index1]) / 2
    del partition_indices

    E_min = np.min(BZMeshEs)
    E_max = np.max(BZMeshEs)
    extra = 0.1 * (E_max - E_min)
    omegas = np.arange(E_min - extra, E_max + extra, 0.01)
    projected_dos = np.array(
        [
            np.dot(Lorentzian(omega, x0=BZMeshEs, gamma=gamma), BZMeshProbs)
            for omega in omegas
        ]
    ) / kpoint_num
    return GE, EF, avg_particle_nums, omegas, projected_dos


if __name__ == "__main__":
    cell = Lattice(ALL_POINTS[0:20], TRANSLATION_VECTORS)

    M = cell.bs[0] / 2
    K = np.dot(np.array([2, 1]), cell.bs) / 3
    kpoints, indices = KPath([np.array([0.0, 0.0]), K, M])

    t0 = -1.0
    t1 = -1.0
    model = "Model1"
    GKMGEs = TightBinding(kpoints, model, return_vectors=False, t0=t0, t1=t1)
    GE, mu, avg_particle_nums, omegas, projected_dos = TypicalSolver(
        cell, model, numk=200, gamma=0.02, t0=t0, t1=t1
    )

    print("GE = {0}".format(GE))
    print("mu = {0}".format(mu))
    print("Total particle number: {0:.6f}".format(np.sum(avg_particle_nums)))
    for i, num in enumerate(avg_particle_nums):
        print("Particle number on {0:2d}th state: {1:.6f}".format(i, num))

    fig_EB, ax_EB = plt.subplots()
    ax_EB.plot(GKMGEs[:, 0::2], lw=2, color="black")
    ax_EB.plot(GKMGEs[:, 1::2], lw=1, color="tab:red", ls="dashed")
    ax_EB.set_xlim(0, kpoints.shape[0] - 1)
    ax_EB.axhline(mu, ls="dashed", color="tab:green", lw=1)
    ax_EB.set_xticks(indices)
    ax_EB.set_xticklabels([r"$\Gamma$", "K", "M", r"$\Gamma$"])
    ax_EB.grid(ls="dashed", color="gray", axis="both")
    fig_EB.set_size_inches(3, 4.2)

    fig_dos0, axes_dos0 = plt.subplots(2, 5, sharex="all", sharey="all")
    fig_dos1, axes_dos1 = plt.subplots(2, 5, sharex="all", sharey="all")
    for index in range(20):
        if index < 10:
            ax = axes_dos0[divmod(index, 5)]
        else:
            ax = axes_dos1[divmod(index - 10, 5)]
        ax.plot(omegas, projected_dos[:, 2 * index], lw=2.0, color="black")
        ax.plot(
            omegas, projected_dos[:, 2 * index + 1],
            lw=1.0, color="tab:red", ls="dashed",
        )
        ax.set_xlim(omegas[0], omegas[-1])
        ax.axvline(mu, ls="dashed", color="tab:green", lw=1.0)
        ax.set_title("index={0}".format(index))
        ax.grid(ls="dashed", color="gray", axis="both")

    fig_dos_avg, axes_dos_avg = plt.subplots()
    dos_avg = np.mean(projected_dos, axis=1)
    axes_dos_avg.plot(omegas, dos_avg, lw=2.0)
    axes_dos_avg.set_xlim(omegas[0], omegas[-1])
    axes_dos_avg.axvline(mu, ls="dashed", color="tab:green", lw=1.0)
    axes_dos_avg.set_title("Averaged DoS")
    axes_dos_avg.grid(ls="dashed", color="gray", axis="both")

    plt.show()
    prefix = "fig/{0}_t0={1:.3f}_t1={2:.3f}".format(model, t0, t1)
    fig_EB.savefig(prefix + "_EB.png", transparent=True)
    fig_dos0.savefig(prefix + "_DOS0.png", transparent=True)
    fig_dos1.savefig(prefix + "_DOS1.png", transparent=True)
    fig_dos_avg.savefig(prefix + "_DOS_AVG.png", transparent=True)
    plt.close("all")
