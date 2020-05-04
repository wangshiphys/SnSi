import matplotlib.pyplot as plt
import numpy as np
from HamiltonianPy import Lattice, KPath

from database import POINTS, VECTORS
from utilities import Lorentzian

DEFAULT_MODEL_PARAMETERS = {"t": -1.0, "mu0": 0.0, "mu1": 0.0}


def TightBinding(kpoints, cell, return_vectors=True, **model_params):
    actual_model_params = dict(DEFAULT_MODEL_PARAMETERS)
    actual_model_params.update(model_params)
    t = actual_model_params["t"]
    mu0 = actual_model_params["mu0"] / 2
    mu1 = actual_model_params["mu1"] / 2

    terms = []
    for point in cell.points:
        index = cell.getIndex(point, fold=False)
        mu = mu1 if index == 12 else mu0
        terms.append((2 * index, 2 * index, mu, np.array([0.0, 0.0])))
        terms.append((2 * index + 1, 2 * index + 1, mu, np.array([0.0, 0.0])))

    intra_bonds_1st, inter_bonds_1st = cell.bonds(nth=1)
    intra_bonds_2nd, inter_bonds_2nd = cell.bonds(nth=2)
    all_bonds = intra_bonds_1st + inter_bonds_1st
    all_bonds += intra_bonds_2nd + inter_bonds_2nd

    for bond in all_bonds:
        p0, p1 = bond.endpoints
        coeff = t / np.dot(p0 - p1, p0 - p1)
        p0_eqv, dR0 = cell.decompose(p0)
        p1_eqv, dR1 = cell.decompose(p1)
        index0 = cell.getIndex(p0_eqv, fold=False)
        index1 = cell.getIndex(p1_eqv, fold=False)
        terms.append((2 * index0, 2 * index1, coeff, dR1 - dR0))
        terms.append((2 * index0 + 1, 2 * index1 + 1, coeff, dR1 - dR0))

    # msg = "({0:2d}, {1:2d}), coeff={2:.6f}, dR=({3:.6f}, {4:.6f})"
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


def TypicalSolver(cell, enum=None, numk=100, gamma=0.01, **model_params):
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

    BZMeshEs, BZMeshVectors = TightBinding(kpoints, cell, **model_params)
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
    cluster_points = np.append(POINTS, [[0.0, 1 / np.sqrt(3)]], axis=0)
    cluster = Lattice(cluster_points, VECTORS)

    M = cluster.bs[0] / 2
    Gamma = np.array([0.0, 0.0])
    K = np.dot(np.array([2, 1]), cluster.bs) / 3
    kpoints, indices = KPath([Gamma, K, M])

    GKMGEs = TightBinding(kpoints, cluster, return_vectors=False)
    GE, mu, avg_particle_nums, omegas, projected_dos = TypicalSolver(
        cluster, numk=200, gamma=0.01
    )
    print("GE = {0}".format(GE))
    print("mu = {0}".format(mu))
    print("Total particle number: {0:.6f}".format(np.sum(avg_particle_nums)))
    for i, num in enumerate(avg_particle_nums):
        print("Particle number on {0:2d}th state: {1:.6f}".format(i, num))

    fig_EB, ax_EB = plt.subplots(num="EB")
    ax_EB.plot(GKMGEs[:, 0::2], lw=2, color="black")
    ax_EB.plot(GKMGEs[:, 1::2], lw=1, color="tab:red", ls="dashed")
    ax_EB.set_xlim(0, kpoints.shape[0] - 1)
    ax_EB.axhline(mu, ls="dashed", color="tab:green", lw=1)
    ax_EB.set_xticks(indices)
    ax_EB.set_xticklabels([r"$\Gamma$", "K", "M", r"$\Gamma$"])
    ax_EB.grid(ls="dashed", color="gray", axis="both")

    for index in range(cluster.point_num):
        fig_dos, ax_dos = plt.subplots(num="index={0}".format(index))
        ax_dos.plot(omegas, projected_dos[:, 2 * index], lw=2.0, color="black")
        ax_dos.plot(
            omegas, projected_dos[:, 2 * index + 1],
            lw=1.0, color="tab:red", ls="dashed",
        )
        ax_dos.set_xlim(omegas[0], omegas[-1])
        ax_dos.axvline(mu, ls="dashed", color="tab:green", lw=1.0)
        ax_dos.set_title("index={0}".format(index))
        ax_dos.grid(ls="dashed", color="gray", axis="both")

    fig_avg_dos, ax_avg_dos = plt.subplots(num="avg_dos")
    dos_avg = np.mean(projected_dos, axis=1)
    ax_avg_dos.plot(omegas, dos_avg, lw=2.0)
    ax_avg_dos.set_xlim(omegas[0], omegas[-1])
    ax_avg_dos.axvline(mu, ls="dashed", color="tab:green", lw=1.0)
    ax_avg_dos.set_title("Averaged DoS")
    ax_avg_dos.grid(ls="dashed", color="gray", axis="both")

    plt.show()
    plt.close("all")
