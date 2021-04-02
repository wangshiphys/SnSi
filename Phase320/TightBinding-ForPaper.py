import matplotlib.pyplot as plt
import numpy as np
from HamiltonianPy import Lattice

from database import ALL_POINTS, TRANSLATION_VECTORS
from utilities import Lorentzian

DEFAULT_MODEL_PARAMETERS = {"t0": -1.0, "t1": -1.0}


def TightBinding(kpoints, return_vectors=True, **model_params):
    actual_model_params = dict(DEFAULT_MODEL_PARAMETERS)
    actual_model_params.update(model_params)
    t0 = actual_model_params["t0"]
    t1 = actual_model_params["t1"]

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
        [6, 11],
        [1, 2], [2, 3], [3, 4], [4, 5],
        [5, 6], [6, 7], [7, 8], [8, 9], [9, 1],
        [11, 12], [12, 13], [13, 14], [14, 15],
        [15, 16], [16, 17], [17, 18], [18, 19], [19, 11],
    ]
    inter_hopping_indices = [[3, 37], [9, 74]]

    terms = []
    zero_dr = np.array([0.0, 0.0], dtype=np.float64)
    for t, ijs in [(t0, intra_hopping_indices0), (t1, intra_hopping_indices1)]:
        for ij in ijs:
            p0, p1 = points_collection[ij]
            index0 = cell.getIndex(p0, fold=False)
            index1 = cell.getIndex(p1, fold=False)
            terms.append((2 * index0, 2 * index1, t, zero_dr))
            terms.append((2 * index0 + 1, 2 * index1 + 1, t, zero_dr))
    for ij in inter_hopping_indices:
        p0, p1 = points_collection[ij]
        p0_eqv, dR0 = cell.decompose(p0)
        p1_eqv, dR1 = cell.decompose(p1)
        index0 = cell.getIndex(p0_eqv, fold=False)
        index1 = cell.getIndex(p1_eqv, fold=False)
        terms.append((2 * index0, 2 * index1, t1, dR1 - dR0))
        terms.append((2 * index0 + 1, 2 * index1 + 1, t1, dR1 - dR0))

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

    BZMeshEs, BZMeshVectors = TightBinding(kpoints, **model_params)
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
    GE, mu, avg_particle_nums, omegas, projected_dos = TypicalSolver(
        cell, numk=100, gamma=0.02, t0=-0.5, t1=-1.0
    )

    interval = 2.0
    fig, ax = plt.subplots()
    line_avg, = ax.plot(omegas, np.mean(projected_dos, axis=1), lw=3.0)
    line_site0, = ax.plot(omegas, projected_dos[:, 0] + 1*interval, lw=3.0)
    line_site1, = ax.plot(omegas, projected_dos[:, 2] + 2*interval, lw=3.0)
    line_site3, = ax.plot(omegas, projected_dos[:, 6] + 3*interval, lw=3.0)
    ax.axvline(mu, ls="dashed", lw=2.0, color="black", zorder=0)
    ax.legend(
        [line_avg, line_site0, line_site1,  line_site3],
        ["avg", "site-0", "site-1", "site-3"],
        ncol=2, columnspacing=1.0, loc="upper left", fontsize=17, frameon=True,
    )

    ax.set_xlim(-3.5, 2.5)
    ax.set_ylim(-0.5, 15.1)
    ax.set_yticks([0.0, 1*interval, 2*interval, 3*interval])
    ax.set_xticks([-3, -2, -1, 0, 1, 2])
    ax.set_yticklabels(["", "", "", ""])
    ax.set_xticklabels(["-3", "-2", "-1", "0", "1", "2"], fontsize=20)
    ax.set_xlabel(r"$\omega/t_1$", fontsize=20)
    ax.set_ylabel("DOS (arb. units)", fontsize=20)
    ax.tick_params(axis="y", left=False)
    ax.grid(axis="y", ls="dashed", lw=1.5, color="gray")

    plt.tight_layout()
    plt.show()
    fig.savefig("TBAForPhase1.pdf", transparent=True)
    plt.close("all")
