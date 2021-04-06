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
    GE, mu, avg_particle_nums, omegas, projected_dos = TypicalSolver(
        cell, numk=200, gamma=0.02, t0=-1.0, t1=-0.5
    )

    step = 1.0
    baselines = [0.0, step, 2 * step, 3 * step, 4 * step]
    labels = ["avg", "site-0", "site-3", "site-4", "site-5"]

    fig, ax = plt.subplots()
    ax.plot(omegas, np.mean(projected_dos, axis=1) + baselines[0], lw=4.0)
    ax.plot(omegas, projected_dos[:, 0] + baselines[1], lw=4.0)
    ax.plot(omegas, projected_dos[:, 3] + baselines[2], lw=4.0)
    ax.plot(omegas, projected_dos[:, 4] + baselines[3], lw=4.0)
    ax.plot(omegas, projected_dos[:, 5] + baselines[4], lw=4.0)
    for baseline, label in zip(baselines, labels):
        ax.text(
            -3.0, baseline + 0.3, label, ha="center", va="bottom", fontsize=20
        )
    ax.axvline(mu, ls="dashed", lw=2.0, color="black", zorder=0)

    ax.set_xlim(-3.7, 2.1)
    ax.set_ylim(-0.1, 6.2)
    ax.set_yticks(baselines)
    ax.set_yticklabels([""] * len(baselines))
    ax.set_xticks([-3, -2, -1, 0, 1, 2])
    ax.set_xticklabels(["-3", "-2", "-1", "0", "1", "2"], fontsize=20)
    ax.set_xlabel(r"$\omega/t_0$", fontsize=20)
    ax.set_ylabel("DOS (arb. units)", fontsize=20)
    ax.tick_params(axis="y", left=False)
    ax.grid(axis="y", ls="dashed", lw=1.5, color="gray")

    # top = 0.99,
    # bottom = 0.15,
    # left = 0.092,
    # right = 0.976,
    # hspace = 0.2,
    # wspace = 0.2

    plt.tight_layout()
    plt.show()
    fig.savefig("fig/TBAForPhase2AndPhase3.pdf", transparent=True)
    plt.close("all")
