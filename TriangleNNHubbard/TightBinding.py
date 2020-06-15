"""
Nearest-neighbor tight-binding model defined on the triangular lattice.
"""


import HamiltonianPy as HP
import matplotlib.pyplot as plt
import numpy as np

from utilities import Lorentzian

DEFAULT_MODEL_PARAMETERS = {"t": -1.0, "mu": 0.0}


def TightBinding(kpoints, cell, return_vectors=True, **model_params):
    """
    Solve the tight-binding model.

    Parameters
    ----------
    kpoints : 2D array with shape (N, 2)
        A collection of k-points in reciprocal space.
    cell : Lattice
        Unit cell of the triangle lattice.
    return_vectors : bool, optional
        Whether to return eigen-vectors.
        Default: True.
    model_params : other keyword argument, optional
        Model parameters.

    Returns
    -------
    values : array
        Eigen values corresponding to the given `kpoints`.
    vectors: array
        The corresponding eigen-vectors.
    """

    actual_model_params = dict(DEFAULT_MODEL_PARAMETERS)
    actual_model_params.update(model_params)
    t = actual_model_params["t"]
    mu = actual_model_params["mu"] / 2

    terms = []
    point_num = cell.point_num
    dR = np.array([0.0, 0.0], dtype=np.float64)
    for point in cell.points:
        index = cell.getIndex(point, fold=False)
        # Correspond to spin up and spin down, respectively
        terms.append((index, index, mu, dR))
        terms.append((index + point_num, index + point_num, mu, dR))

    intra_bonds, inter_bonds = cell.bonds(nth=1)
    for bond in intra_bonds + inter_bonds:
        p0, p1 = bond.endpoints
        p0_eqv, dR0 = cell.decompose(p0)
        p1_eqv, dR1 = cell.decompose(p1)
        index0 = cell.getIndex(p0_eqv, fold=False)
        index1 = cell.getIndex(p1_eqv, fold=False)
        # Correspond to spin up and spin down, respectively
        terms.append((index0, index1, t, dR1- dR0))
        terms.append((index0 + point_num, index1 + point_num, t, dR1 - dR0))

    shape = (kpoints.shape[0], 2 * point_num, 2 * point_num)
    HMs = np.zeros(shape, dtype=np.complex128)
    for i, j, coeff, dR in terms:
        HMs[:, i, j] += coeff * np.exp(1j * np.matmul(kpoints, dR))
    HMs += np.transpose(HMs, axes=(0, 2, 1)).conj()

    if return_vectors:
        return np.linalg.eigh(HMs)
    else:
        return np.linalg.eigvalsh(HMs)


def TypicalSolver(cell, enum, numk=100, gamma=0.01, **model_params):
    """
    Calculate the ground state energy, Fermi energy and density of states.

    Parameters
    ----------
    cell : Lattice
        Unit cell of the triangle lattice.
    enum : float or int
        The number of particle per unit-cell.
        The total particle number `numk * numk * enum` must be integer.
    numk : int, optional
        The number of k-point along each translation vector in reciprocal space.
        Default: 100.
    gamma : float, optional
        Specifying the width of the Lorentzian function.
        Default: 0.01.
    model_params : other keyword argument, optional
        Model parameters.

    Returns
    -------
    GE : float
        Ground state energy of the system.
    EF : float
        Fermi energy of the system.
    avg_particle_nums : array
        Particle number on each single particle state.
    omegas : array
        A collection of omegas.
    projected_dos
        Projected density of states corresponding to `omegas`.
    """

    total_particle_num = numk * numk * enum
    if total_particle_num != int(total_particle_num):
        raise ValueError("Total number of particle must be integer!")

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
    cell = HP.lattice_generator("triangle")
    M = cell.bs[0] / 2
    Gamma = np.array([0.0, 0.0])
    K = np.dot(np.array([2, 1]), cell.bs) / 3
    kpoints, indices = HP.KPath([Gamma, K, M])
    GKMGEs = TightBinding(kpoints, cell, return_vectors=False)
    GE, EF, avg_particle_nums, omegas, projected_dos = TypicalSolver(
        cell, enum=1, numk=500, gamma=0.05,
    )

    print("GE = {0}".format(GE))
    print("E_F = {0}".format(EF))
    print("Total particle number: {0:.6f}".format(np.sum(avg_particle_nums)))
    for i, num in enumerate(avg_particle_nums):
        print("Particle number on {0:2d}th state: {1:.6f}".format(i, num))

    fig, (ax_EB, ax_DOS) = plt.subplots(1, 2, sharey="all")
    ax_EB.plot(GKMGEs[:, 0], lw=4)
    ax_EB.plot(GKMGEs[:, 1], lw=2, ls="dashed")
    ax_EB.axhline(EF, ls="dashed", color="tab:red", lw=1)
    ax_EB.set_xticks(indices)
    ax_EB.set_xlim(0, kpoints.shape[0]-1)
    ax_EB.grid(True, ls="dashed", color="gray")
    ax_EB.set_xticklabels([r"$\Gamma$", "K", "M", r"$\Gamma$"])

    ax_DOS.plot(projected_dos[:, 0], omegas, lw=4)
    ax_DOS.plot(projected_dos[:, 1], omegas, lw=2, ls="dashed")
    ax_DOS.axhline(EF, ls="dashed", color="tab:red", lw=1)
    ax_DOS.grid(True, ls="dashed", color="gray")

    plt.get_current_fig_manager().window.showMaximized()
    plt.tight_layout()
    plt.show()
    plt.close("all")
