"""
Solve the tight-binding model.
"""


__all__ = [
    "TightBinding", "TypicalSolver",
]


import numpy as np

from utilities import Lorentzian


DEFAULT_MODEL_PARAMETERS = {
    "t0": 0.0,
    "t1": 1.0,
}


def TightBinding(kpoints, cell, return_vectors=True, **model_params):
    """
    Solve the tight-binding model.

    Parameters
    ----------
    kpoints : 2D array with shape (N,, 2)
        A collection of points in reciprocal space.
    cell : Lattice
        Unit cell of the model.
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
        The corresponding eigen vectors.
    """

    actual_model_params = dict(DEFAULT_MODEL_PARAMETERS)
    actual_model_params.update(model_params)
    t0 = actual_model_params["t0"]
    t1 = actual_model_params["t1"]

    site_num = cell.point_num
    kpoint_num = kpoints.shape[0]
    intra_1st, inter_1st = cell.bonds(nth=1, only=True)
    intra_3rd, inter_3rd = cell.bonds(nth=3, only=True)
    all_bonds = [intra_1st + inter_1st, intra_3rd + inter_3rd]

    terms = []
    for bonds, coeff in zip(all_bonds, [t0, t1]):
        for bond in bonds:
            p0, p1 = bond.endpoints
            p0_eqv, dR0 = cell.decompose(p0)
            p1_eqv, dR1 = cell.decompose(p1)
            index0 = cell.getIndex(p0_eqv, fold=False)
            index1 = cell.getIndex(p1_eqv, fold=False)
            terms.append((index0, index1, coeff, dR0 - dR1))

    HMs = np.zeros((kpoint_num, site_num, site_num), dtype=np.complex128)
    for i, j, coeff, dR in terms:
        HMs[:, i, j] += coeff * np.exp(1j * np.matmul(kpoints, dR))
    HMs += np.transpose(HMs, (0, 2, 1)).conj()

    if return_vectors:
        return np.linalg.eigh(HMs)
    else:
        return np.linalg.eigvalsh(HMs)


def TypicalSolver(cell, nump=13, numk=100, gamma=0.01, **model_params):
    """
    Calculate the ground state energy, chemical potential and density of states.

    Parameters
    ----------
    cell : Lattice
        Unit cell of the tight-binding model.
    nump : float or int, optional
        The number of particle per unit-cell.
        The total particle number `numk * numk * nump` must be integer.
        Default: 13.
    numk : int, optional
        The number of k-point along each translation vector in reciprocal space.
        Default: 100.
    gamma : float, optional
        Specifying the width of the Lorentzian function
        Default: 0.01.
    model_params : other keyword argument, optional
        Model parameters.

    Returns
    -------
    GE : float
        Ground state energy of the system.
    Mu : float
        Chemical potential of the system.
    avg_particle_nums : array
        Particle number on each lattice site.
    omegas : array
        A collection of omegas.
    projected_dos
        Projected density of states corresponding to `omegas`.
    """

    total_particle_num = numk * numk * nump
    if total_particle_num != int(total_particle_num):
        raise ValueError("The total number of particle must be integer")

    ratio = np.linspace(0, 1, numk, endpoint=False)
    ratio_mesh = np.stack(
        np.meshgrid(ratio, ratio, indexing="ij"), axis=-1
    ).reshape((-1, 2))
    kpoints = np.matmul(ratio_mesh, cell.bs)
    del ratio, ratio_mesh

    BZMeshEs, BZMeshVectors = TightBinding(kpoints, cell, **model_params)
    BZMeshEs = BZMeshEs.reshape((-1, ))
    BZMeshProbs = np.transpose(
        (BZMeshVectors * BZMeshVectors.conj()).real, axes=(0, 2, 1)
    ).reshape((-1, BZMeshVectors.shape[1]))
    del BZMeshVectors

    kth = int(total_particle_num) // 2
    partition_indices = np.argpartition(BZMeshEs, kth=[kth-1, kth])
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
                Lorentzian(xs=omega, x0=BZMeshEs, gamma=gamma), BZMeshProbs
            ) for omega in omegas
        ]
    ) / (numk * numk)
    return GE, mu, avg_particle_nums, omegas, projected_dos



