"""
This module provides utility programs used in this project.
"""


__all__ = [
    "Lorentzian", "Mu",
    "ClusterRGFSolver", "SPINS",
    "CPTPerturbation", "Fourier",
]


import HamiltonianPy as HP
import numpy as np
from scipy.optimize import brentq
from scipy.sparse.linalg import eigsh

SPINS = (HP.SPIN_DOWN, HP.SPIN_UP)


# Simulation of the Delta function
def Lorentzian(x, x0=0.0, gamma=0.01):
    """
    The Lorentzian function.

    Parameters
    ----------
    x : float or array of floats
        The independent variable of the Lorentzian function.
    x0 : float or array of floats, optional
        The center of the Lorentzian function.
        Default: 0.0.
    gamma : float, optional
        Specifying the width of the Lorentzian function.
        Default: 0.01.

    Returns
    -------
    res : float or array of floats
        1. `x` and `x0` are both scalar, then the corresponding function
        value is returned;
        2. `x` and/or `x0` are array of floats, the two parameters are
        broadcasted to calculated the expression `x - x0`, and the
        corresponding function values are returned.

    See also
    --------
    numpy.broadcast
    http://mathworld.wolfram.com/LorentzianFunction.html
    """

    gamma /= 2
    return gamma / np.pi / ((x - x0) ** 2 + gamma ** 2)


def _MuCore(mu, dos, omegas, occupied_num, total_num, reverse=False):
    domega = omegas[1] - omegas[0]
    if reverse:
        num = total_num - occupied_num
        indices = omegas > mu
    else:
        num = occupied_num
        indices = omegas < mu
    return np.sum(dos[indices]) * domega - num


def Mu(dos, omegas, occupied_num, total_num, reverse=False):
    args = (dos, omegas, occupied_num, total_num, reverse)
    mu, info = brentq(
        _MuCore, a=omegas[0], b=omegas[-1], args=args, full_output=True
    )
    if info.converged:
        return mu
    else:
        print(info)
        raise RuntimeError("Not converged!")


def _NumberOfSpinUpAndDown(enum, total_sz):
    for spin_up_num in range(enum + 1):
        spin_down_num = enum - spin_up_num
        sz = (spin_up_num - spin_down_num) / 2
        if sz == total_sz:
            return spin_up_num, spin_down_num
    raise ValueError("The given `enum` and `total_sz` cannot be satisfied.")


def _HMGenerator(HTerms, state_indices_table, basis):
    HM = 0.0
    for term in HTerms:
        HM += term.matrix_repr(state_indices_table, basis)
    HM += HM.getH()
    return HM


# Particle number is conserved
def _ClusterRGFSolver0(HTerms, cluster, omegas, enum, eta):
    creators = [
        HP.AoC(HP.CREATION, site=point, spin=spin)
        for spin in SPINS for point in cluster.points
    ]
    annihilators = [creator.dagger() for creator in creators]
    state_indices_table = HP.IndexTable(
        HP.StateID(site=point, spin=spin)
        for spin in SPINS for point in cluster.points
    )

    states_num = len(state_indices_table)
    basis = HP.base_vectors([states_num, enum])
    basis_p = HP.base_vectors([states_num, enum + 1])
    basis_h = HP.base_vectors([states_num, enum - 1])

    HM = _HMGenerator(HTerms, state_indices_table, basis)
    HM_P = _HMGenerator(HTerms, state_indices_table, basis_p)
    HM_H = _HMGenerator(HTerms, state_indices_table, basis_h)

    # noinspection PyTypeChecker
    values, vectors = eigsh(HM, k=1, which="SA")
    GE = values[0]
    GS = vectors[:, 0]
    del HM

    excited_states_p = {}
    excited_states_h = {}
    for creator in creators:
        excited_states_p[creator] = creator.matrix_repr(
            state_indices_table, basis, left_bases=basis_p
        ).dot(GS)
    for annihilator in annihilators:
        excited_states_h[annihilator] = annihilator.matrix_repr(
            state_indices_table, basis, left_bases=basis_h
        ).dot(GS)
    del state_indices_table, GS, vectors, basis, basis_p, basis_h

    projected_matrices_p, projected_vectors_p = HP.MultiKrylov(
        HM_P, excited_states_p
    )
    del HM_P, excited_states_p
    projected_matrices, projected_vectors = HP.MultiKrylov(
        HM_H, excited_states_h
    )
    del HM_H, excited_states_h
    projected_vectors.update(projected_vectors_p)
    projected_matrices.update(projected_matrices_p)

    gfs_dict = HP.RGFSolverLanczosMultiple(
        omegas=omegas, As=annihilators, Bs=creators, GE=GE,
        projected_matrices=projected_matrices,
        projected_vectors=projected_vectors,
        eta=eta, structure="dict",
    )
    return gfs_dict


# Spin component is conserved
def _ClusterRGFSolver1(HTerms, cluster, omegas, enum, total_sz, eta):
    spin_up_num, spin_down_num = _NumberOfSpinUpAndDown(enum, total_sz)
    state_indices_table = HP.IndexTable(
        HP.StateID(site=site, spin=spin)
        for spin in SPINS for site in cluster.points
    )
    spin_up_state_indices = [
        state_indices_table(HP.StateID(site=site, spin=HP.SPIN_UP))
        for site in cluster.points
    ]
    spin_down_state_indices = [
        state_indices_table(HP.StateID(site=site, spin=HP.SPIN_DOWN))
        for site in cluster.points
    ]
    basis = HP.base_vectors(
        [spin_up_state_indices, spin_up_num],
        [spin_down_state_indices, spin_down_num],
    )

    HM = _HMGenerator(HTerms, state_indices_table, basis)
    # noinspection PyTypeChecker
    values, vectors = eigsh(HM, k=1, which="SA")
    GE = values[0]
    GS = vectors[:, 0]
    del HM

    gfs_dict = {}
    for spin in SPINS:
        if spin == HP.SPIN_UP:
            basis_p = HP.base_vectors(
                [spin_up_state_indices, spin_up_num + 1],
                [spin_down_state_indices, spin_down_num],
            )
            basis_h = HP.base_vectors(
                [spin_up_state_indices, spin_up_num - 1],
                [spin_down_state_indices, spin_down_num],
            )
        else:
            basis_p = HP.base_vectors(
                [spin_up_state_indices, spin_up_num],
                [spin_down_state_indices, spin_down_num + 1],
            )
            basis_h = HP.base_vectors(
                [spin_up_state_indices, spin_up_num],
                [spin_down_state_indices, spin_down_num - 1],
            )

        creators = [HP.AoC(HP.CREATION, site, spin) for site in cluster.points]
        annihilators = [creator.dagger() for creator in creators]
        HM_P = _HMGenerator(HTerms, state_indices_table, basis_p)
        HM_H = _HMGenerator(HTerms, state_indices_table, basis_h)

        excited_states_p = {}
        excited_states_h = {}
        for creator in creators:
            excited_states_p[creator] = creator.matrix_repr(
                state_indices_table, basis, left_bases=basis_p
            ).dot(GS)
        for annihilator in annihilators:
            excited_states_h[annihilator] = annihilator.matrix_repr(
                state_indices_table, basis, left_bases=basis_h
            ).dot(GS)
        del basis_h, basis_p

        projected_matrices, projected_vectors = HP.MultiKrylov(
            HM_P, excited_states_p
        )
        del HM_P, excited_states_p
        projected_matrices_h, projected_vectors_h = HP.MultiKrylov(
            HM_H, excited_states_h
        )
        del HM_H, excited_states_h
        projected_vectors.update(projected_vectors_h)
        projected_matrices.update(projected_matrices_h)

        gfs_spin = HP.RGFSolverLanczosMultiple(
            omegas=omegas, As=annihilators, Bs=creators, GE=GE,
            projected_matrices=projected_matrices,
            projected_vectors=projected_vectors,
            eta=eta, structure="dict",
        )
        gfs_dict.update(gfs_spin)
    return gfs_dict


def ClusterRGFSolver(
        HTerms, cluster, omegas, enum=None, total_sz=None,
        eta=0.01, structure="array",
):
    point_num = cluster.point_num
    if enum is None:
        enum = point_num
    assert isinstance(enum, int) and (0 < enum < (2 * point_num))

    if total_sz is None:
        gfs_dict = _ClusterRGFSolver0(HTerms, cluster, omegas, enum, eta)
    else:
        # `total_sz` must be half-integer
        assert isinstance(total_sz, (int, float)) and (
            int(2 * total_sz) == (2 * total_sz)
        )
        gfs_dict = _ClusterRGFSolver1(
            HTerms, cluster, omegas, enum, total_sz, eta
        )

    if structure == "array":
        creators = [
            HP.AoC(HP.CREATION, site, spin)
            for spin in SPINS for site in cluster.points
        ]
        annihilators = [creator.dagger() for creator in creators]

        dim = len(creators)
        gfs_array = np.zeros((len(omegas), dim, dim), dtype=np.complex128)
        for row, creator in enumerate(creators):
            for col, annihilator in enumerate(annihilators):
                key = (creator.dagger(), annihilator.dagger())
                if key in gfs_dict:
                    gfs_array[:, row, col] = gfs_dict[key]
        return gfs_array
    else:
        return gfs_dict


def CPTPerturbation(VTerms, cluster, kpoints):
    creators = [
        HP.AoC(HP.CREATION, site=site, spin=spin)
        for spin in SPINS for site in cluster.points
    ]
    annihilators = [creator.dagger() for creator in creators]

    dim = len(creators)
    VMatrices = np.zeros((kpoints.shape[0], dim, dim), dtype=np.complex128)
    for term in VTerms:
        creator, annihilator = term.components
        p0 = creator.site
        p1 = annihilator.site
        p0_eqv, dR0 = cluster.decompose(p0)
        p1_eqv, dR1 = cluster.decompose(p1)
        row = creators.index(creator.derive(site=p0_eqv))
        col = annihilators.index(annihilator.derive(site=p1_eqv))
        VMatrices[:, row, col] += term.coeff * np.exp(
            1j * np.dot(kpoints, dR1 - dR0)
        )
    VMatrices += np.transpose(VMatrices, (0, 2, 1)).conj()
    return VMatrices


def Fourier(cell, cluster):
    cell_creators = [
        HP.AoC(HP.CREATION, site=site, spin=spin)
        for spin in SPINS for site in cell.points
    ]
    cluster_creators = [
        HP.AoC(HP.CREATION, site=site, spin=spin)
        for spin in SPINS for site in cluster.points
    ]
    cell_annihilators = [creator.dagger() for creator in cell_creators]
    cluster_annihilators = [creator.dagger() for creator in cluster_creators]

    FTs = []
    for cluster_row, creator in enumerate(cluster_creators):
        p0_eqv, dR0 = cell.decompose(creator.site)
        cell_row = cell_creators.index(creator.derive(site=p0_eqv))
        for cluster_col, annihilator in enumerate(cluster_annihilators):
            p1_eqv, dR1 = cell.decompose(annihilator.site)
            cell_col = cell_annihilators.index(annihilator.derive(site=p1_eqv))
            FTs.append(
                (cluster_row, cluster_col, cell_row, cell_col, dR1 - dR0)
            )
    return FTs
