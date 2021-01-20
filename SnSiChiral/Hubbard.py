import logging
import sys
from time import time

import HamiltonianPy as HP
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import eigsh

from database import POINTS, VECTORS
from utilities import Mu

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(message)s",
)

SPINs = (HP.SPIN_DOWN, HP.SPIN_UP)
DEFAULT_MODEL_PARAMETERS = {"t0": -1.0, "t1": -1.0, "U": 0.0}


def TermsGenerator(model="Model1", **model_params):
    actual_model_params = dict(DEFAULT_MODEL_PARAMETERS)
    actual_model_params.update(model_params)
    t0 = actual_model_params["t0"]
    t1 = actual_model_params["t1"]
    U = actual_model_params["U"] / 2

    cluster = HP.Lattice(points=POINTS, vectors=VECTORS)
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

    VTerms = []
    HTerms = [
        HP.HubbardFactory(site=point, coeff=U) for point in cluster.points
    ]
    for ij in intra_hopping_indices0:
        p0, p1 = points_collection[ij]
        for spin in SPINs:
            HTerms.append(HP.HoppingFactory(p0, p1, spin0=spin, coeff=t0))
    for ij in intra_hopping_indices1:
        p0, p1 = points_collection[ij]
        for spin in SPINs:
            HTerms.append(HP.HoppingFactory(p0, p1, spin0=spin, coeff=t1))
    for ij in inter_hopping_indices:
        p0, p1 = points_collection[ij]
        for spin in SPINs:
            VTerms.append(HP.HoppingFactory(p0, p1, spin0=spin, coeff=t1))
    return cluster, HTerms, VTerms


def ClusterRGFSolver(HTerms, cluster, omegas, eta=0.01):
    creators = [
        HP.AoC(HP.CREATION, site=point, spin=spin)
        for point in cluster.points for spin in SPINs
    ]
    annihilators = [creator.dagger() for creator in creators]
    state_indices_table = HP.IndexTable(
        HP.StateID(site=point, spin=spin)
        for point in cluster.points for spin in SPINs
    )

    point_num = cluster.point_num
    basis = HP.base_vectors([2 * point_num, point_num])
    basis_p = HP.base_vectors([2 * point_num, point_num + 1])
    basis_h = HP.base_vectors([2 * point_num, point_num - 1])

    HM = 0.0
    HM_P = 0.0
    HM_H = 0.0
    for term in HTerms:
        HM += term.matrix_repr(state_indices_table, basis)
        HM_P += term.matrix_repr(state_indices_table, basis_p)
        HM_H += term.matrix_repr(state_indices_table, basis_h)
    HM += HM.getH()
    HM_P += HM_P.getH()
    HM_H += HM_H.getH()

    # noinspection PyTypeChecker
    (GE, ), GS = eigsh(HM, k=1, which="SA")
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
    del state_indices_table, GS, basis, basis_p, basis_h

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

    dim = len(creators)
    gfs_array = np.zeros((len(omegas), dim, dim), dtype=np.complex128)
    for row, creator in enumerate(creators):
        for col, annihilator in enumerate(annihilators):
            key = (creator.dagger(), annihilator.dagger())
            if key in gfs_dict:
                gfs_array[:, row, col] = gfs_dict[key]
    return gfs_array


def CPTPerturbation(VTerms, cluster, kpoints):
    creators = [
        HP.AoC(HP.CREATION, site=point, spin=spin)
        for point in cluster.points for spin in SPINs
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


if __name__ == "__main__":
    model_params = {"model": "Model1", "t0": -1.00, "t1": -1.00, "U": 0.00}
    cluster, HTerms, VTerms = TermsGenerator(**model_params)

    numk = 200
    ratio = np.linspace(0, 1, numk, endpoint=False)
    ratio_mesh = np.stack(
        np.meshgrid(ratio, ratio, indexing="ij"), axis=-1
    ).reshape((-1, 2))
    kpoints = np.matmul(ratio_mesh, cluster.bs)
    del ratio, ratio_mesh

    eta = 0.05
    emin = -10.0
    emax = 10.0
    step = 0.01
    omegas = np.arange(emin, emax + step, step)

    t0 = time()
    cluster_gfs = ClusterRGFSolver(HTerms, cluster, omegas, eta=eta)
    t1 = time()
    logging.info("Time spend on cluster_gfs: %.3fs", t1 - t0)

    t0 = time()
    VMatrices = CPTPerturbation(VTerms, cluster, kpoints)
    t1 = time()
    logging.info("Time spend on VMatrices: %.3fs", t1 - t0)

    dos = []
    for nth, cluster_gf in enumerate(cluster_gfs):
        t0 = time()
        cpt_gfs = np.linalg.inv(np.linalg.inv(cluster_gf) - VMatrices)
        dos.append(np.mean(np.diagonal(cpt_gfs, axis1=1, axis2=2).imag, axis=0))
        t1 = time()
        logging.info("Time spend on %4dth omega: %.3fs", nth, t1 - t0)
    dos = -np.array(dos) / np.pi

    site_num = cluster.point_num
    total_dos = np.sum(dos, axis=1)
    mu_h = Mu(total_dos, omegas, site_num, 2 * site_num, reverse=True)
    mu_p = Mu(total_dos, omegas, site_num, 2 * site_num, reverse=False)
    mu = (mu_p + mu_h) / 2
    particle_num = np.sum(total_dos[omegas < mu]) * step
    local_particle_num = np.sum(dos[omegas < mu], axis=0) * step

    print("Sum of DoS: {0:.8f}".format(np.sum(dos) * step))
    print("mu_p = {0:.8f}, mu_h = {1:.8f}, mu = {2:.8f}".format(mu_p, mu_h, mu))
    print("Particle number: {0:.8f}".format(particle_num))
    for index in range(site_num):
        num = local_particle_num[2*index] + local_particle_num[2*index+1]
        print("Particle number on {0:0>2d}th site: {1:8f}".format(index, num))

    fig, ax = plt.subplots()
    ax.plot(omegas, total_dos, lw=4)
    ax.set_xlim(emin, emax)
    ax.axvline(mu, ls="dashed", color="tab:red", lw=1)
    ax.grid(axis="both", ls="dashed", color="gray")
    plt.show()
    plt.close("all")
    logging.info("Program stop running")
