import logging
import sys
from time import time

import HamiltonianPy as HP
import matplotlib.pyplot as plt
import numpy as np

from database import ALL_POINTS, TRANSLATION_VECTORS
from utilities import ClusterRGFSolver, CPTPerturbation, SPINS

DEFAULT_MODEL_PARAMETERS = {
    "t0": -1.0, "t1": -1.0, "mu0": 0.0, "mu1": 0.0, "U0": 0.0, "U1": 0.0,
}


def TermsGenerator(model="Model1", **model_params):
    actual_model_params = dict(DEFAULT_MODEL_PARAMETERS)
    actual_model_params.update(model_params)
    t0 = actual_model_params["t0"]
    t1 = actual_model_params["t1"]
    U0 = actual_model_params["U0"] / 2
    U1 = actual_model_params["U1"] / 2
    mu0 = actual_model_params["mu0"] / 2
    mu1 = actual_model_params["mu1"] / 2

    ids = ([0, 0], [-1, 0], [1, 0], [0, -1], [0, 1])
    points_collection = np.concatenate(
        [np.dot(ij, TRANSLATION_VECTORS) + ALL_POINTS[0:20] for ij in ids]
    )

    HTerms0 = []
    HTerms1 = []
    for index in range(10):
        point = points_collection[index]
        U, mu = (U0, mu0) if index == 0 else (U1, mu1)
        if U != 0:
            HTerms0.append(HP.HubbardFactory(site=point, coeff=U))
        if mu != 0:
            HTerms0.append(HP.CPFactory(point, spin=HP.SPIN_UP, coeff=mu))
            HTerms0.append(HP.CPFactory(point, spin=HP.SPIN_DOWN, coeff=mu))
    for index in range(10, 20):
        point = points_collection[index]
        U, mu = (U0, mu0) if index == 10 else (U1, mu1)
        if U != 0:
            HTerms1.append(HP.HubbardFactory(site=point, coeff=U))
        if mu != 0:
            HTerms1.append(HP.CPFactory(point, spin=HP.SPIN_UP, coeff=mu))
            HTerms1.append(HP.CPFactory(point, spin=HP.SPIN_DOWN, coeff=mu))

    for ij in [
        [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9]
    ]:
        p0, p1 = points_collection[ij]
        coeff = t0 / np.dot(p1 - p0, p1 - p0)
        for spin in SPINS:
            HTerms0.append(HP.HoppingFactory(p0, p1, spin0=spin, coeff=coeff))
    for ij in [
        [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 1]
    ]:
        p0, p1 = points_collection[ij]
        coeff = t1 / np.dot(p1 - p0, p1 - p0)
        for spin in SPINS:
            HTerms0.append(HP.HoppingFactory(p0, p1, spin0=spin, coeff=coeff))

    for ij in [
        [10, 11], [10, 12], [10, 13], [10, 14],
        [10, 15], [10, 16], [10, 17], [10, 18], [10, 19],
    ]:
        p0, p1 = points_collection[ij]
        coeff = t0 / np.dot(p1 - p0, p1 - p0)
        for spin in SPINS:
            HTerms1.append(HP.HoppingFactory(p0, p1, spin0=spin, coeff=coeff))
    for ij in [
        [11, 12], [12, 13], [13, 14], [14, 15],
        [15, 16], [16, 17], [17, 18], [18, 19], [19, 11],
    ]:
        p0, p1 = points_collection[ij]
        coeff = t1 / np.dot(p1 - p0, p1 - p0)
        for spin in SPINS:
            HTerms1.append(HP.HoppingFactory(p0, p1, spin0=spin, coeff=coeff))

    VTerms0 = []
    VTerms1 = []
    inter_hopping_indices0 = [[6, 11]]
    inter_hopping_indices1 = [[3, 37], [9, 74]]
    if model == "Model2":
        inter_hopping_indices0 += [[6, 12], [6, 19], [5, 11], [7, 11]]
        inter_hopping_indices1 += [
            [3, 36], [3, 38], [2, 37], [4, 37],
            [9, 73], [9, 75], [1, 74], [8, 74],
        ]
    for ij in inter_hopping_indices0:
        p0, p1 = points_collection[ij]
        coeff = t1 / np.dot(p1 - p0, p1 - p0)
        for spin in SPINS:
            VTerms0.append(HP.HoppingFactory(p0, p1, spin0=spin, coeff=coeff))
    for ij in inter_hopping_indices1:
        p0, p1 = points_collection[ij]
        coeff = t1 / np.dot(p1 - p0, p1 - p0)
        for spin in SPINS:
            VTerms1.append(HP.HoppingFactory(p0, p1, spin0=spin, coeff=coeff))

    return HTerms0, HTerms1, VTerms0, VTerms1


def RGFForCluster2(
        cluster0, cluster1, cluster2, HTerms0, HTerms1, VTerms, omegas, eta=0.01
):
    gfs0 = ClusterRGFSolver(
        HTerms0, cluster0, omegas, eta=eta, structure="dict",
    )
    gfs1 = ClusterRGFSolver(
        HTerms1, cluster1, omegas, eta=eta, structure="dict",
    )

    creators = [
        HP.AoC(HP.CREATION, site=point, spin=spin)
        for spin in SPINS for point in cluster2.points
    ]
    annihilators = [creator.dagger() for creator in creators]

    dim = len(creators)
    VMatrix = np.zeros((dim, dim), dtype=np.float64)
    for term in VTerms:
        creator, annihilator = term.components
        row = creators.index(creator)
        col = annihilators.index(annihilator)
        VMatrix[row, col] += term.coeff
    VMatrix += VMatrix.T.conj()

    gfs2 = np.zeros((len(omegas), dim, dim), dtype=np.complex128)
    for row, creator in enumerate(creators):
        for col, annihilator in enumerate(annihilators):
            key = (creator.dagger(), annihilator.dagger())
            if (key in gfs0) and (key in gfs1):
                raise RuntimeError("key in both dict!")
            elif key in gfs0:
                gfs2[:, row, col] = gfs0[key]
            elif key in gfs1:
                gfs2[:, row, col] = gfs1[key]
    return np.linalg.inv(np.linalg.inv(gfs2) - VMatrix)


def EnergyBand(
        omegas, kpoints, cluster0, cluster1, cluster2,
        eta=0.01, model="Model1", **model_params
):
    HTerms0, HTerms1, VTerms0, VTerms1 = TermsGenerator(model, **model_params)
    VMatrices = CPTPerturbation(VTerms1, cluster2, kpoints)

    start = time()
    cluster2_gfs = RGFForCluster2(
        cluster0, cluster1, cluster2, HTerms0, HTerms1, VTerms0, omegas, eta
    )
    end = time()
    logging.info("Time spend on cluster2_gfs: %.3fs", end - start)

    spectrum = []
    for nth, VMatrix in enumerate(VMatrices):
        start = time()
        cpt_gfs = np.linalg.inv(np.linalg.inv(cluster2_gfs) - VMatrix)
        spectrum.append(np.trace(cpt_gfs, axis1=1, axis2=2).imag)
        end = time()
        logging.info("Time spend %4dth VMatrix: %.3fs", nth, end - start)
    return -np.array(spectrum).T / np.pi


def DOS(
        omegas, cluster0, cluster1, cluster2, numk=200,
        eta=0.01, model="Model1", **model_params
):
    ratio = np.linspace(0, 1, numk, endpoint=False)
    ratio_mesh = np.stack(
        np.meshgrid(ratio, ratio, indexing="ij"), axis=-1
    ).reshape((-1, 2))
    kpoints = np.matmul(ratio_mesh, cluster2.bs)
    del ratio, ratio_mesh

    HTerms0, HTerms1, VTerms0, VTerms1 = TermsGenerator(model, **model_params)
    VMatrices = CPTPerturbation(VTerms1, cluster2, kpoints)

    start = time()
    cluster2_gfs = RGFForCluster2(
        cluster0, cluster1, cluster2, HTerms0, HTerms1, VTerms0, omegas, eta
    )
    end = time()
    logging.info("Time spend on cluster2_gfs: %.3fs", end - start)

    dos = []
    for nth, cluster2_gf in enumerate(cluster2_gfs):
        start = time()
        cpt_gfs = np.linalg.inv(np.linalg.inv(cluster2_gf) - VMatrices)
        dos.append(np.mean(np.diagonal(cpt_gfs, axis1=1, axis2=2).imag, axis=0))
        end = time()
        logging.info("Time spend on %4dth omega: %.3fs", nth, end - start)
    return -np.array(dos) / np.pi


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout, level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )

    t0 = -1.00
    t1 = -1.00
    model = "Model1"

    eta  = 0.01
    emin = -3.3
    emax = 2.3
    step = 0.01
    omegas = np.arange(emin, emax + step, step)

    cluster0 = HP.Lattice(ALL_POINTS[0:10], TRANSLATION_VECTORS)
    cluster1 = HP.Lattice(ALL_POINTS[10:20], TRANSLATION_VECTORS)
    cluster2 = HP.Lattice(ALL_POINTS[0:20], TRANSLATION_VECTORS)

    M = cluster2.bs[0] / 2
    K = np.dot(np.array([2, 1]), cluster2.bs) / 3
    kpoints, indices = HP.KPath([np.array([0.0, 0.0]), K, M])
    kpoint_num = kpoints.shape[0]

    spectrum = EnergyBand(
        omegas, kpoints, cluster0, cluster1, cluster2,
        eta=eta, model=model, t0=t0, t1=t1,
    )

    fig, ax = plt.subplots()
    cs = ax.contourf(
        range(kpoint_num), omegas, spectrum, cmap="hot", levels=200,
    )
    fig.colorbar(cs, ax=ax)
    ax.set_xticks(indices)
    ax.set_ylim(emin, emax)
    ax.set_xlim(0, kpoints.shape[0] - 1)
    ax.grid(True, ls="dashed", color="gray")
    ax.set_xticklabels([r"$\Gamma$", "K", "M", r"$\Gamma$"])
    plt.show()
    plt.close("all")
