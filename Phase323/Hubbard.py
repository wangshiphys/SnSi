import logging
import sys
from time import time

import HamiltonianPy as HP
import matplotlib.pyplot as plt
import numpy as np

from database import ALL_POINTS, TRANSLATION_VECTORS
from utilities import ClusterRGFSolver, CPTPerturbation, SPINS

DEFAULT_MODEL_PARAMETERS = {"t": -1.0, "U": 0.0}


def TermsGenerator(**model_params):
    actual_model_params = dict(DEFAULT_MODEL_PARAMETERS)
    actual_model_params.update(model_params)
    U = actual_model_params["U"] / 2
    t = actual_model_params["t"]

    ids = [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [1, 1]]
    points = np.concatenate(
        [np.dot(ij, TRANSLATION_VECTORS) + ALL_POINTS for ij in ids]
    )

    V0_ijs = [[6, 11]]
    V1_ijs = [[15, 20]]
    V2_ijs = [[3, 40], [9, 83], [12, 45], [18, 90], [1, 136], [4, 45], [7, 89]]
    H0_ijs = [
        [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9],
        [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 1],
    ]
    H1_ijs = [
        [10, 11], [10, 12], [10, 13], [10, 14],
        [10, 15], [10, 16], [10, 17], [10, 18], [10, 19],
        [11, 12], [12, 13], [13, 14], [14, 15],
        [15, 16], [16, 17], [17, 18], [18, 19], [19, 11],
    ]
    H2_ijs = [[20, 21], [21, 22], [22, 20]]

    VTerms0 = []
    VTerms1 = []
    VTerms2 = []
    HTerms0 = [HP.HubbardFactory(points[i], coeff=U) for i in range(10)]
    HTerms1 = [HP.HubbardFactory(points[i], coeff=U) for i in range(10, 20)]
    HTerms2 = [HP.HubbardFactory(points[i], coeff=U) for i in range(20, 23)]

    for container, ijs in [
        [VTerms0, V0_ijs], [VTerms1, V1_ijs], [VTerms2, V2_ijs],
        [HTerms0, H0_ijs], [HTerms1, H1_ijs], [HTerms2, H2_ijs],
    ]:
        for ij in ijs:
            p0, p1 = points[ij]
            coeff = t / np.dot(p1 - p0, p1 - p0)
            for spin in SPINS:
                container.append(
                    HP.HoppingFactory(p0, p1, spin0=spin, coeff=coeff)
                )
    return HTerms0, HTerms1, HTerms2, VTerms0, VTerms1, VTerms2


def RGFForCluster3(
        cluster0, cluster1, cluster2, cluster3,
        HTerms0, HTerms1, HTerms2, VTerms01, VTerms12,
        omegas, eta=0.01
):
    gfs0 = ClusterRGFSolver(
        HTerms0, cluster0, omegas, eta=eta, structure="dict",
    )
    gfs1 = ClusterRGFSolver(
        HTerms1, cluster1, omegas, eta=eta, structure="dict",
    )
    gfs2 = ClusterRGFSolver(
        HTerms2, cluster2, omegas, eta=eta, structure="dict",
    )

    creators = [
        HP.AoC(HP.CREATION, site=point, spin=spin)
        for spin in SPINS for point in cluster3.points
    ]
    annihilators = [creator.dagger() for creator in creators]

    dim = len(creators)
    VMatrix01 = np.zeros((dim, dim), dtype=np.float64)
    VMatrix12 = np.zeros((dim, dim), dtype=np.float64)
    for matrix, terms in ((VMatrix01, VTerms01), (VMatrix12, VTerms12)):
        for term in terms:
            creator, annihilator = term.components
            row = creators.index(creator)
            col = annihilators.index(annihilator)
            matrix[row, col] += term.coeff
    VMatrix01 += VMatrix01.T.conj()
    VMatrix12 += VMatrix12.T.conj()

    gfs3 = np.zeros((len(omegas), dim, dim), dtype=np.complex128)
    for row, creator in enumerate(creators):
        for col, annihilator in enumerate(annihilators):
            key = (creator.dagger(), annihilator.dagger())
            if key in gfs0:
                gfs3[:, row, col] = gfs0[key]
            elif key in gfs1:
                gfs3[:, row, col] = gfs1[key]
            elif key in gfs2:
                gfs3[:, row, col] = gfs2[key]

    gfs01 = np.linalg.inv(np.linalg.inv(gfs3) - VMatrix01)
    return np.linalg.inv(np.linalg.inv(gfs01) - VMatrix12)


def EnergyBand(
        omegas, kpoints, cluster0, cluster1, cluster2, cluster3,
        eta=0.01, **model_params
):
    HTerms0, HTerms1, HTerms2, VTerms0, VTerms1, VTerms2 = TermsGenerator(
        **model_params
    )
    VMatrices = CPTPerturbation(VTerms2, cluster3, kpoints)

    t0 = time()
    cluster3_gfs = RGFForCluster3(
        cluster0, cluster1, cluster2, cluster3,
        HTerms0, HTerms1, HTerms2, VTerms0, VTerms1,
        omegas=omegas, eta=eta
    )
    t1 = time()
    logging.info("Time spend on cluster3_gfs: %.3fs", t1 - t0)

    spectrum = []
    for nth, VMatrix in enumerate(VMatrices):
        t0 = time()
        cpt_gfs = np.linalg.inv(np.linalg.inv(cluster3_gfs) - VMatrix)
        spectrum.append(np.trace(cpt_gfs, axis1=1, axis2=2).imag)
        t1 = time()
        logging.info("Time spend %4dth VMatrix: %.3fs", nth, t1 - t0)
    return -np.array(spectrum).T / np.pi


def DOS(
        omegas, cluster0, cluster1, cluster2, cluster3, numk=200,
        eta=0.01, **model_params
):
    ratio = np.linspace(0, 1, numk, endpoint=False)
    ratio_mesh = np.stack(
        np.meshgrid(ratio, ratio, indexing="ij"), axis=-1
    ).reshape((-1, 2))
    kpoints = np.matmul(ratio_mesh, cluster3.bs)
    del ratio, ratio_mesh

    HTerms0, HTerms1, HTerms2, VTerms0, VTerms1, VTerms2 = TermsGenerator(
        **model_params
    )
    VMatrices = CPTPerturbation(VTerms2, cluster3, kpoints)

    t0 = time()
    cluster3_gfs = RGFForCluster3(
        cluster0, cluster1, cluster2, cluster3,
        HTerms0, HTerms1, HTerms2, VTerms0, VTerms1,
        omegas=omegas, eta=eta
    )
    t1 = time()
    logging.info("Time spend on cluster3_gfs: %.3fs", t1 - t0)

    dos = []
    for nth, cluster3_gf in enumerate(cluster3_gfs):
        t0 = time()
        cpt_gfs = np.linalg.inv(np.linalg.inv(cluster3_gf) - VMatrices)
        dos.append(np.mean(np.diagonal(cpt_gfs, axis1=1, axis2=2).imag, axis=0))
        t1 = time()
        logging.info("Time spend on %4dth omega: %.3fs", nth, t1 - t0)
    return -np.array(dos) / np.pi



if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout, level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )

    U = 0.0

    numk = 100
    eta = 0.01

    emin = -3.5
    emax = 2.5
    step = 0.01
    omegas = np.arange(emin, emax + step, 0.01)

    cluster0 = HP.Lattice(ALL_POINTS[0:10], TRANSLATION_VECTORS)
    cluster1 = HP.Lattice(ALL_POINTS[10:20], TRANSLATION_VECTORS)
    cluster2 = HP.Lattice(ALL_POINTS[20:23], TRANSLATION_VECTORS)
    cluster3 = HP.Lattice(ALL_POINTS, TRANSLATION_VECTORS)

    dos = DOS(
        omegas, cluster0, cluster1, cluster2, cluster3,
        numk=numk, eta=eta, U=U,
    )
    sum_of_dos = np.sum(dos) * step
    print("Sum of DoS: {0}".format(sum_of_dos), flush=True)

    fig, ax = plt.subplots(num="DoS")
    ax.plot(omegas, np.mean(dos, axis=1))
    ax.set_xlim(emin, emax)
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel("DoS(a.u.)")
    ax.grid(ls="dashed")
    plt.show()
    plt.close("all")
