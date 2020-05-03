import logging
import sys
from time import time

import HamiltonianPy as HP
import matplotlib.pyplot as plt
import numpy as np

from database import ALL_POINTS, TRANSLATION_VECTORS
from utilities import ClusterRGFSolver, CPTPerturbation, SPINS

DEFAULT_MODEL_PARAMETERS = {
    "t": -1.0, "mu0": 0.0, "mu1": 0.0, "U0": 0.0, "U1": 0.0,
}


def TermsGenerator(cluster0, cluster1, model="Model1", **model_params):
    actual_model_params = dict(DEFAULT_MODEL_PARAMETERS)
    actual_model_params.update(model_params)
    t = actual_model_params["t"]
    U0 = actual_model_params["U0"] / 2
    U1 = actual_model_params["U1"] / 2
    mu0 = actual_model_params["mu0"] / 2
    mu1 = actual_model_params["mu1"] / 2

    HTerms0 = []
    HTerms1 = []
    hopping_indices = [
        [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9],
        [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 1],
    ]
    for container, cluster in [(HTerms0, cluster0), (HTerms1, cluster1)]:
        for point in cluster.points:
            point_index = cluster.getIndex(point, fold=False)
            U, mu = (U0, mu0) if point_index == 0 else (U1, mu1)
            container.append(HP.HubbardFactory(site=point, coeff=U))
            container.append(HP.CPFactory(point, spin=HP.SPIN_UP, coeff=mu))
            container.append(HP.CPFactory(point, spin=HP.SPIN_DOWN, coeff=mu))
        for ij in hopping_indices:
            p0, p1 = cluster.points[ij]
            coeff = t / np.dot(p1 - p0, p1 - p0)
            for spin in SPINS:
                container.append(
                    HP.HoppingFactory(p0, p1, spin0=spin, coeff=coeff)
                )

    if model == "Model1":
        hopping0_indices = [[6, 1]]
        hopping1_indices = [[3, 7]]
        hopping2_indices = [[9, 4]]
    elif model == "Model2":
        hopping0_indices = [[6, 1], [6, 2], [6, 9], [5, 1], [7, 1]]
        hopping1_indices = [[3, 7], [3, 6], [3, 8], [2, 7], [4, 7]]
        hopping2_indices = [[9, 4], [9, 3], [9, 5], [1, 4], [8, 4]]
    else:
        raise ValueError("Invalid `model`: {0}".format(model))

    VTerms0 = []
    for i, j in hopping0_indices:
        p0 = cluster0.points[i]
        p1 = cluster1.points[j]
        coeff = t / np.dot(p1 - p0, p1 - p0)
        for spin in SPINS:
            VTerms0.append(HP.HoppingFactory(p0, p1, spin0=spin, coeff=coeff))

    VTerms1 = []
    temp = [
        (hopping1_indices, cluster1.points - cluster1.vectors[0]),
        (hopping2_indices, cluster1.points - cluster1.vectors[1]),
    ]
    for indices, points in temp:
        for i, j in indices:
            p0 = cluster0.points[i]
            p1 = points[j]
            coeff = t / np.dot(p1 - p0, p1 - p0)
            for spin in SPINS:
                VTerms1.append(
                    HP.HoppingFactory(p0, p1, spin0=spin, coeff=coeff)
                )
    return HTerms0, HTerms1, VTerms0, VTerms1


def RGFForCluster2(
        cluster0, cluster1, cluster2, HTerms0, HTerms1, VTerms, omegas, eta=0.01
):
    gfs0 = ClusterRGFSolver(HTerms0, cluster0, omegas, eta=eta, toarray=False)
    gfs1 = ClusterRGFSolver(HTerms1, cluster1, omegas, eta=eta, toarray=False)

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
    HTerms0, HTerms1, VTerms0, VTerms1 = TermsGenerator(
        cluster0, cluster1, model, **model_params
    )
    VMatrices = CPTPerturbation(VTerms1, cluster2, kpoints)

    t0 = time()
    cluster2_gfs = RGFForCluster2(
        cluster0, cluster1, cluster2, HTerms0, HTerms1, VTerms0, omegas, eta
    )
    t1 = time()
    logging.info("Time spend on cluster2_gfs: %.3fs", t1 - t0)

    spectrum = []
    for nth, VMatrix in enumerate(VMatrices):
        t0 = time()
        cpt_gfs = np.linalg.inv(np.linalg.inv(cluster2_gfs) - VMatrix)
        spectrum.append(np.trace(cpt_gfs, axis1=1, axis2=2).imag)
        t1= time()
        logging.info("Time spend %4dth VMatrix: %.3fs", nth, t1 - t0)
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

    HTerms0, HTerms1, VTerms0, VTerms1 = TermsGenerator(
        cluster0, cluster1, model, **model_params
    )
    VMatrices = CPTPerturbation(VTerms1, cluster2, kpoints)

    t0 = time()
    cluster2_gfs = RGFForCluster2(
        cluster0, cluster1, cluster2, HTerms0, HTerms1, VTerms0, omegas, eta
    )
    t1 = time()
    logging.info("Time spend on cluster2_gfs: %.3fs", t1 - t0)

    dos = []
    for nth, cluster2_gf in enumerate(cluster2_gfs):
        t0 = time()
        cpt_gfs = np.linalg.inv(np.linalg.inv(cluster2_gf) - VMatrices)
        dos.append(np.mean(np.diagonal(cpt_gfs, axis1=1, axis2=2).imag, axis=0))
        t1 = time()
        logging.info("Time spend on %4dth omega: %.3fs", nth, t1 - t0)
    return -np.array(dos) / np.pi


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout, level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )

    U0 = U1 = 0.0
    model = "Model1"

    numk = 100
    eta  = 0.02
    emin = -4.0
    emax =  4.0
    step = 0.01
    omegas = np.arange(emin, emax + step, step)

    cluster0 = HP.Lattice(ALL_POINTS[0:10], TRANSLATION_VECTORS)
    cluster1 = HP.Lattice(ALL_POINTS[10:20], TRANSLATION_VECTORS)
    cluster2 = HP.Lattice(ALL_POINTS[0:20], TRANSLATION_VECTORS)

    dos = DOS(
        omegas, cluster0, cluster1, cluster2,
        numk=numk, eta=eta, model=model, U0=U0, U1=U1
    )
    sum_dos = np.sum(dos) * step
    print("Sum of DoS: {0}".format(sum_dos), flush=True)

    fig, ax = plt.subplots()
    ax.plot(omegas, np.sum(dos, axis=1))
    ax.set_xlim(emin, emax)
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel("DoS(a.u.)")
    ax.grid()
    plt.show()
    plt.close("all")
