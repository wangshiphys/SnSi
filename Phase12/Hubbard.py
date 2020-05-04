import logging
from time import time

import HamiltonianPy as HP
import numpy as np

from utilities import ClusterRGFSolver, CPTPerturbation, SPINS

DEFAULT_MODEL_PARAMETERS = {
    "t": -1.0, "mu0": 0.0, "mu1": 0.0, "U0": 0.0, "U1": 0.0,
}


def TermsGenerator(cluster, amplitude=0.0, **model_params):
    assert isinstance(amplitude, float) and (0.0 <= amplitude < 1.0)

    actual_model_params = dict(DEFAULT_MODEL_PARAMETERS)
    actual_model_params.update(model_params)
    t = actual_model_params["t"]
    U0 = actual_model_params["U0"] / 2
    U1 = actual_model_params["U1"] / 2
    mu0 = actual_model_params["mu0"] / 2
    mu1 = actual_model_params["mu1"] / 2

    HTerms = []
    VTerms = []
    for point in cluster.points:
        index = cluster.getIndex(point, fold=False)
        mu, U = (mu1, U1) if index == 12 else (mu0, U0)
        if U != 0.0:
            HTerms.append(HP.HubbardFactory(site=point, coeff=U))
        if mu != 0.0:
            HTerms.append(HP.CPFactory(point, spin=HP.SPIN_UP, coeff=mu))
            HTerms.append(HP.CPFactory(point, spin=HP.SPIN_DOWN, coeff=mu))

    intra_bonds_1st, inter_bonds_1st = cluster.bonds(nth=1)
    intra_bonds_2nd, inter_bonds_2nd = cluster.bonds(nth=2)
    for bond in intra_bonds_1st + intra_bonds_2nd:
        p0, p1 = bond.endpoints
        coeff = t / np.dot(p0 - p1, p0 - p1)
        coeff *= (2 * amplitude * np.random.random() + 1 - amplitude)
        for spin in SPINS:
            HTerms.append(HP.HoppingFactory(p0, p1, spin0=spin, coeff=coeff))
    for bond in inter_bonds_1st + inter_bonds_2nd:
        p0, p1 = bond.endpoints
        coeff = t / np.dot(p0 - p1, p0 - p1)
        coeff *= (2 * amplitude * np.random.random() + 1 - amplitude)
        for spin in SPINS:
            VTerms.append(HP.HoppingFactory(p0, p1, spin0=spin, coeff=coeff))
    return HTerms, VTerms


def EnergyBand(
        omegas, cluster, kpoints, amplitude=0.0,
        enum=None, total_sz=None, eta=0.01, **model_params
):
    HTerms, VTerms = TermsGenerator(cluster, amplitude, **model_params)
    VMatrices = CPTPerturbation(VTerms, cluster, kpoints)
    cluster_gfs = ClusterRGFSolver(
        HTerms, cluster, omegas, enum=enum, total_sz=total_sz, eta=eta
    )

    spectrum = []
    for nth, VMatrix in enumerate(VMatrices):
        t0 = time()
        cpt_gfs = np.linalg.inv(np.linalg.inv(cluster_gfs) - VMatrix)
        spectrum.append(np.trace(cpt_gfs, axis1=1, axis2=2).imag)
        t1 = time()
        logging.info("Time spend %4dth VMatrix: %.3fs", nth, t1 - t0)
    return -np.array(spectrum).T / np.pi


def DOS(
        omegas, cluster, numk=200, amplitude=0.0,
        enum=None, total_sz=None, eta=0.01, **model_params
):
    ratio = np.linspace(0, 1, numk, endpoint=False)
    ratio_mesh = np.stack(
        np.meshgrid(ratio, ratio, indexing="ij"), axis=-1
    ).reshape((-1, 2))
    kpoints = np.matmul(ratio_mesh, cluster.bs)
    del ratio, ratio_mesh

    HTerms, VTerms = TermsGenerator(cluster, amplitude, **model_params)
    VMatrices = CPTPerturbation(VTerms, cluster, kpoints)
    cluster_gfs = ClusterRGFSolver(
        HTerms, cluster, omegas, enum=enum, total_sz=total_sz, eta=eta
    )

    dos = []
    for nth, cluster_gf in enumerate(cluster_gfs):
        t0 = time()
        cpt_gfs = np.linalg.inv(np.linalg.inv(cluster_gf) - VMatrices)
        dos.append(np.mean(np.diagonal(cpt_gfs, axis1=1, axis2=2).imag, axis=0))
        t1 = time()
        logging.info("Time spend on %4dth omega: %.3fs", nth, t1 - t0)
    return -np.array(dos) / np.pi
