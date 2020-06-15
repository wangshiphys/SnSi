"""
CPT study of the spin-1/2 Hubbard model defined on the triangular lattice
with nearest-neighbor (NN) hoppings.
"""


import HamiltonianPy as HP
import matplotlib.pyplot as plt
import numpy as np

from utilities import *

DEFAULT_MODEL_PARAMETERS = {"t": -1.0, "U": 0.0}


def TermsGenerator(cluster, **model_params):
    actual_model_params = dict(DEFAULT_MODEL_PARAMETERS)
    actual_model_params.update(model_params)
    t = actual_model_params["t"]
    U = actual_model_params["U"] / 2

    VTerms = []
    intra_bonds, inter_bonds = cluster.bonds(nth=1)
    HTerms = [HP.HubbardFactory(point, coeff=U) for point in cluster.points]
    for container, bonds in [(HTerms, intra_bonds), (VTerms, inter_bonds)]:
        for bond in bonds:
            p0, p1 = bond.endpoints
            for spin in SPINS:
                container.append(HP.HoppingFactory(p0, p1, spin0=spin, coeff=t))
    return HTerms, VTerms


def EnergyBand(
        omegas, kpoints, num0=3, num1=4, enum=None, total_sz=None,
        eta=0.05, **model_params,
):
    kpoint_num = kpoints.shape[0]
    cell = HP.lattice_generator("triangle", num0=1, num1=1)
    cluster = HP.lattice_generator("triangle", num0=num0, num1=num1)

    FTs = Fourier(cell, cluster)
    HTerms, VTerms = TermsGenerator(cluster, **model_params)
    VMatrices = CPTPerturbation(VTerms, cluster, kpoints)
    cluster_gfs = ClusterRGFSolver(
        HTerms, cluster, omegas, enum=enum, total_sz=total_sz, eta=eta,
    )

    spectrum = np.zeros((len(omegas), kpoint_num), dtype=np.float64)
    for index in range(kpoint_num):
        kpoint = kpoints[index]
        VMatrix = VMatrices[index]
        cpt_gfs = np.linalg.inv(np.linalg.inv(cluster_gfs) - VMatrix)

        tmp = np.zeros((len(omegas), 2, 2), dtype=np.complex128)
        for i, j, m, n, dR in FTs:
            tmp[:, m, n] += cpt_gfs[:, i, j] * np.exp(1j * np.dot(kpoint, dR))
        spectrum[:, index] = np.trace(tmp, axis1=1, axis2=2).imag
    spectrum = -spectrum / np.pi / (num0 * num1)
    return spectrum


def DOS(
        omegas, num0=3, num1=4, enum=None, total_sz=None,
        numk=200, eta=0.05, **model_params
):
    cluster = HP.lattice_generator("triangle", num0=num0, num1=num1)
    ratio = np.linspace(0, 1, numk, endpoint=False)
    ratio_mesh = np.stack(
        np.meshgrid(ratio, ratio, indexing="ij"), axis=-1
    ).reshape((-1, 2))
    kpoints = np.matmul(ratio_mesh, cluster.bs)
    del ratio, ratio_mesh

    HTerms, VTerms = TermsGenerator(cluster, **model_params)
    VMatrices = CPTPerturbation(VTerms, cluster, kpoints)
    cluster_gfs = ClusterRGFSolver(
        HTerms, cluster, omegas, enum=enum, total_sz=total_sz, eta=eta,
    )

    dos = []
    for cluster_gf in cluster_gfs:
        cpt_gfs = np.linalg.inv(np.linalg.inv(cluster_gf) - VMatrices)
        dos.append(np.mean(np.diagonal(cpt_gfs, axis1=1, axis2=2).imag, axis=0))
    dos = -np.array(dos) / np.pi
    return dos


if __name__ == "__main__":
    t = -1.0
    U = 0.0

    num0 = 3
    num1 = 3
    site_num = num0 * num1

    emin = -7.0
    emax = 5.0
    step = 0.02
    omegas = np.arange(emin, emax + step, step)

    cell = HP.lattice_generator("triangle")
    M = cell.bs[0] / 2
    Gamma = np.array([0.0, 0.0])
    K = np.dot(np.array([2, 1]), cell.bs) / 3
    kpoints, indices = HP.KPath([Gamma, K, M])
    kpoint_num = kpoints.shape[0]

    spectrum = EnergyBand(omegas, kpoints, num0=num0, num1=num1, t=t, U=U)
    dos = DOS(omegas, num0=num0, num1=num1, t=t, U=U)

    total_dos = np.sum(dos, axis=1)
    mu_h = Mu(total_dos, omegas, site_num, 2*site_num, reverse=True)
    mu_p = Mu(total_dos, omegas, site_num, 2*site_num, reverse=False)
    mu = (mu_p + mu_h) / 2
    print("mu_p = {0:.6f}".format(mu_p))
    print("mu_h = {0:.6f}".format(mu_h))
    print("mu = {0:.6f}".format(mu))

    fig, (ax_EB, ax_DOS) = plt.subplots(1, 2, sharey="all")
    cs = ax_EB.contourf(
        range(kpoint_num), omegas, spectrum, cmap="hot", levels=500,
    )
    fig.colorbar(cs, ax=ax_EB)
    ax_EB.axhline(mu, ls="dashed", color="tab:red", lw=1)
    ax_EB.set_xticks(indices)
    ax_EB.set_ylim(emin, emax)
    ax_EB.set_xlim(0, kpoints.shape[0] - 1)
    ax_EB.grid(True, ls="dashed", color="gray")
    ax_EB.set_xticklabels([r"$\Gamma$", "K", "M", r"$\Gamma$"])

    ax_DOS.plot(total_dos, omegas, lw=4)
    ax_DOS.axhline(mu, ls="dashed", color="tab:red", lw=1)
    ax_DOS.grid(True, ls="dashed", color="gray")

    plt.show()
    plt.close("all")
