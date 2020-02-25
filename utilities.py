"""
This module provides utility programs used in this project.
"""


__all__ = [
    "Lorentzian",
]


import numpy as np


# Simulation of the Delta function
def Lorentzian(xs, x0=0.0, gamma=0.01):
    """
    The Lorentzian function.

    Parameters
    ----------
    xs : float or array of floats
        The independent variable of the Lorentzian function
    x0 : float or array of floats, optional
        The center of the Lorentzian function
        Default: 0.0
    gamma : float, optional
        Specifying the width of the Lorentzian function
        Default: 0.01

    Returns
    -------
    res : float or array of floats
        1. `xs` and `x0` are both scalar, then the corresponding function
        value is returned;
        2. `xs` and/or `x0` are array of floats, the two parameters are
        broadcasted to calculated the expression `xs -x0`, and the
        corresponding function values are returned.

    See also
    --------
    numpy.broadcast
    http://mathworld.wolfram.com/LorentzianFunction.html
    """

    gamma /= 2
    return gamma / np.pi / ((xs - x0) ** 2 + gamma ** 2)
