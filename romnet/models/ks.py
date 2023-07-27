"""Kuramoto-Sivashinsky equation"""

from typing import Callable

import numpy as np
import scipy as sp

from .. import BilinearModel
from ..typing import Vector

__all__ = ["KuramotoSivashinsky", "freq_to_space", "space_to_freq"]


def fft_multiply(yf: Vector, zf: Vector) -> Vector:
    """Multiply two Fourier series using fast Fourier transform"""
    nmodes = len(yf) - 1
    y = sp.fft.irfft(yf)
    z = sp.fft.irfft(zf)
    yz = y * z
    return 2 * nmodes * sp.fft.rfft(yz)


def freq_to_space(yf: Vector) -> Vector:
    nmodes = len(yf) - 1
    return 2 * nmodes * sp.fft.irfft(yf)


def space_to_freq(y: Vector) -> Vector:
    nmodes = len(y) // 2
    return sp.fft.rfft(y) / (2 * nmodes)


class KuramotoSivashinsky(BilinearModel):
    r"""Kuramoto-Sivashinsky equation

    u_t + u u_x + u_xx + u_xxxx = 0

    with periodic boundary conditions, for 0 <= x <= L
    The equation is solved using a spectral collocation or spectral Galerkin
    method

    .. math::
        u(x,t) = \sum_{k=-n}^n u_k(t) \exp(2\pi i k x / L)

    Since u is real, this implies :math:`u_{-k} = \overline{u_k}`

    The state is represented as a vector of Fourier coefficients u_k,
    for k = 0, ..., nmodes.
    """

    def __init__(self, nmodes: int, L: float):
        """summary here

        Args:
            nmodes: Number of (complex) Fourier modes to use
            L: Length of domain
        """
        self.nmodes = nmodes
        self.L = L
        k = np.arange(nmodes + 1)
        ksq = (2 * np.pi / L) ** 2 * k**2
        # Linear part = -u_xx - u_xxxx
        self._linear_factor = ksq * (1 - ksq)
        self._deriv = (2j * np.pi / L) * k

    def get_solver(self, alpha: float) -> Callable[[Vector], Vector]:
        def solver(u: Vector) -> Vector:
            return u / (1 - alpha * self._linear_factor)

        return solver

    def get_adjoint_solver(self, alpha: float) -> Callable[[Vector], Vector]:
        # linear part is self-adjoint
        return self.get_solver(alpha)

    def linear(self, u: Vector) -> Vector:
        return u * self._linear_factor

    def adjoint(self, u: Vector) -> Vector:
        # linear part is self-adjoint
        return self.linear(u)

    def bilinear(self, a: Vector, b: Vector) -> Vector:
        # a * b_x
        return fft_multiply(a, self._deriv * b)

    def nonlinear(self, u: Vector) -> Vector:
        return self.bilinear(u, u)

    def adjoint_nonlinear(self, u: Vector, w: Vector) -> Vector:
        return self.bilinear(w, u) - self.bilinear(u, w)
