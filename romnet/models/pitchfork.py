"""Model of pitchfork bifurcation"""

import numpy as np
from numpy.typing import NDArray

from .. import Model
from ..typing import Vector

__all__ = ["Pitchfork"]


class Pitchfork(Model):
    """
    2-state ODE model, normal form of pitchfork bifurcation
    """
    num_states = 2

    def __init__(
        self,
        lam1: float = 0.1,
        lam2: float = 10.,
        a: float = 1.
    ):
        self.lam1 = lam1
        self.lam2 = lam2
        self.a = a

    def rhs(self, x: Vector) -> Vector:
        out = np.zeros_like(x)
        out[..., 0] = self.lam1 * x[..., 0] * (self.a**2 - x[..., 0]**2)
        out[..., 1] = self.lam2 * (x[..., 0]**2 - x[..., 1])
        return out

    def jac(self, x: Vector) -> NDArray[np.float64]:
        out = np.zeros(x.shape + (self.num_states,))
        out[..., 0, 0] = self.lam1 * self.a**2 * np.ones_like(x[..., 0]) - 3 * self.lam1 * x[..., 0]**2
        out[..., 1, 0] = np.zeros_like(x[..., 0])
        out[..., 0, 1] = 2 * self.lam2 * x[..., 0]
        out[..., 1, 1] = -self.lam2 * np.ones_like(x[..., 0])
        return out

    def adjoint_rhs(self, x: Vector, v: Vector) -> Vector:
        return self.jac(x).T @ v

    def slow_manifold(self, x1: float) -> float:
        ep = 1 / self.lam2
        h0 = x1**2
        h1 = (-2 * self.lam1 * self.a**2 * x1**2
              + 2 * self.lam1 * x1**4)
        h2 = (4 * self.lam1**2 * self.a**4 * x1**2
              - 12 * self.lam1**2 * self.a**2 * x1**4
              + 8 * self.lam1**2 * x1**6)
        h3 = (-8 * self.lam1**3 * self.a**6 * x1**2
              + 56 * self.lam1**3 * self.a**4 * x1**4
              - 96 * self.lam1**3 * self.a**2 * x1**6
              + 48 * self.lam1**3 * x1**8)
        x2 = h0 + ep * h1 + ep**2 * h2 + ep**3 * h3
        return x2
