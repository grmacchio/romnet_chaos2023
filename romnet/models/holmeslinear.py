"""Non-normal linear model from P. Holmes (2012)"""

from typing import Optional

import torch
from torch import Tensor
import numpy as np

from .. import Model
from ..typing import Vector

__all__ = ["HolmesLinear"]


class HolmesLinear(Model):

    def __init__(self):
        self.A = np.array([[-1.0, 0.0, 100.0],
                           [0.0, -2.0, 100.0],
                            [0.0, 0.0, -5.0]])
        self.A_tensor = torch.tensor(self.A)
        self.B = np.array([[1.0],[1.0],[1.0]])
        self.B_tensor = torch.tensor(self.B)
        self.C = np.array([[1.0, 1.0, 1.0]])
        self.C_tensor = torch.tensor(self.C)
        self.D = np.array([[0.0]])
        self.D_tensor = torch.tensor(self.D)

    @property
    def num_states(self) -> int:
        return self.A.shape[0]
    
    @property
    def num_outputs(self) -> int:
        return self.C.shape[0]

    def rhs(self, x: Vector) -> Vector:
        return x @ self.A.T
    
    def rhs_tensor(self, x: Tensor) -> Tensor:
        return x @ self.A_tensor.T
    
    def output(self, x: Vector) -> Vector:
        return x @ self.C.T
    
    def output_tensor(self, x: Tensor) -> Tensor:
        return x @ self.C_tensor.T

    def adjoint_rhs(self, x: Vector, v: Vector) -> Vector:
        return v @ self.A
    
    def adjoint_output(self, _: Vector, v: Vector) -> Vector:
        return v @ self.C

    def random_ic(self, max_amplitude: Optional[float] = 6.) -> Vector:
        xmax = max_amplitude
        zmin = -1 * max_amplitude
        zmax = max_amplitude
        x = xmax * (2 * np.random.rand() - 1)
        y = xmax * (2 * np.random.rand() - 1)
        z = zmin + (zmax - zmin) * np.random.rand()
        return np.array((x, y, z))
