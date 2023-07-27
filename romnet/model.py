"""model - Define how a given state evolves in time."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import lu_factor, lu_solve

from .timestepper import SemiImplicit, Timestepper
from .typing import Vector, VectorField, VectorList

__all__ = ["Model", "SemiLinearModel", "BilinearModel", "LUSolver", "project", "DiscreteModel"]


class Model(ABC):
    """
    Abstract base class for an ODE dx/dt = f(x)
    """

    @abstractmethod
    def rhs(self, x: Vector) -> Vector:
        """Return the right-hand-side of the ODE x' = f(x)."""

    def adjoint_rhs(self, x: Vector, v: Vector) -> Vector:
        """For the right-hand-side function f(x), return Df(x)^T v."""
        raise NotImplementedError(
            "Adjoint not implemented for class %s" % self.__class__.__name__
        )

    def get_stepper(self, dt: float, method: str = "rk2") -> Callable[[Vector], Vector]:
        """Return a discrete-time model, for the given timestep."""
        cls = Timestepper.lookup(method)
        stepper = cls(dt)
        return DiscreteModel(self.rhs, stepper)

    def get_adjoint_stepper(
        self, dt: float, method: str = "rk2"
    ) -> Callable[[Vector, Vector], Vector]:
        """Return a discrete-time model, for the given timestep."""
        cls = Timestepper.lookup(method)
        stepper = cls(dt)
        return DiscreteAdjoint(self.adjoint_rhs, stepper)


def project(rhs: VectorField, V: VectorList, W: Optional[VectorList] = None) -> VectorField:
    """
    Returns a reduced-order model that projects onto linear subspaces

    Rows of V determine the subspace to project onto
    Rows of W determine the direction of projection

    That is, the projection is given by
        V' (WV')^{-1} W

    The number of states in the reduced-order model is the number of rows
    in V (or W).

    If W is not specified, it is assumed W = V
    """
    n = len(V)
    if W is None:
        W = V
    assert len(W) == n

    # Let W1 = (W V')^{-1} W
    G = np.array([[np.dot(W[i], V[j]) for j in range(n)] for i in range(n)])
    W1 = np.linalg.solve(G, W)
    # Now projection is given by P = V' W1, and W1 V' = Identity

    def rom_rhs(z: Vector) -> Vector:
        x = sum(mode * c for mode, c in zip(V, z))
        fx = rhs(x)
        return np.array([np.dot(W1[i], fx) for i in range(len(W1))])

    return rom_rhs


class SemiLinearModel(Model):
    """Abstract base class for semi-linear models.

    Subclasses describe a model of the form
        x' = A x + N(x)
    """

    @abstractmethod
    def linear(self, x: Vector) -> Vector:
        """Return the linear part A x."""
        ...

    @abstractmethod
    def nonlinear(self, x: Vector) -> Vector:
        """Return the nonlinear part N(x)."""
        ...

    @abstractmethod
    def get_solver(self, alpha: float) -> Callable[[Vector], Vector]:
        """Return a solver for the linear part

        The returned solver is a callable object that, when called with
        argument b, returns a solution of the system

            x - alpha * linear(x) = b
        """
        ...

    def rhs(self, x: Vector) -> Vector:
        return self.linear(x) + self.nonlinear(x)

    def adjoint(self, x: Vector) -> Vector:
        """Return the adjoint of the linear part A^T x."""
        raise NotImplementedError(
            "Adjoint not implemented for class %s" % self.__class__.__name__
        )

    def get_adjoint_solver(self, alpha: float) -> Callable[[Vector], Vector]:
        """Return a solver for the adjoint of the linear part

        The returned solver is a callable object that, when called with
        argument b, returns a solution of the system

            x - alpha * adjoint(x) = b
        """
        raise NotImplementedError(
            "Adjoint not implemented for class %s" % self.__class__.__name__
        )

    def adjoint_nonlinear(self, x: Vector, v: Vector) -> Vector:
        """Return the adjoint of DN(x) applied to the vector v"""
        raise NotImplementedError(
            "Adjoint not implemented for class %s" % self.__class__.__name__
        )

    def adjoint_rhs(self, x: Vector, v: Vector) -> Vector:
        return self.adjoint(v) + self.adjoint_nonlinear(x, v)

    def get_stepper(
        self, dt: float, method: str = "rk2cn"
    ) -> Callable[[Vector], Vector]:
        try:
            cls = SemiImplicit.lookup(method)
        except NotImplementedError:
            return super().get_stepper(dt, method)
        stepper = cls(dt, self.linear, self.get_solver)
        return DiscreteModel(self.nonlinear, stepper)

    def get_adjoint_stepper(
        self, dt: float, method: str = "rk2cn"
    ) -> Callable[[Vector, Vector], Vector]:
        try:
            cls = SemiImplicit.lookup(method)
        except NotImplementedError:
            return super().get_adjoint_stepper(dt, method)
        stepper = cls(dt, self.adjoint, self.get_adjoint_solver)
        return DiscreteAdjoint(self.adjoint_nonlinear, stepper)


class LUSolver:
    """A class for solving linear systems A x = b

    Args:
        mat(array): the matrix A

    When instantiated, an LU factorization of A is computed, and this is
    used when solving the system for a given right-hand side b.
    """

    def __init__(self, mat: ArrayLike):
        self.LU = lu_factor(mat)

    def __call__(self, rhs: Vector) -> Vector:
        """Solve the system A x = rhs for x

        Args:
            rhs(array): the right-hand side of the equation to be solved

        Returns:
            The solution x
        """
        return lu_solve(self.LU, rhs)


class BilinearModel(SemiLinearModel):
    """Model where the right-hand side is a bilinear function of the state

    Models have the form

        x' = c + L x + B(x, x)

    where B is bilinear

    Args:
        c(array_like): vector containing the constant terms c
        L(array_like): matrix containing the linear map L
        B(array_like): rank-3 tensor describing the bilinear map B
    """

    def __init__(self, c: ArrayLike, L: ArrayLike, B: ArrayLike):
        self._affine = np.array(c)
        self._linear = np.array(L)
        self._bilinear = np.array(B)
        self.state_dim = self._linear.shape[0]

    def linear(self, x: Vector) -> Vector:
        return self._linear.dot(x)

    def adjoint(self, x: Vector) -> Vector:
        return self._linear.T.dot(x)

    def get_solver(self, alpha: float) -> Callable[[Vector], Vector]:
        mat = np.eye(self.state_dim) - alpha * self._linear
        return LUSolver(mat)

    def get_adjoint_solver(self, alpha: float) -> Callable[[Vector], Vector]:
        mat = np.eye(self.state_dim) - alpha * self._linear.T
        return LUSolver(mat)

    def bilinear(self, a: Vector, b: Vector) -> Vector:
        """Evaluate the bilinear term B(a, b)"""
        return self._bilinear.dot(a).dot(b)

    def nonlinear(self, x: Vector) -> Vector:
        return self._affine + self.bilinear(x, x)

    def adjoint_nonlinear(self, x: Vector, v: Vector) -> Vector:
        w = np.einsum("kji, j, k", self._bilinear, x, v)
        w += np.einsum("jik, j, k", self._bilinear, v, x)
        return w

    def project(self, V: VectorList, W: Optional[VectorList] = None) -> "BilinearModel":
        """Return a reduced-order model by projecting onto linear subspaces

        Rows of V determine the subspace to project onto, and
        rows of W determine the direction of projection

        That is, the projection is given by

        .. math:: V^T (WV^T)^{-1} W

        The number of states in the reduced-order model is the number of rows
        in V (or W).

        Args:
            V(list): List of modes to project onto.
                The model is projected onto a subspace spanned by the elements
                of this list.
            W(list): List of adjoint modes.
                The nullspace of the projection is the orthogonal complement
                of the subspace spanned by the elements of W.
                If not specified, an orthogonal projection is used (W = V)

        Returns:
            A :class:`BilinearModel` containing the desired projection
        """
        n = len(V)
        if W is None:
            W = V
        assert len(W) == n

        # Let W1 = (W V')^{-1} W
        G = np.array([[np.dot(W[i], V[j]) for j in range(n)] for i in range(n)])
        W1 = np.linalg.solve(G, W)
        # Now projection is given by P = V' W1, and W1 V' = Identity

        c = np.array([np.dot(W1[i], self._affine) for i in range(n)])
        L = np.array(
            [[np.dot(W1[i], self.linear(V[j])) for j in range(n)] for i in range(n)]
        )
        B = np.array(
            [
                [
                    [np.dot(W1[i], self.bilinear(V[j], V[k])) for k in range(n)]
                    for j in range(n)
                ]
                for i in range(n)
            ]
        )
        return BilinearModel(c, L, B)


@dataclass
class DiscreteModel:
    rhs: VectorField
    timestepper: Union[Timestepper, SemiImplicit]

    def __call__(self, x: Vector) -> Vector:
        return self.timestepper.step(x, self.rhs)


@dataclass
class DiscreteAdjoint:
    adjoint_rhs: Callable[[Vector, Vector], Vector]
    timestepper: Union[Timestepper, SemiImplicit]

    def __call__(self, x: Vector, v: Vector) -> Vector:
        f = partial(self.adjoint_rhs, x)
        return self.timestepper.step(v, f)
