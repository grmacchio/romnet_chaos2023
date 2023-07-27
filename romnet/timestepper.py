"""timestepper - step ordinary differential equations forward in time"""

import abc
from typing import Callable, Dict, List, Type

from .typing import Vector, VectorField

__all__ = ["Timestepper", "SemiImplicit"]


class Timestepper(abc.ABC):
    """Abstract base class for timesteppers."""

    # registry for subclasses, mapping names to constructors
    __registry: Dict[str, Type["Timestepper"]] = {}

    def __init_subclass__(cls) -> None:
        name = cls.__name__.lower()
        cls.__registry[name] = cls

    @classmethod
    def lookup(cls, method: str) -> Type["Timestepper"]:
        """Return the subclass corresponding to the string in `method`."""
        try:
            return cls.__registry[method.lower()]
        except KeyError as exc:
            raise NotImplementedError(f"Method '{method}' unknown") from exc

    def __init__(self, dt: float):
        self.dt = dt

    @abc.abstractmethod
    def step(self, x: Vector, rhs: VectorField) -> Vector:
        """Advance the state x by one timestep, for the ODE x' = rhs(x)."""

    @classmethod
    def methods(cls) -> List[str]:
        return list(cls.__registry.keys())


class SemiImplicit(abc.ABC):
    """Abstract base class for semi-implicit timesteppers."""

    # registry for subclasses, mapping names to constructors
    __registry: Dict[str, Type["SemiImplicit"]] = {}

    def __init_subclass__(cls) -> None:
        name = cls.__name__.lower()
        cls.__registry[name] = cls

    @classmethod
    def lookup(cls, method: str) -> Type["SemiImplicit"]:
        """Return the subclass corresponding to the string in `method`."""
        try:
            return cls.__registry[method.lower()]
        except KeyError as exc:
            raise NotImplementedError(f"Method '{method}' unknown") from exc

    def __init__(
        self,
        dt: float,
        linear: VectorField,
        solver_factory: Callable[[float], Callable[[Vector], Vector]],
    ):
        self._dt = dt
        self.linear = linear
        self.get_solver = solver_factory
        self.update()

    @property
    def dt(self) -> float:
        return self._dt

    @dt.setter
    def dt(self, value: float) -> None:
        self._dt = value
        self.update()

    @abc.abstractmethod
    def update(self) -> None:
        """Update quantities used in the semi-implicit solve.

        This routine is called when the timestepper is created, and whenever
        the timestep is changed
        """

    @abc.abstractmethod
    def step(self, x: Vector, nonlinear: VectorField) -> Vector:
        """Advance the state forward by one step"""

    @classmethod
    def methods(cls) -> List[str]:
        return list(cls.__registry.keys())


class Euler(Timestepper):
    """Explicit Euler timestepper."""

    def step(self, x: Vector, rhs: VectorField) -> Vector:
        return x + self.dt * rhs(x)


class RK2(Timestepper):
    """Second-order Runge-Kutta timestepper."""

    def step(self, x: Vector, rhs: VectorField) -> Vector:
        k1 = self.dt * rhs(x)
        k2 = self.dt * rhs(x + k1)
        return x + (k1 + k2) / 2.0

class RK4(Timestepper):
    """Fourth-order Runge-Kutta timestepper."""

    def step(self, x: Vector, rhs: VectorField) -> Vector:
        k1 = self.dt * rhs(x)
        k2 = self.dt * rhs(x + k1 / 2.0)
        k3 = self.dt * rhs(x + k2 / 2.0)
        k4 = self.dt * rhs(x + k3)
        return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


class RK2CN(SemiImplicit):
    """Semi-implicit timestepper: Crank-Nicolson + 2nd-order Runge-Kutta.

    See Peyret p148-149
    """

    def update(self) -> None:
        self.solve = self.get_solver(0.5 * self.dt)

    def step(self, x: Vector, nonlinear: VectorField) -> Vector:
        rhs_linear = x + 0.5 * self.dt * self.linear(x)
        Nx = nonlinear(x)

        rhs1 = rhs_linear + self.dt * Nx
        x1 = self.solve(rhs1)

        rhs2 = rhs_linear + 0.5 * self.dt * (Nx + nonlinear(x1))
        x2 = self.solve(rhs2)
        return x2


class RK3CN(SemiImplicit):
    """Semi-implicit timestepper: Crank-Nicolson + 3rd-order Runge-Kutta.

    Peyret, p.146 and 149
    """

    A = [0, -5.0 / 9, -153.0 / 128]
    B = [1.0 / 3, 15.0 / 16, 8.0 / 15]
    Bprime = [1.0 / 6, 5.0 / 24, 1.0 / 8]

    def update(self) -> None:
        self.solvers = [self.get_solver(b * self.dt) for b in self.Bprime]

    def step(self, x: Vector, nonlinear: VectorField) -> Vector:
        A = self.A
        B = self.B
        Bprime = self.Bprime

        Q1 = self.dt * nonlinear(x)
        rhs1 = x + B[0] * Q1 + Bprime[0] * self.dt * self.linear(x)
        x1 = self.solvers[0](rhs1)

        Q2 = A[1] * Q1 + self.dt * nonlinear(x1)
        rhs2 = x1 + B[1] * Q2 + Bprime[1] * self.dt * self.linear(x1)
        x2 = self.solvers[1](rhs2)

        Q3 = A[2] * Q2 + self.dt * nonlinear(x2)
        rhs3 = x2 + B[2] * Q3 + Bprime[2] * self.dt * self.linear(x2)
        x3 = self.solvers[2](rhs3)
        return x3
