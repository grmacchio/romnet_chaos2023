#!/usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy as sp
from typing import Callable
from matplotlib.axes import Axes
from numpy.typing import NDArray
import torch

import romnet
from romnet.models import Pitchfork
from romnet.typing import Vector
from romnet.sample import TrajectoryList

matplotlib.rc('xtick', labelsize=22) 
matplotlib.rc('ytick', labelsize=22) 


def plot_manifolds(
    model: Pitchfork,
    P: Callable[[Vector], Vector],
    ax: Axes,
    learned: bool = False,
    vert_shift: float = 0.0
) -> None:
    ax.set_xlim((-0.1, 1.75))
    ax.set_ylim((-0.25-vert_shift, 1.75-vert_shift))
    ax.set_xlabel("$x_1$", fontsize=30)
    ax.set_xticks([0.0, 1.0, 1.75])
    ax.set_yticks([0.0-vert_shift, 1.0-vert_shift, 1.75-vert_shift])
    ax.margins(x=0, y=0)
    x1 = np.linspace(-1.75, 1.75, 100)
    x2 = model.slow_manifold(x1)
    x_slow = np.vstack((x1, x2))
    ax.plot(
        x_slow[0, :], x_slow[1, :], color="m",
        alpha=0.4, linewidth=6, zorder=2
    )
    if learned:
        x_learn = np.zeros_like(x_slow)
        for i in range(x_slow.shape[1]):
            x_learn[:, i] = P(x_slow[:, i])
        ax.plot(
            x_learn[0, :], x_learn[1, :], color="g",
            alpha=0.5, linewidth=6, zorder=2
        )


def plot_projected_ic(
    P: Callable[[Vector], Vector],
    ic: Vector,
    true_traj: TrajectoryList,
    proj_traj: TrajectoryList,
    ax: Axes,
    proj_k: int = 0
) -> None:
    P_ic = P(ic)
    ax.plot(
        ic[0], ic[1], marker="P", color="blue",
        markersize=20, zorder=1, label="$x_0$"
    )
    ax.plot(
        P_ic[0], P_ic[1], marker="P", color="red",
        markersize=20, zorder=2, label="$P(x_0)$"
    )
    ax.plot(
        true_traj.traj[0, :, 0], true_traj.traj[0, :, 1],
        color="blue", linewidth=4, zorder=2, label="$x(t; x_0)$"
    )
    ax.plot(
        proj_traj.traj[0, :, 0], proj_traj.traj[0, :, 1],
        color="red", linewidth=4, linestyle="--", zorder=2, label="$\hat{x}(t; x_0)$"
    )
    fixed_pt1 = np.array([1, 1])
    ax.plot(
        fixed_pt1[0], fixed_pt1[1], marker='o',
        color="blue", markersize="20"
    )
    fixed_pt2 = np.array([0, 0])
    ax.plot(
        fixed_pt2[0], fixed_pt2[1], marker='o',
        color="blue", markersize="20"
    )

    # true trajectories arrow
    for k in [0, 20]:
        dx = true_traj.traj[0, k+1, :] - true_traj.traj[0, k, :]
        mult = 0.4
        base = true_traj.traj[0, k+1, :] - mult * dx
        diff = mult * dx
        ax.arrow(
            base[0], base[1],
            mult * diff[0], mult * diff[1],
            color="blue", width=0.016, length_includes_head=True,
            zorder=1
        )

    # projected trajectories arrow
    for k in [proj_k]:
        dx = proj_traj.traj[0, k+1, :] - proj_traj.traj[0, k, :]
        mult = 0.1
        base = proj_traj.traj[0, k+1, :] - mult * dx
        diff = mult * dx
        ax.arrow(
            base[0], base[1],
            mult * diff[0], mult * diff[1],
            color="red", width=0.016, length_includes_head=True,
            zorder=1
        )


def plot_projected_dynamics(
        model: Pitchfork,
        P: Callable[[Vector], Vector],
        DP: Callable[[Vector, Vector], Vector],
        x1_learned: NDArray[np.float64],
        x1_learn_fixed_pt: NDArray[np.float64],
        mult: float,
        ax: Axes,
        label: str
    ) -> None:
        
        # plot vectors function
        def plot_vectors(
            x: NDArray[np.float64],
            v: NDArray[np.float64],
            mult: float,
            ax: Axes,
            color: str,
            zorder: int,
            width: float = 0.02
        ) -> None:
            for i in range(x.shape[0]):
                ax.arrow(
                    x[i, 0], x[i, 1],
                    mult * v[i, 0], mult * v[i, 1],
                    color=color, width=width, length_includes_head=True,
                    zorder=zorder
                )
        
        # plot dotted project lines function
        def plot_lines(
            base_pts: NDArray[np.float64],
            end_pts: NDArray[np.float64],
            ax_in: Axes = ax
        ) -> None:
            for i in range(base_pts.shape[0]):
                ax_in.plot(
                    [base_pts[i, 0], end_pts[i, 0]],
                    [base_pts[i, 1], end_pts[i, 1]],
                    color="black", linewidth=0.75, linestyle='--'
                )
        
        # project function
        def project(x1):
            x = np.zeros((2, len(x1)))
            for i in range(len(x1)):
                x_in = np.array([x1[i], model.slow_manifold(x1[i])])
                x[:, i] = P(x_in)
            return x
        
        # plotting slow manifold dynamics
        x1_slow_1 = np.array([0.28, 0.61, 0.875])
        x_slow_1 = np.vstack((x1_slow_1, model.slow_manifold(x1_slow_1)))
        f_flow_1 = model.rhs(x_slow_1) / np.linalg.norm(model.rhs(x_slow_1), axis=0)
        plot_vectors(x_slow_1.T, f_flow_1.T, 0.18, ax, "blue", 5)
        x1_slow_2 = np.array([1.11])
        x_slow_2 = np.vstack((x1_slow_2, model.slow_manifold(x1_slow_2)))
        f_flow_2 = model.rhs(x_slow_2) / np.linalg.norm(model.rhs(x_slow_2), axis=0)
        plot_vectors(x_slow_2.T, f_flow_2.T, 0.18, ax, "blue", 5)
        fixed_pt1 = np.array([np.sqrt(model.a), np.sqrt(model.a)])
        ax.plot(
            fixed_pt1[0], fixed_pt1[1], marker='o',
            color="blue", markersize="20"
        )
        fixed_pt2 = np.array([0, 0])
        ax.plot(
            fixed_pt2[0], fixed_pt2[1], marker='o',
            color="blue", markersize="20"
        )

        # plotting FOM dynamics on learned manifold
        x_learn = project(x1_learned)
        f_learned = model.rhs(x_learn)

        # plotting ROM dynamics on learned manifold
        temp = np.array([DP(x_learn[:, i], f_learned[:, i])
                                for i in range(x_learn.shape[1])]).T
        DPf_learned = temp / np.linalg.norm(temp, axis=0)
        plot_vectors(x_learn.T, DPf_learned.T, mult, ax, "red", 4)
        x_learn_fixed_pt = project(x1_learn_fixed_pt)
        ax.plot(
            x_learn_fixed_pt[0, 0], x_learn_fixed_pt[1, 0],
            marker="o", color="red", markersize=10, zorder=4
        )
        ax.plot(
            x_learn_fixed_pt[0, 1], x_learn_fixed_pt[1, 1],
            marker="o", color="red", markersize=10, zorder=4
        )

        # plotting inset
        ax_inset = ax.inset_axes([0.46, 0.0, 0.52, 0.52])
        plot_manifolds(model, P, ax_inset, learned=True, vert_shift=0.3)
        ax_inset.set_xticklabels([])
        ax_inset.set_yticklabels([])
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        ax_inset.set_xlabel("")
        ax_inset.set_ylabel("")
        zoom = 0.05
        x_center = 0.6
        x_shift = 0.000
        xlim = (x_center - zoom + x_shift, x_center + zoom + x_shift)
        ax_inset.set_xlim(xlim)
        y_center = model.slow_manifold(x_center) + 0.001
        ylim = (y_center - zoom, y_center + zoom)
        ax_inset.set_ylim(ylim)
        ax.indicate_inset_zoom(ax_inset, edgecolor="black")
        div = 2*3 + 1
        x1_grid = np.linspace(xlim[0], xlim[1], div)
        x2_grid = np.linspace(ylim[0], ylim[1], div)
        x1_back, x2_back = np.meshgrid(x1_grid, x2_grid)
        x_back = np.array([x1_back, x2_back])
        f_back = np.array([[model.rhs(x_back[:, i, j])
                            for i in range(div)]for j in range(div)]).T
        f_back = f_back / np.linalg.norm(f_back, axis=0)
        ax_inset.quiver(
            x_back[0], x_back[1],
            f_back[0], f_back[1],
            color="gray", alpha=0.3, angles="xy", width=0.015, scale=7
        )
        multmult = 0.26
        x_center_learned = np.array([x_center, 0.95 * x_center**2 + 0.05]).reshape(-1, 1)
        f_center_learned = model.rhs(x_center_learned)
        plot_vectors(x_center_learned.T, f_center_learned.T, multmult, ax_inset, "black", 4, width=0.0025)
        DPf_center_learned = np.array([DP(x_center_learned[:, i], f_center_learned[:, i])
                                for i in range(x_center_learned.shape[1])]).T
        plot_vectors(x_center_learned.T, DPf_center_learned.T, multmult, ax_inset, "red", 4, width=0.0025)
        base_pts = x_center_learned + multmult * f_center_learned
        end_pts = x_center_learned + multmult * DPf_center_learned
        plot_lines(base_pts.T, end_pts.T, ax_in=ax_inset)
        x1_center_slow = x_center + 0.017
        x_center_slow = np.array([x1_center_slow, model.slow_manifold(x1_center_slow)]).reshape(-1, 1)
        f_center_slow = model.rhs(x_center_slow)
        plot_vectors(x_center_slow.T, f_center_slow.T, multmult, ax_inset, "blue", 4, width=0.0025)
        ax_inset.set_aspect("equal")
        # ax.arrow(
        #     -2, 0, 0, 1, color="black", width=0.009,
        #     length_includes_head=True, label="$f(\\hat{x})$"
        # )
        # ax.arrow(
        #     -2, 0, 0, 1, color="red", width=0.009,
        #     length_includes_head=True, label=label
        # )


def compare_projections():

    # model
    model = Pitchfork()

    # true trajectory
    dt = 0.1
    t_final = 800
    n = int(t_final / dt) + 1
    ic = np.array([1.25, 0])
    t = np.arange(0, n) * dt
    step = model.get_stepper(dt, method="rk4")
    def ic_func():
        return ic
    true_traj = romnet.sample(step, ic_func, 1, n)

    # learned manifold
    def learned_manifold(x1: float) -> float:
        return 0.95 * x1**2 + 0.05
    def learned_manifold_deriv(x1: float) -> float:
        return 0.95 * 2 * x1

    # slow manifold projections, P_slow
    def P_orth_slow(x: Vector) -> Vector:
        def objective(x1: float) -> float:
            x_manifold = np.array([x1, model.slow_manifold(x1)])
            return np.linalg.norm(x - x_manifold)
        x0 = 1
        sol = sp.optimize.minimize(objective, x0)
        return np.array([sol.x.item(), model.slow_manifold(sol.x).item()])
    def P_oblique_slow(x: Vector) -> Vector:
        x1 = x[0]
        x2 = model.slow_manifold(x[0])
        return np.array([x1, x2])

    # learned manifold projections, P
    def P_orth(x: Vector) -> Vector:
        def objective(x1: float) -> float:
            x_manifold = np.array([x1, learned_manifold(x1)])
            return np.linalg.norm(x - x_manifold)
        x0 = 1
        sol = sp.optimize.minimize(objective, x0)
        return np.array([sol.x.item(), learned_manifold(sol.x).item()])
    def P_oblique(x: Vector) -> Vector:
        x1 = x[0]
        x2 = learned_manifold(x[0])
        return np.array([x1, x2])

    # derivative of learned manifold projections, DP
    def DP_orth(x: Vector, v: Vector) -> Vector:
        Phi = np.array([[1], [learned_manifold_deriv(x[0])]])
        Psi = Phi
        DP = Phi @ Psi.T / (Psi.T @ Phi)
        return DP @ v
    def DP_oblique(x: Vector, v: Vector) -> Vector:
        Phi = np.array([[1], [learned_manifold_deriv(x[0])]])
        Psi = np.array([[1], [0]])
        DP = Phi @ Psi.T / (Psi.T @ Phi)
        return DP @ v

    # plot projected ic for P_orth
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(1, 2, 1)
    ax.set_aspect("equal")
    proj_orth_ic = P_orth_slow(ic)
    def ic_func() -> Vector:
        return proj_orth_ic
    proj_orth_traj = romnet.sample(step, ic_func, 1, n)
    plot_manifolds(model, P_orth_slow, ax)
    plot_projected_ic(P_orth_slow, ic, true_traj, proj_orth_traj, ax, proj_k=87)
    ax.legend(
            loc="upper left", labelspacing=1, borderpad=0.5, fontsize=22.5#15
    )
    ax.set_ylabel("$x_2$", fontsize=30)

    # plot projected ic for P_oblique
    ax = fig.add_subplot(1, 2, 2)
    ax.set_aspect("equal")
    proj_oblique_ic = P_oblique_slow(ic)
    def ic_func() -> Vector:
        return proj_oblique_ic
    proj_oblique_traj = romnet.sample(step, ic_func, 1, n)
    plot_manifolds(model, P_oblique_slow, ax)
    plot_projected_ic(P_oblique_slow, ic, true_traj, proj_oblique_traj, ax, proj_k=57)
    ax.set_yticklabels([])
    fig.savefig("./results/pitchfork/pitchfork_example_ic.pdf", format="pdf")

    # plot error plots
    matplotlib.rc('xtick', labelsize=26) 
    matplotlib.rc('ytick', labelsize=26) 
    t_plot_idx = int(5 / dt)
    fig = plt.figure(figsize=(18, 9))
    ax = fig.add_subplot(1, 2, 1)
    error_orth = np.linalg.norm(true_traj.traj[0, :, :] - proj_orth_traj.traj[0, : ,:], axis=1)
    error_oblique = np.linalg.norm(true_traj.traj[0, :, :] - proj_oblique_traj.traj[0, : ,:], axis=1)
    orth_asym_error = np.exp( -2 * model.lam1 * t)
    oblique_asym_error = np.exp( -1 * model.lam2 * t)
    ax.semilogy(t[:t_plot_idx], error_orth[:t_plot_idx], color="red", linewidth=4, linestyle="--", zorder=2, label="$\|x(t;x_0) - \hat{x}(t;x_0)\|_2$")
    ax.semilogy(t[:t_plot_idx], orth_asym_error[:t_plot_idx], color="k", linewidth=4, zorder=1, label="$e^{-2 \lambda t}$")
    ax.set_xlabel("$t$", fontsize=34)
    ax.legend(loc="upper right", labelspacing=1, borderpad=0.5, fontsize=26) #17
    ax = fig.add_subplot(1, 2, 2)
    ax.semilogy(t[:t_plot_idx], error_oblique[:t_plot_idx], color="red", linewidth=4, linestyle="--", zorder=2, label="$\|x(t;x_0) - \hat{x}(t;x_0)\|_2$")
    ax.semilogy(t[:t_plot_idx], oblique_asym_error[:t_plot_idx], color="k", linewidth=4, zorder=1, label="$e^{- t / \epsilon}$")
    ax.set_xlabel("$t$", fontsize=34)
    ax.legend(loc="upper right", labelspacing=1, borderpad=0.5, fontsize=26) #17
    fig.savefig("./results/pitchfork/pitchfork_example_error.pdf", format="pdf")

    # plotting projected dynamics for P_orth
    matplotlib.rc('xtick', labelsize=22) 
    matplotlib.rc('ytick', labelsize=22) 
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(1, 2, 1)
    ax.set_aspect("equal")
    x1_learn = np.array([0.3, 0.57, 0.84, 1.14])
    x1_learn_fixed_pt = np.array([0.0, 1.0])
    plot_manifolds(model, P_orth, ax, learned=True, vert_shift=0.3)
    plot_projected_dynamics(
        model, P_orth, DP_orth, x1_learn, x1_learn_fixed_pt, 0.18, ax,
        "Orthogonal"#"$\\mathrm{d}P[\\mathrm{d}P^T \\mathrm{d}P]^{-1}\\mathrm{d}P^T(\\hat{x}) f(\\hat{x})$"
    )
    ax.set_ylabel("$x_2$", fontsize=30)

    # plotting projected dynamics for P_oblique
    ax = fig.add_subplot(1, 2, 2)
    ax.set_aspect("equal")
    x1_learn = np.array([0.11, 0.46, 0.74, 1.22])
    x1_learn_fixed_pt = np.array([0.0, 1.0])
    plot_manifolds(model, P_oblique, ax, learned=True, vert_shift=0.3)
    plot_projected_dynamics(
        model, P_orth, DP_oblique, x1_learn, x1_learn_fixed_pt, 0.18, ax,
        "Oblique"#"$\\mathrm{d}P(\\hat{x})f(\\hat{x})$"
    )
    ax.set_yticklabels([])
    fig.savefig("./results/pitchfork/pitchfork_example_dyn.pdf", format="pdf")


if __name__ == "__main__":
    compare_projections()
