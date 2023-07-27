#!/usr/bin/env python

import sys
import time
from typing import Callable, Tuple

import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes
import numpy as np
from numpy.typing import NDArray
import torch

import romnet
from romnet.models import NoackModel
from romnet.sample import TrajectoryList
from romnet.autoencoder import AE
from noack_train import rom


def slow_manifold_input(xmax, div):
    model = NoackModel()
    X1 = np.linspace(-xmax, xmax, div)
    X2 = np.linspace(-xmax, xmax, div)
    X = np.zeros((div * div, model.num_states))
    k = 0
    for i in range(div):
        for j in range(div):
            X[k, 0] = X1[i]
            X[k, 1] = X2[j]
            r = np.sqrt(X1[i] ** 2 + X2[j] ** 2)
            X[k, 2] = model.slow_manifold(r)
            k = k + 1
    return X


def surface(X, div):
    X1_graph = np.zeros((div, div))
    X2_graph = np.zeros((div, div))
    X3_graph = np.zeros((div, div))
    k = 0
    for j in range(div):
        for i in range(div):
            X1_graph[i, j] = X[k, 0]
            X2_graph[i, j] = X[k, 1]
            X3_graph[i, j] = X[k, 2]
            k = k + 1
    return X1_graph, X2_graph, X3_graph


def plot_Rr(ax: Axes, autoencoder: AE, z_rom_traj: TrajectoryList, test_traj: TrajectoryList, traj_num: int = 0):
    z_test = autoencoder.enc(test_traj.traj[traj_num]).numpy()
    z_rom = z_rom_traj.traj[traj_num]
    ax.plot(z_test[:, 0], z_test[:, 1], "blue", label="FOM")
    ax.plot(z_rom[:, 0], z_rom[:, 1], "red", linestyle='--', label="ROM")


def plot_Rn(ax: Axes, autoencoder: AE, z_rom_traj: TrajectoryList, test_traj: TrajectoryList, traj_num: int = 0):
    xmax = 1.
    div = 20
    x_rom_traj = autoencoder.dec(z_rom_traj.traj).numpy()
    slow_m = slow_manifold_input(xmax, div)
    range_P = autoencoder.forward(slow_m).numpy()
    slow_m_x1, slow_m_x2, slow_m_x3 = surface(slow_m, div)
    range_P_x1, range_P_x2, range_P_x3 = surface(range_P, div)
    ax.plot_surface(slow_m_x1, slow_m_x2, slow_m_x3, color="m", alpha=0.2)
    ax.plot_surface(range_P_x1, range_P_x2, range_P_x3, color="green", alpha=0.3)
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-1, 3])
    ax.plot3D(
        test_traj.traj[traj_num][:, 0],
        test_traj.traj[traj_num][:, 1],
        test_traj.traj[traj_num][:, 2],
        "blue",
    )
    ax.plot3D(
        x_rom_traj[traj_num][:, 0],
        x_rom_traj[traj_num][:, 1],
        x_rom_traj[traj_num][:, 2],
        "red", linestyle='--'
    )


def find_traj_near_origin(test_traj: TrajectoryList) -> int:
    xmax = 1.
    cond1 = test_traj.traj[:, 0, 0] <= xmax
    cond2 = test_traj.traj[:, 0, 1] <= xmax
    cond3 = -xmax <= test_traj.traj[:, 0, 0]
    cond4 = -xmax <= test_traj.traj[:, 0, 1]
    return np.where(cond1 & cond2 & cond3 & cond4)[0][0]


def manifold_error(exp_num: int, basename: str = "") -> float:
    xmax = 1.
    div = 20
    slow_m = slow_manifold_input(xmax, div)
    autoencoder = romnet.load_romnet(basename + str(exp_num) + ".romnet")
    range_P = autoencoder.forward(slow_m).numpy()
    error = np.mean(np.square(np.linalg.norm(range_P - slow_m, axis=1)))
    return error


def ic_colorbar(x_star: NDArray, rbg_basis: Tuple[int, int, int]=(0, 1, 0), basename: str = "") -> Tuple[Callable[[int], Tuple[float, float, float]], Callable[[Axes], None]]:
    test_traj = romnet.load(basename + "_test.traj")
    norm = np.linalg.norm(test_traj.traj[:, 0, :] - x_star, axis=1)
    max = np.max(norm)
    min = np.min(norm)
    C = 1 - (norm - min) / (max - min)
    def color_func(traj_num: int) -> Tuple[float, float, float]:
        return (rbg_basis[0] * C[traj_num], rbg_basis[1] * C[traj_num], rbg_basis[2] * C[traj_num])
    colors_idxs = np.argsort(norm)
    colors = [color_func(idx) for idx in colors_idxs]
    def colorbar(ax: Axes):
        color_plot = plt.scatter([0,0,0],[0,0,0], c=np.linspace(min, max, 3), s=0, cmap=plt.cm.colors.ListedColormap(colors))
        plt.colorbar(color_plot, ticks=[min, max], ax=ax)
    return color_func, colorbar


def test_rom(savefig: bool = False):

    # Note: Here, we use the best model based on the lowest validation loss.
    #       Model selection for large dimensional models would use this methodology.
    #       In this case, the dimensionality is small so we can use the validation
    #       trajectories to select the best model.

    data_basename = "./data/noack/Course/noack"
    results_basename = "./results/noack/Course_StandAE_REC/noack"

    with torch.no_grad():

        # start time
        tic = time.time()

        # gather experiments
        print("----------------------------------------- Loading experiments")
        num_model = 64
        exp_list = romnet.ExperimentList([ romnet.load_exp(results_basename + str(i) + ".exp") for i in range(num_model)])

        # plot training and validation loss
        print("----------------------------------------- Plotting losses")
        val_exp_num = exp_list.val_ranking[0]
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        exp_list.plot_train_loss(ax, exp_num=val_exp_num)
        exp_list.plot_val_loss(ax, exp_num=val_exp_num)
        ax.set_ylabel("Training Loss (blue) and Validation Loss (green)")
        ax.set_xlabel("Epochs")
        if savefig:
            fig.savefig(results_basename + "_tvloss.pdf", format="pdf")

        # plot rom loss
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        exp_list.plot_rom_loss(ax, exp_num=val_exp_num)
        ax.set_ylabel("ROM Loss")
        ax.set_xlabel("Epochs")
        ax.set_ylim(top=100)
        if savefig:
            fig.savefig(results_basename + "_romloss.pdf", format="pdf")

        # print the rankings
        print("----------------------------------------- Printing loss rankings")
        exp_list.print_rankings()

        # plot rom error
        print("----------------------------------------- Plotting rom error")
        test_traj = romnet.load(data_basename + "_test.traj")
        traj_num = find_traj_near_origin(test_traj)
        y_top = 1e3
        y_bot = 1e-6
        x_star = np.zeros(3)
        color_func, colorbar = ic_colorbar(x_star, basename=data_basename)
        rom_exp_num = exp_list.trained_rom_ranking[0]
        orthrom_exp_num = exp_list.trained_orthrom_ranking[0]
        model = NoackModel()
        autoencoder_val = romnet.load_romnet(results_basename + str(val_exp_num) + ".romnet")
        _, rom_error_val, rom_traj_val = rom(model, autoencoder_val, test_traj)
        autoencoder_rom = romnet.load_romnet(results_basename + str(rom_exp_num) + ".romnet")
        _, rom_error_rom, rom_traj_rom = rom(model, autoencoder_rom, test_traj)
        autoencoder_orthrom = romnet.load_romnet(results_basename + str(orthrom_exp_num) + ".romnet")
        _, rom_error_orthrom, rom_traj_orthrom = rom(model, autoencoder_orthrom, test_traj, dec_rom=True)
        dt = 0.1
        t_final = 20.
        n = int(t_final / dt) + 1
        t = np.arange(0, n, 1) * dt
        fig = plt.figure()
        ax = fig.add_subplot(1, 3, 1)
        exp_list.plot_rom_error(ax, t, exp_num=val_exp_num, color_func=color_func)
        ax.semilogy(t, rom_error_val[traj_num], color='r', linewidth='1.5')
        ax.set_title("Best Val. Model: Model {}".format(val_exp_num))
        ax.set_ylabel("ROM Error")
        ax.set_xlabel("Time, $t$")
        ax.set_ylim([y_bot, y_top])
        ax = fig.add_subplot(1, 3, 2)
        exp_list.plot_rom_error(ax, t, exp_num=rom_exp_num, color_func=color_func)
        ax.semilogy(t, rom_error_rom[traj_num], color='r', linewidth='1.5')
        ax.set_title("Best EncROM Model: Model {}".format(rom_exp_num))
        ax.set_xlabel("Time, $t$")
        ax.set_ylim([y_bot, y_top])
        ax = fig.add_subplot(1, 3, 3)
        exp_list.plot_rom_error(ax, t, exp_num=orthrom_exp_num, color_func=color_func, orthrom=True)
        ax.semilogy(t, rom_error_orthrom[traj_num], color='r', linewidth='1.5')
        ax.set_title("Best DecROM Model: Model {}".format(orthrom_exp_num))
        ax.set_xlabel("Time, $t$")
        ax.set_ylim([y_bot, y_top])
        colorbar(ax)
        if savefig:
            fig.savefig(results_basename + "_error.pdf", format="pdf")

        # plot trajectories in Rr
        print("----------------------------------------- Plotting trajectories in Rr")
        fig = plt.figure()
        ax = fig.add_subplot(1, 3, 1)
        plot_Rr(ax, autoencoder_val, rom_traj_val, test_traj, traj_num)
        ax.set_title("Best Val. Loss: Model {}".format(val_exp_num))
        ax.set_xlabel("$z_1$")
        ax.set_xlabel("$z_2$")
        ax = fig.add_subplot(1, 3, 2)
        plot_Rr(ax, autoencoder_rom, rom_traj_rom, test_traj, traj_num)
        ax.set_title("Best EncROM Loss: Model {}".format(rom_exp_num))
        ax.set_xlabel("$z_1$")
        ax = fig.add_subplot(1, 3, 3)
        plot_Rr(ax, autoencoder_orthrom, rom_traj_orthrom, test_traj, traj_num)
        ax.set_title("Best DecROM Loss: Model {}".format(orthrom_exp_num))
        ax.set_xlabel("$z_1$")
        if savefig:
            fig.savefig(results_basename + "_trajRr.pdf", format="pdf")

        # plot trajectories in Rn
        print("----------------------------------------- Plotting trajectories in Rn")
        fig = plt.figure()
        ax = fig.add_subplot(1, 3, 1, projection="3d")
        plot_Rn(ax, autoencoder_val, rom_traj_val, test_traj, traj_num)
        ax.set_title("Best Val. Loss: Model {}".format(val_exp_num))
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_zlabel("$x_3$")
        ax = fig.add_subplot(1, 3, 2, projection="3d")
        plot_Rn(ax, autoencoder_rom, rom_traj_rom, test_traj, traj_num)
        ax.set_title("Best EncROM Loss: Model {}".format(rom_exp_num))
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_zlabel("$x_3$")
        ax = fig.add_subplot(1, 3, 3, projection="3d")
        plot_Rn(ax, autoencoder_orthrom, rom_traj_orthrom, test_traj, traj_num)
        ax.set_title("Best DecROM Loss: Model {}".format(orthrom_exp_num))
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_zlabel("$x_3$")
        if savefig:
            fig.savefig(results_basename + "_trajRn.pdf", format="pdf")

        # plot train time
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        exp_list.plot_train_time(ax)
        ax.set_ylabel("Training Time, [sec]")
        if savefig:
            fig.savefig(results_basename + "_time.pdf", format="pdf")

        # calculate manifold error
        print("----------------------------------------- Calculate manifold error")
        for exp_num in range(num_model):
            setattr(exp_list.experiments[exp_num], "manifold_error", manifold_error(exp_num, basename=results_basename))
        print("Best Val. Manifold Error: {}".format(exp_list.experiments[val_exp_num].manifold_error))
        print("Best EncROM Manifold Error: {}".format(exp_list.experiments[rom_exp_num].manifold_error))
        print("Best DecROM Manifold Error: {}".format(exp_list.experiments[orthrom_exp_num].manifold_error))

        # rom_loss
        print("----------------------------------------- Printing rom_loss")
        print("Best Val. ROM Loss: {}".format(exp_list.experiments[val_exp_num].trained_rom_loss))
        print("Best EncROM ROM Loss: {}".format(exp_list.experiments[rom_exp_num].trained_rom_loss))
        print("Best DecROM ROM Loss: {}".format(exp_list.experiments[orthrom_exp_num].trained_orthrom_loss))

        # print mean training time
        print("----------------------------------------- Printing mean training time")
        print("Mean Training Time: {}".format(np.mean(exp_list.train_time)))

        # plot
        if not savefig:
            plt.show()

        # end time
        toc = time.time()
        print("----------------------------------------- Done")
        print("Run Time: {}".format(toc - tic))


if __name__ == "__main__":
    if len(sys.argv) == 1:
        test_rom()
    elif (len(sys.argv) == 2) and (sys.argv[1] == "savefig"):
        test_rom(savefig=True)
