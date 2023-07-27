
from typing import Callable, Tuple, Optional
import re
import os
import sys

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import torch
import pickle
from matplotlib.pyplot import Axes

import romnet
from romnet.sample import TrajectoryList
from romnet.models import NoackModel
from noack_train import rom
from noack_test import plot_Rn, plot_Rr, manifold_error


class ROMResults():

    def __init__(self, rom_loss: float, rom_error: NDArray, rom_traj: TrajectoryList) -> None:
        self.rom_loss = rom_loss
        self.rom_error = rom_error
        self.rom_traj = rom_traj

    def save(self, fname: str) -> None:
        with open(fname, "wb") as fp:
            pickle.dump(self, fp, pickle.HIGHEST_PROTOCOL)


def plot_rom_error(ax: Axes, rom_error: NDArray, t: NDArray, num_plot_traj: Optional[int]=None, color_func: Callable[[int], Tuple[float, float, float]]=lambda n: (0, 0, 0), label: str = "") -> None:
    num_traj = len(rom_error)
    if num_plot_traj is None:
        num_plot_traj = num_traj
    step = np.max((int(num_traj / num_plot_traj), 1))
    traj_idxs = np.arange(0, num_traj, step=step)
    for traj_idx in traj_idxs[:-1]:
        color = color_func(traj_idx)
        ax.semilogy(t, rom_error[traj_idx], color=color, linewidth=0.5, alpha=0.5)
    color = color_func(traj_idxs[-1])
    ax.semilogy(t, rom_error[traj_idx], color=color, linewidth=0.5, alpha=0.5, label=label)


def generate_rom_results():

    data_basename = "./data/noack"
    results_basename = "./results/noack"

    with torch.no_grad():

        # experiment types
        exp_types = [results_basename + "/Fine_ProjAE_REC/",
                     results_basename + "/Fine_ProjAE_GAP/",
                     results_basename + "/Fine_ProjAE_RVP/",
                     results_basename + "/Fine_StandAE_REC/",
                     results_basename + "/Fine_StandAE_GAP/",
                     results_basename + "/Fine_StandAE_RVP/",
                     results_basename + "/Course_ProjAE_REC/",
                     results_basename + "/Course_ProjAE_GAP/",
                     results_basename + "/Course_ProjAE_RVP/",
                     results_basename + "/Course_StandAE_REC/",
                     results_basename + "/Course_StandAE_GAP/",
                     results_basename + "/Course_StandAE_RVP/"]
        
        # load test data set
        test_traj = romnet.load(data_basename + "/Fine/noack_test.traj")

        # experiment type loop
        num_exp = 64
        for exp_type_num in range(len(exp_types)):
            print("Experiment Type Number: {}".format(exp_type_num))
            exp_type = exp_types[exp_type_num]

            if "Fine" in exp_type:
                val_traj = romnet.load(data_basename + "/Fine/noack_val.traj")
            elif "Course" in exp_type:
                val_traj = romnet.load(data_basename + "/Course/noack_val.traj")

            # model
            model = NoackModel()

            # determine the best EncROM and DecROM models based on their associated validation data sets
            best_rom_num = 0
            best_rom_loss = np.inf
            best_orthrom_num = 0
            best_orthrom_loss = np.inf
            for exp_num in range(num_exp):
                sys.stdout = None
                print(exp_num)
                autoencoder = romnet.load_romnet(exp_type + "noack" + str(exp_num) + ".romnet")
                rom_loss, _, _ = rom(model, autoencoder, val_traj)
                orthrom_loss, _, _ = rom(model, autoencoder, val_traj, dec_rom=True)
                if rom_loss < best_rom_loss:
                    best_rom_num = exp_num
                    best_rom_loss = rom_loss
                if orthrom_loss < best_orthrom_loss:
                    best_orthrom_num = exp_num
                    best_orthrom_loss = orthrom_loss
                sys.stdout = sys.__stdout__
            
            # delete already saved .romresults and .orthromresults files
            rom_file_pattern = re.compile(r'noack(\d+)' + '.romresults')
            orthrom_file_pattern = re.compile(r'noack(\d+)' + '.orthromresults')
            for filename in os.listdir(exp_type):
                rom_match = rom_file_pattern.match(filename)
                orthrom_match = orthrom_file_pattern.match(filename)
                if rom_match:
                    os.remove(os.path.join(exp_type, filename))
                if orthrom_match:
                    os.remove(os.path.join(exp_type, filename))

            # generate EncROM and DecROM predicted test trajectories on TestData
            best_rom_autoencoder = romnet.load_romnet(exp_type + "noack" + str(best_rom_num) + ".romnet")
            best_rom_loss, best_rom_error, best_rom_traj = rom(model, best_rom_autoencoder, test_traj)
            best_orthrom_autoencoder = romnet.load_romnet(exp_type + "noack" + str(best_orthrom_num) + ".romnet")
            best_orthrom_loss, best_orthrom_error, best_orthrom_traj = rom(model, best_orthrom_autoencoder, test_traj, dec_rom=True)

            # save the best EncROM and DecROM rom trajectories
            rom_results = ROMResults(best_rom_loss, best_rom_error, best_rom_traj)
            orthrom_results = ROMResults(best_orthrom_loss, best_orthrom_error, best_orthrom_traj)
            rom_results.save(exp_type + "noack" + str(best_rom_num) + ".romresults")
            orthrom_results.save(exp_type + "noack" + str(best_orthrom_num) + ".orthromresults")


def generate_table():

    results_basename = "./results/noack"

    with torch.no_grad():

        # experiment types
        exp_types = [results_basename + "/Fine_ProjAE_REC/",
                     results_basename + "/Fine_ProjAE_GAP/",
                     results_basename + "/Fine_ProjAE_RVP/",
                     results_basename + "/Fine_StandAE_REC/",
                     results_basename + "/Fine_StandAE_GAP/",
                     results_basename + "/Fine_StandAE_RVP/",
                     results_basename + "/Course_ProjAE_REC/",
                     results_basename + "/Course_ProjAE_GAP/",
                     results_basename + "/Course_ProjAE_RVP/",
                     results_basename + "/Course_StandAE_REC/",
                     results_basename + "/Course_StandAE_GAP/",
                     results_basename + "/Course_StandAE_RVP/"]

        # initialize Dataset1 and Dataset2 manifold error and prediction error tables
        table1 = [
            ["Type", "Rec. Manf.", "Rec. Pred.", "GAP Manf.", "GAP Pred.", "RVP Manf.", "RVP Pred."],
            ["ProjAE, EncROM", None, None, None, None, None, None],
            ["StandAE, EncROM", None, None, None, None, None, None],
            ["ProjAE, DecROM", None, None, None, None, None, None],
            ["StandAE, DecROM", None, None, None, None, None, None]
        ]
        table2 = [
            ["Type", "Rec. Manf.", "Rec. Pred.", "GAP Manf.", "GAP Pred.", "RVP Manf.", "RVP Pred."],
            ["ProjAE, EncROM", None, None, None, None, None, None],
            ["StandAE, EncROM", None, None, None, None, None, None],
            ["ProjAE, DecROM", None, None, None, None, None, None],
            ["StandAE, DecROM", None, None, None, None, None, None]
        ]

        # experiment type loop
        for exp_type_num in range(len(exp_types)):
            exp_type = exp_types[exp_type_num]

            # load EncROM and DecROM results
            rom_file_pattern = re.compile(r'noack(\d+)' + '.romresults')
            orthrom_file_pattern = re.compile(r'noack(\d+)' + '.orthromresults')
            for filename in os.listdir(exp_type):
                rom_match = rom_file_pattern.match(filename)
                orthrom_match = orthrom_file_pattern.match(filename)
                if rom_match:
                    rom_exp_num = int(rom_match.group(1))
                if orthrom_match:
                    orthrom_exp_num = int(orthrom_match.group(1))
            rom_results = romnet.load(exp_type + "noack" + str(rom_exp_num) + ".romresults")
            orthrom_results = romnet.load(exp_type + "noack" + str(orthrom_exp_num) + ".orthromresults")

            # calculate the EncROM and DecROM manifold error
            rom_manifold_error = manifold_error(rom_exp_num, exp_type + "noack")
            orthrom_manifold_error = manifold_error(orthrom_exp_num, exp_type + "noack")

            # insert manifold errors and prediction errors into table
            if exp_type_num < 6:
                table = table1
            elif exp_type_num < 12:
                table = table2
            table[1 + np.mod(int(exp_type_num / 3), 2)][1 + 2 * np.mod(exp_type_num, 3)] = str(np.round(rom_manifold_error, 5))
            table[3 + np.mod(int(exp_type_num / 3), 2)][1 + 2 * np.mod(exp_type_num, 3)] = str(np.round(orthrom_manifold_error, 5))
            table[1 + np.mod(int(exp_type_num / 3), 2)][2 + 2 * np.mod(exp_type_num, 3)] = str(np.round(rom_results.rom_loss, 5))
            table[3 + np.mod(int(exp_type_num / 3), 2)][2 + 2 * np.mod(exp_type_num, 3)] = str(np.round(orthrom_results.rom_loss, 5))

        # print tables
        for row in table1:
            print(row)
        print()
        for row in table2:
            print(row)


def generate_traj_plots():

    data_basename = "./data/noack"
    results_basename = "./results/noack"

    with torch.no_grad():

        # experiment type list
        exp_types = [results_basename + "/Fine_ProjAE_REC/",
                     results_basename + "/Fine_ProjAE_GAP/",
                     results_basename + "/Fine_ProjAE_RVP/"]
        exp_xticks = [[-3.0, -2.0, -1.0, 0.0],
                      [-0.3, 0.0, 0.3, 0.6],
                      [0.0, 0.5, 1.0]]
        exp_yticks = [[3.0, 2.0, 1.0, 0.0, -1.0, -2.0],
                      [-0.6, -0.3, 0.0, 0.3, 0.6],
                      [-1.5, -1.0, -0.5, 0.0, 0.5]]

        # load test trajectories
        test_traj = romnet.load(data_basename + "/Fine/noack_test.traj")

        # experiment type loop
        for i in range(len(exp_types)):
            exp_type = exp_types[i]

            # load EncROM results
            rom_file_pattern = re.compile(r'noack(\d+)' + '.romresults')
            for filename in os.listdir(exp_type):
                rom_match = rom_file_pattern.match(filename)
                if rom_match:
                    rom_exp_num = int(rom_match.group(1))
            rom_results = romnet.load(exp_type + "noack" + str(rom_exp_num) + ".romresults")

            # generate rom trajectories
            autoencoder = romnet.load_romnet(exp_type + "noack" + str(rom_exp_num) + ".romnet")

            # plot an example trajectory in Rn for best model
            fig = plt.figure()
            ax = fig.add_subplot(1, 2, 1, projection="3d")
            fig.subplots_adjust(wspace=0.5)
            traj_num = 4
            plot_Rn(ax, autoencoder, rom_results.rom_traj, test_traj, traj_num)
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")
            ax.set_zlabel("$x_3$")
            ax.set_xticks([-2, -1, 0, 1, 2])
            ax.set_yticks([-2, -1, 0, 1, 2])
            ax.set_zticks([-2, -1, 0, 1, 2])
            box_len = 1.5
            ax.set_xlim([-box_len, box_len])
            ax.set_ylim([-box_len, box_len])
            ax.set_zlim([-box_len, box_len])
            ax.view_init(elev=21, azim=26, roll=0)
            ax.axis('equal')
            ax.dist = 8

            # plot an example trajectory in Rr for best model
            ax = fig.add_subplot(1, 2, 2)
            plot_Rr(ax, autoencoder, rom_results.rom_traj, test_traj, traj_num)
            ax.set_xlabel("$z_1$")
            ax.set_ylabel("$z_2$")
            ax.set_xticks(exp_xticks[i])
            ax.set_yticks(exp_yticks[i])
            ax.axis('equal')
            ax.grid()
            legend = ax.legend()
            for line in legend.get_lines():
                line.set_linewidth(3.0)

        plt.show()


def generate_error_plots():

    results_basename = "./results/noack"

    with torch.no_grad():
        _, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

        # load (ProjAE, EncROM, RVP LOSS) and (ProjAE, DecROM, RVP LOSS)'s results
        exp_type = results_basename + "/Fine_ProjAE_RVP/"
        rom_file_pattern = re.compile(r'noack(\d+)' + '.romresults')
        orthrom_file_pattern = re.compile(r'noack(\d+)' + '.orthromresults')
        for filename in os.listdir(exp_type):
            rom_match = rom_file_pattern.match(filename)
            orthrom_match = orthrom_file_pattern.match(filename)
            if rom_match:
                rom_exp_num = int(rom_match.group(1))
            if orthrom_match:
                orthrom_exp_num = int(orthrom_match.group(1))
        ProjAE_RVP_rom_results = romnet.load(exp_type + "noack" + str(rom_exp_num) + ".romresults")
        ProjAE_RVP_orthrom_results = romnet.load(exp_type + "noack" + str(orthrom_exp_num) + ".orthromresults")

        # load (StandAE, EncROM, RVP LOSS)'s results
        exp_type = results_basename + "/Fine_StandAE_RVP/"
        rom_file_pattern = re.compile(r'noack(\d+)' + '.romresults')
        for filename in os.listdir(exp_type):
            rom_match = rom_file_pattern.match(filename)
            if rom_match:
                rom_exp_num = int(rom_match.group(1))
        StandAE_RVP_rom_results = romnet.load(exp_type + "noack" + str(rom_exp_num) + ".romresults")

        # load (ProjAE, EncROM, REC LOSS)'s results
        exp_type = results_basename + "/Fine_ProjAE_REC/"
        rom_file_pattern = re.compile(r'noack(\d+)' + '.romresults')
        for filename in os.listdir(exp_type):
            rom_match = rom_file_pattern.match(filename)
            if rom_match:
                rom_exp_num = int(rom_match.group(1))
        ProjAE_REC_rom_results = romnet.load(exp_type + "noack" + str(rom_exp_num) + ".romresults")

        # plot left panel
        dt = 0.1
        t_final = 20.
        n = int(t_final / dt) + 1
        t = np.arange(0, n) * dt
        def black_color_func(traj_idx: int) -> Tuple[float, float, float]:
            return (0, 0, 0)
        def green_color_func(traj_idx: int) -> Tuple[float, float, float]:
            return (34/255, 139/255, 34/255)
        plot_rom_error(ax1, ProjAE_RVP_rom_results.rom_error, t, 50, black_color_func, label="ProjAE, EncROM, RVP")
        ax1.set_xlabel("Time, $t$")
        ax1.set_ylabel("$\|\hat{x}(t) - x(t)\|_2^2$")
        plot_rom_error(ax1, StandAE_RVP_rom_results.rom_error, t, 50, green_color_func, label="StandAE, EncROM, RVP")
        legend = ax1.legend()
        for line in legend.get_lines():
            line.set_linewidth(3.0)

        # plot right panel
        def orange_color_func(traj_idx: int) -> Tuple[float, float, float]:
            return (204/255, 102/255, 0/255)
        def rose_color_func(traj_idx: int) -> Tuple[float, float, float]:
            return (227/255, 38/255, 54/255)
        plot_rom_error(ax2, ProjAE_RVP_orthrom_results.rom_error, t, 50, orange_color_func, label="ProjAE, DecROM, RVP")
        ax2.set_xlabel("Time, $t$")
        plot_rom_error(ax2, ProjAE_REC_rom_results.rom_error, t, 50, rose_color_func, label="ProjAE, EncROM, Rec.")
        legend = ax2.legend()
        for line in legend.get_lines():
            line.set_linewidth(3.0)
        
        plt.show()


if __name__ == "__main__":
    # generate_rom_results()
    # generate_table()
    # generate_traj_plots()
    generate_error_plots()
