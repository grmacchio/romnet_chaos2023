#!/usr/bin/env python

import time

import numpy as np

import romnet
from romnet.models import NoackModel
from romnet.typing import Vector
from romnet.sample import TrajectoryDataset, TrajectoryList


def adj_output(x: Vector, eta: Vector) -> Vector:
    return eta


def generate_traj(initial_time: float = 0.0):

    # start time
    tic = time.time()

    # getting discrete model stepper
    model = NoackModel()
    dt = 0.1
    step = model.get_stepper(dt, method="rk4")

    # defining initial conditions
    resolution = 10
    box_len = 2
    if resolution == 10:
        side_grid = np.linspace(-box_len / 2, box_len / 2, resolution)
    if resolution == 4:
        side_grid = np.linspace(-box_len / 2, box_len / 2, 10)
        sub_grid = side_grid[np.where((side_grid < 0.15) & (side_grid > -0.15))]
        side_grid = np.linspace(-box_len / 2, box_len / 2, resolution)
        side_grid = np.hstack((side_grid, sub_grid))
    x_grid, y_grid, z_grid = np.meshgrid(side_grid, side_grid, side_grid)
    train_ics = np.vstack((x_grid.flatten(), y_grid.flatten(), z_grid.flatten())).T
    train_ics_iter = iter(train_ics)
    train_ic = lambda: next(train_ics_iter)
    val_ic = lambda: model.random_ic(max_amplitude=box_len / 2)
    select_ic = lambda: model.random_ic(max_amplitude=box_len / 2)
    test_ic = lambda: model.random_ic(max_amplitude=box_len / 2)
    num_train = len(train_ics)
    num_val = num_train
    num_select = num_train
    num_test = num_train

    # generating trajectories
    print("----------------------------------------- Generating TrajectoryList")
    t_final = 20.
    n = int(t_final / dt) + 1
    select_traj = romnet.sample(step, select_ic, num_select, n)
    select_traj.save("noack_select.traj")
    training_traj = romnet.sample(step, train_ic, num_train, n)
    val_traj = romnet.sample(step, val_ic, num_val, n)
    test_traj = romnet.sample(step, test_ic, num_test, n)
    training_traj.save("noack_train.traj")
    val_traj.save("noack_val.traj")
    test_traj.save("noack_test.traj")

    # number of training state samples
    print("Number of Training Samples: {}".format(len(training_traj)))

    # end time
    toc = time.time()
    print("----------------------------------------- Done")
    print("Run Time: {}".format(toc - tic))

    # final time
    return (toc - tic) + initial_time


def generate_gds(initial_time: float = 0.0):

    # start time
    tic = time.time()

    # time
    dt = 0.1

    # model
    model = NoackModel()

    # loading trajectories
    training_traj = romnet.load("noack_train.traj")
    val_traj = romnet.load("noack_val.traj")

    # GradientDataset (gds)
    s = 10
    L = 20
    print("----------------------------------------- Generating GradientDataset")
    adj_step = model.get_adjoint_stepper(dt, method="rk4")
    training_data, _ = romnet.sample_gradient_long_traj(training_traj, adj_step, adj_output, model.num_states, s, L)
    val_data, _ = romnet.sample_gradient_long_traj(val_traj, adj_step, adj_output, model.num_states, s, L)
    if training_data.X.shape[0] > len(training_traj):
        training_data.data_length = len(training_traj)
    training_data.save("noack_train.gds")
    val_data.save("noack_val.gds")

    # number of training state samples
    print("Number of Available Training Samples: {}".format(training_data.X.shape[0]))
    print("Number of Training Samples: {}".format(training_data.data_length))

    # end time
    toc = time.time()
    print("----------------------------------------- Done")
    print("Run Time: {}".format(toc - tic))

    # final time
    return (toc - tic) + initial_time


def generate_tds(initial_time: float = 0.0):

    # start time
    tic = time.time()
    
    # model
    model = NoackModel()

    # loading trajectories
    training_traj = romnet.load("noack_train.traj")
    val_traj = romnet.load("noack_val.traj")

    # TrajectoryDataset (tds)
    print("----------------------------------------- Generating TrajectoryDataset")
    training_data_tds = TrajectoryDataset(training_traj.traj, training_traj.traj, model.rhs(training_traj.traj))
    val_data_tds = TrajectoryDataset(val_traj.traj, val_traj.traj, model.rhs(val_traj.traj))
    training_data_tds.save("noack_train.tds")
    val_data_tds.save("noack_val.tds")

    # number of training state samples
    print("Number of Training Samples: {}".format(len(training_data_tds)))

    # end time
    toc = time.time()
    print("----------------------------------------- Done")
    print("Run Time: {}".format(toc - tic))

    # final time
    return (toc - tic) + initial_time


if __name__ == "__main__":
    t1 = generate_traj()
    t2 = generate_gds(t1)
    t3 = generate_tds(t2)
    print("----------------------------------------- END")
    print("Total Run Time: {}".format(t3))
