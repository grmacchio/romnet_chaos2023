#!/usr/bin/env python

import sys
from typing import Callable
import time

import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torch.distributed as dist
import numpy as np

import romnet
from romnet import AE
from romnet.models import NoackModel
from romnet.model import Model
from romnet.timestepper import Timestepper
from romnet.model import DiscreteModel
from romnet.sample import TrajectoryList, TrajectoryDataset, GradientDataset


def sample_faster_tensor(
    step: Callable[[Tensor], Tensor], random_state: Callable[[], Tensor], num_traj: int, n: int
) -> TrajectoryList:
    """
    Faster sample() method with a 95%-85% reduction in computation time

    Sample num_traj trajectories each with length n

    random_state() generates a random initial state x
    step(x) advances the state forward in time

    Returns a TrajectoryList object
    """
    ic0 = random_state()
    traj_traj = torch.zeros((num_traj, n, ic0.shape[0]))
    traj_traj[0, 0, :] = ic0
    for i in range(num_traj - 1):
        traj_traj[i + 1, 0, :] = random_state()
    for i in range(n - 1):
        traj_traj[:, i + 1, :] = step(traj_traj[:, i, :])
    return TrajectoryList(traj_traj)


def rom(model: Model, autoencoder: AE, traj: TrajectoryList, dec_rom: bool = False):

    with torch.no_grad():
        
        try:

            # rom
            def enc_rom_rhs(z: Tensor) -> Tensor:
                x = autoencoder.dec(z)
                _, v = autoencoder.d_enc(x, model.rhs_tensor(x))
                return v
            def dec_rom_rhs(z: Tensor) -> Tensor:
                x = autoencoder.dec(z)
                I = torch.eye(2).unsqueeze(1)
                d_decT = autoencoder.d_dec(z, I)[1].transpose(0,1)
                d_dec = d_decT.transpose(1, 2)
                rhs = model.rhs_tensor(x).unsqueeze(-1)
                z_dot = torch.linalg.solve((d_decT @ d_dec), d_decT @ rhs)
                return z_dot.squeeze(-1)
            if not dec_rom:
                rom_rhs = enc_rom_rhs
            else:
                rom_rhs = dec_rom_rhs

            # rom discrete time model
            dt = 0.1
            rk4 = Timestepper.lookup("rk4")
            stepper = rk4(dt)
            step = DiscreteModel(rom_rhs, stepper)

            # rom_traj
            ics = autoencoder.enc(traj.traj[:, 0, :])
            ics_itr = iter(ics)
            def ic():
                return next(ics_itr)
            rom_traj = sample_faster_tensor(step, ic, traj.num_traj, traj.n)

            # rom_error
            x_rom_traj = autoencoder.dec(rom_traj.traj).numpy()
            rom_error = np.square(np.linalg.norm(traj.traj - x_rom_traj, axis=2))

            # rom_loss
            rom_loss = np.mean(rom_error)
            print(f"ROM Epoch Loss: {rom_loss:>7f}")

        except:

            rom_loss = np.inf
            rom_error = np.full((traj.num_traj, traj.n), np.inf)
            rom_traj = TrajectoryList(np.full((traj.num_traj, traj.n, 2), np.inf))
            print(f"ROM Epoch Loss: {rom_loss:>7f}")

    return rom_loss, rom_error, rom_traj


def train_autoencoder(computer: str = "local", job_num: str = "0"):

    debug_basename = ""
    # debug_basename = "/chaos2023"
    data_basename = "." + debug_basename + "/data/noack/Fine/noack"
    results_basename = "." + debug_basename + "/results/noack/Fine_StandAE_RVP/noack"

    ##################################
    #           Recon Loss           #
    ##################################

    '''
    # local or cluster
    if computer == "local":
        dist.init_process_group("gloo")
        job_num = dist.get_rank()
        # job_num = 0
    elif computer == "cluster":
        job_num = int(job_num)

    # model loop
    for model_num in range(8):

        model_id = str(job_num + 8 * model_num)
        
        # start time
        tic = time.time()

        # time
        dt = 0.1
        t_final = 20.

        # transient 1
        t_trans1 = 0.0 # (0.2s)
        idx_trans1 = int(t_trans1 / dt)

        # transient 2
        t_trans2 = t_final # (2.5s)
        idx_trans2 = int(t_trans2 / dt)

        # model
        model = NoackModel()

        # load data
        print("----------------------------------------- Loading data")
        training_traj = romnet.load(data_basename + "_train.traj")
        val_traj = romnet.load(data_basename + "_val.traj")

        # cut transients
        training_traj = TrajectoryList(training_traj.traj[:, idx_trans1:idx_trans2 + 1, :])
        val_traj = TrajectoryList(val_traj.traj[:, idx_trans1:idx_trans2 + 1, :])

        # limit number of val trajectories
        num_val_traj = len(val_traj.traj)
        val_traj = TrajectoryList(val_traj.traj[0:num_val_traj, :, :])

        # hyperparameters
        learning_rate = 1.0e-3
        batch_size = 400
        num_epochs = 300 * 3
        dims = [3, 3, 3, 3, 3, 2]
        gamma_reg = 1.0e-5

        # initialize autoencoder, save initial autoencoder, and initialize experiment
        # autoencoder = romnet.StandAE(dims)
        # autoencoder = romnet.ProjAE(dims)
        # romnet.save_romnet(autoencoder, basename + model_id + "_initial" + ".romnet")
        # experiment = romnet.Experiment()

        # load initial autoencoder and initialize experiment
        autoencoder = romnet.load_romnet(results_basename + model_id + "_initial.romnet")
        experiment = romnet.Experiment()

        # load autoencoder and experiment
        # autoencoder = romnet.load_romnet(results_basename + model_id + ".romnet")
        # experiment = romnet.load_exp(results_basename + model_id + ".exp", reset_best=True)

        # loss function
        def loss_fn(X_pred, X):
            loss = romnet.recon_loss(X_pred, X)
            # reg = gamma_reg * autoencoder.regularizer()
            return loss  # + reg
    
        # validation function
        def val_func() -> None:
            romnet.save_romnet(autoencoder, results_basename + model_id + ".romnet")
            experiment.save(results_basename + model_id + "" + ".exp")

        # train autoencoder
        print("----------------------------------------- Training model {}".format(model_id))
        tic_train = time.time()
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50)
        train_dataloader = DataLoader(training_traj, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_traj, batch_size=batch_size, shuffle=True)
        for epoch in range(num_epochs):
            print(f"----------------------------------------- Epoch {epoch+1}")
            train_loss = romnet.train_loop(train_dataloader, autoencoder, loss_fn, optimizer)
            val_loss = romnet.val_loop(val_dataloader, autoencoder, loss_fn)
            experiment.add_loss(train_loss, val_loss, val_func, *rom(model, autoencoder, val_traj))
            scheduler.step(val_loss)
            for param_group in optimizer.param_groups:
                print("Next Learning Rate: {}".format(param_group['lr']))
        toc_train = time.time()
        experiment.add_train_time(toc_train - tic_train)

        # save final autoencoder
        print("----------------------------------------- Saving final autoencoder")
        romnet.save_romnet(autoencoder, results_basename + model_id + "_final" + ".romnet")

        # save orthogonal rom
        print("----------------------------------------- Saving orthogonal ROM")
        autoencoder = romnet.load_romnet(results_basename + model_id + ".romnet")
        experiment.add_trained_orthrom_traj(*rom(model, autoencoder, val_traj, dec_rom=True))

        # save experiment
        print("----------------------------------------- Saving experiment")
        print(experiment)
        experiment.save(results_basename + model_id + "" + ".exp")

        # end time
        toc = time.time()
        print("----------------------------------------- Done")
        print("Run Time: {}".format(toc - tic))
    '''

    ##################################
    #            GAP Loss            #
    ##################################

    '''
    # local or cluster
    if computer == "local":
        dist.init_process_group("gloo")
        job_num = dist.get_rank()
        # job_num = 0
    elif computer == "cluster":
        job_num = int(job_num)

    # model loop
    for model_num in range(8):

        model_id = str(job_num + 8 * model_num)

        # start time
        tic = time.time()

        # time
        dt = 0.1
        t_final = 20.

        # transient 1
        t_trans1 = 0.0 # (0.2s)
        idx_trans1 = int(t_trans1 / dt)

        # transient 2
        t_trans2 = t_final # (2.5s)
        idx_trans2 = int(t_trans2 / dt)

        # model
        model = NoackModel()

        # load data
        print("----------------------------------------- Loading data")
        training_data = romnet.load(data_basename + "_train.gds")
        val_data = romnet.load(data_basename + "_val.gds")
        val_traj = romnet.load(data_basename + "_val.traj")

        # cut transients
        train_trans_idxs = np.where(np.logical_and(idx_trans1 <= training_data.T[:len(training_data)], training_data.T[:len(training_data)] <= idx_trans2))[0]
        val_trans_idxs = np.where(np.logical_and(idx_trans1 <= val_data.T[:len(val_data)], val_data.T[:len(val_data)] <= idx_trans2))[0]
        training_data = GradientDataset(training_data.X[train_trans_idxs], training_data.G[train_trans_idxs])
        val_data = GradientDataset(val_data.X[val_trans_idxs], val_data.G[val_trans_idxs])

        # limit number of val trajectories
        num_val_traj = len(val_traj.traj)
        val_traj = TrajectoryList(val_traj.traj[0:num_val_traj, :, :])

        # hyperparameters
        learning_rate = 1.0e-3
        batch_size = 400
        num_epochs = 300 * 3
        dims = [3, 3, 3, 3, 3, 2]
        gamma_reg = 1.0e-5

        # initialize autoencoder, save initial autoencoder, and initialize experiment
        # autoencoder = romnet.StandAE(dims)
        # autoencoder = romnet.ProjAE(dims)
        # romnet.save_romnet(autoencoder, results_basename + model_id + "_initial" + ".romnet")
        # experiment = romnet.Experiment()

        # load initial autoencoder and initialize experiment
        autoencoder = romnet.load_romnet(results_basename + model_id + "_initial.romnet")
        experiment = romnet.Experiment()

        # load autoencoder and experiment
        # autoencoder = romnet.load_romnet(results_basename + model_id + ".romnet")
        # experiment = romnet.load_exp(results_basename + model_id + ".exp", reset_best=True)

        # loss function
        def loss_fn(X_pred, X, G):
            loss = romnet.GAP_loss(X_pred, X, G)
            # reg = gamma_reg * autoencoder.regularizer()
            return loss  # + reg
        
        # validation function
        def val_func() -> None:
            romnet.save_romnet(autoencoder, results_basename + model_id + ".romnet")
            experiment.save(results_basename + model_id + "" + ".exp")

        # train autoencoder
        print("----------------------------------------- Training model {}".format(model_id))
        tic_train = time.time()
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50)
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
        for epoch in range(num_epochs):
            print(f"----------------------------------------- Epoch {epoch+1}")
            train_loss = romnet.train_loop(train_dataloader, autoencoder, loss_fn, optimizer)
            val_loss = romnet.val_loop(val_dataloader, autoencoder, loss_fn)
            experiment.add_loss(train_loss, val_loss, val_func, *rom(model, autoencoder, val_traj))
            scheduler.step(val_loss)
            for param_group in optimizer.param_groups:
                print("Next Learning Rate: {}".format(param_group['lr']))
        toc_train = time.time()
        experiment.add_train_time(toc_train - tic_train)

        # save final autoencoder
        print("----------------------------------------- Saving final autoencoder")
        romnet.save_romnet(autoencoder, results_basename + model_id + "_final" + ".romnet")

        # save orthogonal rom
        print("----------------------------------------- Saving orthogonal ROM")
        autoencoder = romnet.load_romnet(results_basename + model_id + ".romnet")
        experiment.add_trained_orthrom_traj(*rom(model, autoencoder, val_traj, dec_rom=True))

        # save experiment
        print("----------------------------------------- Saving experiment")
        print(experiment)
        experiment.save(results_basename + model_id + "" + ".exp")

        # end time
        toc = time.time()
        print("----------------------------------------- Done")
        print("Run Time: {}".format(toc - tic))
    '''
        
    #################################
    #    Velocity Projection Loss   #
    #################################

    # local or cluster
    if computer == "local":
        dist.init_process_group("gloo")
        job_num = dist.get_rank()
        # job_num = 0
    elif computer == "cluster":
        job_num = int(job_num)

    # model loop
    for model_num in range(8):

        model_id = str(job_num + 8 * model_num)

        # start time
        tic = time.time()

        # time
        dt = 0.1
        t_final = 20.

        # transient 1
        t_trans1 = 0.0 # (0.2s)
        idx_trans1 = int(t_trans1 / dt)

        # transient 2
        t_trans2 = t_final # (2.5s)
        idx_trans2 = int(t_trans2 / dt)

        # model
        model = NoackModel()

        # load data
        print("----------------------------------------- Loading data")
        training_data = romnet.load(data_basename + "_train.tds")
        val_data = romnet.load(data_basename + "_val.tds")
        val_traj = romnet.load(data_basename + "_val.traj")

        # cut transients
        training_data = TrajectoryDataset(training_data.x_traj[:, idx_trans1:idx_trans2 + 1, :], training_data.y_traj[:, idx_trans1:idx_trans2 + 1, :], training_data.fx_traj[:, idx_trans1:idx_trans2 + 1, :])
        val_data = TrajectoryDataset(val_data.x_traj[:, idx_trans1:idx_trans2 + 1, :], val_data.y_traj[:, idx_trans1:idx_trans2 + 1, :], val_data.fx_traj[:, idx_trans1:idx_trans2 + 1, :])

        # limit number of val trajectories
        num_val_traj = len(val_traj.traj)
        val_traj = TrajectoryList(val_traj.traj[0:num_val_traj, :, :])

        # hyperparameters
        learning_rate = 1.0e-3
        batch_size = 2
        num_epochs = 300 * 3
        dims = [3, 3, 3, 3, 3, 2]
        gamma_reg = 1.0e-5
        t_batch = t_final
        gamma_vpl = np.linalg.norm(np.eye(3), ord=2)
        L = 1 / t_batch

        # initialize autoencoder, save initial autoencoder, and initialize experiment
        # autoencoder = romnet.StandAE(dims)
        # autoencoder = romnet.ProjAE(dims)
        # romnet.save_romnet(autoencoder, results_basename + model_id + "_initial" + ".romnet")
        # experiment = romnet.Experiment()

        # load initial autoencoder and initialize experiment
        autoencoder = romnet.load_romnet(results_basename + model_id + "_initial.romnet")
        experiment = romnet.Experiment()

        # load autoencoder and experiment
        # autoencoder = romnet.load_romnet(results_basename + model_id + ".romnet")
        # experiment = romnet.load_exp(results_basename + model_id + ".exp", reset_best=True)

        # loss function
        n_t_batch = int(t_batch / dt) + 1
        training_data.set_time_batch(n_t_batch)
        t = torch.arange(0, n_t_batch) * dt
        def loss_fn(X_pred, X, Y, F_x):
            weights = romnet.weight_func(t, t_batch, L)
            # weights = romnet.weight_func_L0(t, t_batch)
            Y_pred = X_pred
            _, dP = autoencoder.d_autoenc(X, F_x)
            _, dP_pred = autoencoder.d_autoenc(X_pred, model.rhs_tensor(X_pred))
            state_proj_loss = romnet.integral_loss(Y_pred, Y, t)
            vel_proj_loss = romnet.integral_loss(dP_pred, dP, t, weights)
            loss = state_proj_loss + gamma_vpl * vel_proj_loss
            # reg = gamma_reg * autoencoder.regularizer()
            return loss  # + reg
        
        # validation function
        val_data.set_time_batch(n_t_batch)
        def val_func() -> None:
            romnet.save_romnet(autoencoder, results_basename + model_id + ".romnet")
            experiment.save(results_basename + model_id + "" + ".exp")

        # train autoencoder
        print("----------------------------------------- Training model {}".format(model_id))
        tic_train = time.time()
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50)
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
        for epoch in range(num_epochs):
            print(f"----------------------------------------- Epoch {epoch+1}")
            train_loss = romnet.train_loop(train_dataloader, autoencoder, loss_fn, optimizer)
            val_loss = romnet.val_loop(val_dataloader, autoencoder, loss_fn)
            experiment.add_loss(train_loss, val_loss, val_func, *rom(model, autoencoder, val_traj))
            scheduler.step(val_loss)
            for param_group in optimizer.param_groups:
                print("Next Learning Rate: {}".format(param_group['lr']))
        toc_train = time.time()
        experiment.add_train_time(toc_train - tic_train)

        # save final autoencoder
        print("----------------------------------------- Saving final autoencoder")
        romnet.save_romnet(autoencoder, results_basename + model_id + "_final" + ".romnet")

        # save orthogonal rom
        print("----------------------------------------- Saving orthogonal ROM")
        autoencoder = romnet.load_romnet(results_basename + model_id + ".romnet")
        experiment.add_trained_orthrom_traj(*rom(model, autoencoder, val_traj, dec_rom=True))

        # save experiment
        print("----------------------------------------- Saving experiment")
        print(experiment)
        experiment.save(results_basename + model_id + "" + ".exp")

        # end time
        toc = time.time()
        print("----------------------------------------- Done")
        print("Run Time: {}".format(toc - tic))


if __name__ == "__main__":
    if len(sys.argv) == 1:
        train_autoencoder()  # term: torchrun --nproc_per_node=8 noack_train.py
    elif len(sys.argv) > 1:
            train_autoencoder(computer="cluster", job_num=sys.argv[2])  
