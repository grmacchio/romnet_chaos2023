
from typing import Tuple
import time

import numpy as np
from scipy.stats import ortho_group
from scipy.linalg import solve_continuous_lyapunov as lyap
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch import nn
import control  # https://python-control.readthedocs.io/en/0.9.4/generated/control.gram.html
from control.matlab import bode
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import romnet
from romnet.models import HolmesLinear
from romnet.sample import TrajectoryDataset, TrajectoryList
from romnet.autoencoder import AE
from noack_train import rom
from romnet.typing import Vector


def generate_data():

    # getting discrete model stepper
    model = HolmesLinear()
    dt = 0.01
    step = model.get_stepper(dt, method="rk4")

    # defining initial condition
    def ic():
        return np.ones(3)
    
    # generating TrajectoryDataset
    t_final = 6.
    n = int(t_final / dt) + 1
    x_traj = (romnet.sample(step, ic, 1, n)).traj
    y_traj = model.output(x_traj)
    fx_traj = model.rhs(x_traj)
    tds = TrajectoryDataset(x_traj, y_traj, fx_traj)

    # saving TrajectoryDataset
    tds.save("./data/linear/linear.tds")


def train_POD():
    
    # loading TrajectoryDataset
    tds = romnet.load("./data/linear/linear.tds")

    # finding POD modes
    Phi, _, _ = np.linalg.svd(tds.x_traj[0].T, full_matrices=False, compute_uv=True)

    # saving POD modes
    np.save("./results/linear/linear_POD.npy", Phi)


def train_BT():

    # load model
    model = HolmesLinear()

    # finding BT modes
    Wc = lyap(model.A, - model.B @ model.B.T)
    Wo = lyap(model.A.T, - model.C.T @ model.C)
    L = np.linalg.cholesky(Wc)
    L_inv = np.linalg.inv(L)
    S_sqr, K = np.linalg.eig(np.conj(L.T) @ (Wo @ L))
    S_sqrroot = np.diag(np.sqrt(np.sqrt(S_sqr)))
    S_sqrroot_inv = np.diag(1. / np.sqrt(np.sqrt(S_sqr)))
    Phi = L @ (K @ S_sqrroot_inv)
    PsiT = S_sqrroot @ (np.conj(K.T) @ L_inv)
    PhiPsi = np.array([Phi, PsiT.T])

    # saving BT modes
    np.save("./results/linear/linear_BT.npy", PhiPsi)


class LinearAE(AE):

    def __init__(self, dim_in: int, dim_out: int, D: Tensor = None, X: Tensor = None) -> None:
        super().__init__()
        self.dim_in_ = dim_in
        self.dim_out_ = dim_out
        Q = ortho_group.rvs(dim_in)[:, :dim_out]
        if D is None:
            self.D = nn.Parameter(torch.tensor(Q))
        else:
            self.D = nn.Parameter(D)
        if X is None:
            self.X = nn.Parameter(torch.tensor(Q.transpose()))
        else:
            self.X = nn.Parameter(X)
        self.update()

    def dim_in(self) -> int:
        return self.dim_in_
    
    def dim_out(self) -> int:
        return self.dim_out_
    
    def update(self) -> None:
        self.E = (self.X @ self.D).inverse() @ self.X

    def enc(self, x: Vector) -> Tensor:
        x = torch.as_tensor(x)
        return x @ self.E.T
    
    def dec(self, x: Vector) -> Tensor:
        x = torch.as_tensor(x)
        return x @ self.D.T
    
    def d_enc(self, x: Vector, v: Vector) -> Tuple[Tensor, Tensor]:
        x = torch.as_tensor(x)
        v = torch.as_tensor(v)
        return self.enc(x), self.enc(v)
    
    def d_dec(self, x: Vector, v: Vector) -> Tuple[Tensor, Tensor]:
        x = torch.as_tensor(x)
        v = torch.as_tensor(v)
        return self.dec(x), self.dec(v)


def train_RVP():

    basename = "linear"

    # loading TrajectoryDataset
    training_data = romnet.load("./data/linear/linear.tds")
    val_data = romnet.load("./data/linear/linear.tds")
    val_traj = TrajectoryList(val_data.x_traj)

    # temporal parameters
    dt = 0.1
    t_final = 6.

    # defining the model
    model = HolmesLinear()

    # hyperparameters
    learning_rate = 1.0e-3
    batch_size = 2
    num_epochs = 10000
    t_batch = t_final
    gamma_vpl = np.linalg.norm(model.C, ord=2)
    L = 1 / t_batch

    # defining autoencoder and experiment
    autoencoder = LinearAE(dim_in=3, dim_out=2)
    experiment = romnet.Experiment()

    # defining loss function
    n_t_batch = int(t_batch / dt) + 1
    training_data.set_time_batch(n_t_batch)
    def loss_fn(X_pred, X, Y, F_x):
        t = torch.arange(0, n_t_batch) * dt
        weights = romnet.weight_func(t, t_batch, L)
        Y_pred = model.output_tensor(X_pred)
        _, dP = autoencoder.d_autoenc(X, F_x)
        _, dP_pred = autoencoder.d_autoenc(X_pred, model.rhs_tensor(X_pred))
        state_proj_loss = romnet.integral_loss(Y_pred, Y, t)
        vel_proj_loss = romnet.integral_loss(dP_pred, dP, t, weights)
        loss = state_proj_loss + gamma_vpl * vel_proj_loss
        return loss
    
    # validation function
    val_data.set_time_batch(n_t_batch)
    def val_func() -> None:
        romnet.save_romnet(autoencoder, basename + ".romnet")
        experiment.save(basename + ".exp")

    # train autoencoder
    tic_train = time.time()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    for epoch in range(num_epochs):
        print(f"----------------------------------------- Epoch {epoch+1}")
        train_loss = romnet.train_loop(train_dataloader, autoencoder, loss_fn, optimizer)
        val_loss = romnet.val_loop(val_dataloader, autoencoder, loss_fn)
        experiment.add_loss(train_loss, val_loss, val_func, *rom(model, autoencoder, val_traj))
    toc_train = time.time()
    experiment.add_train_time(toc_train - tic_train)

    # save final autoencoder
    print("----------------------------------------- Saving final autoencoder")
    romnet.save_romnet(autoencoder, basename + "_final" + ".romnet")

    # save orthogonal rom
    print("----------------------------------------- Saving orthogonal ROM")
    autoencoder = romnet.load_romnet(basename + ".romnet")
    experiment.add_trained_orthrom_traj(*rom(model, autoencoder, val_traj, dec_rom=True))

    # save experiment
    print("----------------------------------------- Saving experiment")
    print(experiment)
    experiment.save(basename + "" + ".exp")


def generate_plots():
    
    # loading TrajectoryDataset
    tds = romnet.load("./data/linear/linear.tds")
    x_traj = TrajectoryList(tds.x_traj).traj

    # temporal parameters
    dt = 0.1
    t_final = 6.
    t = torch.arange(0, int(t_final / dt) + 1) * dt

    # latent dimension
    r = 2

    # define model
    model = HolmesLinear()

    # Full Model
    s = control.StateSpace(model.A, model.B, model.C, model.D)

    with torch.no_grad():

        # POD rom
        Phi_POD = torch.tensor(np.load("./results/linear/linear_POD.npy"))[:, :r]
        autoencoder_POD = LinearAE(dim_in=3, dim_out=2, D=Phi_POD, X=Phi_POD.T)
        P_POD = np.array(autoencoder_POD.D @ autoencoder_POD.E)
        A_POD = P_POD @ model.A
        B_POD = P_POD @ model.B
        C_POD = model.C
        D_POD = model.D
        s_POD = control.StateSpace(A_POD, B_POD, C_POD, D_POD)

        # BT rom
        Phi_BT, Psi_BT = tuple(torch.tensor(np.load("./results/linear/linear_BT.npy"))[:, :, :r])
        autoencoder_BT = LinearAE(dim_in=3, dim_out=2, D=Phi_BT, X=Psi_BT.T)
        P_BT = np.array(autoencoder_BT.D @ autoencoder_BT.E)
        A_BT = P_BT @ model.A
        B_BT = P_BT @ model.B
        C_BT = model.C
        D_BT = model.D
        s_BT = control.StateSpace(A_BT, B_BT, C_BT, D_BT)

        # RVP rom
        autoencoder_RVP = romnet.load_romnet("./results/linear/linear.romnet")
        P_RVP = np.array(autoencoder_RVP.D @ autoencoder_RVP.E)
        A_RVP = P_RVP @ model.A
        B_RVP = P_RVP @ model.B
        C_RVP = model.C
        D_RVP = model.D
        s_RVP = control.StateSpace(A_RVP, B_RVP, C_RVP, D_RVP)

        # plot impulse response
        t, y = control.impulse_response(s, t)
        t_POD, y_POD = control.impulse_response(s_POD, t)
        t_BT, y_BT = control.impulse_response(s_BT, t)
        t_RVP, y_RVP = control.impulse_response(s_RVP, t)
        fig = plt.figure()
        gs = gridspec.GridSpec(1, 2)
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(t, y, label="Full Model", linewidth=6, color="b")
        ax1.plot(t_POD, y_POD, label="Reconstruction Loss", linewidth=4, color="k")
        ax1.plot(t_RVP, y_RVP, label="RVP Loss", linestyle="dashed", linewidth=4, color="r")
        ax1.plot(t_BT, y_BT, label="Balanced Truncation", linestyle="dotted", linewidth=4, color="orange")
        ax1.set_xlabel(r"$t$", fontsize=16, labelpad=9)
        ax1.set_ylabel(r"$y(t)$", fontsize=16, labelpad=7)
        ax1.tick_params(axis='both', labelsize=13)
        ax1.legend(fontsize=12)

        # plot bode
        mag, phase, omega = bode(s, omega_limits=(1e-2, 1e4), plot=False)
        mag_POD, phase_POD, omega_POD = bode(s_POD, omega_limits=(1e-2, 1e4), plot=False)
        mag_RVP, phase_RVP, omega_RVP = bode(s_RVP, omega_limits=(1e-2, 1e4), plot=False)
        mag_BT, phase_BT, omega_BT = bode(s_BT, omega_limits=(1e-2, 1e4), plot=False)
        gs_right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1])
        def mag2db(mag):
            return 20 * np.log10(mag)
        _ax21 = fig.add_subplot(gs_right[0])
        ax21 = _ax21.twinx()
        _ax21.yaxis.set_visible(False)
        _ax21.xaxis.set_visible(False)
        ax21.semilogx(omega, mag2db(mag), label="Full Model", linewidth=6, color="b")
        ax21.semilogx(omega_POD, mag2db(mag_POD), label="Reconstruction Loss", linewidth=4, color="k")
        ax21.semilogx(omega_RVP, mag2db(mag_RVP), label="RVP Loss", linestyle="dashed", linewidth=4, color="r")
        ax21.semilogx(omega_BT, mag2db(mag_BT), label="Balanced Truncation", linestyle="dotted", linewidth=4, color="orange")
        ax21.set_ylim([-60, 60])
        ax21.set_ylabel(r"$|G(i\omega)|$, [dB]", fontsize=15, labelpad=17.5)
        ax21.tick_params(axis='both', labelsize=13)
        _ax22 = fig.add_subplot(gs_right[1])
        ax22 = _ax22.twinx()
        _ax22.yaxis.set_visible(False)
        def rad2deg(rad):
            return 180 * rad / np.pi
        phase_POD = phase_POD - 2 * np.pi * np.floor(phase_POD / 2 / np.pi) - 2 * np.pi
        idx = 342
        k = 1
        phase_POD = np.hstack((phase_POD[:idx - k], phase_POD[idx + 1 + k:]))  # smoothing POD phase plot
        omega_POD = np.hstack((omega_POD[:idx - k], omega_POD[idx + 1 + k:]))
        ax22.semilogx(omega, rad2deg(phase), label="Full Model", linewidth=6, color="b")
        ax22.semilogx(omega_POD, rad2deg(phase_POD), label="Reconstruction Loss", linewidth=4, color="k")
        ax22.semilogx(omega_RVP, rad2deg(phase_RVP), label="RVP Loss", linestyle="dashed", linewidth=4, color="r")
        ax22.semilogx(omega_BT, rad2deg(phase_BT), label="Balanced Truncation", linestyle="dotted", linewidth=4, color="orange")
        ax22.set_ylim([-360 - 20, 0 + 20])
        ax22.set_yticks([-360, -270, -180, -90, 0])
        ax22.set_ylabel(r"$\angle G(i\omega)$, [deg]", fontsize=16, labelpad=9)
        _ax22.set_xlabel(r"$\omega$, [rad/s]", fontsize=16, labelpad=7)
        _ax22.tick_params(axis='both', labelsize=13)
        ax22.tick_params(axis='both', labelsize=13)

        plt.show()


if __name__ == "__main__":
    # generate_data()
    # train_POD()
    # train_BT()
    # train_RVP()
    generate_plots()
