import pickle
from typing import Any, Callable, List, Tuple, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray
from torch.utils.data import Dataset
from matplotlib.pyplot import Axes

from .typing import Vector, VectorField

__all__ = ["sample", "sample_gradient", "load",
           "sample_gradient_long_traj", "sample_fast",
           "Experiment", "ExperimentList", "load_exp",
           "sample_faster_tensor", "rom"]


class TrajectoryList(Dataset[Vector]):
    """
    Container for samples of trajectories

    Suppose traj is a numpy array of dimension (num_traj, n, *)
    That is, there are num_traj trajectories, each of length n

    dataset = TrajectoryList(traj)
    dataset.traj[i] is trajectory i (an array with n samples)
    dataset[j] is sample j (from all trajectories, concatenated together)

    This class is compatible with torch.DataLoader:

    training = DataLoader(dataset, batch_size=64, shuffle=True)
    """

    def __init__(self, traj: ArrayLike, time_batch: Optional[int] = None):
        self.traj = np.array(traj)
        self.num_traj = self.traj.shape[0]
        self.n = self.traj.shape[1]
        if time_batch is None:
            time_batch = 1
        self.set_time_batch(time_batch)
        newshape = list(self.traj.shape)
        newshape[1] *= newshape[0]
        newshape.pop(0)
        self.state_data = self.traj.view()
        self.state_data.shape = tuple(newshape)

    def set_time_batch(self, time_batch: int) -> None:
        self.time_batch = time_batch
        self.time_div = (self.n // self.time_batch)
        self.data_length = self.num_traj * self.time_div

    def __len__(self) -> int:
        return self.data_length

    def __getitem__(self, i: int) -> Tuple[Vector]:
        traj_idx = i // self.time_div
        time_idx1 = (i - traj_idx * self.time_div) * self.time_batch
        time_idx2 = (i + 1 - traj_idx * self.time_div) * self.time_batch
        return (self.traj[traj_idx, time_idx1:time_idx2, :], )

    def save(self, fname: str) -> None:
        with open(fname, "wb") as fp:
            pickle.dump(self, fp, pickle.HIGHEST_PROTOCOL)


class GradientDataset(Dataset[Tuple[Vector, Vector]]):
    def __init__(self, X: ArrayLike, G: ArrayLike, T: ArrayLike=None):
        self.X = np.array(X)
        self.G = np.array(G)
        self.T = np.array(T)
        self.data_length = len(self.X)

    def __len__(self) -> int:
        return self.data_length

    def __getitem__(self, i: int) -> Tuple[Vector, Vector]:
        return self.X[i], self.G[i]

    def save(self, fname: str) -> None:
        with open(fname, "wb") as fp:
            pickle.dump(self, fp, pickle.HIGHEST_PROTOCOL)


class TrajectoryDataset(Dataset[Tuple[Vector, Vector, Vector]]):
    def __init__(
        self,
        x_traj: ArrayLike,
        y_traj: ArrayLike,
        fx_traj: ArrayLike,
        time_batch: Optional[int] = None
    ):
        self.x_traj = np.array(x_traj)
        self.y_traj = np.array(y_traj)
        self.fx_traj = np.array(fx_traj)
        self.num_traj = self.x_traj.shape[0]
        self.n = self.x_traj.shape[1]
        if time_batch is None:
            time_batch = 1
        self.set_time_batch(time_batch)

    def set_time_batch(self, time_batch: int) -> None:
        self.time_batch = time_batch
        self.time_div = (self.n // self.time_batch)
        self.data_length = int(self.num_traj * self.time_div)

    def __len__(self) -> int:
        return self.data_length

    def __getitem__(self, i: int) -> Tuple[Vector, Vector, Vector]:
        traj_idx = i // self.time_div
        time_idx1 = (i - traj_idx * self.time_div) * self.time_batch
        time_idx2 = (i + 1 - traj_idx * self.time_div) * self.time_batch
        return (
            self.x_traj[traj_idx, time_idx1:time_idx2, :],
            self.y_traj[traj_idx, time_idx1:time_idx2, :],
            self.fx_traj[traj_idx, time_idx1:time_idx2, :],
        )

    def save(self, fname: str) -> None:
        with open(fname, "wb") as fp:
            pickle.dump(self, fp, pickle.HIGHEST_PROTOCOL)


def load(fname: str) -> Any:
    with open(fname, "rb") as fp:
        data = pickle.load(fp)
    return data


def sample(
    step: VectorField,
    random_state: Callable[[], Vector],
    num_traj: int,
    n: int
) -> TrajectoryList:
    """
    Sample num_traj trajectories each with length n

    random_state() generates a random initial state x
    step(x) advances the state forward in time

    Returns a TrajectoryList object
    """
    traj_list = list()
    for _ in range(num_traj):
        traj = list()
        x = random_state()
        traj.append(x)
        for _ in range(n - 1):
            x = step(x)
            x = np.array(x)
            traj.append(x)
        traj_list.append(traj)
    return TrajectoryList(traj_list)


def sample_fast(
    step: VectorField,
    random_state: Callable[[], Vector],
    num_traj: int,
    n: int
) -> TrajectoryList:
    """
    Faster sample() method with a 95%-85% reduction in computation time

    Sample num_traj trajectories each with length n

    random_state() generates a random initial state x
    step(x) advances the state forward in time

    Returns a TrajectoryList object
    """
    ic0 = random_state()
    traj_traj = np.zeros((num_traj, n, ic0.shape[0]))
    traj_traj[0, 0, :] = ic0
    for i in range(num_traj - 1):
        traj_traj[i + 1, 0, :] = random_state()
    for i in range(n - 1):
        traj_traj[:, i + 1, :] = step(traj_traj[:, i, :])
    return TrajectoryList(traj_traj)


def sample_gradient(
    traj_list: TrajectoryList,
    adj_step: Callable[[Vector, Vector], Vector],
    adj_output: Callable[[Vector, Vector], Vector],
    num_outputs: int,
    samples_per_traj: int,
    L: int,
) -> GradientDataset:
    """Sample the gradient using the standard method discussed in Section 3 of
    [1].

    Args:
        traj_list (TrajectoryList): data structure that is used to define a
            list of trajectories.
        model (Model): dynamical system being sampled.
        samples_per_traj (int): number of gradient samples calculated
            per trajectory.
        L (int): time horizon used for advancing the adjoint variable.

    Returns:
        GradientDataset: State and gradient data structure
        compatible with PyTorch's dataloader. GradientDataset.X[i] is the ith
        state sample and GradientDataset.G[i] is the ith gradient sample.

    References:
        [1] Otto, S.E., Padovan, A. and Rowley, C.W., 2022. Model Reduction
        for Nonlinear Systems by Balanced Truncation of State and
        Gradient Covariance.
    """
    T = list()
    X = list()
    G = list()
    N = traj_list.n  # num pts in each trajectory
    for x in traj_list.traj:
        for _ in range(samples_per_traj):
            # choose a time t in [0..N-1-L]
            t = np.random.randint(N - L)
            # choose a tau in [0..L]
            tau = np.random.randint(L + 1)
            # choose random direction eta for gradient
            eta = np.sqrt(L + 1) * np.random.randn(num_outputs)
            lam = adj_output(x[t + tau], eta)
            for i in range(1, tau):
                lam = adj_step(x[t + tau - i], lam)
            T.append(t)
            X.append(x[t])
            G.append(lam)
    return GradientDataset(X, G, T)


def sample_gradient_long_traj(
    traj_list: TrajectoryList,
    adj_step: Callable[[Vector, Vector], Vector],
    adj_output: Callable[[Vector, Vector], Vector],
    num_outputs: int,
    samples_per_traj: int,
    L: int
) -> Tuple[GradientDataset, NDArray[np.float64]]:
    """Sample the gradient using the method of long trajectories discussed in
    Algorithm 3.1 of [1].

    Args:
        traj_list (TrajectoryList): data structure that is used to define a
            list of trajectories.
        model (Model): dynamical system being sampled.
        samples_per_traj (int): number of gradient samples calculated
            per trajectory.
        L (int): time horizon used for advancing the adjoint variable.

    Returns:
        tuple:
            GradientDataset (GradientDataset): State and gradient data
            structure compatible with PyTorch's dataloader.
            GradientDataset.X[i] is the ith state sample and
            GradientDataset.G[i] is the ith gradient sample.

            D (ndarray): Vector of scaling factors, D, taking the form

            .. math:: \\frac{1}{\\sqrt{1 - \\tau_{max} - \\tau_{min}}}

            given in Algorithm 3.1 of [1]. The
            matrix Y in Algorithm 3.1 can be computed using
            Y = D * GradientDataset.G.

    References:
        [1] Otto, S.E., Padovan, A. and Rowley, C.W., 2022. Model Reduction
        for Nonlinear Systems by Balanced Truncation of State and
        Gradient Covariance.
    """
    T = list()
    X = list()
    G = list()
    D = list()
    N = traj_list.n  # num pts in each trajectory
    for x in traj_list.traj:
        for _ in range(samples_per_traj):
            t = np.random.randint(N - L)
            tau = np.random.randint(L + 1)
            eta = np.sqrt(L + 1) * np.random.randn(num_outputs)
            tau_min = np.max((0, t + tau - (N - L - 1)))
            tau_max = np.min((L, t + tau))
            nu = 1 + tau_max - tau_min
            T_ = list()
            X_ = list()
            Lam = list()
            T_.append(t + tau)
            X_.append(x[t + tau])
            Lam.append(adj_output(x[t + tau], eta))
            for i in range(1, tau_max):
                T_.append(t + tau - i)
                X_.append(x[t + tau - i])
                Lam.append(adj_step(x[t + tau - i], Lam[i - 1]))
            T.extend(T_[tau_min:tau_max])
            X.extend(X_[tau_min:tau_max])
            G.extend(Lam[tau_min:tau_max])
            D.extend([1 / np.sqrt(nu)] * len(Lam[tau_min:tau_max]))
    return GradientDataset(X, G, T), np.array(D).reshape(-1, 1)


def null_function() -> None:
    pass


class Experiment:

    def __init__(self):
        self.train_loss = []
        self.val_loss = []
        self.rom_loss = []
        self.best_train_loss = np.inf
        self.best_train_epoch = -1
        self.best_val_loss = np.inf
        self.best_val_epoch = -1
        self.best_rom_loss = np.inf
        self.best_rom_epoch = -1
        self.train_time = []
        self.orthrom_loss = np.nan
        self.add_trained_rom_traj()
        self.add_trained_orthrom_traj()

    def add_loss(
        self,
        train_loss: float,
        val_loss: Optional[float]=np.nan,
        val_func: Optional[Callable[[], None]]=null_function,
        rom_loss: Optional[float]=np.nan,
        rom_error: Optional[NDArray]=None,
        rom_traj: Optional[TrajectoryList]=None
    )-> None:
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.rom_loss.append(rom_loss)
        if train_loss < self.best_train_loss:
            self.best_train_loss = train_loss
            self.best_train_epoch = len(self.train_loss) - 1
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_val_epoch = len(self.val_loss) - 1
            self.add_trained_rom_traj(rom_loss, rom_error, rom_traj)
            val_func()
        if rom_loss < self.best_rom_loss:
            self.best_rom_loss = rom_loss
            self.best_rom_epoch = len(self.rom_loss) - 1

    def add_train_time(self, train_time: float):
        self.train_time.append(train_time)

    def add_trained_rom_traj(
        self,
        rom_loss: Optional[float]=np.nan,
        rom_error: Optional[NDArray]=None,
        rom_traj: Optional[TrajectoryList]=None
    ) -> None:
        self.trained_rom_loss = rom_loss
        self.trained_rom_error = rom_error
        self.trained_rom_traj = rom_traj

    def add_trained_orthrom_traj(
        self,
        rom_loss: Optional[float]=np.nan,
        rom_error: Optional[NDArray]=None,
        rom_traj: Optional[TrajectoryList]=None
    ) -> None:
        self.trained_orthrom_loss = rom_loss
        self.trained_orthrom_error = rom_error
        self.trained_orthrom_traj = rom_traj

    def __str__(self) -> str:
        return "Best Training Loss: {}".format(self.best_train_loss) + "\n" + \
        "Best Training Loss Epoch: {}".format(self.best_train_epoch + 1) + "\n" + \
        "Best Validation Loss: {}".format(self.best_val_loss) + "\n" + \
        "Best Validation Loss Epoch: {}".format(self.best_val_epoch + 1) + "\n" + \
        "Best ROM Loss: {}".format(self.best_rom_loss) + "\n" + \
        "Best ROM Loss Epoch: {}".format(self.best_rom_epoch + 1) + "\n" + \
        "Trained ROM Loss: {}".format(self.trained_rom_loss) + "\n" + \
        "Trained Orthogonal ROM Loss: {}".format(self.trained_orthrom_loss) + "\n" + \
        "Total Train Time: {}".format(np.sum(self.train_time))

    def save(self, fname: str) -> None:
        with open(fname, "wb") as fp:
            pickle.dump(self, fp, pickle.HIGHEST_PROTOCOL)


class ExperimentList:

    def __init__(self, experiments: List[Experiment]):
        # experiment information
        self.experiments = experiments
        self.num_exp = len(self.experiments)
        self.num_epoch = len(self.experiments[0].train_loss)
        self.epochs = np.arange(0, self.num_epoch)
        # ranking information
        self.train_ranking = np.argsort([exp.best_train_loss for exp in self.experiments])
        self.val_ranking = np.argsort([exp.best_val_loss for exp in self.experiments])
        self.rom_ranking = np.argsort([exp.best_rom_loss for exp in self.experiments])
        self.trained_rom_ranking = np.argsort([exp.trained_rom_loss for exp in self.experiments])
        try:
            self.trained_orthrom_ranking = np.argsort([exp.trained_orthrom_loss for exp in self.experiments])
        except AttributeError:
            raise AttributeError("No orthogonal ROMs have been trained.")
        # mean information
        train_loss = np.vstack([exp.train_loss for exp in self.experiments])
        val_loss = np.vstack([exp.val_loss for exp in self.experiments])
        rom_loss = np.vstack([exp.rom_loss for exp in self.experiments])
        self.mean_train_loss = np.mean(np.nan_to_num(train_loss, nan=10**9), axis=0)
        self.mean_val_loss = np.mean(np.nan_to_num(val_loss, nan=10**9), axis=0)
        self.mean_rom_loss = np.mean(np.nan_to_num(rom_loss, nan=10**9), axis=0)
        # max information
        self.max_train_loss = np.max(np.nan_to_num(train_loss, nan=10**9), axis=0)
        self.max_val_loss = np.max(np.nan_to_num(val_loss, nan=10**9), axis=0)
        self.max_rom_loss = np.max(np.nan_to_num(rom_loss, nan=10**9), axis=0)
        # min information
        self.min_train_loss = np.min(np.nan_to_num(train_loss, nan=10**9), axis=0)
        self.min_val_loss = np.min(np.nan_to_num(val_loss, nan=10**9), axis=0)
        self.min_rom_loss = np.min(np.nan_to_num(rom_loss, nan=10**9), axis=0)
        # train time
        self.train_time = [np.sum(exp.train_time) for exp in self.experiments]

    def plot_train_loss(self, ax: Axes, exp_num: Optional[int]=0, color: Optional[chr]='b') -> None:
        exp = self.experiments[exp_num]
        ax.set_xlim([0, self.num_epoch-1])
        ax.semilogy(self.epochs, self.mean_train_loss, color=color)
        ax.fill_between(self.epochs, self.min_train_loss, self.max_train_loss, alpha=0.5, color=color)
        ax.semilogy(self.epochs, exp.train_loss, color=color, linestyle="--")

    def plot_val_loss(self, ax: Axes, exp_num: Optional[int]=0, color: Optional[chr]='g') -> None:
        exp = self.experiments[exp_num]
        ax.set_xlim([0, self.num_epoch-1])
        ax.semilogy(self.epochs, self.mean_val_loss, color=color)
        ax.fill_between(self.epochs, self.min_val_loss, self.max_val_loss, alpha=0.5, color=color)
        ax.semilogy(self.epochs, exp.val_loss, color=color, linestyle="--")

    def plot_rom_loss(self, ax: Axes, exp_num: Optional[int]=0, color: Optional[chr]='r') -> None:
        exp = self.experiments[exp_num]
        ax.set_xlim([0, self.num_epoch-1])
        ax.semilogy(self.epochs, self.mean_rom_loss, color=color)
        ax.fill_between(self.epochs, self.min_rom_loss, self.max_rom_loss, alpha=0.5, color=color)
        ax.semilogy(self.epochs, exp.rom_loss, color=color, linestyle="--")

    def plot_rom_error(self, ax: Axes, t: NDArray, exp_num: Optional[int]=0, num_plot_traj: Optional[int]=None, color_func: Callable[[int], Tuple[float, float, float]]=lambda n: (0, 0, 0), orthrom: bool = False, label: str = "") -> None:
        exp = self.experiments[exp_num]
        if not orthrom:
            num_traj = len(exp.trained_rom_error)
            if num_plot_traj is None:
                num_plot_traj = num_traj
            step = np.max((int(num_traj / num_plot_traj), 1))
            traj_idxs = np.arange(0, num_traj, step=step)
            for traj_idx in traj_idxs[:-1]:
                color = color_func(traj_idx)
                ax.semilogy(t, exp.trained_rom_error[traj_idx], color=color, linewidth=0.5, alpha=0.5)
            color = color_func(traj_idxs[-1])
            ax.semilogy(t, exp.trained_rom_error[traj_idxs[-1]], color=color, linewidth=0.5, alpha=0.5, label=label)
        else:
            num_traj = len(exp.trained_orthrom_error)
            if num_plot_traj is None:
                num_plot_traj = num_traj
            step = np.max((int(num_traj / num_plot_traj), 1))
            traj_idxs = np.arange(0, num_traj, step=step)
            for traj_idx in traj_idxs[:-1]:
                color = color_func(traj_idx)
                ax.semilogy(t, exp.trained_orthrom_error[traj_idx], color=color, linewidth=0.5, alpha=0.5)
            color = color_func(traj_idxs[-1])
            ax.semilogy(t, exp.trained_orthrom_error[traj_idxs[-1]], color=color, linewidth=0.5, alpha=0.5, label=label)

    def plot_train_time(self, ax: Axes):
        ax.boxplot(self.train_time)

    def print_rankings(self):
        print("Training Loss Ranking: {}".format(self.train_ranking))
        print("Validation Loss Ranking: {}".format(self.val_ranking))
        print("ROM Loss Ranking: {}".format(self.rom_ranking))
        print("EncROM ROM Loss Ranking: {}".format(self.trained_rom_ranking))
        print("DecROM ROM Loss Ranking: {}".format(self.trained_orthrom_ranking))


def load_exp(filename, reset_best=False):
    with open(filename, "rb") as fp:
        experiment = pickle.load(fp)
    if reset_best:
        experiment.best_train_loss = np.inf
        experiment.best_val_loss = np.inf
        experiment.best_rom_loss = np.inf
    return experiment


import torch
from torch import Tensor
from romnet.autoencoder import AE
from romnet.model import Model, DiscreteModel
from romnet.timestepper import Timestepper
from scipy.integrate import solve_ivp


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


def rom(model: Model, autoencoder: AE, traj: TrajectoryList, dec_rom: bool = False, scipy: bool = False, dt: float = 0.1):

    with torch.no_grad():
        
        try:

            # rom
            def enc_rom_rhs(z: Vector) -> Tensor:
                x = autoencoder.dec(z)
                _, v = autoencoder.d_enc(x, model.rhs_tensor(x))
                return v
            def dec_rom_rhs(z: Vector) -> Tensor:
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

            # using romnet.timestepper
            if not scipy:

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
            
            # using scipy.integrate.solve_ivp
            else:

                # solve IVP
                rom_traj = []
                ics = autoencoder.enc(traj.traj[:, 0, :])
                ics_itr = iter(ics)
                def ic():
                    return next(ics_itr)
                for i in range(traj.num_traj):
                    sol = solve_ivp(
                        lambda _, z: rom_rhs(z).numpy(),
                        t_span=[0, dt * traj.n],
                        y0=ic().numpy(),
                        method="BDF",
                        t_eval=np.linspace(0, dt * traj.n, traj.n)
                    )
                    rom_traj.append(sol.y.T)
                rom_traj = TrajectoryList(rom_traj)

            # rom_error
            y_traj = model.output(traj.traj)
            y_rom_traj = model.output_tensor(autoencoder.dec(rom_traj.traj)).numpy()
            rom_error = np.square(np.linalg.norm(y_traj - y_rom_traj, axis=2))

            # rom_loss
            rom_loss = np.mean(rom_error)
            print(f"ROM Epoch Loss: {rom_loss:>7f}")

        except:

            rom_loss = np.inf
            rom_error = np.full((traj.num_traj, model.num_outputs), np.inf)
            rom_traj = TrajectoryList(np.full((traj.num_traj, traj.n, 2), np.inf))
            print(f"ROM Epoch Loss: {rom_loss:>7f}")

    return rom_loss, rom_error, rom_traj
