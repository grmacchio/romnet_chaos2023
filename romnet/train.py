import pickle
from typing import Callable, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from numpy.typing import ArrayLike, NDArray

from .typing import Vector

__all__ = ["train_loop", "val_loop", "CoBRAS"]


def train_loop(dataloader, autoencoder, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    epoch_loss = 0.0
    for batch, data_tuple in enumerate(dataloader):
        X = data_tuple[0]
        Xpred = autoencoder(X)
        batch_loss = loss_fn(Xpred, *data_tuple)

        with torch.no_grad():
            epoch_loss += batch_loss.item() / num_batches

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        autoencoder.update()

        if batch % 100 == 0:
            batch_loss, current = batch_loss.item(), (batch + 1) * len(X)
            print(f"Batch Loss: {batch_loss:>7f} [{current:>5d}/{size:>5d}]")

    print(f"Training Epoch Loss: {epoch_loss:>7f}")

    return epoch_loss


def val_loop(dataloader, autoencoder, loss_fn):
    num_batches = len(dataloader)
    epoch_loss = 0.0

    with torch.no_grad():
        for data_tuple in dataloader:
            X = data_tuple[0]
            Xpred = autoencoder(X)
            epoch_loss += loss_fn(Xpred, *data_tuple).item() / num_batches

    print(f"Validation Epoch Loss: {epoch_loss:>7f}")

    return epoch_loss


class ProjectedGradientDataset(Dataset[Tuple[Vector, Vector, Vector]]):
    def __init__(self, X: ArrayLike, G: ArrayLike, XdotG: ArrayLike, T: ArrayLike=None):
        self.X = np.array(X)
        self.G = np.array(G)
        self.XdotG = np.array(XdotG)
        self.T = np.array(T)
        self.data_length = self.X.shape[0]

    def __len__(self) -> int:
        return self.data_length

    def __getitem__(self, i: int) -> Tuple[Vector, Vector, Vector]:
        return self.X[i], self.G[i], self.XdotG[i]

    def save(self, fname: str) -> None:
        with open(fname, "wb") as fp:
            pickle.dump(self, fp, pickle.HIGHEST_PROTOCOL)


class CoBRAS:
    """Calculate and store the linear projection associated with Theorem 2.3 in
    [1].

    Attributes:
        U (ndarray): Matrix of left singular vectors.
        s (array): Array of singular values.
        VH (ndarray): Matrix of right singular vectors conjugate transpose.
        Phi (ndarray): Matrix of direct modes.
        Psi (ndarray): Matrix of adjoint modes.

    References:
        [1] Otto, S.E., Padovan, A. and Rowley, C.W., 2022. Model Reduction
        for Nonlinear Systems by Balanced Truncation of State and
        Gradient Covariance.
    """

    def __init__(self, X: NDArray[np.float64], Y: NDArray[np.float64]):
        """Calculate U, s, VH, Phi, and Psi.

        Args:
            X (ndarray): state sample matrix where X[i] is the ith state sample.
            Y (ndarray): gradient sample matrix where Y[i] is ith gradient
                sample.

        Note:
            The X and Y used here are the transposes of X and Y in [1].
        """
        self.U, self.s, self.VH = np.linalg.svd(
            np.dot(Y, X.T), full_matrices=False, compute_uv=True
        )
        self.Phi = np.dot(X.T, self.VH.T) / np.sqrt(self.s)
        self.Psi = np.dot(Y.T, self.U) / np.sqrt(self.s)

    def projectors(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return Phi and Psi."""
        return self.Phi, self.Psi

    def save_projectors(self, fname: str) -> None:
        """Save the tuple (Phi, Psi)."""
        with open(fname, "wb") as fp:
            pickle.dump((self.Phi, self.Psi), fp, pickle.HIGHEST_PROTOCOL)

    def project(
        self, X: NDArray[np.float64], G: NDArray[np.float64], T: NDArray[np.float64], rank: int
    ) -> ProjectedGradientDataset:
        """Project the gradient and state samples onto the direct modes, Phi,
        and adjoint modes, Psi.

        Args:
            X (ndarray): non-normalized state sample matrix where X[i] is the
                ith state sample.
            G (ndarray): non-normalized gradient sample matrix where G[i] is
                the ith gradient sample.
            rank (int): the number of leading direct and adjoint modes used.

        Returns:
            ProjectedGradientDataset: A data structure of gradient and state
            samples projected onto the direct and adjoint modes. This data
            structure is compatible with PyTorch's dataloader. In particular,
            if `x` and `g` are state and gradient samples, then

            .. math:: \\xi = \\Psi^T x, \\quad \\gamma = \\Phi^T g, \\quad a =
                \\langle x, g \\rangle

            represent the projected state, projected gradient, and
            state-gradient inner product, respectfully.
            ProjectedGradientDataset.X[i], ProjectedGradientDataset.G[i], and
            ProjectedGradientDataset.XdotG[i] are the ith projected
            state, projected gradient, and state-gradient inner product,
            respectfully.
        """
        XdotG = np.array([np.dot(x, g) for x, g in zip(X, G)])
        Xproj = X @ self.Psi[:, :rank]
        Gproj = G @ self.Phi[:, :rank]
        return ProjectedGradientDataset(Xproj, Gproj, XdotG, T)


class ProjectedStateDataset(Dataset[Tuple[Vector, Vector, Vector]]):
    def __init__(self, A: ArrayLike, D: ArrayLike, XdotX: ArrayLike, M: ArrayLike, T: ArrayLike=None):
        self.A = np.array(A)
        self.D = np.array(D)
        self.XdotX = np.array(XdotX)
        self.M = np.array(M)
        self.T = np.array(T)
        self.data_length = self.A.shape[0]

    def __len__(self) -> int:
        return self.data_length

    def __getitem__(self, i: int) -> Tuple[Vector, Vector, Vector]:
        return self.A[i], self.D[i], self.XdotX[i]

    def save(self, fname: str) -> None:
        with open(fname, "wb") as fp:
            pickle.dump(self, fp, pickle.HIGHEST_PROTOCOL)
