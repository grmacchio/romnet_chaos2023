from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.stats import ortho_group
from torch import Tensor, nn
from abc import ABC, abstractmethod

from .typing import Vector

__all__ = ["ProjAE", "GAP_loss", "reduced_GAP_loss", "load_romnet", "save_romnet",
           "recon_loss", "reduced_recon_loss", "AE", "MultiLinear", "AEList", "integral_loss",
           "weight_func", "SReLU", "SReLUAE", "StandAE"]

# for better compatibility with numpy arrays
torch.set_default_dtype(torch.float64)

# custom dtype
TVector = Union[Vector, Tensor]


# activation functions
def SReLU(x: Tensor, a: Tensor) -> Tensor:
    return torch.where(
        x < 0,
        torch.log(1 + torch.exp(a * x)) / a,
        torch.log(1 + torch.exp(-a * x)) / a + x
        )


def d_SReLU(x: Tensor, a: Tensor) -> Tensor:
    return torch.where(
        x < 0,
        torch.exp(a * x) / (1 + torch.exp(a * x)),
        1 / (1 + torch.exp(-a * x))
    )


def SLReLU(x: Tensor, a: Tensor, b: Tensor, m: Tensor) -> Tensor:
    return SReLU(x, a) - SReLU(-m * x, b)


def d_SLReLU(x: Tensor, a: Tensor, b: Tensor, m: Tensor) -> Tensor:
    return d_SReLU(x, a) + m * d_SReLU(-m * x, b)


def PHI(x: Tensor) -> Tensor:
    return (1/2) * (1 + torch.special.erf(x / np.sqrt(2)))


def d_erf(x: Tensor) -> Tensor:
    return 2 / np.sqrt(np.pi) * torch.exp(-x**2)


def d_PHI(x: Tensor) -> Tensor:
    return d_erf(x / np.sqrt(2)) / (2 * np.sqrt(2))


class GeLU(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return nn.functional.gelu(x)
    
    def d_forward(self, x: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        xout = self.forward(x)
        vout = (PHI(x) + x * d_PHI(x)) * v
        return xout, vout


# layers
class Linear(nn.Module):

    def __init__(self, dim_in: int, dim_out: int, bias: bool=True) -> None:
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        if self.dim_in > self.dim_out:
            self.A = nn.Parameter(torch.tensor(ortho_group.rvs(self.dim_in)[:self.dim_out, :] / np.sqrt(0.425 + 0.444)))  # 2017 - Hendrycks - Adjusting for Dropout Variance in Batch Normalization and Weight Initialization 
        elif (self.dim_in <=self.dim_out) & ((self.dim_in != 1) or (self.dim_out != 1)):
            self.A = nn.Parameter(torch.tensor(ortho_group.rvs(self.dim_out)[:, :self.dim_in] / np.sqrt(0.425 + 0.444)))
        else:
            self.A = nn.Parameter(torch.tensor(1.))
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.dim_out, 1))
        else:
            self.bias = torch.zeros(self.dim_out, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(-1)
        return (self.A @ x + self.bias).squeeze(-1)
    
    def d_forward(self, x: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        xout = self.forward(x)
        v = v.unsqueeze(-1)
        vout = (self.A @ v).squeeze(-1)
        return xout, vout


# layer pairs
class LayerPair(nn.Module, ABC):

    def __init__(self):
        super().__init__()

    def update(self) -> None:
        ...

    @abstractmethod
    def enc_activ(self, x: Tensor) -> Tensor:
        ...
    
    @abstractmethod
    def dec_activ(self, x: Tensor) -> Tensor:
        ...

    @abstractmethod
    def d_enc_activ(self, x: Tensor) -> Tensor:
        ...

    @abstractmethod
    def d_dec_activ(self, x: Tensor) -> Tensor:
        ...

    @abstractmethod
    def enc(self, x: Tensor) -> Tensor:
        ...

    @abstractmethod
    def dec(self, x: Tensor) -> Tensor:
        ...

    def forward(self, x: Tensor) -> Tensor:
        return self.dec(self.enc(x))

    @abstractmethod
    def d_enc(self, x: Tensor, v: Tensor) -> Tensor:
        ...

    @abstractmethod
    def d_dec(self, x: Tensor, v: Tensor) -> Tensor:
        ...

    def regularizer(self) -> float:
        """Regularizer"""
        raise NotImplementedError(
            "Regularizer not implemented for layer pair %s" % self.__class__.__name__
        )


class ProjLayerPair(LayerPair):
    def __init__(self, dim_in: int, dim_out: int, angle: Optional[float] = None):
        super().__init__()

        # activation function parameters
        if angle is None:
            angle = np.pi / 8
        self.a = 1.0 / np.sin(angle) ** 2 - 1.0 / np.cos(angle) ** 2
        self.b = 1.0 / np.sin(angle) ** 2 + 1.0 / np.cos(angle) ** 2
        self.d = self.b**2 - self.a**2
        self.x_star = np.sqrt((2 * self.a) / ((self.b + self.a)**2 - self.d))
        self.y_star = self.orig_dec_activ(torch.tensor(-self.x_star)).item()

        # input and output dimensions
        self.dim_in = dim_in
        self.dim_out = dim_out

        # initialize weights
        Q = ortho_group.rvs(dim_in)[:, :dim_out]
        self.D = nn.Parameter(torch.tensor(Q))  # decoding matrix
        self.X = nn.Parameter(torch.tensor(Q.transpose()))  # encoding matrix
        self.update()

        # initialize biases
        self.bias = nn.Parameter(
            -np.sqrt(2 * self.a) / self.a * (self.D @ torch.ones(self.dim_out, 1))
        )

    def extra_repr(self) -> str:
        return "%d, %d" % (self.dim_in, self.dim_out)

    def update(self) -> None:
        self.XD = self.X @ self.D
        self.XD_INV = self.XD.inverse()
        self.E = (self.XD_INV) @ self.X

    def orig_enc_activ(self, x: Tensor) -> Tensor:
        """Activation function for encoder"""
        return (self.b * x - torch.sqrt(self.d * x**2 + 2 * self.a)) / self.a

    def orig_dec_activ(self, x: Tensor) -> Tensor:
        """Activation function for decoder"""
        return (self.b * x + torch.sqrt(self.d * x**2 + 2 * self.a)) / self.a
    
    def enc_activ(self, x: Tensor) -> Tensor:
        # return self.orig_enc_activ(x)
        return self.orig_enc_activ(x + self.x_star) + self.y_star
    
    def dec_activ(self, x: Tensor) -> Tensor:
        # return self.orig_dec_activ(x)
        return self.orig_dec_activ(x - self.x_star) - self.y_star

    def orig_d_enc_activ(self, x: Tensor) -> Tensor:
        return self.b / self.a - self.d * x / (
            self.a * torch.sqrt(self.d * x**2 + 2 * self.a)
        )

    def orig_d_dec_activ(self, x: Tensor) -> Tensor:
        return self.b / self.a + self.d * x / (
            self.a * torch.sqrt(self.d * x**2 + 2 * self.a)
        )
    
    def d_enc_activ(self, x: Tensor) -> Tensor:
        # return self.orig_d_enc_activ(x)
        return self.orig_d_enc_activ(x + self.x_star)

    def d_dec_activ(self, x: Tensor) -> Tensor:
        # return self.orig_d_dec_activ(x)
        return self.orig_d_dec_activ(x - self.x_star)

    def enc(self, x: TVector) -> Tensor:
        """Encoder"""
        x = torch.as_tensor(x).unsqueeze(-1)
        return self.enc_activ(self.E @ (x - self.bias)).squeeze(-1)

    def dec(self, x: TVector) -> Tensor:
        """Decoder"""
        x = torch.as_tensor(x).unsqueeze(-1)
        return (torch.matmul(self.D, self.dec_activ(x)) + self.bias).squeeze(-1)

    def d_enc(self, x: TVector, v: TVector) -> Tensor:
        """Tangent map of encoder"""
        x = torch.as_tensor(x).unsqueeze(-1)
        v = torch.as_tensor(v).unsqueeze(-1)
        return (self.d_enc_activ(self.E @ (x - self.bias)) * self.E @ v).squeeze(-1)

    def d_dec(self, x: TVector, v: TVector) -> Tensor:
        """Tangent map of decoder"""
        x = torch.as_tensor(x).unsqueeze(-1)
        v = torch.as_tensor(v).unsqueeze(-1)
        return torch.matmul(self.D, self.d_dec_activ(x) * v).squeeze(-1)

    def regularizer(self) -> Tensor:
        norm1 = torch.norm(self.XD - torch.eye(self.dim_out))
        norm2 = torch.norm(self.XD_INV)
        return (norm1 * norm2) ** 2


class SReLULayerPair(LayerPair):

    def __init__(self, dim_in: int, dim_out: int, a: Optional[float] = 100.):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.a = a
        self.linear_enc = nn.Linear(dim_in, dim_out)
        nn.init.kaiming_normal_(self.linear_enc.weight, mode='fan_in', nonlinearity='relu')
        self.linear_dec = nn.Linear(dim_out, dim_in)
        nn.init.kaiming_normal_(self.linear_dec.weight, mode='fan_in', nonlinearity='relu')

    def extra_repr(self) -> str:
        return "%d, %d" % (self.dim_in, self.dim_out)

    def enc_activ(self, x: Tensor) -> Tensor:
        return SReLU(x, self.a)
    
    def dec_activ(self, x: Tensor) -> Tensor:
        return SReLU(x, self.a)
    
    def d_enc_activ(self, x: Tensor) -> Tensor:
        return d_SReLU(x, self.a)
    
    def d_dec_activ(self, x: Tensor) -> Tensor:
        return d_SReLU(x, self.a)
    
    def enc(self, x: TVector) -> Tensor:
        return self.enc_activ(self.linear_enc(x))
    
    def dec(self, x: TVector) -> Tensor:
        return self.dec_activ(self.linear_dec(x))
    
    def d_enc(self, x: TVector, v: TVector) -> Tensor:
        return self.d_enc_activ(self.linear_enc(x)) * (v @ self.linear_enc.weight)

    def d_dec(self, x: TVector, v: TVector) -> Tensor:
        return self.d_dec_activ(self.linear_dec(x)) * (v @ self.linear_dec.weight)


# fully connected networks
class FullyConnected(nn.Module):

    def __init__(self, dims: List[int]) -> None:
        super().__init__()
        self.dims = dims
        self.num_layers = len(dims) - 1
        self.num_modules = 2 * self.num_layers + 1
        self.module_list = nn.ModuleList([])
        for i in range(self.num_layers):
            self.module_list.append(Linear(self.dims[i], self.dims[i + 1]))
            self.module_list.append(GeLU())
        self.module_list.append(Linear(self.dims[-1], self.dims[-1], bias=False))

    def forward(self, x: Tensor) -> Tensor:
        xout = torch.as_tensor(x)
        for module in self.module_list:
            xout = module(xout)
        return xout
    
    def d_forward(self, x: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        xout = torch.as_tensor(x)
        vout = torch.as_tensor(v)
        for module in self.module_list:
            xout, vout = module.d_forward(xout, vout)
        return xout, vout


# autoencoders
class AE(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def dim_in(self) -> int:
        """Input dimension"""

    @property
    @abstractmethod
    def dim_out(self) -> int:
        """Output dimension"""

    def update(self) -> None:
        """
        Update model parameters pre- and post-optimization step.

        Default set to null function.
        """
        pass

    @abstractmethod
    def enc(self, x: TVector) -> Tensor:
        """Encoder"""

    @abstractmethod
    def dec(self, x: TVector) -> Tensor:
        """Decoder"""

    def forward(self, x: TVector) -> Tensor:
        """Decoder composed with encoder"""
        return self.dec(self.enc(x))

    @abstractmethod
    def d_enc(self, x: TVector, v: TVector) -> Tuple[Tensor, Tensor]:
        """Derivative of encoder"""

    @abstractmethod
    def d_dec(self, x: TVector, v: TVector) -> Tuple[Tensor, Tensor]:
        """Derivative of decoder"""

    def d_autoenc(self, x: TVector, v: TVector) -> Tuple[Tensor, Tensor]:
        """Derivative of decoder composed with encoder"""
        z, vz = self.d_enc(x, v)
        return self.d_dec(z, vz)

    def regularizer(self) -> float:
        """Total regularizer"""
        raise NotImplementedError(
            "Total regularizer not implemented for class %s" % self.__class__.__name__
        )

    def save(self, fname: str) -> None:
        """Save autoencoder"""
        torch.save(self, fname)


class ProjAE(AE):
    """
    Autoencoder constrained to be a projection

    The autoencoder is built from a sequence of LayerPair objects
    """

    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims
        self.num_layers = len(dims) - 1
        self.layers = nn.ModuleList(
            [ProjLayerPair(self.dims[i], self.dims[i + 1]) for i in range(self.num_layers)]
        )
        #self.preserve_fixed_point()

    @property
    def dim_in(self) -> int:
        return self.dims[0]

    @property
    def dim_out(self) -> int:
        return self.dims[-1]

    def orig_update(self) -> None:
        for pair in self.layers:
            pair.update()

    def preserve_fixed_point(self) -> None:
        x_eq = torch.zeros((self.dims[0], 1))
        dec_zero = torch.zeros((self.dims[-1]))
        for layer in reversed(self.layers[1:]):
            dec_zero = layer.dec(dec_zero)
        inner_dec = self.layers[0].D @ self.layers[0].dec_activ(dec_zero).unsqueeze(-1)
        self.layers[0].bias = nn.Parameter(x_eq - inner_dec)
        self.layers[0].bias.requires_grad = False

    def update(self) -> None:
        self.orig_update()
        #self.preserve_fixed_point()

    def enc(self, x: TVector) -> Tensor:
        xout = torch.as_tensor(x)
        for layer in self.layers:
            xout = layer.enc(xout)
        return xout

    def dec(self, x: TVector) -> Tensor:
        xout = torch.as_tensor(x)
        for layer in reversed(self.layers):
            xout = layer.dec(xout)
        return xout

    def d_enc(self, x: TVector, v: TVector) -> Tuple[Tensor, Tensor]:
        xout = torch.as_tensor(x)
        vout = torch.as_tensor(v)
        for layer in self.layers:
            vout = layer.d_enc(xout, vout)
            xout = layer.enc(xout)
        return xout, vout

    def d_dec(self, x: TVector, v: TVector) -> Tuple[Tensor, Tensor]:
        xout = torch.as_tensor(x)
        vout = torch.as_tensor(v)
        for layer in reversed(self.layers):
            vout = layer.d_dec(xout, vout)
            xout = layer.dec(xout)
        return xout, vout

    def regularizer(self) -> float:
        total_regularizer = 0.0
        for layer in self.layers:
            total_regularizer += layer.regularizer()
        return total_regularizer

    def enc_parameters(self) -> List[Tensor]:
        params = []
        for layer in self.layers:
            params.append(layer.X)
        return params


class SReLUAE(AE):

    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims
        self.num_layers = len(dims) - 1
        self.layers = nn.ModuleList(
            [SReLULayerPair(self.dims[i], self.dims[i + 1]) for i in range(self.num_layers)]
        )

    @property
    def dim_in(self) -> int:
        return self.dims[0]

    @property
    def dim_out(self) -> int:
        return self.dims[-1]

    def enc(self, x: TVector) -> Tensor:
        xout = torch.as_tensor(x)
        for layer in self.layers:
            xout = layer.enc(xout)
        return xout

    def dec(self, x: TVector) -> Tensor:
        xout = torch.as_tensor(x)
        for layer in reversed(self.layers):
            xout = layer.dec(xout)
        return xout

    def d_enc(self, x: TVector, v: TVector) -> Tuple[Tensor, Tensor]:
        xout = torch.as_tensor(x)
        vout = torch.as_tensor(v)
        for layer in self.layers:
            vout = layer.d_enc(xout, vout)
            xout = layer.enc(xout)
        return xout, vout

    def d_dec(self, x: TVector, v: TVector) -> Tuple[Tensor, Tensor]:
        xout = torch.as_tensor(x)
        vout = torch.as_tensor(v)
        for layer in reversed(self.layers):
            vout = layer.d_dec(xout, vout)
            xout = layer.dec(xout)
        return xout, vout


class MultiLinear(AE):
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        if self.num_layers < 1:
            raise ValueError("num_layers must be greater than or equal to 1.")
        self.layers = [
            nn.Parameter(torch.tensor(ortho_group.rvs(dim)))
            for _ in range(self.num_layers)
        ]
        self.update()

    def extra_repr(self) -> str:
        return "dim={}, num_layers={}".format(self.dim, self.dim)

    @property
    def dim_in(self) -> int:
        return self.dim

    @property
    def dim_out(self) -> int:
        return self.dim

    def update(self) -> None:
        self.inv_layers = [layer.inverse() for layer in self.layers]
        self.E = torch.eye(self.dim)
        self.D = torch.eye(self.dim)
        for layer in self.layers:
            self.E = torch.matmul(layer, self.E)
        for invlayer in reversed(self.inv_layers):
            self.D = torch.matmul(invlayer, self.D)

    def enc(self, x: TVector) -> Tensor:
        x = torch.as_tensor(x).unsqueeze(-1)
        return (self.E @ x).squeeze(-1)

    def dec(self, x: TVector) -> Tensor:
        x = torch.as_tensor(x).unsqueeze(-1)
        return (self.D @ x).squeeze(-1)

    def d_enc(self, x: TVector, v: TVector) -> Tuple[Tensor, Tensor]:
        v = torch.as_tensor(v).unsqueeze(-1)
        xout = self.enc(x)
        vout = (self.E @ v).squeeze(-1)
        return xout, vout

    def d_dec(self, x: TVector, v: TVector) -> Tuple[Tensor, Tensor]:
        v = torch.as_tensor(v).unsqueeze(-1)
        xout = self.dec(x)
        vout = (self.D @ v).squeeze(-1)
        return xout, vout

    def regularizer(self) -> float:
        total_regularizer = 0.0
        for invlayer in self.inv_layers:
            total_regularizer += torch.sum(invlayer * invlayer).item()
        return total_regularizer


class AEList(AE):
    def __init__(self, ae_list: List[AE]):
        super().__init__()
        self.ae_list = nn.ModuleList(ae_list)
        self.num_ae = len(ae_list)
        if self.num_ae < 2:
            raise ValueError(
                "projae_list needs length >=2, current length is {}".format(self.num_ae)
            )
        for i in range(self.num_ae - 1):
            dim_out = ae_list[i].dim_out
            dim_in = ae_list[i + 1].dim_in
            if dim_out != dim_in:
                raise ValueError(
                    "Element {} of ae_list has dim_out = {} and next element has dim_in = {}".format(i, dim_out, dim_in)
                )

    def dim_in(self) -> int:
        return self.ae_list[0].dim_in

    def dim_out(self) -> int:
        return self.ae_list[-1].dim_out

    def update(self) -> None:
        for ae in self.ae_list:
            ae.update()

    def enc(self, x: TVector) -> Tensor:
        xout = torch.as_tensor(x)
        for ae in self.ae_list:
            xout = ae.enc(xout)
        return xout

    def dec(self, x: TVector) -> Tensor:
        xout = torch.as_tensor(x)
        for ae in reversed(self.ae_list):
            xout = ae.dec(xout)
        return xout

    def d_enc(self, x: TVector, v: TVector) -> Tuple[Tensor, Tensor]:
        xout = torch.as_tensor(x)
        vout = torch.as_tensor(v)
        for ae in self.ae_list:
            xout, vout = ae.d_enc(xout, vout)
        return xout, vout

    def d_dec(self, x: TVector, v: TVector) -> Tuple[Tensor, Tensor]:
        xout = torch.as_tensor(x)
        vout = torch.as_tensor(v)
        for ae in self.ae_list:
            xout, vout = ae.d_dec(xout, vout)
        return xout, vout

    def regularizer(self) -> float:
        total_regularizer = 0.0
        for ae in self.ae_list:
            total_regularizer += ae.regularizer()
        return total_regularizer

    def regularizer_component(self, ae_idx: int) -> float:
        return self.ae_list[ae_idx].regularizer()


class StandAE(AE):

    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims
        self.enc_net = FullyConnected(self.dims)
        self.dims.reverse()
        self.dec_net = FullyConnected(self.dims)
        self.dims.reverse()

    def dim_in(self) -> int:
        return self.dims[0]
    
    def dim_out(self) -> int:
        return self.dims[-1]
    
    def enc(self, x: TVector) -> Tensor:
        return self.enc_net(x)

    def dec(self, x: TVector) -> Tensor:
        return self.dec_net(x)

    def d_enc(self, x: TVector, v: TVector) -> Tuple[Tensor, Tensor]:
        return self.enc_net.d_forward(x, v)
    
    def d_dec(self, x: TVector, v: TVector) -> Tuple[Tensor, Tensor]:
        return self.dec_net.d_forward(x, v)


# loading and saving methods
def load_romnet(fname: str) -> AE:
    net = torch.load(fname)
    net.update()
    return net


def save_romnet(autoencoder: AE, fname: str) -> None:
    torch.save(autoencoder, fname)


# loss functions
def GAP_loss(X_pred: Tensor, X: Tensor, G: Tensor) -> Tensor:
    return torch.mean(torch.square(torch.sum(G * (X_pred - X), dim=1)))


def reduced_GAP_loss(X_pred: Tensor, X: Tensor, G: Tensor, XdotG: Tensor) -> Tensor:
    return torch.mean(torch.square(XdotG - torch.sum(G * X_pred, dim=1)))


def recon_loss(X_pred: Tensor, X: Tensor) -> Tensor:
    E = X - X_pred
    return torch.mean(torch.sum(E * E, dim=2))


def reduced_recon_loss(
    X_pred: Tensor,
    X: Tensor,
    G: Tensor,
    XdotX: Tensor,
    M: Tensor
) -> Tensor:
    term1 = - 2 * torch.sum(G * X_pred, dim=1)
    term2 = torch.sum(X_pred * (X_pred @ M.T), dim=1)
    return torch.mean(XdotX + term1 + term2)


def integral_loss(
    pred: Tensor,
    true: Tensor,
    t: Tensor,
    weights: Tensor = torch.ones(1)
) -> Tensor:
    error = pred - true
    error_integrand = torch.sum(error * error, dim=2) * weights
    error_integral = torch.trapz(error_integrand, t, dim=1) / t[-1]
    return torch.mean(error_integral)


def weight_func(t: Tensor, t_final: float, L: float) -> Tensor:
    m = 2 * L
    term1 = np.exp(m * t_final) - m * t_final
    term2 = np.exp(m * t) - m * t
    return (term1 - term2) / (4 * (L**2))


def weight_func_L0(t: Tensor, t_final: float) -> Tensor:
    return (t_final**2 - t**2) / 2
