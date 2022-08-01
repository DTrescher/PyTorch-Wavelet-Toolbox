"""Utility methods to compute wavelet decompositions from a dataset."""
import sys
from typing import Sequence, Tuple, Union

if sys.version_info < (3, 8):
    from typing_extensions import Protocol
else:
    from typing import Protocol

import pywt
import torch
import numpy as np


class Wavelet(Protocol):
    """Wavelet object interface, based on the pywt wavelet object."""

    name: str
    dec_lo: Sequence[float]
    dec_hi: Sequence[float]
    rec_lo: Sequence[float]
    rec_hi: Sequence[float]
    dec_len: int
    rec_len: int
    filter_bank: Tuple[
        Sequence[float], Sequence[float], Sequence[float], Sequence[float]
    ]


def _as_wavelet(wavelet: Union[Wavelet, str]) -> Wavelet:
    """Ensure the input argument to be a pywt wavelet compatible object.

    Args:
        wavelet (Wavelet or str): The input argument, which is either a
            pywt wavelet compatible object or a valid pywt wavelet name string.

    Returns:
        Wavelet: the input wavelet object or the pywt wavelet object described by the
            input str.
    """
    if isinstance(wavelet, str):
        return pywt.Wavelet(wavelet)
    else:
        return wavelet


def _is_boundary_mode_supported(boundary_mode: str) -> bool:
    return boundary_mode == "qr" or boundary_mode == "gramschmidt"


def _outer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Torch implementation of numpy's outer for 1d vectors."""
    a_flat = torch.reshape(a, [-1])
    b_flat = torch.reshape(b, [-1])
    a_mul = torch.unsqueeze(a_flat, dim=-1)
    b_mul = torch.unsqueeze(b_flat, dim=0)
    return a_mul * b_mul


def _wavelet_as_tensor(wavelet: Union[Wavelet, str],
                       device: Union[torch.device, str],
                       dtype: torch.dtype = torch.float32
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    wavelet = _as_wavelet(wavelet)
    dec_lo, dec_hi, rec_lo, rec_hi = wavelet.filter_bank
    return (torch.tensor(dec_lo, device=device, dtype=dtype),
            torch.tensor(dec_hi, device=device, dtype=dtype),
            torch.tensor(rec_lo, device=device, dtype=dtype),
            torch.tensor(rec_hi, device=device, dtype=dtype))


def _up_arrow_op(dec: torch.Tensor, j: int) -> torch.Tensor:
    if j == 0:
        return torch.tensor([[[1.]]])
    N = dec.shape[-1]
    dec_n = torch.zeros(2 ** (j - 1) * (N - 1) + 1)
    for i in range(N):
        dec_n[2 ** (j - 1) * i] = dec[..., i]
    return dec_n.view(1, 1, -1)


def _period_list(d: torch.Tensor, N: int) -> torch.Tensor:
    n_app = N - np.mod(d.shape[-1], N)
    d = torch.cat([d, torch.zeros(d.shape[0], d.shape[1], n_app)], dim=-1)
    if d.shape[-1] < 2 * N:
        return d
    else:
        return d.view(-1, N).sum(dim=0)


def _circular_convolve_d(dec: torch.Tensor, data: torch.Tensor, j: int) -> torch.Tensor:
    # data: [batch, features, window]
    assert torch.is_floating_point(dec)
    assert torch.is_floating_point(data)

    N = data.shape[-1]
    L = dec.shape[-1]

    w_j = torch.zeros(data.shape)
    for b in range(data.shape[0]):
        for f in range(data.shape[1]):
            for t in range(N):
                index = torch.remainder(t - 2 ** (j - 1) * torch.arange(L), N)
                v_p = torch.gather(data[b, f, :], -1, index)
                w_j[b, f, t] = (dec * v_p).sum()
    return w_j


def _circular_convolve_s(dec_hi: torch.Tensor, dec_lo: torch.Tensor,
                         c_j: torch.Tensor, v_j: torch.Tensor, j: int) -> torch.Tensor:
    # data: [batch, features, window]
    N = v_j.shape[-1]
    L = dec_hi.shape[-1]

    v_j_1 = torch.zeros(v_j.shape)
    for b in range(v_j.shape[0]):
        for f in range(v_j.shape[1]):
            for t in range(N):
                index = torch.remainder(t + 2 ** (j - 1) * torch.arange(L), N)
                c_p = torch.gather(c_j[b, f, :], -1, index)
                v_p = torch.gather(v_j[b, f, :], -1, index)
                v_j_1[b, f, t] = (dec_hi * c_p).sum() + (dec_lo * v_p).sum()
    return v_j_1


def _circular_convolve_mra(h_j_o: torch.Tensor, c_j: torch.Tensor) -> torch.Tensor:
    N = c_j.shape[-1]
    D_j = torch.zeros(c_j.shape)
    for b in range(c_j.shape[0]):
        for f in range(c_j.shape[1]):
            for t in range(N):
                index = torch.remainder(t + torch.arange(N), N)
                w_j_p = torch.gather(c_j[b, f, :], -1, index)
                D_j[b, f, t] = (h_j_o * w_j_p).sum()
    return D_j
