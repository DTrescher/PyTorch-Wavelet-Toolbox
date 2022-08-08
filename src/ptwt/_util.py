"""Utility methods to compute wavelet decompositions from a dataset."""
import sys
from typing import Sequence, Tuple, Union

if sys.version_info < (3, 8):
    from typing_extensions import Protocol
else:
    from typing import Protocol

import pywt
import torch


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


@torch.jit.script
def _conv(d: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    s_k = kernel.shape[-1] - 1
    return torch.nn.functional.conv1d(
        torch.nn.functional.pad(
            d,
            (s_k, s_k),
            'constant', 0.
        ),
        kernel.flip(-1)
    )


@torch.jit.script
def _up_arrow_op(dec: torch.Tensor, j: int) -> torch.Tensor:
    if j == 0:
        return torch.tensor([[[1.]]], device=dec.device, dtype=dec.dtype)

    N = dec.shape[-1]
    dec_n = torch.zeros(int(2 ** (j - 1) * (N - 1) + 1), device=dec.device, dtype=dec.dtype)
    dec_n[int(2 ** (j - 1)) * torch.arange(N)] = dec[torch.arange(N)]
    return dec_n.view(1, 1, -1)


@torch.jit.script
def _period_list(d: torch.Tensor, N: int) -> torch.Tensor:
    n_app = N - torch.remainder(d.shape[-1], N)
    d = torch.cat([d, torch.zeros(d.shape[0], d.shape[1], n_app, device=d.device)], dim=-1)
    if d.shape[-1] < 2 * N:
        return d.view(-1)
    else:
        return d.view(-1, N).sum(dim=0)


@torch.jit.script
def _circular_convolve_d(dec: torch.Tensor, data: torch.Tensor, j: int) -> torch.Tensor:
    # data: [batch, features, window]
    assert torch.is_floating_point(dec)
    assert torch.is_floating_point(data)

    N = data.shape[-1]
    L = dec.shape[-1]

    """OLD:
    w_j = torch.zeros(data.shape, device=data.device, dtype=data.dtype)
    for b in range(data.shape[0]):
        for f in range(data.shape[1]):
            for t in range(N):
                index = torch.remainder(t - 2 ** (j - 1) * torch.arange(L), N)
                v_p = torch.gather(data[b, f, :], -1, index.to(device=data.device, dtype=torch.int64))
                w_j[b, f, t] = (dec * v_p).sum(-1)
    print(w_j.shape)
    return w_j
    """

    index = torch.remainder(torch.arange(N).view(-1, 1) - 2 ** (j - 1) * torch.arange(L), N) \
        .to(device=data.device, dtype=torch.int64)\
        .view(-1).repeat(data.shape[0], data.shape[1], 1)

    v_p = torch.gather(data, -1, index)
    return torch.matmul(v_p.view(data.shape[0], data.shape[1], -1, L), dec)


@torch.jit.script
def _circular_convolve_s(dec_hi: torch.Tensor, dec_lo: torch.Tensor,
                         c_j: torch.Tensor, v_j: torch.Tensor, j: int) -> torch.Tensor:
    # data: [batch, features, window]
    N = v_j.shape[-1]
    L = dec_hi.shape[-1]

    """OLD:
    v_j_1 = torch.zeros(v_j.shape, device=v_j.device, dtype=v_j.dtype)
    for b in range(v_j.shape[0]):
        for f in range(v_j.shape[1]):
            for t in range(N):
                index = torch.remainder(t + 2 ** (j - 1) * torch.arange(L), N)
                c_p = torch.gather(c_j[b, f, :], -1, index.to(device=v_j.device, dtype=torch.int64))
                v_p = torch.gather(v_j[b, f, :], -1, index.to(device=v_j.device, dtype=torch.int64))
                v_j_1[b, f, t] = (dec_hi * c_p).sum() + (dec_lo * v_p).sum()
    return v_j_1
    """

    index = torch.remainder(torch.arange(N).view(-1, 1) + 2 ** (j - 1) * torch.arange(L), N)\
        .to(device=v_j.device, dtype=torch.int64) \
        .view(-1).repeat(v_j.shape[0], v_j.shape[1], 1)

    c_p = torch.gather(c_j, -1, index)
    v_p = torch.gather(v_j, -1, index)
    return torch.matmul(c_p.view(v_j.shape[0], v_j.shape[1], -1, L), dec_hi) + \
           torch.matmul(v_p.view(v_j.shape[0], v_j.shape[1], -1, L), dec_lo)


@torch.jit.script
def _circular_convolve_mra(h_j_o: torch.Tensor, c_j: torch.Tensor) -> torch.Tensor:
    N = c_j.shape[-1]

    """OLD:
    D_j = torch.zeros(c_j.shape, device=c_j.device, dtype=c_j.dtype)
    for b in range(c_j.shape[0]):
        for f in range(c_j.shape[1]):
            for t in range(N):
                index = torch.remainder(t + torch.arange(N), N)
                w_j_p = torch.gather(c_j[b, f, :], -1, index.to(c_j.device))
                D_j[b, f, t] = (h_j_o * w_j_p).sum()
    return D_j
    """

    # index: [N, N]
    index = torch.remainder(torch.arange(N).view(-1, 1) + torch.arange(N), N) \
        .to(device=c_j.device, dtype=torch.int64)\
        .view(-1).repeat(c_j.shape[0], c_j.shape[1], 1)

    w_j_p = torch.gather(c_j, -1, index)
    return torch.matmul(w_j_p.view(c_j.shape[0], c_j.shape[1], -1, N), h_j_o)
