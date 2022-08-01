"""Fast wavelet transformation code with edge-padding."""
# Created by moritz wolter, 14.04.20
from typing import List, Optional, Sequence, Tuple, Union

import pywt
import torch

from ._util import Wavelet, _as_wavelet, _wavelet_as_tensor, _circular_convolve_d, _circular_convolve_s, _up_arrow_op, \
    _period_list, _circular_convolve_mra


def get_filter_tensors(
        wavelet: Union[Wavelet, str],
        flip: bool,
        device: Union[torch.device, str],
        dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert input wavelet to filter tensors.

    Args:
        wavelet (Wavelet or str): A pywt wavelet compatible object or
                the name of a pywt wavelet.
        flip (bool): If true filters are flipped.
        device (torch.device) : PyTorch target device.
        dtype (torch.dtype): The data type sets the precision of the
               computation. Default: torch.float32.

    Returns:
        tuple: Tuple containing the four filter tensors
        dec_lo, dec_hi, rec_lo, rec_hi

    """
    wavelet = _as_wavelet(wavelet)

    def _create_tensor(filter: Sequence[float]) -> torch.Tensor:
        if flip:
            if isinstance(filter, torch.Tensor):
                return filter.flip(-1).unsqueeze(0).to(device)
            else:
                return torch.tensor(filter[::-1], device=device, dtype=dtype).unsqueeze(0)
        else:
            if isinstance(filter, torch.Tensor):
                return filter.unsqueeze(0).to(device)
            else:
                return torch.tensor(filter, device=device, dtype=dtype).unsqueeze(0)

    dec_lo, dec_hi, rec_lo, rec_hi = wavelet.filter_bank
    dec_lo_tensor = _create_tensor(dec_lo)
    dec_hi_tensor = _create_tensor(dec_hi)
    rec_lo_tensor = _create_tensor(rec_lo)
    rec_hi_tensor = _create_tensor(rec_hi)
    return dec_lo_tensor, dec_hi_tensor, rec_lo_tensor, rec_hi_tensor


def _get_pad(data_len: int, filt_len: int) -> Tuple[int, int]:
    """Compute the required padding.

    Args:
        data_len (int): The length of the input vector.
        filt_len (int): The length of the used filter.

    Returns:
        tuple: The numbers to attach on the edges of the input.

    """
    # pad to ensure we see all filter positions and
    # for pywt compatability.
    # convolution output length:
    # see https://arxiv.org/pdf/1603.07285.pdf section 2.3:
    # floor([data_len - filt_len]/2) + 1
    # should equal pywt output length
    # floor((data_len + filt_len - 1)/2)
    # => floor([data_len + total_pad - filt_len]/2) + 1
    #    = floor((data_len + filt_len - 1)/2)
    # (data_len + total_pad - filt_len) + 2 = data_len + filt_len - 1
    # total_pad = 2*filt_len - 3

    # we pad half of the total requried padding on each side.
    padr = (2 * filt_len - 3) // 2
    padl = (2 * filt_len - 3) // 2

    # pad to even singal length.
    if data_len % 2 != 0:
        padr += 1

    return padr, padl


def _translate_boundary_strings(pywt_mode: str) -> str:
    """Translate pywt mode strings to pytorch mode strings.

    We support constant, zero, reflect and periodic.
    Unfortunately "constant" has different meanings in the
    pytorch and pywavelet communities.

    Raises:
        ValueError: If the padding mode is not supported.

    """
    if pywt_mode == "constant":
        pt_mode = "replicate"
    elif pywt_mode == "zero":
        pt_mode = "constant"
    elif pywt_mode == "reflect":
        pt_mode = pywt_mode
    elif pywt_mode == "periodic":
        pt_mode = "circular"
    else:
        raise ValueError("Padding mode not supported.")
    return pt_mode


def fwt_pad(
        data: torch.Tensor, wavelet: Union[Wavelet, str], mode: str = "reflect"
) -> torch.Tensor:
    """Pad the input signal to make the fwt matrix work.

    Args:
        data (torch.Tensor): Input data [batch_size, 1, time]
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet.
        mode (str): The desired way to pad.
            Supported modes are "reflect", "zero", "constant" and "periodic".
            Refection padding mirrors samples along the border.
            Zero padding pads zeros.
            Constant padding replicates border values.
            periodic padding repeats samples in a cyclic fashing.
            Defaults to reflect.

    Returns:
        torch.Tensor: A pytorch tensor with the padded input data

    """
    wavelet = _as_wavelet(wavelet)
    # convert pywt to pytorch convention.
    mode = _translate_boundary_strings(mode)

    padr, padl = _get_pad(data.shape[-1], len(wavelet.dec_lo))
    data_pad = torch.nn.functional.pad(data, [padl, padr], mode=mode)
    return data_pad


def _flatten_2d_coeff_lst(
        coeff_lst_2d: List[
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        ],
        flatten_tensors: bool = True,
) -> List[torch.Tensor]:
    """Flattens a list of tensor tuples into a single list.

    Args:
        coeff_lst_2d (list): A pywt-style coefficient list of torch tensors.
        flatten_tensors (bool): If true, 2d tensors are flattened. Defaults to True.

    Returns:
        list: A single 1-d list with all original elements.
    """
    flat_coeff_lst = []
    for coeff in coeff_lst_2d:
        if isinstance(coeff, tuple):
            for c in coeff:
                if flatten_tensors:
                    flat_coeff_lst.append(c.flatten())
                else:
                    flat_coeff_lst.append(c)
        else:
            if flatten_tensors:
                flat_coeff_lst.append(coeff.flatten())
            else:
                flat_coeff_lst.append(coeff)
    return flat_coeff_lst


def wavedec(
        data: torch.Tensor,
        wavelet: Union[Wavelet, str],
        level: Optional[int] = None,
        mode: str = "reflect",
) -> List[torch.Tensor]:
    """Compute the analysis (forward) 1d fast wavelet transform.

    Args:
        data (torch.Tensor): Input time series of shape [batch_size, 1, time]
                             1d inputs are interpreted as [time],
                             2d inputs are interpreted as [batch_size, time].
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet.
        level (int): The scale level to be computed.
                               Defaults to None.
        mode (str): The desired padding mode. Padding extends the singal along
            the edges. Supported modes are "reflect", "zero", "constant"
            and "periodic". Defaults to "reflect".

    Returns:
        list: A list [cA_n, cD_n, cD_n-1, …, cD2, cD1]
        containing the wavelet coefficients. A denotes
        approximation and D detail coefficients.

    Example:
        >>> import torch
        >>> import ptwt, pywt
        >>> import numpy as np
        >>> # generate an input of even length.
        >>> data = np.array([0, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0])
        >>> data_torch = torch.from_numpy(data.astype(np.float32))
        >>> # compute the forward fwt coefficients
        >>> ptwt.wavedec(data_torch, pywt.Wavelet('haar'),
                         mode='zero', level=2)

    """
    if data.dim() == 1:
        # assume time series
        data = data.unsqueeze(0).unsqueeze(0)
    elif data.dim() == 2:
        # assume batched time series
        data = data.unsqueeze(1)

    dec_lo, dec_hi, _, _ = get_filter_tensors(
        wavelet, flip=True, device=data.device, dtype=data.dtype
    )
    filt_len = dec_lo.shape[-1]
    filt = torch.stack([dec_lo, dec_hi], 0)

    if level is None:
        level = pywt.dwt_max_level(data.shape[-1], filt_len)

    result_lst = []
    res_lo = data
    for _ in range(level):
        res_lo = fwt_pad(res_lo, wavelet, mode=mode)
        res = torch.nn.functional.conv1d(res_lo, filt, stride=2)
        res_lo, res_hi = torch.split(res, 1, 1)
        result_lst.append(res_hi.squeeze(1))
    result_lst.append(res_lo.squeeze(1))
    return result_lst[::-1]


def waverec(coeffs: List[torch.Tensor], wavelet: Union[Wavelet, str]) -> torch.Tensor:
    """Reconstruct a signal from wavelet coefficients.

    Args:
        coeffs (list): The wavelet coefficient list produced by wavedec.
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet.

    Returns:
        torch.Tensor: The reconstructed signal.

    Example:
        >>> import torch
        >>> import ptwt, pywt
        >>> import numpy as np
        >>> # generate an input of even length.
        >>> data = np.array([0, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0])
        >>> data_torch = torch.from_numpy(data.astype(np.float32))
        >>> # invert the fast wavelet transform.
        >>> ptwt.waverec(ptwt.wavedec(data_torch, pywt.Wavelet('haar'),
                                      mode='zero', level=2),
                         pywt.Wavelet('haar'))

    """
    _, _, rec_lo, rec_hi = get_filter_tensors(
        wavelet, flip=False, device=coeffs[-1].device, dtype=coeffs[-1].dtype
    )
    filt_len = rec_lo.shape[-1]
    filt = torch.stack([rec_lo, rec_hi], 0)

    res_lo = coeffs[0]
    for c_pos, res_hi in enumerate(coeffs[1:]):
        res_lo = torch.stack([res_lo, res_hi], 1)
        res_lo = torch.nn.functional.conv_transpose1d(res_lo, filt, stride=2).squeeze(1)

        # remove the padding
        padl = (2 * filt_len - 3) // 2
        padr = (2 * filt_len - 3) // 2
        if c_pos < len(coeffs) - 2:
            pred_len = res_lo.shape[-1] - (padl + padr)
            next_len = coeffs[c_pos + 2].shape[-1]
            if next_len != pred_len:
                padr += 1
                pred_len = res_lo.shape[-1] - (padl + padr)
                assert (
                        next_len == pred_len
                ), "padding error, please open an issue on github "
        if padl > 0:
            res_lo = res_lo[..., padl:]
        if padr > 0:
            res_lo = res_lo[..., :-padr]
    return res_lo


def modwt(data: torch.Tensor,
          wavelet: Union[Wavelet, str],
          level: Optional[int] = None
          ) -> torch.Tensor:
    """
    Args:
        data (torch.Tensor): Input time series.
            Shapes:
                [window]
                [batch, window]
                [batch, features, window]
        wavelet (Wavelet or str): A pywt wavelet compatible object or the name of a pywt wavelet.
        level (int): The scale level to be computed. Defaults to None.
    Example:
        >>> import torch
        >>> import ptwt, pywt
        >>> import numpy as np
        >>> # generate an input of even length.
        >>> d = torch.tensor([0., 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0])
        >>> # compute the forward fwt coefficients
        >>> ptwt.modwt(d, pywt.Wavelet('haar'), level=2)
    """
    if data.dim() == 1:
        # assume time series
        data = data.view(1, 1, -1)
    elif data.dim() == 2:
        # assume batched time series
        data = data.unsqueeze(1)

    dec_lo, dec_hi, _, _ = _wavelet_as_tensor(wavelet, data.device, data.dtype)

    dec_lo = dec_lo / torch.sqrt(torch.tensor(2.))
    dec_hi = dec_hi / torch.sqrt(torch.tensor(2.))

    if level is None:
        level = pywt.dwt_max_level(data.shape[-1], dec_lo.shape[-1])

    wavecoeff = []
    v_j_1 = data
    for j in range(level):
        w = _circular_convolve_d(dec_hi, v_j_1, j + 1)
        v_j_1 = _circular_convolve_d(dec_lo, v_j_1, j + 1)
        wavecoeff.append(w)
    wavecoeff.append(v_j_1)
    return torch.stack(wavecoeff)


def imodwt(coeffs: torch.Tensor, wavelet: Union[Wavelet, str]) -> torch.Tensor:
    """
    Args:
        coeffs (torch.Tensor): The wavelet coefficients produced by modwt.
        wavelet (Wavelet or str): A pywt wavelet compatible object or the name of a pywt wavelet.

    Example:
        >>> import torch
        >>> import ptwt, pywt
        >>> import numpy as np
        >>> # generate an input of even length.
        >>> d = torch.tensor([0., 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0])
        >>> # invert the fast wavelet transform.
        >>> ptwt.imodwt(ptwt.modwt(d, pywt.Wavelet('haar'), level=2), pywt.Wavelet('haar'))
    """
    dec_lo, dec_hi, _, _ = _wavelet_as_tensor(wavelet, coeffs.device, coeffs.dtype)

    dec_lo = dec_lo / torch.sqrt(torch.tensor(2.))
    dec_hi = dec_hi / torch.sqrt(torch.tensor(2.))

    level = coeffs.shape[0] - 1

    v_j = coeffs[-1]
    for jp in range(level):
        j = level - jp - 1
        v_j = _circular_convolve_s(dec_hi, dec_lo, coeffs[j], v_j, j + 1)
    return v_j.squeeze()


def modwtmra(coeffs: torch.Tensor,
             wavelet: Union[Wavelet, str]
             ) -> torch.Tensor:

    dec_lo, dec_hi, _, _ = _wavelet_as_tensor(wavelet, coeffs.device, coeffs.dtype)
    level, N = coeffs.shape[0], coeffs.shape[-1]
    level = level - 1

    def _conv(d, kernel):
        s_k = kernel.shape[-1] - 1
        return torch.nn.functional.conv1d(
            torch.nn.functional.pad(
                d,
                (s_k, s_k),
                'constant', 0
            ),
            kernel.flip(-1)
        )

    # Details
    D = []
    g_j_part = torch.tensor([[[1.]]])
    for j in range(level):
        # g_j_part
        g_j_up = _up_arrow_op(dec_lo, j)
        g_j_part = _conv(g_j_up, g_j_part)

        # h_j_o
        h_j_up = _up_arrow_op(dec_hi, j + 1)
        h_j = _conv(h_j_up, g_j_part)
        h_j_t = h_j / (2 ** ((j + 1) / 2.))
        if j == 0:
            h_j_t = dec_hi.view(1, 1, -1) / torch.sqrt(torch.tensor(2.))
        h_j_t_o = _period_list(h_j_t, N)
        D.append(_circular_convolve_mra(h_j_t_o, coeffs[j]))

    # Scale
    j = level - 1
    g_j_up = _up_arrow_op(dec_lo, j + 1)
    g_j = _conv(g_j_up, g_j_part)
    g_j_t = g_j / (2 ** ((j + 1) / 2.))
    g_j_t_o = _period_list(g_j_t, N)
    D.append(_circular_convolve_mra(g_j_t_o, coeffs[-1]))
    return torch.stack(D).squeeze()
