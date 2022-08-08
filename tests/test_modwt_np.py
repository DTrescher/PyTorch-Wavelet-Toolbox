import numpy as np
import torch
import pywt
import time
from matplotlib import pyplot as plt

from ptwt._util import _up_arrow_op, _wavelet_as_tensor, _period_list, _circular_convolve_mra, _circular_convolve_d
from ptwt.conv_transform import get_filter_tensors, modwt, imodwt, wavedec, modwtmra


def upArrow_op(li, j):
    if j == 0:
        return [1]
    N = len(li)
    li_n = np.zeros(2 ** (j - 1) * (N - 1) + 1)
    for i in range(N):
        li_n[2 ** (j - 1) * i] = li[i]
    return li_n


def period_list(li, N):
    #print('np-in:', li.shape)
    n = len(li)
    # append [0 0 ...]
    n_app = N - np.mod(n, N)
    li = list(li)
    li = li + [0] * n_app
    #print('np-d:', li)
    if len(li) < 2 * N:
        return np.array(li)
    else:
        li = np.array(li)
        li = np.reshape(li, [-1, N])
        li = np.sum(li, axis=0)
        return li


def circular_convolve_mra(h_j_o, w_j):
    ''' calculate the mra D_j'''
    N = len(w_j)
    l = np.arange(N)
    D_j = np.zeros(N)
    for t in range(N):
        index = np.mod(t + l, N)
        w_j_p = np.array([w_j[ind] for ind in index])
        D_j[t] = (np.array(h_j_o) * w_j_p).sum()
    return D_j


def circular_convolve_d(h_t, v_j_1, j):
    '''
    jth level decomposition
    h_t: \tilde{h} = h / sqrt(2)
    v_j_1: v_{j-1}, the (j-1)th scale coefficients
    return: w_j (or v_j)
    '''
    N = len(v_j_1)
    L = len(h_t)
    w_j = np.zeros(N)
    l = np.arange(L)
    for t in range(N):
        index = np.mod(t - 2 ** (j - 1) * l, N)
        v_p = np.array([v_j_1[ind] for ind in index])
        w_j[t] = (np.array(h_t) * v_p).sum()
    return w_j


def circular_convolve_s(h_t, g_t, w_j, v_j, j):
    '''
    (j-1)th level synthesis from w_j, w_j
    see function circular_convolve_d
    '''
    N = len(v_j)
    L = len(h_t)
    v_j_1 = np.zeros(N)
    l = np.arange(L)
    for t in range(N):
        index = np.mod(t + 2 ** (j - 1) * l, N)
        w_p = np.array([w_j[ind] for ind in index])
        v_p = np.array([v_j[ind] for ind in index])
        v_j_1[t] = (np.array(h_t) * w_p).sum()
        v_j_1[t] = v_j_1[t] + (np.array(g_t) * v_p).sum()
    return v_j_1


def modwt_np(x, filters, level):
    '''
    filters: 'db1', 'db2', 'haar', ...
    return: see matlab
    '''
    # filter
    wavelet = pywt.Wavelet(filters)
    h = wavelet.dec_hi
    g = wavelet.dec_lo
    h_t = np.array(h) / np.sqrt(2)
    g_t = np.array(g) / np.sqrt(2)
    wavecoeff = []
    v_j_1 = x
    for j in range(level):
        w = circular_convolve_d(h_t, v_j_1, j + 1)
        v_j_1 = circular_convolve_d(g_t, v_j_1, j + 1)
        wavecoeff.append(w)
    wavecoeff.append(v_j_1)
    return np.vstack(wavecoeff)


def imodwt_np(w, filters):
    ''' inverse modwt '''
    # filter
    wavelet = pywt.Wavelet(filters)
    h = wavelet.dec_hi
    g = wavelet.dec_lo
    h_t = np.array(h) / np.sqrt(2)
    g_t = np.array(g) / np.sqrt(2)
    level = len(w) - 1
    v_j = w[-1]
    for jp in range(level):
        j = level - jp - 1
        v_j = circular_convolve_s(h_t, g_t, w[j], v_j, j + 1)
    return v_j


def modwtmra_np(w, filters):
    ''' Multiresolution analysis based on MODWT'''
    # filter
    wavelet = pywt.Wavelet(filters)
    h = wavelet.dec_hi
    g = wavelet.dec_lo
    # D
    level, N = w.shape
    level = level - 1
    D = []
    g_j_part = [1]
    for j in range(level):
        # g_j_part
        g_j_up = upArrow_op(g, j)
        #print(f'np-{j}:', f'conv({g_j_part}, {g_j_up})')
        g_j_part = np.convolve(g_j_part, g_j_up)
        #print(f'np-part-{j}:', g_j_part)
        # h_j_o
        h_j_up = upArrow_op(h, j + 1)
        #print(f'np-{j}:', f'conv({g_j_part}, {h_j_up})')
        h_j = np.convolve(g_j_part, h_j_up)
        #print(f'np-{j}:', h_j)

        h_j_t = h_j / (2 ** ((j + 1) / 2.))
        if j == 0: h_j_t = h / np.sqrt(2)
        h_j_t_o = period_list(h_j_t, N)
        D.append(circular_convolve_mra(h_j_t_o, w[j]))
    # S
    j = level - 1
    g_j_up = upArrow_op(g, j + 1)
    g_j = np.convolve(g_j_part, g_j_up)
    g_j_t = g_j / (2 ** ((j + 1) / 2.))
    g_j_t_o = period_list(g_j_t, N)
    S = circular_convolve_mra(g_j_t_o, w[-1])
    D.append(S)
    return np.vstack(D)


def test_modwt_np():
    s1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ws = modwt_np(s1, 'db2', 3)
    print(ws)
    s1p = imodwt_np(ws, 'db2')
    print(s1p)
    mra = modwtmra_np(ws, 'db2')
    print(mra)


"""
def circular_convolve_d_torch(w, d, j):
    w = w.squeeze()
    d = d.view(1, 1, -1)

    N = d.shape[-1]
    L = w.shape[-1]

    w_j = torch.zeros(N)
    for t in range(N):
        index = torch.remainder(t - 2 ** (j - 1) * torch.arange(L), N)
        v_p = torch.gather(d, -1, index.view(1, 1, -1))
        w_j[t] = (w * v_p).sum()
    return w_j
"""


def test_circular_convolve_d():
    data = np.array([0, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0])
    data_torch = torch.from_numpy(data.astype(np.float32))

    wavelet = pywt.Wavelet('haar')
    dec_lo, dec_hi, _, _ = get_filter_tensors(
        wavelet, flip=False, device=data_torch.device, dtype=data_torch.dtype
    )

    level = pywt.dwt_max_level(data.shape[-1], dec_lo.shape[-1])

    dec_hi_torch = dec_hi / torch.sqrt(torch.tensor(2.))
    dec_hi = np.array(wavelet.dec_hi) / np.sqrt(2)
    print()
    print(dec_hi_torch)
    print(dec_hi)

    #for j in range(level):
        #_np = circular_convolve_d(dec_hi, data, j + 1)
        #print('np:', _np)
        #_torch = circular_convolve_d_torch(dec_hi_torch, data_torch, j + 1)
        #print('torch:', _torch)
        #assert np.array_equal(_np, _torch.numpy())


def test_modwt_and_imodwt():
    data = np.array([0, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0])
    data_torch = torch.from_numpy(data.astype(np.float32))

    wavelet = pywt.Wavelet('haar')

    _np = modwt_np(data, 'haar', level=3)
    _torch = modwt(data_torch, wavelet, level=3)
    print(_torch)
    assert np.array_equal(_np, _torch.numpy())

    _np = imodwt_np(_np, 'haar')
    _torch = imodwt(_torch, wavelet)
    assert np.array_equal(_np, _torch.numpy())


def test_modwt_and_imodwt_multi_dim():
    # shape: [batch, window]
    data = torch.tensor([0., 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0])
    data_batched = torch.tensor([[0., 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0], [1., 2, 3, 4, 5, 5, 6, 7, 8, 8, 7, 6]])
    data_features_batched = torch.tensor([[[0., 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0], [1., 2, 3, 4, 5, 5, 6, 7, 8, 8, 7, 6]], [[0., 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0], [1., 2, 3, 4, 5, 5, 6, 7, 8, 8, 7, 6]]])

    wavelet = pywt.Wavelet('haar')

    print()
    _torch = modwt(data, wavelet, level=3)
    assert torch.equal(imodwt(_torch, wavelet), data)
    print(_torch)
    _torch = modwt(data_batched, wavelet, level=3)
    assert torch.equal(imodwt(_torch, wavelet), data_batched)
    print(imodwt(_torch, wavelet))
    _torch = modwt(data_features_batched, wavelet, level=3)
    assert torch.equal(imodwt(_torch, wavelet), data_features_batched)
    print(imodwt(_torch, wavelet))


#TODO: test multi-dim input
def test_impl_modwtmra():
    data = np.array([0., 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0, 2, 1, 5, 3, 8, 6, 9])
    data_torch = torch.from_numpy(data)

    wavelet = pywt.Wavelet('haar')

    """
    _np = modwtmra_np(modwt_np(data, 'haar', level=3), 'haar')
    _torch = modwtmra(modwt(data_torch, wavelet, level=3), wavelet)
    print()
    print(_np)

    plt.plot(data)
    plt.show()
    for i in _np:
        plt.plot(i)
        plt.show()
    """
    print()
    _np = modwtmra_np(modwt_np(data, 'haar', level=3), 'haar')

    coeffs = modwt(data_torch, wavelet, level=3)

    dec_lo, dec_hi, _, _ = _wavelet_as_tensor(wavelet, coeffs.device, coeffs.dtype)

    level, N = coeffs.shape[0], coeffs.shape[-1]
    level = level - 1

    def _conv(d, kernel):
        s_k = kernel.shape[-1] - 1
        return torch.nn.functional.conv1d(
            torch.nn.functional.pad(
                d.view(1, 1, -1),
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
        #print(f'torch-{j}:', f'conv({g_j_part}, {g_j_up})')
        g_j_part = _conv(g_j_up, g_j_part)
        #print(f'torch-part-{j}:', g_j_part)

        # h_j_o
        h_j_up = _up_arrow_op(dec_hi, j + 1)
        #print(f'torch-{j}:', f'conv({g_j_part}, {h_j_up})')
        h_j = _conv(h_j_up, g_j_part)
        #print(f'torch-{j}:', h_j)

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

    #print(torch.from_numpy(_np.astype(np.float32)))
    #print(torch.stack(D).squeeze())
    print(torch.from_numpy(_np) - torch.stack(D).squeeze())
    assert torch.allclose(torch.stack(D).squeeze(), torch.from_numpy(_np.astype(np.float32)), atol=1e-5)


def test_modwtmra():
    data = torch.tensor([0., 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0])
    data_batched = torch.tensor([[0., 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0], [1., 2, 3, 4, 5, 5, 6, 7, 8, 8, 7, 6]])
    data_features_batched = torch.tensor(
        [[[0., 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0], [1., 2, 3, 4, 5, 5, 6, 7, 8, 8, 7, 6]],
         [[0., 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0], [1., 2, 3, 4, 5, 5, 6, 7, 8, 8, 7, 6]]])

    data = torch.tensor([43.0200, 43.0200, 43.0700, 43.0800, 43.1000, 43.0900, 43.1100, 43.0900,
            43.0700, 43.1300, 43.1900, 43.2200, 43.2400, 43.2900, 43.3400, 43.3200,
            43.3500, 43.3500, 43.4200, 43.4900, 43.5000, 43.5300, 43.5600, 43.4800,
            43.4500, 43.4200, 43.4400, 43.4000, 43.4000, 43.3600, 43.3400, 43.3300,
            43.3000, 43.2400, 43.2600, 43.2100, 43.2100, 43.2300, 43.2100, 43.1600,
            43.1500, 43.2200, 43.2400, 43.2300, 43.2500, 43.2200, 43.2000, 43.2000,
            43.1900, 43.1700, 43.1300, 43.1500, 43.1600, 43.1400, 43.1800, 43.2000,
            43.2200, 43.1900, 43.1900, 43.2200, 43.2400, 43.2400, 43.2500, 43.2400,
            43.2500, 43.2600, 43.2500, 43.2700, 43.2600, 43.2600, 43.2500, 43.2500,
            43.2500, 43.2500, 43.2500, 43.2300, 43.1900, 43.2000, 43.2000, 43.2000,
            43.2000, 43.2100, 43.1800, 43.1900, 43.1900, 43.1600, 43.1610, 43.1400,
            43.1400, 43.1300, 43.1400, 43.1400, 43.1400, 43.1300, 43.1300, 43.1300,
            43.1800, 43.2200, 43.2700, 43.2800, 43.2700, 43.2500, 43.2200, 43.2500,
            43.2700, 43.2500, 43.2900, 43.2600, 43.2400, 43.2400, 43.2500, 43.2500,
            43.2410, 43.2700, 43.2500, 43.2300, 43.2400, 43.2400, 43.2000, 43.2100,
            43.2100, 43.2100, 43.1900, 43.1890, 43.1700, 43.1700, 43.1800, 43.1600,
            43.1800, 43.1800, 43.1700, 43.1700, 43.1400, 43.1400, 43.1400, 43.1300,
            43.1210, 43.1300, 43.1100, 43.1000, 43.1300, 43.0900, 43.1000, 43.0900,
            43.0800, 43.0300, 43.0200, 42.9800, 43.0000, 42.9900, 42.9500, 42.9000,
            42.9010, 42.9100, 42.9300, 42.9300, 42.9100, 42.9000, 42.9000, 42.8800,
            42.8700, 42.8900, 42.8400, 42.8300, 42.7900, 42.7600, 42.6900, 42.7500,
            42.6990, 42.6500, 42.6700, 42.6600, 42.6900, 42.6900, 42.7000, 42.6700,
            42.6900, 42.7000, 42.7300, 42.8000, 42.8200, 42.8600, 42.8300, 42.8300,
            42.8400, 42.8500, 42.8200, 42.8400, 42.8600, 42.8500, 42.8600, 42.8700,
            42.8100, 42.8500, 42.8500, 42.8300, 42.8400, 42.8400, 42.8300, 42.8500,
            42.8400, 42.8300, 42.8000, 42.7700, 42.7500, 42.7380, 42.7200, 42.7300,
            42.7200, 42.6900, 42.7100, 42.7100, 42.7200, 42.7200, 42.7200, 42.7110,
            42.7500, 42.7500, 42.7500, 42.7400, 42.7500, 42.7400, 42.7500, 42.8200,
            42.8600, 42.7900, 42.7700, 42.7700, 42.7800, 42.7900, 42.8400, 42.8000,
            42.7900, 42.7500, 42.7500, 42.7500, 42.7600, 42.7800, 42.7800, 42.7700,
            42.7600, 42.7600, 42.7600, 42.7500, 42.7700, 42.7600, 42.7800, 42.8200,
            42.7900, 42.7900, 42.8000, 42.8100, 42.8100, 42.8300, 42.8100, 42.8000,
            42.7800, 42.8000, 42.7900, 42.7700, 42.7400, 42.7000, 42.7100, 42.7400,
            42.7400, 42.7500, 42.7200, 42.7700, 42.7000, 42.7500, 42.7000, 42.6500,
            42.7920, 42.6600, 42.6500, 42.6400, 42.7100, 42.6800, 42.6800, 42.6900,
            42.6900, 42.7000, 42.6900, 42.7000, 42.7000, 42.7820, 42.8500, 42.8500,
            42.8700, 42.8600, 42.9200, 42.8600, 42.8600, 42.6900, 42.9100, 42.6800,
            42.7200, 42.7200, 42.7200, 42.7400, 42.7500, 42.6500, 42.5110, 42.4600,
            42.4900, 42.6000, 42.5900, 42.5700, 42.4400, 42.4700, 42.4500, 42.3300,
            42.3800, 42.2400, 42.1500, 42.2300, 42.2300, 42.2900, 42.2700, 42.2500,
            42.3200, 42.4500, 42.4600, 42.5100, 42.5500, 42.5000, 42.5000, 42.5200,
            42.4600, 42.4700, 42.4900, 42.5800, 42.6000, 42.4900, 42.5000, 42.5800,
            42.5610, 42.5600, 42.5300, 42.5000, 42.5000, 42.5110, 42.4900, 42.5100,
            42.4900, 42.5100, 42.5200, 42.5400, 42.5100, 42.5100, 42.5100, 42.5200,
            42.4600, 42.5000, 42.4800, 42.4700, 42.4600, 42.4500, 42.4100, 42.4100,
            42.4400, 42.4900, 42.5000, 42.5000, 42.5000, 42.4900, 42.4900, 42.5000,
            42.4900, 42.4900, 42.4700, 42.4500, 42.4600, 42.4600, 42.4500, 42.4400,
            42.5000, 42.5000, 42.4800, 42.4500, 42.4100, 42.4300, 42.4000, 42.4600,
            42.4400, 42.4600, 42.4300, 42.4600, 42.4100, 42.4320, 42.4300, 42.4300,
            42.4400, 42.4400, 42.4300, 42.4400, 42.4400, 42.4500, 42.4500, 42.4400,
            42.4800, 42.4600, 42.4600, 42.4600, 42.4700, 42.4700, 42.4800, 42.4900,
            42.5100, 42.5500, 42.5400, 42.5300, 42.5400, 42.5300, 42.5300, 42.5400,
            42.5400, 42.5400, 42.4800, 42.5100, 42.5400, 42.5400, 42.5500, 42.5400,
            42.5400, 42.5900, 42.6000, 42.5990, 42.6000, 42.5900, 42.5900, 42.5900,
            42.5900, 42.5800, 42.5900, 42.6000, 42.6000, 42.6000, 42.6300, 42.6200,
            42.6300, 42.6200, 42.6100, 42.6000, 42.6100, 42.6200, 42.6000, 42.6000,
            42.6000, 42.6000, 42.5300, 42.5500, 42.5600, 42.5600, 42.5500, 42.4900,
            42.4600, 42.4510, 42.5000, 42.4500, 42.4600, 42.4500, 42.4600, 42.4600,
            42.4600, 42.4400, 42.4400, 42.4400, 42.4900, 42.4800, 42.4800, 42.4900,
            42.5000, 42.5120, 42.4900, 42.5100, 42.5200, 42.5200, 42.5100, 42.5000,
            42.5000, 42.4900, 42.4900, 42.4900, 42.5200, 42.5100, 42.5300, 42.5300,
            42.5300, 42.5210, 42.5300, 42.5300, 42.5210, 42.5300, 42.5200, 42.5200,
            42.5300, 42.5300, 42.5200, 42.5300, 42.6000, 42.5900, 42.5800, 42.5800,
            42.5900, 42.5800, 42.5700, 42.5600, 42.5600, 42.5700, 42.5700, 42.5800])

    print()
    print(data.shape)

    wavelet = pywt.Wavelet('haar')

    _np = modwtmra_np(modwt_np(data.numpy(), 'haar', level=9), 'haar')
    _torch = modwtmra(modwt(data, wavelet), wavelet)
    print(_np[-1])
    print(_torch[-1])
    assert torch.allclose(_torch, torch.from_numpy(_np.astype(np.float32)), atol=1e-5)

    """
    plt.plot(data, label='np-orig')
    plt.legend()
    plt.show()
    for i in _np:
        plt.plot(i, label=f'')
        plt.legend()
        plt.show()
    """

    plt.plot(data, label='torch-orig')
    plt.legend()
    plt.show()
    for i in _torch:
        plt.plot(i, label=f'')
        plt.legend()
        plt.show()

    plt.plot(_torch.sum(dim=0), label='torch-reconstruction')
    plt.legend()
    plt.show()

    _torch = modwtmra(modwt(data_batched, wavelet, level=3), wavelet)
    #print(_torch)

    _torch = modwtmra(modwt(data_features_batched, wavelet, level=3), wavelet)
    #print(_torch)

    #plt.plot(data_features_batched[0][0])
    #plt.show()


def test_stuff():
    data = torch.tensor([43.0200, 43.0200, 43.0700, 43.0800, 43.1000, 43.0900, 43.1100, 43.0900,
                         43.0700, 43.1300, 43.1900, 43.2200, 43.2400, 43.2900, 43.3400, 43.3200,
                         43.3500, 43.3500, 43.4200, 43.4900, 43.5000, 43.5300, 43.5600, 43.4800,
                         43.4500, 43.4200, 43.4400, 43.4000, 43.4000, 43.3600, 43.3400, 43.3300,
                         43.3000, 43.2400, 43.2600, 43.2100, 43.2100, 43.2300, 43.2100, 43.1600,
                         43.1500, 43.2200, 43.2400, 43.2300, 43.2500, 43.2200, 43.2000, 43.2000,
                         43.1900, 43.1700, 43.1300, 43.1500, 43.1600, 43.1400, 43.1800, 43.2000,
                         43.2200, 43.1900, 43.1900, 43.2200, 43.2400, 43.2400, 43.2500, 43.2400,
                         43.2500, 43.2600, 43.2500, 43.2700, 43.2600, 43.2600, 43.2500, 43.2500,
                         43.2500, 43.2500, 43.2500, 43.2300, 43.1900, 43.2000, 43.2000, 43.2000,
                         43.2000, 43.2100, 43.1800, 43.1900, 43.1900, 43.1600, 43.1610, 43.1400,
                         43.1400, 43.1300, 43.1400, 43.1400, 43.1400, 43.1300, 43.1300, 43.1300,
                         43.1800, 43.2200, 43.2700, 43.2800, 43.2700, 43.2500, 43.2200, 43.2500,
                         43.2700, 43.2500, 43.2900, 43.2600, 43.2400, 43.2400, 43.2500, 43.2500,
                         43.2410, 43.2700, 43.2500, 43.2300, 43.2400, 43.2400, 43.2000, 43.2100,
                         43.2100, 43.2100, 43.1900, 43.1890, 43.1700, 43.1700, 43.1800, 43.1600,
                         43.1800, 43.1800, 43.1700, 43.1700, 43.1400, 43.1400, 43.1400, 43.1300,
                         43.1210, 43.1300, 43.1100, 43.1000, 43.1300, 43.0900, 43.1000, 43.0900,
                         43.0800, 43.0300, 43.0200, 42.9800, 43.0000, 42.9900, 42.9500, 42.9000,
                         42.9010, 42.9100, 42.9300, 42.9300, 42.9100, 42.9000, 42.9000, 42.8800,
                         42.8700, 42.8900, 42.8400, 42.8300, 42.7900, 42.7600, 42.6900, 42.7500,
                         42.6990, 42.6500, 42.6700, 42.6600, 42.6900, 42.6900, 42.7000, 42.6700,
                         42.6900, 42.7000, 42.7300, 42.8000, 42.8200, 42.8600, 42.8300, 42.8300,
                         42.8400, 42.8500, 42.8200, 42.8400, 42.8600, 42.8500, 42.8600, 42.8700,
                         42.8100, 42.8500, 42.8500, 42.8300, 42.8400, 42.8400, 42.8300, 42.8500,
                         42.8400, 42.8300, 42.8000, 42.7700, 42.7500, 42.7380, 42.7200, 42.7300,
                         42.7200, 42.6900, 42.7100, 42.7100, 42.7200, 42.7200, 42.7200, 42.7110,
                         42.7500, 42.7500, 42.7500, 42.7400, 42.7500, 42.7400, 42.7500, 42.8200,
                         42.8600, 42.7900, 42.7700, 42.7700, 42.7800, 42.7900, 42.8400, 42.8000,
                         42.7900, 42.7500, 42.7500, 42.7500, 42.7600, 42.7800, 42.7800, 42.7700,
                         42.7600, 42.7600, 42.7600, 42.7500, 42.7700, 42.7600, 42.7800, 42.8200,
                         42.7900, 42.7900, 42.8000, 42.8100, 42.8100, 42.8300, 42.8100, 42.8000,
                         42.7800, 42.8000, 42.7900, 42.7700, 42.7400, 42.7000, 42.7100, 42.7400,
                         42.7400, 42.7500, 42.7200, 42.7700, 42.7000, 42.7500, 42.7000, 42.6500,
                         42.7920, 42.6600, 42.6500, 42.6400, 42.7100, 42.6800, 42.6800, 42.6900,
                         42.6900, 42.7000, 42.6900, 42.7000, 42.7000, 42.7820, 42.8500, 42.8500,
                         42.8700, 42.8600, 42.9200, 42.8600, 42.8600, 42.6900, 42.9100, 42.6800,
                         42.7200, 42.7200, 42.7200, 42.7400, 42.7500, 42.6500, 42.5110, 42.4600,
                         42.4900, 42.6000, 42.5900, 42.5700, 42.4400, 42.4700, 42.4500, 42.3300,
                         42.3800, 42.2400, 42.1500, 42.2300, 42.2300, 42.2900, 42.2700, 42.2500,
                         42.3200, 42.4500, 42.4600, 42.5100, 42.5500, 42.5000, 42.5000, 42.5200,
                         42.4600, 42.4700, 42.4900, 42.5800, 42.6000, 42.4900, 42.5000, 42.5800,
                         42.5610, 42.5600, 42.5300, 42.5000, 42.5000, 42.5110, 42.4900, 42.5100,
                         42.4900, 42.5100, 42.5200, 42.5400, 42.5100, 42.5100, 42.5100, 42.5200,
                         42.4600, 42.5000, 42.4800, 42.4700, 42.4600, 42.4500, 42.4100, 42.4100,
                         42.4400, 42.4900, 42.5000, 42.5000, 42.5000, 42.4900, 42.4900, 42.5000,
                         42.4900, 42.4900, 42.4700, 42.4500, 42.4600, 42.4600, 42.4500, 42.4400,
                         42.5000, 42.5000, 42.4800, 42.4500, 42.4100, 42.4300, 42.4000, 42.4600,
                         42.4400, 42.4600, 42.4300, 42.4600, 42.4100, 42.4320, 42.4300, 42.4300,
                         42.4400, 42.4400, 42.4300, 42.4400, 42.4400, 42.4500, 42.4500, 42.4400,
                         42.4800, 42.4600, 42.4600, 42.4600, 42.4700, 42.4700, 42.4800, 42.4900,
                         42.5100, 42.5500, 42.5400, 42.5300, 42.5400, 42.5300, 42.5300, 42.5400,
                         42.5400, 42.5400, 42.4800, 42.5100, 42.5400, 42.5400, 42.5500, 42.5400,
                         42.5400, 42.5900, 42.6000, 42.5990, 42.6000, 42.5900, 42.5900, 42.5900,
                         42.5900, 42.5800, 42.5900, 42.6000, 42.6000, 42.6000, 42.6300, 42.6200,
                         42.6300, 42.6200, 42.6100, 42.6000, 42.6100, 42.6200, 42.6000, 42.6000,
                         42.6000, 42.6000, 42.5300, 42.5500, 42.5600, 42.5600, 42.5500, 42.4900,
                         42.4600, 42.4510, 42.5000, 42.4500, 42.4600, 42.4500, 42.4600, 42.4600,
                         42.4600, 42.4400, 42.4400, 42.4400, 42.4900, 42.4800, 42.4800, 42.4900,
                         42.5000, 42.5120, 42.4900, 42.5100, 42.5200, 42.5200, 42.5100, 42.5000,
                         42.5000, 42.4900, 42.4900, 42.4900, 42.5200, 42.5100, 42.5300, 42.5300,
                         42.5300, 42.5210, 42.5300, 42.5300, 42.5210, 42.5300, 42.5200, 42.5200,
                         42.5300, 42.5300, 42.5200, 42.5300, 42.6000, 42.5900, 42.5800, 42.5800,
                         42.5900, 42.5800, 42.5700, 42.5600, 42.5600, 42.5700, 42.5700, 42.5800], dtype=torch.float32, device='cuda:0')

    wavelet = pywt.Wavelet('haar')

    #plt.plot(data, label='torch-orig')
    #plt.legend()
    #plt.show()

    #modwtmra(modwt(data, wavelet), wavelet)

    print()
    for i in range(6):
        t0 = time.time()
        x_mra = modwtmra(modwt(data, wavelet, level=9), wavelet)
        t1 = time.time()
        print(t1 - t0, 'seconds')

    assert x_mra.sum(0).allclose(data)

    #for i in x:
    #    plt.plot(i.squeeze())
    #    #plt.legend()
    #    plt.show()
