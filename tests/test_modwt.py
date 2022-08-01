import numpy as np
import pytest
import pywt
import torch
from scipy import signal

from src.ptwt._mackey_glass import MackeyGenerator
from src.ptwt.conv_transform import modwt, imodwt, modwtmra

@pytest.mark.parametrize("wavelet_string", ["db1", "db2", "db3", "db4", "db5", "sym5"])
@pytest.mark.parametrize("level", [1, 2, None])
@pytest.mark.parametrize("length", [64, 65])
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_conv_modwt(wavelet_string, level, length, batch_size, dtype):
    generator = MackeyGenerator(
        batch_size=batch_size, tmax=length, delta_t=1, device="cpu"
    )

    mackey_data_1 = torch.squeeze(generator(), -1).type(dtype)
    wavelet = pywt.Wavelet(wavelet_string)
    ptcoeff = modwt(mackey_data_1, wavelet, level=level)
    print(ptcoeff)


@pytest.mark.parametrize("wavelet_string", ["db1", "db2", "db3", "db4", "db5", "sym5"])
@pytest.mark.parametrize("level", [1, 2, None])
def test(wavelet_string, level):
    # generate an input of even length.
    data = np.array([0.12345, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, -0.2345])
    data_torch = torch.from_numpy(data.astype(np.float32))
    # compute the forward fwt coefficients
    ptcoeff = modwt(data_torch, pywt.Wavelet('haar'), level=level)
    print(ptcoeff)
    res = imodwt(ptcoeff, wavelet_string)
    print(res)
    assert np.allclose(data, res.numpy())
