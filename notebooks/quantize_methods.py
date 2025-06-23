"""Quantization methods."""

import numpy as np
import torch
from typing import List



# Converters
def _convert_to_fp16(val):
    return np.float16(val)

def quantize_row_q40_tinyml(fp32_weight: torch.Tensor, STORE_FP16: bool) -> List[np.ndarray]:
    """Apply q40 quantization.

    layout of 64 weights: 0, 1, 2 ... 32, 0', 1', 2' .... 32'
    expected layout of weights: (0, 0'), (1, 1'), (2, 2'), (3, 3') ... = 32 x 8bit
    """
    qk = 32
    assert fp32_weight.ndim == 2, "Input Tensor Linear Weight should be 2-d"
    assert fp32_weight.device == torch.device("cpu")
    n_element = fp32_weight.nelement()
    nb = n_element // qk
    x = fp32_weight.numpy()

    # Reshape x to be a 2D array with shape (nb, qk)
    x = x.reshape(nb, qk) # nb: number of blocks; qk: block size

    # Get the indices of maximum absolute values along axis 1
    idx_max_abs = np.argmax(np.abs(x), axis=1)
    max_vals = x[np.arange(x.shape[0]), idx_max_abs]
    min_vals = np.zeros(nb, dtype=np.float32)
    d_vals = max_vals / -8

    id_vals = 1.0 / d_vals
    id_vals[d_vals == 0] = 0.0

    if STORE_FP16:
        d = _convert_to_fp16(d_vals)  # scaling factors
        m = _convert_to_fp16(min_vals)  # offsets
    else:
        d = np.float32(d_vals)  # scaling factors
        m = np.float32(min_vals)  # offsets
    zp = np.full(d.shape, 8, dtype=np.int8)  # zero point

    qs = np.zeros((nb // 2, qk), dtype=np.uint8)

    xi = ((x * id_vals[:, np.newaxis]) + 8.5).clip(0, 15).astype(np.uint8)
    
    # Ink: get quantization error
    d_xi = (xi.astype(np.float32) - 8) * d_vals[:, np.newaxis]
    # get maximum error
    max_error = (x - d_xi).max()
    mean_error = (x - d_xi).mean()
    relative_error = np.abs((x - d_xi) / (x + 1e-8)).mean()  # Avoid division by zero
    print("Max error: {:.6f}, mean error: {:.6f}, relative error: {:.6f}%".format(max_error, mean_error, relative_error*100))

    # Support blocksize 32 un
    assert nb % 2 == 0
    assert qk == 32
    xi = xi.reshape(nb // 2, qk * 2)

    for e in range(32):
        qs[:, e] = xi[:, e] | (xi[:, 32 + e] << 4)

    return qs, d, m, zp

def quantize_row_q41_tinyml(fp32_weight: torch.Tensor, STORE_FP16: bool):
    """
    Extending q41 in llama.cpp
    """
    qk = 32
    assert fp32_weight.ndim == 2, "Input Tensor Linear Weight should be 2-d"
    assert fp32_weight.device == torch.device("cpu")
    n_element = fp32_weight.nelement()
    nb = n_element // qk
    x = fp32_weight.numpy()

    # Reshape x to be a 2D array with shape (nb, qk)
    x = x.reshape(nb, qk) # nb: number of blocks; qk: block size

    # Get the indices of maximum absolute values along axis 1
    idx_max = np.argmax(x, axis=1)
    max_vals = x[np.arange(x.shape[0]), idx_max]
    idx_min = np.argmin(x, axis=1)
    min_vals = x[np.arange(x.shape[0]), idx_min]
    d_vals = (max_vals - min_vals) / (2 ** 4 - 1)

    id_vals = 1.0 / d_vals
    id_vals[d_vals == 0] = 0.0

    if STORE_FP16:
        raise NotImplementedError("FP16 is not implemented yet")
        d = _convert_to_fp16(d_vals)  # scaling factors
        m = _convert_to_fp16(min_vals)  # offsets
        zp = _convert_to_fp16(8.0)  # zero point
    else:
        d = np.float32(d_vals)  # scaling factors
        m = np.float32(min_vals)  # offsets
        zp = np.full(d.shape, 0, dtype=np.int8)  # zero point

    qs = np.zeros((nb // 2, qk), dtype=np.uint8)

    xi = (np.round( (x - m[:, np.newaxis]) * id_vals[:, np.newaxis])).clip(0, 15).astype(np.uint8)
    # xi = ((x - m[:, np.newaxis]) * id_vals[:, np.newaxis]+0.5).clip(0, 15).astype(np.uint8)
    # xi = np.round( (x - m[:, np.newaxis]) * id_vals[:, np.newaxis].clip(0, 15)).astype(np.uint8)
    
    # Ink: get quantization error
    d_xi = xi.astype(np.float32)  * d_vals[:, np.newaxis] + m[:, np.newaxis]
    # get maximum error
    max_error = (x - d_xi).max()
    mean_error = (x - d_xi).mean()
    relative_error = np.abs((x - d_xi) / (x + 1e-8)).mean()  # Avoid division by zero
    # relative_error = 0
    print("Max error: {:.6f}, mean error: {:.6f}, relative error: {:.6f}%".format(max_error, mean_error, relative_error*100))

    # Support blocksize 32 un
    assert nb % 2 == 0
    assert qk == 32
    xi = xi.reshape(nb // 2, qk * 2)

    for e in range(32):
        qs[:, e] = xi[:, e] | (xi[:, 32 + e] << 4)

    return qs, d, m, zp