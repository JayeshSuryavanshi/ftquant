import numpy as np
import pytest

from ftquant.quant import gguf_legacy, mlx_affine, parse


@pytest.mark.parametrize("bits", [8, 6, 4, 3, 2])
def test_mlx_affine_matches_mlx(bits):
    mx = pytest.importorskip("mlx.core")
    rng = np.random.default_rng(bits)
    w = (rng.standard_normal((64, 256)) * 0.02).astype(np.float32)
    ours = mlx_affine(bits)(w)
    q, s, b = mx.quantize(mx.array(w), group_size=64, bits=bits, stream=mx.cpu)
    assert np.array_equal(ours, np.array(mx.dequantize(q, s, b, group_size=64, bits=bits, stream=mx.cpu)))
    q, s, b = mx.quantize(mx.array(w), group_size=64, bits=bits, stream=mx.gpu)
    gpu = np.array(mx.dequantize(q, s, b, group_size=64, bits=bits, stream=mx.gpu))
    step = (w.max() - w.min()) / (2**bits - 1)
    assert np.abs(ours - gpu).max() < 1e-4 * step


def test_gguf_q8_0_roundtrip_is_close():
    w = np.random.default_rng(0).standard_normal((8, 256)).astype(np.float32)
    out = gguf_legacy("Q8_0")(w)
    assert out.shape == w.shape
    assert np.linalg.norm(out - w) / np.linalg.norm(w) < 0.01


def test_parse_rejects_k_quants():
    with pytest.raises(ValueError):
        parse("gguf:Q4_K_M")
    assert parse("mlx:3g32")[0] == "MLX 3-bit (group 32)"
