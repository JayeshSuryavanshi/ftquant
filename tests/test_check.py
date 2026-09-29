import json

import numpy as np
import pytest

from ftquant.check import noise_to_signal, pairs
from ftquant.quant import parse
from ftquant.tensors import Checkpoint, round_to

from helpers import write_safetensors

NAMES = [
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
]


@pytest.fixture
def base(tmp_path):
    rng = np.random.default_rng(0)
    d = tmp_path / "base"
    d.mkdir()
    w = {
        n: ((rng.standard_normal((64, 128)) * 0.02).astype(np.float32), "BF16")
        for n in NAMES
    }
    write_safetensors(d / "model.safetensors", w)
    return d


def lora(rank=4, seed=1, size=0.05):
    rng = np.random.default_rng(seed)
    return {
        n: (
            (rng.standard_normal((128, rank)) * size).astype(np.float32),
            (rng.standard_normal((rank, 64)) * size).astype(np.float32),
        )
        for n in NAMES
    }


def mlx_adapter(path, lo, scale):
    path.mkdir()
    t = {}
    for n, (a, b) in lo.items():
        stem = n[: -len(".weight")]
        t[stem + ".lora_a"] = (a, "F32")
        t[stem + ".lora_b"] = (b, "F32")
    write_safetensors(path / "adapters.safetensors", t)
    (path / "adapter_config.json").write_text(
        json.dumps({"fine_tune_type": "lora", "lora_parameters": {"scale": scale}})
    )


def peft_adapter(path, lo, r, alpha):
    path.mkdir()
    t = {}
    for n, (a, b) in lo.items():
        stem = "base_model.model." + n[: -len(".weight")]
        t[stem + ".lora_A.weight"] = (a.T, "F32")
        t[stem + ".lora_B.weight"] = (b.T, "F32")
    write_safetensors(path / "adapter_model.safetensors", t)
    (path / "adapter_config.json").write_text(json.dumps({"r": r, "lora_alpha": alpha}))


def test_bf16_reader_roundtrip(base):
    ck = Checkpoint(base)
    x = ck.get(NAMES[0])
    assert x.dtype == np.float32 and x.shape == (64, 128)
    assert np.array_equal(round_to(x, "BF16"), x)


def test_mlx_and_peft_adapters_agree(base, tmp_path):
    lo = lora()
    mlx_adapter(tmp_path / "mlx", lo, scale=2.0)
    peft_adapter(tmp_path / "peft", lo, r=4, alpha=8)
    fmts = [parse("mlx:4"), parse("mlx:3")]
    _, a = noise_to_signal(Checkpoint(base), str(tmp_path / "mlx"), fmts)
    _, b = noise_to_signal(Checkpoint(base), str(tmp_path / "peft"), fmts)
    for k in a:
        assert a[k]["noise_to_signal"] == pytest.approx(
            b[k]["noise_to_signal"], rel=1e-6
        )


def test_smaller_updates_have_higher_noise(base, tmp_path):
    mlx_adapter(tmp_path / "big", lora(size=0.1), scale=2.0)
    mlx_adapter(tmp_path / "small", lora(size=0.01), scale=2.0)
    fmts = [parse("mlx:4")]
    _, big = noise_to_signal(Checkpoint(base), str(tmp_path / "big"), fmts)
    _, small = noise_to_signal(Checkpoint(base), str(tmp_path / "small"), fmts)
    label = fmts[0][0]
    assert small[label]["noise_to_signal"] > big[label]["noise_to_signal"]


def test_identical_model_raises(base):
    with pytest.raises(ValueError):
        noise_to_signal(Checkpoint(base), str(base), [parse("mlx:4")])


def test_full_model_folder_is_supported(base, tmp_path):
    ft = tmp_path / "ft"
    ft.mkdir()
    ck = Checkpoint(base)
    rng = np.random.default_rng(3)
    write_safetensors(
        ft / "model.safetensors",
        {
            n: (
                ck.get(n) + rng.standard_normal((64, 128)).astype(np.float32) * 0.01,
                "BF16",
            )
            for n in NAMES
        },
    )
    assert len(list(pairs(ck, str(ft)))) == 2


def test_prediction_only_within_tested_scope():
    from ftquant.check import predict

    assert predict(0.989, 5.52, 1.2067, "mlx-q3") == pytest.approx(0.414, abs=0.01)
    assert predict(1.0, 0.5, 0.004, "mlx-q8") is None
    assert predict(1.0, 0.5, None, "mlx-q4") is None
    assert predict(1.0, 0.5, 0.1, None) is None
