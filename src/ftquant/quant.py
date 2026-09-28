from collections.abc import Callable

import numpy as np

Roundtrip = Callable[[np.ndarray], np.ndarray]


def mlx_affine(bits: int, group: int = 64) -> Roundtrip:
    n_bins = 2**bits - 1

    def roundtrip(w: np.ndarray) -> np.ndarray:
        g = w.reshape(-1, group).astype(np.float32)
        w_max = g.max(axis=1, keepdims=True)
        w_min = g.min(axis=1, keepdims=True)
        mask = np.abs(w_min) > np.abs(w_max)
        scales = np.maximum((w_max - w_min) / n_bins, 1e-7)
        scales = np.where(mask, scales, -scales)
        edge = np.where(mask, w_min, w_max)
        q0 = np.round(edge / scales)
        scales = np.where(q0 != 0, edge / np.where(q0 != 0, q0, 1), scales)
        biases = np.where(q0 == 0, 0, edge)
        q = np.clip(np.round((g - biases) / scales), 0, n_bins)
        return (q * scales + biases).reshape(w.shape)

    return roundtrip


def gguf_legacy(name: str) -> Roundtrip:
    from gguf.constants import GGMLQuantizationType
    from gguf.quants import dequantize, quantize

    qtype = getattr(GGMLQuantizationType, name)

    def roundtrip(w: np.ndarray) -> np.ndarray:
        return dequantize(quantize(w.astype(np.float32), qtype), qtype).reshape(w.shape)

    return roundtrip


def parse(spec: str) -> tuple[str, Roundtrip]:
    s = spec.strip().lower()
    if s.startswith("mlx:"):
        parts = s[4:].split("g")
        bits = int(parts[0])
        group = int(parts[1]) if len(parts) > 1 else 64
        return f"MLX {bits}-bit (group {group})", mlx_affine(bits, group)
    if s.startswith("gguf:"):
        name = spec.split(":", 1)[1].upper()
        if name not in {"Q8_0", "Q5_0", "Q4_0", "Q4_1", "Q5_1"}:
            raise ValueError(
                f"{name}: k-quants need llama.cpp; use 'ftquant check-gguf' with converted GGUF files"
            )
        return f"GGUF {name}", gguf_legacy(name)
    raise ValueError(f"unknown format '{spec}', use e.g. mlx:4, mlx:3g32, gguf:Q8_0")
