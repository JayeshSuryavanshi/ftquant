import argparse
import json
import shutil
from pathlib import Path

import gguf
import mlx.core as mx
import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize


def dequantized(
    quant: Path, ref: Path, arch: gguf.MODEL_ARCH, n_layers: int
) -> dict[str, mx.array]:
    names = gguf.get_tensor_name_map(arch, n_layers)
    ref_w = mx.load(str(ref / "model.safetensors"))
    tensors = {t.name: t for t in GGUFReader(str(quant)).tensors}
    out = {}
    for hf, arr in ref_w.items():
        t = tensors.pop(names.get_name(hf, try_suffixes=(".weight", ".bias")))
        values = dequantize(t.data, t.tensor_type).astype(np.float32).reshape(arr.shape)
        out[hf] = mx.array(values).astype(mx.bfloat16)
    if tensors:
        raise SystemExit(
            f"tensors in {quant.name} with no counterpart in {ref}: {sorted(tensors)}"
        )
    return out


def build(
    quant: Path, ref: Path, dst: Path, arch: gguf.MODEL_ARCH, n_layers: int
) -> None:
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True)
    for f in ref.iterdir():
        if f.is_file() and f.name != "model.safetensors":
            shutil.copyfile(f, dst / f.name)
    mx.save_safetensors(
        str(dst / "model.safetensors"),
        dequantized(quant, ref, arch, n_layers),
        metadata={"format": "mlx"},
    )


def verify(
    quant: Path, ref: Path, dst: Path, arch: gguf.MODEL_ARCH, n_layers: int
) -> dict:
    saved = mx.load(str(dst / "model.safetensors"))
    expected = dequantized(quant, ref, arch, n_layers)
    unequal = [
        k
        for k, v in expected.items()
        if k not in saved or not mx.array_equal(saved[k], v).item()
    ]
    extra = sorted(set(saved) - set(expected))
    return {
        "tensors": len(expected),
        "unequal": unequal,
        "extra": extra,
        "passed": not unequal and not extra,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="QWEN3")
    ap.add_argument("--layers", type=int, default=28)
    args = ap.parse_args()
    arch = gguf.MODEL_ARCH[args.arch]
    build(Path(args.quant), Path(args.ref), Path(args.out), arch, args.layers)
    print(
        json.dumps(
            verify(Path(args.quant), Path(args.ref), Path(args.out), arch, args.layers)
        )
    )


if __name__ == "__main__":
    main()
