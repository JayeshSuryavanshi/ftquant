import argparse
import json
import subprocess
import sys
from pathlib import Path

import gguf
import mlx.core as mx
import numpy as np
from gguf import GGUFReader

ROOT = Path(__file__).resolve().parents[2]
CONVERT_LORA = ROOT / "vendor" / "llama.cpp" / "convert_lora_to_gguf.py"


def mlx_adapter(
    adapter: Path,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], float, int]:
    params = json.loads((adapter / "adapter_config.json").read_text())[
        "lora_parameters"
    ]
    w = mx.load(str(adapter / "adapters.safetensors"))
    mods = {}
    for k in w:
        if k.endswith(".lora_a"):
            m = k[: -len(".lora_a")]
            a = np.array(w[k].astype(mx.float32))
            b = np.array(w[m + ".lora_b"].astype(mx.float32))
            mods[m] = (a, b)
    return mods, float(params["scale"]), int(params["rank"])


def to_peft(adapter: Path, dst: Path) -> None:
    mods, scale, rank = mlx_adapter(adapter)
    dst.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for m, (a, b) in mods.items():
        tensors[f"base_model.model.{m}.lora_A.weight"] = mx.array(
            np.ascontiguousarray(a.T)
        )
        tensors[f"base_model.model.{m}.lora_B.weight"] = mx.array(
            np.ascontiguousarray(b.T)
        )
    mx.save_safetensors(
        str(dst / "adapter_model.safetensors"), tensors, metadata={"format": "pt"}
    )
    config = {
        "peft_type": "LORA",
        "r": rank,
        "lora_alpha": scale * rank,
        "target_modules": sorted({m.rsplit(".", 1)[-1] for m in mods}),
        "lora_dropout": 0.0,
        "bias": "none",
        "fan_in_fan_out": False,
        "use_rslora": False,
    }
    (dst / "adapter_config.json").write_text(json.dumps(config, indent=1))


def convert(adapter: Path, base: Path, out: Path, work: Path) -> None:
    peft = work / "peft"
    to_peft(adapter, peft)
    cmd = [
        sys.executable,
        str(CONVERT_LORA),
        "--base",
        str(base),
        "--outtype",
        "f32",
        "--outfile",
        str(out),
        str(peft),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"convert_lora_to_gguf failed:\n{p.stderr[-3000:]}")


def check(adapter: Path, lora: Path, arch: gguf.MODEL_ARCH, n_layers: int) -> dict:
    mods, scale, rank = mlx_adapter(adapter)
    reader = GGUFReader(str(lora))
    field = reader.fields["adapter.lora.alpha"]
    alpha = float(field.parts[field.data[0]][0])
    tensors = {t.name: np.array(t.data, dtype=np.float64) for t in reader.tensors}
    names = gguf.get_tensor_name_map(arch, n_layers)
    worst, checked = 0.0, 0
    for m, (a, b) in mods.items():
        g = names.get_name(m + ".weight", try_suffixes=(".weight",))
        ga, gb = tensors.pop(g + ".lora_a"), tensors.pop(g + ".lora_b")
        if ga.shape != (rank, a.shape[0]) or gb.shape != (b.shape[1], rank):
            raise SystemExit(f"{g}: unexpected LoRA shapes {ga.shape}, {gb.shape}")
        delta_gguf = (alpha / ga.shape[0]) * gb @ ga
        delta_mlx = scale * (a.astype(np.float64) @ b.astype(np.float64)).T
        worst = max(
            worst, float(np.abs(delta_gguf - delta_mlx).max() / np.abs(delta_mlx).max())
        )
        checked += 1
    return {
        "modules": checked,
        "alpha": alpha,
        "rank": rank,
        "mlx_scale": scale,
        "max_relative_difference": worst,
        "unmatched_tensors": sorted(tensors),
        "passed": checked == len(mods) and not tensors and worst <= 1e-6,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="QWEN3")
    ap.add_argument("--layers", type=int, default=28)
    args = ap.parse_args()
    out = Path(args.out)
    convert(Path(args.adapter), Path(args.base), out, out.parent)
    print(
        json.dumps(
            check(Path(args.adapter), out, gguf.MODEL_ARCH[args.arch], args.layers)
        )
    )


if __name__ == "__main__":
    main()
