import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

LINEAR = (
    "attn_q.",
    "attn_k.",
    "attn_v.",
    "attn_output.",
    "ffn_gate.",
    "ffn_up.",
    "ffn_down.",
)


def _index(path: Path) -> dict:
    from gguf import GGUFReader

    return {
        t.name: t
        for t in GGUFReader(str(path)).tensors
        if t.name.endswith(".weight") and any(k in t.name for k in LINEAR)
    }


def _vec(t) -> np.ndarray:
    from gguf.quants import dequantize

    return dequantize(t.data, t.tensor_type).astype(np.float32).reshape(-1)


def _quantize(src: Path, dst: Path, qtype: str, llama_quantize: str) -> None:
    subprocess.run(
        [llama_quantize, str(src), str(dst), qtype], check=True, capture_output=True
    )


def run(
    base_gguf: str,
    ft_gguf: str,
    types: list[str],
    llama_quantize: str = "llama-quantize",
) -> dict:
    if shutil.which(llama_quantize) is None:
        raise FileNotFoundError(
            f"{llama_quantize} not found; install llama.cpp or pass --llama-quantize"
        )
    base, ft = Path(base_gguf), Path(ft_gguf)
    b16, f16 = _index(base), _index(ft)
    names = [n for n in f16 if n in b16]
    den = 0.0
    for n in names:
        d = _vec(f16[n]) - _vec(b16[n])
        den += float(d @ d)
    if den == 0:
        raise ValueError("the two GGUF files have identical linear weights")
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for t in types:
            bq, fq = Path(tmp) / f"base-{t}.gguf", Path(tmp) / f"ft-{t}.gguf"
            _quantize(base, bq, t, llama_quantize)
            _quantize(ft, fq, t, llama_quantize)
            bi, fi = _index(bq), _index(fq)
            dot = qq = 0.0
            for n in names:
                d = _vec(f16[n]) - _vec(b16[n])
                dq = _vec(fi[n]) - _vec(bi[n])
                dot += float(dq @ d)
                qq += float(dq @ dq)
            nsr = max(qq - dot * dot / den, 0.0) ** 0.5 / den**0.5
            rows.append(
                {
                    "format": f"GGUF {t}",
                    "noise_to_signal": nsr,
                    "signal_retained": dot / den,
                }
            )
            bq.unlink(missing_ok=True)
            fq.unlink(missing_ok=True)
    return {"layers": len(names), "results": rows}


def main(argv: list[str] | None = None) -> None:
    from ftquant.check import reading

    ap = argparse.ArgumentParser(
        prog="ftquant check-gguf",
        description="Noise-to-update ratio for llama.cpp k-quants, from bf16/f16 GGUF files of the base and the fine-tune.",
    )
    ap.add_argument("--base-gguf", required=True)
    ap.add_argument("--finetuned-gguf", required=True)
    ap.add_argument(
        "--types", nargs="+", default=["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M"]
    )
    ap.add_argument("--llama-quantize", default="llama-quantize")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    out = run(args.base_gguf, args.finetuned_gguf, args.types, args.llama_quantize)
    if args.json:
        print(json.dumps(out, indent=1))
        return
    print(f"{out['layers']} linear tensors compared\n")
    print(f"{'format':14s} {'noise/update':>12s} {'update kept':>12s}  reading")
    for x in out["results"]:
        print(
            f"{x['format']:14s} {x['noise_to_signal']:12.2f} {100 * x['signal_retained']:11.0f}%  {reading(x['signal_retained'], x['noise_to_signal'])}"
        )


if __name__ == "__main__":
    main()
