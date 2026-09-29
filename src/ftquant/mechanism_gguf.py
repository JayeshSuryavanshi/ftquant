import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs" / "week1"
WORK = ROOT / "runs" / "mechanism-gguf"
CONVERT = ROOT / "vendor" / "llama.cpp" / "convert_hf_to_gguf.py"
TYPES = ["Q8_0", "Q6_K", "Q4_K_M", "Q3_K_M", "Q2_K"]
LINEAR = (
    "attn_q.",
    "attn_k.",
    "attn_v.",
    "attn_output.",
    "ffn_gate.",
    "ffn_up.",
    "ffn_down.",
)


def sh(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True)


def bf16_gguf(cfg: str, model_id: str, runs: Path) -> Path:
    out = WORK / cfg
    out.mkdir(parents=True, exist_ok=True)
    path = out / "bf16.gguf"
    if path.exists():
        return path
    tmp = path.with_suffix(".partial")
    if cfg.endswith("-base"):
        src = Path(snapshot_download(model_id))
        sh(
            [
                sys.executable,
                str(CONVERT),
                str(src),
                "--outtype",
                "bf16",
                "--outfile",
                str(tmp),
            ]
        )
        tmp.rename(path)
        return path
    fused = out / "fused"
    sh(
        [
            sys.executable,
            "-m",
            "mlx_lm",
            "fuse",
            "--model",
            model_id,
            "--adapter-path",
            str(runs / cfg / "adapter"),
            "--save-path",
            str(fused),
        ]
    )
    sh(
        [
            sys.executable,
            str(CONVERT),
            str(fused),
            "--outtype",
            "bf16",
            "--outfile",
            str(tmp),
        ]
    )
    shutil.rmtree(fused, ignore_errors=True)
    tmp.rename(path)
    return path


def quantized(bf16: Path, qtype: str) -> Path:
    p = bf16.with_name(f"{qtype}.gguf")
    if not p.exists():
        tmp = p.with_suffix(".partial")
        sh(["llama-quantize", str(bf16), str(tmp), qtype])
        tmp.rename(p)
    return p


def index(path: Path) -> dict:
    return {
        t.name: t
        for t in GGUFReader(str(path)).tensors
        if t.name.endswith(".weight") and any(k in t.name for k in LINEAR)
    }


def vec(t) -> np.ndarray:
    return dequantize(t.data, t.tensor_type).astype(np.float32).reshape(-1)


def param_count(path: Path) -> int:
    return sum(int(np.prod(t.shape)) for t in GGUFReader(str(path)).tensors)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", default="week1")
    args = ap.parse_args()
    runs = RUNS.parent / args.run
    base_bf16 = bf16_gguf(args.base, args.model, runs)
    n = param_count(base_bf16)
    bpw = {"bf16": round(base_bf16.stat().st_size * 8 / n, 3)}
    base_q = {}
    for t in TYPES:
        base_q[t] = quantized(base_bf16, t)
        bpw[t] = round(base_q[t].stat().st_size * 8 / n, 3)
    b_bf16 = index(base_bf16)
    b_q = {t: index(p) for t, p in base_q.items()}
    results = {"model": args.model, "params": n, "bpw": bpw, "configs": {}}
    for cfg in args.configs:
        ft_bf16 = bf16_gguf(cfg, args.model, runs)
        f_bf16 = index(ft_bf16)
        names = list(f_bf16)
        den = 0.0
        for name in names:
            d = vec(f_bf16[name]) - vec(b_bf16[name])
            den += float(d @ d)
        row = {"delta_norm": den**0.5}
        for t in TYPES:
            ft_t = quantized(ft_bf16, t)
            f_q = index(ft_t)
            dot = qq = 0.0
            for name in names:
                d = vec(f_bf16[name]) - vec(b_bf16[name])
                dq = vec(f_q[name]) - vec(b_q[t][name])
                dot += float(dq @ d)
                qq += float(dq @ dq)
            row[t] = {
                "signal_retained": dot / den,
                "noise_to_signal": max(qq - dot * dot / den, 0.0) ** 0.5 / den**0.5,
            }
            del f_q
            ft_t.unlink(missing_ok=True)
        results["configs"][cfg] = row
        print(
            cfg,
            {
                t: (
                    round(row[t]["signal_retained"], 3),
                    round(row[t]["noise_to_signal"], 2),
                )
                for t in TYPES
            },
            flush=True,
        )
        del f_bf16
        shutil.rmtree(WORK / cfg, ignore_errors=True)
    Path(args.out).write_text(json.dumps(results, indent=1))
    print("bpw", bpw)
    shutil.rmtree(WORK / args.base, ignore_errors=True)


if __name__ == "__main__":
    main()
