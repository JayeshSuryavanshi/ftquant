import argparse
import json
import shutil
from pathlib import Path

from ftquant.check import Checkpoint, noise_to_signal, resolve
from ftquant.check_gguf import run as gguf_run
from ftquant.mechanism_gguf import WORK, bf16_gguf
from ftquant.quant import parse

RUNS = Path(__file__).resolve().parents[2] / "runs"


def mlx(model: str, configs: list[str], bits: list[int], runs: Path) -> list[dict]:
    base = Checkpoint(resolve(model))
    out = []
    for cfg in configs:
        formats = [parse(f"mlx:{b}") for b in bits]
        _, res = noise_to_signal(base, str(runs / cfg / "adapter"), formats)
        out.append(
            {
                "config": cfg,
                "signal_retained": {
                    str(b): res[f[0]]["signal_retained"] for b, f in zip(bits, formats)
                },
                "noise_to_signal": {
                    str(b): res[f[0]]["noise_to_signal"] for b, f in zip(bits, formats)
                },
            }
        )
        print(cfg, out[-1], flush=True)
    return out


def gguf(
    model: str, base_cfg: str, configs: list[str], types: list[str], runs: Path
) -> dict:
    base = bf16_gguf(base_cfg, model, runs)
    res: dict = {"model": model, "configs": {}}
    for cfg in configs:
        ft = bf16_gguf(cfg, model, runs)
        rows = gguf_run(str(base), str(ft), types)["results"]
        res["configs"][cfg] = {
            t: {
                "signal_retained": r["signal_retained"],
                "noise_to_signal": r["noise_to_signal"],
            }
            for t, r in zip(types, rows)
        }
        print(cfg, res["configs"][cfg], flush=True)
        shutil.rmtree(WORK / cfg, ignore_errors=True)
    shutil.rmtree(WORK / base_cfg, ignore_errors=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Low-memory, low-disk NSR and kept for large bases."
    )
    ap.add_argument("--model", required=True)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--bits", nargs="+", type=int, default=[6, 4, 3])
    ap.add_argument("--base", help="base config name for GGUF; omit to skip GGUF")
    ap.add_argument("--types", nargs="+", default=["Q6_K", "Q4_K_M", "Q3_K_M", "Q2_K"])
    ap.add_argument("--mlx-out")
    ap.add_argument("--gguf-out")
    args = ap.parse_args()
    runs = RUNS / args.run
    if args.mlx_out:
        Path(args.mlx_out).write_text(
            json.dumps(mlx(args.model, args.configs, args.bits, runs), indent=1)
        )
    if args.gguf_out and args.base:
        Path(args.gguf_out).write_text(
            json.dumps(
                gguf(args.model, args.base, args.configs, args.types, runs), indent=1
            )
        )


if __name__ == "__main__":
    main()
