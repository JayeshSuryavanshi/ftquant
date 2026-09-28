import argparse
import json
from pathlib import Path

import mlx.core as mx
import yaml
from huggingface_hub import snapshot_download

RUNS = Path(__file__).resolve().parents[2] / "runs" / "week1"


def base_weights(model_id: str) -> dict[str, mx.array]:
    d = Path(snapshot_download(model_id))
    w: dict[str, mx.array] = {}
    for f in sorted(d.glob("*.safetensors")):
        w.update(mx.load(str(f)))
    return w


def fine_tuned_weights(cfg: str, base: dict[str, mx.array], runs: Path = RUNS) -> dict[str, mx.array]:
    run = runs / cfg
    train = yaml.safe_load((run / "train.yaml").read_text())
    ad = mx.load(str(run / "adapter" / "adapters.safetensors"))
    out: dict[str, mx.array] = {}
    if train["fine_tune_type"] == "full":
        for k, v in ad.items():
            if k.endswith(".weight") and k in base and base[k].ndim == 2 and "embed" not in k:
                out[k] = v
        return out
    scale = float(train["lora_parameters"]["scale"])
    for k in ad:
        if k.endswith(".lora_a"):
            stem = k[: -len(".lora_a")]
            a = ad[k].astype(mx.float32)
            b = ad[stem + ".lora_b"].astype(mx.float32)
            wk = stem + ".weight"
            out[wk] = (base[wk].astype(mx.float32) + scale * (b.T @ a.T)).astype(base[wk].dtype)
    return out


def affine_roundtrip(w: mx.array, bits: int, group: int = 64) -> mx.array:
    q, s, b = mx.quantize(w, group_size=group, bits=bits)
    return mx.dequantize(q, s, b, group_size=group, bits=bits)


def surviving_fraction(cfg: str, model_id: str, bits_list: list[int], runs: Path = RUNS) -> dict:
    mx.set_default_device(mx.cpu)
    base = base_weights(model_id)
    ft = fine_tuned_weights(cfg, base, runs)
    dot = {b: 0.0 for b in bits_list}
    qq = {b: 0.0 for b in bits_list}
    den = 0.0
    step_ratio = {b: [] for b in bits_list}
    for k, wft in ft.items():
        wb = base[k].astype(mx.float32)
        wf = wft.astype(mx.float32)
        d = wf - wb
        den += float(mx.sum(d * d))
        for bits in bits_list:
            dq = affine_roundtrip(wf, bits) - affine_roundtrip(wb, bits)
            dot[bits] += float(mx.sum(dq * d))
            qq[bits] += float(mx.sum(dq * dq))
            g = wb.reshape(wb.shape[0], -1, 64)
            step = (mx.max(g, axis=-1) - mx.min(g, axis=-1)) / (2**bits - 1)
            dg = mx.abs(d).reshape(d.shape[0], -1, 64)
            step_ratio[bits].append(float(mx.median(mx.max(dg, axis=-1) / (step + 1e-12))))
        mx.eval(mx.array(0))
    return {"config": cfg, "layers": len(ft), "delta_norm": den ** 0.5,
            "signal_retained": {b: dot[b] / den for b in bits_list},
            "noise_to_signal": {b: max(qq[b] - dot[b] ** 2 / den, 0.0) ** 0.5 / (den ** 0.5) for b in bits_list},
            "median_delta_to_step": {b: sorted(v)[len(v) // 2] for b, v in step_ratio.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--bits", nargs="+", type=int, default=[8, 6, 4, 3, 2])
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", default="week1")
    args = ap.parse_args()
    runs = RUNS.parent / args.run
    res = [surviving_fraction(c, args.model, args.bits, runs) for c in args.configs]
    Path(args.out).write_text(json.dumps(res, indent=1))
    for r in res:
        print(r["config"], "delta_norm", round(r["delta_norm"], 3),
              "| signal kept", {b: round(v, 3) for b, v in r["signal_retained"].items()},
              "| noise/signal", {b: round(v, 2) for b, v in r["noise_to_signal"].items()},
              "| median max|delta|/step", {b: round(v, 3) for b, v in r["median_delta_to_step"].items()})


if __name__ == "__main__":
    main()
