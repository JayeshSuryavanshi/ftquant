import argparse
import json
import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from ftquant.quant import Roundtrip, parse
from ftquant.tensors import Checkpoint, SafeTensors, round_to

RESOURCES = Path(__file__).resolve().parent / "resources"
LINEAR = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def resolve(model: str) -> Path:
    p = Path(model)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model, allow_patterns=["*.safetensors", "*.json"]))


def is_linear(name: str) -> bool:
    parts = name.split(".")
    return name.endswith(".weight") and len(parts) > 2 and parts[-2] in LINEAR


def pairs(
    base: Checkpoint, finetuned: str
) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    p = Path(finetuned)
    if (p / "adapters.safetensors").exists():
        cfg = json.loads((p / "adapter_config.json").read_text())
        ad = SafeTensors(p / "adapters.safetensors")
        if cfg.get("fine_tune_type") == "full":
            for n in ad.names():
                if n in base.where and is_linear(n):
                    yield n, base.get(n), ad.get(n)
            return
        scale = float(cfg["lora_parameters"]["scale"])
        for n in ad.names():
            if n.endswith(".lora_a"):
                stem = n[: -len(".lora_a")]
                w = base.get(stem + ".weight")
                a, b = ad.get(n), ad.get(stem + ".lora_b")
                yield stem + ".weight", w, w + scale * (b.T @ a.T)
        return
    peft = list(p.glob("adapter_model*.safetensors")) if p.exists() else []
    if peft:
        cfg = json.loads((p / "adapter_config.json").read_text())
        r = float(cfg["r"])
        alpha = float(cfg.get("lora_alpha", r))
        scale = alpha / math.sqrt(r) if cfg.get("use_rslora") else alpha / r
        ad = SafeTensors(peft[0])
        for n in ad.names():
            if ".lora_A." in n:
                stem, tail = n.split(".lora_A.")
                name = stem.removeprefix("base_model.model.") + ".weight"
                if name not in base.where:
                    continue
                w = base.get(name)
                yield (
                    name,
                    w,
                    w + scale * (ad.get(stem + ".lora_B." + tail) @ ad.get(n)),
                )
        return
    ft = Checkpoint(resolve(finetuned))
    for n in base.names():
        if is_linear(n) and n in ft.where:
            yield n, base.get(n), ft.get(n)


def noise_to_signal(
    base: Checkpoint, finetuned: str, formats: list[tuple[str, Roundtrip]]
) -> tuple[int, dict]:
    den = 0.0
    dot = {f: 0.0 for f, _ in formats}
    qq = {f: 0.0 for f, _ in formats}
    layers = 0
    for name, wb, wf in pairs(base, finetuned):
        wf = round_to(wf, base.dtype(name))
        d = wf - wb
        dd = float(np.vdot(d, d))
        if dd == 0:
            continue
        layers += 1
        den += dd
        for label, rt in formats:
            dq = rt(wf) - rt(wb)
            dot[label] += float(np.vdot(dq, d))
            qq[label] += float(np.vdot(dq, dq))
    if den == 0:
        raise ValueError("the fine-tuned weights are identical to the base weights")
    out = {
        f: {
            "signal_retained": dot[f] / den,
            "noise_to_signal": max(qq[f] - dot[f] ** 2 / den, 0.0) ** 0.5 / den**0.5,
        }
        for f, _ in formats
    }
    return layers, out


def load_resource(name: str) -> dict:
    p = RESOURCES / name
    return json.loads(p.read_text()) if p.exists() else {}


def reading(kept: float, nsr: float, base_kld: float | None = None) -> str:
    if kept < 0.6:
        return "update rounded away"
    noise = "low noise" if nsr < 3 else "moderate noise" if nsr < 5 else "high noise"
    return noise + (
        ", base model breaks" if base_kld is not None and base_kld >= 1 else ""
    )


def predict(
    kept: float, nsr: float, base_kld: float | None, key: str | None
) -> float | None:
    model = load_resource("predictor.json")
    if not model or base_kld is None or key not in model["tested_formats"]:
        return None
    t = model["theta"]
    z = (
        t[0]
        + t[1] * (math.log(max(kept, 1e-6)) - math.log(max(nsr, 1e-6)))
        + t[2] * math.log(max(base_kld, 1e-6))
    )
    return 1.0 / (1.0 + math.exp(-z))


def kld_key(spec: str) -> str | None:
    s = spec.strip().lower()
    if s.startswith("mlx:"):
        bits, _, group = s[4:].partition("g")
        return f"mlx-q{bits}" if group in ("", "64") else None
    if s.startswith("gguf:"):
        return "gguf-" + spec.split(":", 1)[1].upper()
    return None


def run(base_model: str, finetuned: str, specs: list[str]) -> dict:
    formats = [parse(s) for s in specs]
    base = Checkpoint(resolve(base_model))
    layers, res = noise_to_signal(base, finetuned, formats)
    kld = load_resource("kld.json").get(base_model.split("/")[-1], {})
    rows = []
    for spec, (label, _) in zip(specs, formats):
        nsr = res[label]["noise_to_signal"]
        kept = res[label]["signal_retained"]
        key = kld_key(spec)
        k = kld.get(key)
        rows.append(
            {
                "format": label,
                "noise_to_signal": nsr,
                "signal_retained": kept,
                "base_kld": k,
                "predicted_retention": predict(kept, nsr, k, key),
                "reading": reading(kept, nsr, k),
            }
        )
    return {
        "base": base_model,
        "finetuned": finetuned,
        "layers": layers,
        "results": rows,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="ftquant check",
        description="Measure, before quantizing, how much rounding noise a quantization format adds to a fine-tune's update.",
    )
    ap.add_argument(
        "--base", required=True, help="base model: Hugging Face id or local folder"
    )
    ap.add_argument(
        "--finetuned",
        required=True,
        help="MLX adapter folder, PEFT adapter folder, or fine-tuned model",
    )
    ap.add_argument(
        "--formats",
        nargs="+",
        default=["mlx:8", "mlx:6", "mlx:4", "mlx:3"],
        help="mlx:BITS[gGROUP] or gguf:Q8_0|Q5_0|Q4_0|Q4_1",
    )
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    out = run(args.base, args.finetuned, args.formats)
    if args.json:
        print(json.dumps(out, indent=1))
        return
    print(f"{out['layers']} fine-tuned linear layers compared against {args.base}\n")
    print_table(out["results"], 26)


def print_table(results: list[dict], width: int) -> None:
    print(
        f"{'format':{width}s} {'noise/update':>12s} {'update kept':>12s} {'base damage':>12s} {'predicted gain kept':>20s}  reading"
    )
    for x in results:
        k = f"{x['base_kld']:.3f}" if x["base_kld"] is not None else "n/a"
        r = x["predicted_retention"]
        r = f"{100 * r:.0f}%" if r is not None else "n/a"
        print(
            f"{x['format']:{width}s} {x['noise_to_signal']:12.2f} {100 * x['signal_retained']:11.0f}% {k:>12s} {r:>20s}  {x['reading']}"
        )
    if any(x["predicted_retention"] is None for x in results):
        print(
            "\nn/a: the prediction is shown only for the formats it was tested on (MLX 6/4/3-bit group 64,"
            " GGUF Q6_K/Q4_K_M/Q3_K_M/Q2_K) and bases with a measured base damage."
        )


if __name__ == "__main__":
    main()
