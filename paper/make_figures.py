import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42

import ftquant.plots as P  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
PRED_XLABEL = "Predicted from the weights (%)"
PRED_ARMS_W4 = {
    "olmo1": ("OLMo-2 1B, banking77, seed 1", P.NEUTRAL, "s"),
    "mas06": ("Qwen3-0.6B, MASSIVE, seed 1", P.ACCENT, "o"),
    "q4b": ("Qwen3-4B, banking77", P.INK2, "^"),
}


def replication_data() -> dict:
    pairs = json.loads((P.RUNS / "revision" / "revision.json").read_text())["pairs"]["pairs"]
    data = {}
    for name, p in pairs.items():
        bf = p["bf16"]["mlx-bf16"]
        data[name] = {
            label: {
                "acc_bf16": bf[role],
                "accuracy": {v: c[f"A_{role}"] for v, c in p["formats"].items()},
            }
            for label, role in (("LoRA 1e-5", "gentle"), ("LoRA 1e-4", "strong"))
        }
    return data


def main() -> None:
    OUT.mkdir(exist_ok=True)
    P.GROUND = "#ffffff"
    P.CONFIG_STYLE["lora-lowlr"] = (P.CONFIG_STYLE["lora-lowlr"][0], P.WARM)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unknown infodict keyword")
        P.retention_chart("q06", OUT / "retention-q06.pdf")
        P.lr_sweep_chart(OUT / "lr-sweep.pdf")
        P.mechanism_chart(
            "q06",
            OUT / "mechanism-q06.pdf",
            note="Circles: MLX   Diamonds: GGUF   (MLX 2-bit and GGUF Q2_K not shown)",
            xlabel="Noise-to-update ratio, NSR (log scale)",
        )
        P.replication_chart(OUT / "replication.pdf", replication_data())
        for run, arms in (("week3", None), ("week4", PRED_ARMS_W4)):
            P.predictor_v2_chart(
                OUT / f"predictor-v2-{run}.pdf",
                run,
                arms,
                figsize=(4.2, 3.9),
                xlabel=PRED_XLABEL,
                ylabel="Measured gain retained (%)",
                show_legend=False,
            )
    print(sorted(p.name for p in OUT.glob("*.pdf")))


if __name__ == "__main__":
    main()
