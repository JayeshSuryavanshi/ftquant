import json
from pathlib import Path

RUNS = Path(__file__).resolve().parents[2] / "runs"
VARIANTS = [
    "mlx-q6",
    "mlx-q4",
    "mlx-q3",
    "gguf-Q6_K",
    "gguf-Q4_K_M",
    "gguf-Q3_K_M",
    "gguf-Q2_K",
]
SETTINGS = {
    "Qwen3-0.6B, banking77": (
        "analysis-all.json",
        "q06-lora-lowlr",
        "q06-lora",
        "q06-full",
    ),
    "Qwen3-1.7B, banking77": ("analysis-all.json", "q17-lora-lowlr", "q17-lora", None),
    "OLMo-2 1B, banking77": (
        "week3/analysis.json",
        "olmo1-lora-lowlr",
        "olmo1-lora",
        None,
    ),
    "Qwen3-0.6B, MASSIVE": (
        "week3/analysis.json",
        "mas06-lora-lowlr",
        "mas06-lora",
        "mas06-full",
    ),
}


def rows(path: str) -> dict[tuple[str, str], dict]:
    return {
        (r["config"], r["variant"]): r
        for r in json.loads((RUNS / path).read_text())["retention"]
    }


def summary(by: dict, cfg: str) -> dict:
    fp = {e: by[(cfg, f"{e}-bf16")] for e in ("mlx", "gguf")}
    out = {
        "acc_bf16": fp["mlx"]["acc_ft"],
        "gain": {},
        "accuracy": {},
        "coverage_pts": {},
    }
    for v in VARIANTS:
        r = by.get((cfg, v))
        if r is None:
            continue
        ref = fp["gguf" if v.startswith("gguf") else "mlx"]
        out["gain"][v] = r["retention"]
        out["accuracy"][v] = r["acc_ft"] / ref["acc_ft"]
        out["coverage_pts"][v] = 100 * (r["q_coverage"] - r["fp_coverage"])
    return out


def main() -> None:
    res = {}
    for name, (path, gentle, strong, full) in SETTINGS.items():
        by = rows(path)
        res[name] = {
            label: summary(by, cfg)
            for label, cfg in (
                ("LoRA 1e-5", gentle),
                ("LoRA 1e-4", strong),
                ("full 1e-5", full),
            )
            if cfg
        }
    (RUNS / "week3" / "summary.json").write_text(json.dumps(res, indent=1))
    head = " ".join(f"{v.split('-', 1)[1]:>7s}" for v in VARIANTS)
    for name, cfgs in res.items():
        print(f"\n{name}")
        for metric in ("gain", "accuracy", "coverage_pts"):
            print(f"  {metric:13s} {'bf16 acc':>8s} {head}")
            for label, s in cfgs.items():
                fmt = "{:7.1f}" if metric == "coverage_pts" else "{:7.3f}"
                vals = " ".join(
                    fmt.format(s[metric][v]) if v in s[metric] else "      -"
                    for v in VARIANTS
                )
                print(f"  {label:13s} {s['acc_bf16']:8.3f} {vals}")


if __name__ == "__main__":
    main()
