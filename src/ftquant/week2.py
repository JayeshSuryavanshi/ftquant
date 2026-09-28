import json
from pathlib import Path

import numpy as np

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
CONFIGS = {
    "q06-lora-lr3e-6": ("0.6B", "LoRA", 3e-6, 0),
    "q06-lora-lowlr": ("0.6B", "LoRA", 1e-5, 0),
    "q06-lora-lowlr-s1": ("0.6B", "LoRA", 1e-5, 1),
    "q06-lora-lr3e-5": ("0.6B", "LoRA", 3e-5, 0),
    "q06-lora": ("0.6B", "LoRA", 1e-4, 0),
    "q06-lora-s1": ("0.6B", "LoRA", 1e-4, 1),
    "q06-lora-lr3e-4": ("0.6B", "LoRA", 3e-4, 0),
    "q06-full-lr3e-6": ("0.6B", "full", 3e-6, 0),
    "q06-full": ("0.6B", "full", 1e-5, 0),
    "q06-full-lr3e-5": ("0.6B", "full", 3e-5, 0),
    "q17-lora-lowlr": ("1.7B", "LoRA", 1e-5, 0),
    "q17-lora": ("1.7B", "LoRA", 1e-4, 0),
}


def delta_norms() -> dict[str, float]:
    out = {}
    for run in ("week1", "week2"):
        for tag in ("q06", "q17"):
            p = RUNS / run / f"mechanism-{tag}-mlx.json"
            if p.exists():
                out.update(
                    {r["config"]: r["delta_norm"] for r in json.loads(p.read_text())}
                )
    return out


def spearman(x: list[float], y: list[float]) -> float:
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    rows = json.loads((RUNS / "analysis-all.json").read_text())["retention"]
    by = {(r["config"], r["variant"]): r for r in rows}
    dn = delta_norms()
    table = []
    for cfg, (model, kind, lr, seed) in CONFIGS.items():
        fp_m, fp_g = by[(cfg, "mlx-bf16")], by[(cfg, "gguf-bf16")]
        rec = {
            "config": cfg,
            "model": model,
            "kind": kind,
            "lr": lr,
            "seed": seed,
            "delta_norm": dn.get(cfg),
            "acc_ft_bf16": fp_m["acc_ft"],
            "acc_base_bf16": fp_m["acc_base"],
            "gain_retention": {},
            "accuracy_retention": {},
            "coverage_change_pts": {},
        }
        for v in VARIANTS:
            r = by.get((cfg, v))
            if r is None:
                continue
            fp = fp_g if v.startswith("gguf") else fp_m
            rec["gain_retention"][v] = r["retention"]
            rec["accuracy_retention"][v] = (
                r["acc_ft"] / fp["acc_ft"] if fp["acc_ft"] else float("nan")
            )
            rec["coverage_change_pts"][v] = 100 * (r["q_coverage"] - r["fp_coverage"])
        table.append(rec)

    trend = {}
    for kind in ("LoRA", "full"):
        for v in ("mlx-q4", "mlx-q3", "gguf-Q4_K_M", "gguf-Q3_K_M"):
            pts = [
                (t["lr"], t["gain_retention"][v])
                for t in table
                if t["model"] == "0.6B"
                and t["kind"] == kind
                and t["seed"] == 0
                and not np.isnan(t["gain_retention"].get(v, np.nan))
            ]
            trend[f"{kind} {v}"] = {
                "n": len(pts),
                "spearman_lr_vs_R": spearman(*zip(*pts)),
            }

    (RUNS / "week2" / "summary.json").write_text(
        json.dumps({"configs": table, "lr_trend": trend}, indent=1)
    )

    print(
        f"{'config':18s} {'dNorm':>6s} {'acc':>5s} | gain retention: "
        + " ".join(f"{v[4:]:>7s}" for v in VARIANTS)
    )
    for t in table:
        gr = " ".join(f"{t['gain_retention'].get(v, np.nan):7.3f}" for v in VARIANTS)
        print(
            f"{t['config']:18s} {t['delta_norm'] or 0:6.1f} {t['acc_ft_bf16']:5.3f} | {gr}"
        )
    print("\naccuracy retention (acc_ft(q) / acc_ft(bf16), same engine)")
    for t in table:
        ar = " ".join(
            f"{t['accuracy_retention'].get(v, np.nan):7.3f}" for v in VARIANTS
        )
        print(f"{t['config']:18s} {ar}")
    print("\ncoverage change at fixed tau (pts)")
    for t in table:
        cc = " ".join(
            f"{t['coverage_change_pts'].get(v, np.nan):7.1f}" for v in VARIANTS
        )
        print(f"{t['config']:18s} {cc}")
    print("\nlearning-rate trend (Spearman, seed 0, 0.6B)")
    for k, v in trend.items():
        print(f"  {k:18s} n={v['n']}  rho={v['spearman_lr_vs_R']:+.2f}")


if __name__ == "__main__":
    main()
