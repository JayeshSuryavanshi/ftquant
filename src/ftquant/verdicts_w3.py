import argparse
import json
import math
from pathlib import Path

import numpy as np

from ftquant.analyze import load, retention
from ftquant.predictor_v2 import FORMS, kld_lookup, mech_lookup, metrics, sigmoid

RUNS = Path(__file__).resolve().parents[2] / "runs"
LADDER = [
    "mlx-q6",
    "mlx-q4",
    "mlx-q3",
    "gguf-Q6_K",
    "gguf-Q4_K_M",
    "gguf-Q3_K_M",
    "gguf-Q2_K",
]
ARMS = {
    "olmo1": ("OLMo-2-0425-1B-Instruct", "olmo1-lora", "olmo1-lora-lowlr"),
    "mas06": ("Qwen3-0.6B", "mas06-lora", "mas06-lora-lowlr"),
}


def arrays(run: Path, cfg: str, base: str, v: str) -> tuple[np.ndarray, ...]:
    fp = "gguf-bf16" if v.startswith("gguf") else "mlx-bf16"
    return tuple(
        x["correct"] if x is not None else None
        for x in (
            load(run, cfg, v),
            load(run, base, v),
            load(run, cfg, fp),
            load(run, base, fp),
        )
    )


def h1_test(
    run: Path, prefix: str, strong: str, gentle: str, v: str, reps: int = 2000
) -> dict:
    base = f"{prefix}-base"
    a, b = arrays(run, strong, base, v), arrays(run, gentle, base, v)
    if any(x is None for x in (*a, *b)):
        return {"verdict": "inconclusive", "reason": "missing evaluation"}
    point = retention(*a) - retention(*b)
    if math.isnan(point):
        return {
            "verdict": "inconclusive",
            "reason": "undefined retention",
            "diff": point,
        }
    rng = np.random.default_rng(0)
    n = len(a[0])
    vals = []
    for _ in range(reps):
        i = rng.integers(0, n, n)
        vals.append(retention(*(x[i] for x in a)) - retention(*(x[i] for x in b)))
    lo, hi = (float(x) for x in np.nanpercentile(vals, [2.5, 97.5]))
    if lo > 0:
        verdict = "supported"
    elif hi < 0 or point <= 0:
        verdict = "not supported"
    else:
        verdict = "inconclusive"
    return {
        "R_strong": retention(*a),
        "R_gentle": retention(*b),
        "diff": point,
        "ci": [lo, hi],
        "verdict": verdict,
    }


def config_ci(pts: list[dict], key: str, reps: int = 2000) -> list[float]:
    y = np.clip(np.array([p["retention"] for p in pts]), 0.0, 1.0)
    err = np.abs(np.array([p[key] for p in pts]) - y)
    groups = [
        err[[p["config"] == c for p in pts]] for c in sorted({p["config"] for p in pts})
    ]
    rng = np.random.default_rng(0)
    vals = [
        np.concatenate(
            [groups[i] for i in rng.integers(0, len(groups), len(groups))]
        ).mean()
        for _ in range(reps)
    ]
    return [float(x) for x in np.percentile(vals, [2.5, 97.5])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="week3")
    ap.add_argument("--analysis", default=None)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run = RUNS / args.run
    analysis = Path(args.analysis) if args.analysis else run / "analysis.json"
    rows = json.loads(analysis.read_text())["retention"]
    v2 = json.loads((RUNS / "predictor" / "predictor-v2.json").read_text())
    v1 = json.loads((RUNS / "predictor" / "predictor-v1.json").read_text())
    arms = {k: ARMS[k] for k in args.arms}

    pts, dropped = [], []
    for prefix, (tag, _, _) in arms.items():
        mech = mech_lookup(args.run, prefix)
        kld = kld_lookup(tag)
        for r in rows:
            if (
                r["run"] != args.run
                or not r["config"].startswith(prefix + "-")
                or r["variant"] not in LADDER
            ):
                continue
            key = (r["config"], r["variant"])
            if math.isnan(r["retention"]):
                dropped.append(key)
                continue
            nsr, kept = mech[key]
            p = {
                "config": r["config"],
                "variant": r["variant"],
                "nsr": nsr,
                "kept": kept,
                "kld": kld[r["variant"]],
                "retention": r["retention"],
            }
            x2 = np.array(FORMS[v2["form"]](p))
            x1 = np.array(FORMS["nsr+kld"](p))
            xk = np.array(FORMS["kld-only"](p))
            p["pred_v2"] = float(sigmoid(x2 @ np.array(v2["theta"])))
            p["pred_v1"] = float(sigmoid(x1 @ np.array(v1["theta"])))
            p["pred_kld_only"] = float(sigmoid(xk @ np.array(v2["theta_kld_only"])))
            p["pred_bits_only"] = v2["variant_means"][r["variant"]]
            pts.append(p)

    res: dict = {"n_points": len(pts), "dropped_undefined_R": dropped, "models": {}}
    y = np.array([p["retention"] for p in pts])
    for key in ("pred_v2", "pred_v1", "pred_kld_only", "pred_bits_only"):
        m = metrics(y, np.array([p[key] for p in pts]))
        res["models"][key] = {**m, "mae_ci_config_bootstrap": config_ci(pts, key)}
    mv2 = res["models"]["pred_v2"]
    res["W3-H2"] = (
        "supported"
        if mv2["mae"] <= 0.10 and mv2["safe_agreement"] >= 0.85
        else "not supported"
    )
    res["W3-H3"] = (
        "supported"
        if mv2["mae"] < res["models"]["pred_kld_only"]["mae"]
        and mv2["mae"] < res["models"]["pred_bits_only"]["mae"]
        else "not supported"
    )
    h1 = {}
    for prefix, (_, strong, gentle) in arms.items():
        for v in ("mlx-q3", "gguf-Q3_K_M"):
            h1[f"{prefix} {v}"] = h1_test(run, prefix, strong, gentle, v)
    res["W3-H1_tests"] = h1
    res["W3-H1"] = (
        "supported"
        if all(t["verdict"] == "supported" for t in h1.values())
        else "not supported"
        if any(t["verdict"] == "not supported" for t in h1.values())
        else "inconclusive"
    )
    res["points"] = pts
    out = Path(args.out) if args.out else run / "verdicts.json"
    out.write_text(json.dumps(res, indent=1))

    for name, t in h1.items():
        extra = (
            f"R {t['R_strong']:.3f} vs {t['R_gentle']:.3f}, diff {t['diff']:+.3f} [{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}]"
            if "ci" in t
            else t.get("reason", "")
        )
        print(f"H1 {name:18s} {t['verdict']:14s} {extra}")
    for key, m in res["models"].items():
        lo, hi = m["mae_ci_config_bootstrap"]
        print(
            f"{key:15s} n={m['n']:2d} MAE {m['mae']:.3f} [{lo:.3f}, {hi:.3f}] safe {m['safe_agreement']:.3f}"
        )
    print("dropped:", dropped)
    print("W3-H1", res["W3-H1"], "| W3-H2", res["W3-H2"], "| W3-H3", res["W3-H3"])


if __name__ == "__main__":
    main()
