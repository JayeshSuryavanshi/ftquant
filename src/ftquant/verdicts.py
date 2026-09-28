import json
import math
from pathlib import Path

import numpy as np

RUNS = Path(__file__).resolve().parents[2] / "runs"
PRED = RUNS / "predictor"
MODELS = {
    "nsr_model": "predicted",
    "kld_only": "predicted_kld_only",
    "bits_only": "predicted_bits_only",
}


def load(test: str) -> tuple[list[dict], list[dict]]:
    pts = json.loads((PRED / f"eval-{test}.json").read_text())["points"]
    ok = [p for p in pts if not math.isnan(p["retention"])]
    dropped = [p for p in pts if math.isnan(p["retention"])]
    return ok, dropped


def score(pts: list[dict], key: str) -> dict[str, float]:
    y = np.clip(np.array([p["retention"] for p in pts]), 0.0, 1.0)
    yhat = np.array([p[key] for p in pts])
    return {
        "n": len(pts),
        "mae": float(np.mean(np.abs(yhat - y))),
        "safe_agreement": float(np.mean((y >= 0.9) == (yhat >= 0.9))),
    }


def mae_ci(
    pts: list[dict], key: str, reps: int = 2000, seed: int = 0
) -> tuple[float, float]:
    y = np.clip(np.array([p["retention"] for p in pts]), 0.0, 1.0)
    err = np.abs(np.array([p[key] for p in pts]) - y)
    configs = sorted({p["config"] for p in pts})
    by_cfg = [err[[p["config"] == c for p in pts]] for c in configs]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        pick = rng.integers(0, len(by_cfg), len(by_cfg))
        vals.append(np.concatenate([by_cfg[i] for i in pick]).mean())
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def verdict(ok: bool) -> str:
    return "met" if ok else "not met"


def main() -> None:
    t1, d1 = load("T1")
    t2, d2 = load("T2")
    t3, d3 = load("T3")
    pooled = t1 + t3
    res: dict = {
        "dropped_undefined_R": [(p["config"], p["variant"]) for p in d1 + d2 + d3]
    }
    for name, pts in (("T2", t2), ("T1+T3", pooled), ("T1", t1), ("T3", t3)):
        res[name] = {
            m: {**score(pts, k), "mae_ci_config_bootstrap": mae_ci(pts, k)}
            for m, k in MODELS.items()
        }
    t2n = res["T2"]["nsr_model"]
    p3n = res["T1+T3"]["nsr_model"]
    res["P1"] = verdict(t2n["mae"] <= 0.10 and t2n["safe_agreement"] >= 0.85)
    res["P2"] = verdict(
        t2n["mae"] < res["T2"]["kld_only"]["mae"]
        and t2n["mae"] < res["T2"]["bits_only"]["mae"]
    )
    res["P3"] = verdict(p3n["mae"] <= 0.15 and p3n["safe_agreement"] >= 0.80)
    worst = sorted(
        t2 + pooled,
        key=lambda p: -abs(p["predicted"] - min(max(p["retention"], 0.0), 1.0)),
    )[:8]
    res["largest_errors"] = [
        {k: p[k] for k in ("config", "variant", "nsr", "kld", "retention", "predicted")}
        for p in worst
    ]
    (PRED / "verdicts.json").write_text(json.dumps(res, indent=1))
    for name in ("T2", "T1+T3"):
        for m in MODELS:
            s = res[name][m]
            lo, hi = s["mae_ci_config_bootstrap"]
            print(
                f"{name:6s} {m:10s} n={s['n']:2d} MAE {s['mae']:.3f} [{lo:.3f}, {hi:.3f}]"
                f"  safe {s['safe_agreement']:.3f}"
            )
    print("dropped (R undefined):", res["dropped_undefined_R"])
    print("P1", res["P1"], "| P2", res["P2"], "| P3", res["P3"])
    for w in res["largest_errors"]:
        print(
            f"  {w['config']:18s} {w['variant']:12s} NSR {w['nsr']:5.2f} KLD {w['kld']:.3f}"
            f"  R {w['retention']:.3f}  pred {w['predicted']:.3f}"
        )


if __name__ == "__main__":
    main()
