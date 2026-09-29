import json
import math
from pathlib import Path

import numpy as np

from ftquant.predictor_v2 import FORMS, kld_lookup, mech_lookup, metrics, sigmoid
from ftquant.verdicts_w3 import LADDER, config_ci, h1_test

RUNS = Path(__file__).resolve().parents[2] / "runs"
RUN = "week4"
SEEDS = {
    "olmo1": ("OLMo-2-0425-1B-Instruct", "olmo1-lora-s1", "olmo1-lora-lowlr-s1"),
    "mas06": ("Qwen3-0.6B", "mas06-lora-s1", "mas06-lora-lowlr-s1"),
}
SCALE = {"q4b": ("Qwen3-4B", "q4b-lora", "q4b-lora-lowlr")}


def overall(tests: dict) -> str:
    if all(t["verdict"] == "supported" for t in tests.values()):
        return "supported"
    if any(t["verdict"] == "not supported" for t in tests.values()):
        return "not supported"
    return "inconclusive"


def main() -> None:
    run = RUNS / RUN
    rows = json.loads((run / "analysis.json").read_text())["retention"]
    v2 = json.loads((RUNS / "predictor" / "predictor-v2.json").read_text())
    v1 = json.loads((RUNS / "predictor" / "predictor-v1.json").read_text())
    arms = {**SEEDS, **SCALE}
    fine_tunes = {c for _, s, g in arms.values() for c in (s, g)}

    pts, dropped = [], []
    for prefix, (tag, _, _) in arms.items():
        mech = mech_lookup(RUN, prefix)
        kld = kld_lookup(tag)
        for r in rows:
            if (
                r["config"] not in fine_tunes
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
            p["pred_v2"] = float(
                sigmoid(np.array(FORMS[v2["form"]](p)) @ np.array(v2["theta"]))
            )
            p["pred_v1"] = float(
                sigmoid(np.array(FORMS["nsr+kld"](p)) @ np.array(v1["theta"]))
            )
            p["pred_kld_only"] = float(
                sigmoid(np.array(FORMS["kld-only"](p)) @ np.array(v2["theta_kld_only"]))
            )
            p["pred_bits_only"] = v2["variant_means"][r["variant"]]
            pts.append(p)

    res: dict = {"n_points": len(pts), "dropped_undefined_R": dropped, "models": {}}
    y = np.array([p["retention"] for p in pts])
    for key in ("pred_v2", "pred_v1", "pred_kld_only", "pred_bits_only"):
        m = metrics(y, np.array([p[key] for p in pts]))
        res["models"][key] = {**m, "mae_ci_config_bootstrap": config_ci(pts, key)}
    mv2 = res["models"]["pred_v2"]
    tests = {}
    for group, name in ((SEEDS, "W4-H1"), (SCALE, "W4-H2")):
        tests[name] = {
            f"{prefix} {v}": h1_test(run, prefix, strong, gentle, v)
            for prefix, (_, strong, gentle) in group.items()
            for v in ("mlx-q3", "gguf-Q3_K_M")
        }
        res[f"{name}_tests"] = tests[name]
        res[name] = overall(tests[name])
    res["W4-H3"] = (
        "supported"
        if mv2["mae"] <= 0.10 and mv2["safe_agreement"] >= 0.85
        else "not supported"
    )
    res["W4-H4"] = (
        "supported"
        if mv2["mae"] < res["models"]["pred_kld_only"]["mae"]
        and mv2["mae"] < res["models"]["pred_bits_only"]["mae"]
        else "not supported"
    )
    res["points"] = pts
    (run / "verdicts.json").write_text(json.dumps(res, indent=1))

    for name, group in tests.items():
        for label, t in group.items():
            extra = (
                f"R {t['R_strong']:.3f} vs {t['R_gentle']:.3f}, diff {t['diff']:+.3f} [{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}]"
                if "ci" in t
                else t.get("reason", "")
            )
            print(f"{name} {label:18s} {t['verdict']:14s} {extra}")
    for key, m in res["models"].items():
        lo, hi = m["mae_ci_config_bootstrap"]
        print(
            f"{key:15s} n={m['n']:2d} MAE {m['mae']:.3f} [{lo:.3f}, {hi:.3f}] safe {m['safe_agreement']:.3f}"
        )
    print("dropped:", dropped)
    print(" | ".join(f"{h} {res[h]}" for h in ("W4-H1", "W4-H2", "W4-H3", "W4-H4")))


if __name__ == "__main__":
    main()
