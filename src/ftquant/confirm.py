import json
from pathlib import Path

import numpy as np

from ftquant.analyze import ROOT, load, retention

RUN = ROOT / "week1"
CFG, BASE = "q17-lora", "q17-base"


def arrays(variant: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fp = "gguf-bf16" if variant.startswith("gguf") else "mlx-bf16"
    ref = variant.removesuffix("-unfused")
    return (
        load(RUN, CFG, variant)["correct"],
        load(RUN, BASE, ref)["correct"],
        load(RUN, CFG, fp)["correct"],
        load(RUN, BASE, fp)["correct"],
    )


def paired_diff(
    a: str, b: str, reps: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    xa, xb = arrays(a), arrays(b)
    point = retention(*xa) - retention(*xb)
    rng = np.random.default_rng(seed)
    n = len(xa[0])
    vals = []
    for _ in range(reps):
        i = rng.integers(0, n, n)
        vals.append(retention(*(x[i] for x in xa)) - retention(*(x[i] for x in xb)))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def main() -> None:
    rows = {
        r["variant"]: r
        for r in json.loads((RUN / "analysis.json").read_text())["retention"]
        if r["config"] == CFG
    }
    out = {}

    h1 = {
        v: (rows[v]["retention"], rows[v]["ci"])
        for v in ("mlx-q8", "mlx-q6", "gguf-Q8_0", "gguf-Q6_K")
    }
    ok = all(r >= 0.95 and ci[0] >= 0.90 for r, ci in h1.values())
    out["H1"] = {"verdict": "supported" if ok else "not supported", "detail": h1}

    h2 = {}
    parts = []
    for q3, q4 in (("mlx-q3", "mlx-q4"), ("gguf-Q3_K_M", "gguf-Q4_K_M")):
        d = paired_diff(q3, q4)
        lower = rows[q3]["retention"] < rows[q4]["retention"]
        ci_ok = rows[q3]["ci"][1] < 0.95
        h2[f"{q3} vs {q4}"] = {
            "R3": rows[q3]["retention"],
            "R4": rows[q4]["retention"],
            "R3_ci": rows[q3]["ci"],
            "paired_diff_R3_minus_R4": d,
            "lower": lower,
            "upper_ci_below_0.95": ci_ok,
        }
        parts.append(lower and ci_ok)
    for q2 in ("mlx-q2", "gguf-Q2_K"):
        h2[q2] = {"R": rows[q2]["retention"], "below_0.5": rows[q2]["retention"] < 0.5}
        parts.append(rows[q2]["retention"] < 0.5)
    straddle = any(
        v.get("R3_ci", [1, 0])[0] <= 0.95 <= v.get("R3_ci", [1, 0])[1]
        for v in h2.values()
        if "R3_ci" in v
    )
    straddle |= any(
        rows[q]["ci"][0] <= 0.5 <= rows[q]["ci"][1] for q in ("mlx-q2", "gguf-Q2_K")
    )
    out["H2"] = {
        "verdict": "supported"
        if all(parts)
        else ("inconclusive" if straddle else "not supported"),
        "detail": h2,
    }

    h3 = {}
    parts = []
    for b in ("q4", "q3"):
        d = paired_diff(f"mlx-{b}-unfused", f"mlx-{b}")
        h3[f"mlx-{b}"] = {
            "R_unfused": rows[f"mlx-{b}-unfused"]["retention"],
            "R_fused": rows[f"mlx-{b}"]["retention"],
            "paired_diff_unfused_minus_fused": d,
        }
        parts.append(d[1] > 0)
    inconclusive = any(
        v["paired_diff_unfused_minus_fused"][1]
        <= 0
        <= v["paired_diff_unfused_minus_fused"][2]
        for v in h3.values()
    )
    out["H3"] = {
        "verdict": "supported"
        if all(parts)
        else ("inconclusive" if inconclusive else "not supported"),
        "detail": h3,
    }

    h4 = {}
    parts = []
    for v in ("mlx-q4", "gguf-Q4_K_M"):
        r = rows[v]
        d_err = 100 * (r["q_error_accepted"] - r["fp_error_accepted"])
        d_cov = 100 * (r["q_coverage"] - r["fp_coverage"])
        hit = d_err >= 1.0 or abs(d_cov) >= 3.0
        h4[v] = {
            "R": r["retention"],
            "error_change_pts": d_err,
            "coverage_change_pts": d_cov,
            "fp_coverage": r["fp_coverage"],
            "q_coverage": r["q_coverage"],
            "drift": hit,
        }
        parts.append(hit)
    out["H4"] = {
        "verdict": "supported" if all(parts) else ("not supported"),
        "detail": h4,
    }

    Path(RUN / "confirmatory.json").write_text(json.dumps(out, indent=1))
    for h, v in out.items():
        print(h, v["verdict"])


if __name__ == "__main__":
    main()
