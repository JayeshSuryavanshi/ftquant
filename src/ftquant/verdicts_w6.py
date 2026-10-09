import argparse
import json
import math
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np

RUNS = Path(__file__).resolve().parents[2] / "runs"
W6 = RUNS / "week6"
REPS, SEED = 2000, 0
PRECONDITION_MAX_R = 0.90
MIN_GAIN = 0.35
ACC_TOLERANCE, LOAD_ONLY_AGREEMENT = 0.01, 0.99
FORMATS_A = [
    ("mlx-q4", "mlx-bf16"),
    ("mlx-q3", "mlx-bf16"),
    ("gguf-Q4_K_M", "gguf-bf16"),
    ("gguf-Q3_K_M", "gguf-bf16"),
    ("gguf-Q2_K", "gguf-bf16"),
]

Spec = tuple[str, str]


class Missing(Exception):
    pass


def records(key: str, variant: str) -> list[dict]:
    p = W6 / key / "eval" / f"{variant}.jsonl"
    if not p.exists():
        raise Missing(f"{key}/{variant}")
    return [json.loads(line) for line in open(p)]


def aligned(*specs: Spec) -> list[list[dict]]:
    recs = [records(*s) for s in specs]
    texts = [r["text"] for r in recs[0]]
    for s, rs in zip(specs, recs, strict=True):
        if [r["text"] for r in rs] != texts:
            raise SystemExit(f"item order differs: {s}")
    return recs


def correct(*specs: Spec) -> list[np.ndarray]:
    return [np.array([r["correct"] for r in rs], dtype=float) for rs in aligned(*specs)]


def share(x_q, base_q, ft_fp, base_fp) -> float:
    gain = ft_fp.mean() - base_fp.mean()
    return float((x_q.mean() - base_q.mean()) / gain) if gain > 0 else math.nan


def boot(fn: Callable[..., float], arrays: list[np.ndarray]) -> dict:
    point = fn(*arrays)
    rng = np.random.default_rng(SEED)
    n = len(arrays[0])
    vals = np.array(
        [
            fn(*(a[i] for a in arrays))
            for i in (rng.integers(0, n, n) for _ in range(REPS))
        ]
    )
    nans = int(np.isnan(vals).sum())
    lo, hi = np.percentile(vals, [2.5, 97.5]) if nans == 0 else (math.nan, math.nan)
    return {"ci": [float(point), float(lo), float(hi)], "nan_resamples": nans}


def usable(b: dict) -> bool:
    return b["nan_resamples"] == 0 and not math.isnan(b["ci"][0])


def above(ci: list[float]) -> str:
    return (
        "supported" if ci[1] > 0 else ("not supported" if ci[2] < 0 else "inconclusive")
    )


def overall(verdicts: list[str]) -> str:
    tested = [v for v in verdicts if v != "not testable"]
    if not tested:
        return "not testable"
    if len(tested) == len(verdicts) and all(v == "supported" for v in tested):
        return "supported"
    return "not supported" if "not supported" in tested else "inconclusive"


# ---- W6-H1 and W6-H2: the GGUF remedy


def halfway(x, g1, bq, ft, bb) -> float:
    return share(x, bq, ft, bb) - (1 + share(g1, bq, ft, bb)) / 2


def g_rows(lr: str, fmt: str) -> dict:
    core = [
        (f"q06-{lr}", f"gguf-{fmt}"),
        ("q06-base", f"gguf-{fmt}"),
        (f"q06-{lr}", "gguf-bf16"),
        ("q06-base", "gguf-bf16"),
    ]
    try:
        g1, bq, ft, bb = correct(*core)
    except Missing as e:
        return {"missing": str(e)}
    row = {
        "R_G1": boot(share, [g1, bq, ft, bb]),
        "acc": {
            "G1": float(g1.mean()),
            "base": float(bq.mean()),
            "ft_bf16": float(ft.mean()),
            "base_bf16": float(bb.mean()),
        },
    }
    arms = {}
    for arm, spec in (
        ("G2", (f"q06-{lr}", f"gguf-{fmt}-lora")),
        ("G3", (f"q06dq{fmt[1]}-{lr}", f"gguf-{fmt}-lora")),
    ):
        try:
            x = correct(spec, *core)[0]
        except Missing as e:
            row[f"{arm}_missing"] = str(e)
            continue
        arms[arm] = x
        row["acc"][arm] = float(x.mean())
        row[f"S_{arm}"] = boot(share, [x, bq, ft, bb])
        row[f"S_{arm}_minus_halfway"] = boot(halfway, [x, g1, bq, ft, bb])
        row[f"agreement_{arm}_with_G1"] = float(
            np.mean(
                [
                    a["pred"] == b["pred"]
                    for a, b in zip(*aligned(spec, core[0]), strict=True)
                ]
            )
        )
    if len(arms) == 2:
        row["S_G3_minus_S_G2"] = boot(
            lambda a, b, q, f, z: share(a, q, f, z) - share(b, q, f, z),
            [arms["G3"], arms["G2"], bq, ft, bb],
        )
    return row


def h1_h2(row: dict) -> tuple[dict, dict]:
    if "missing" in row:
        out = {"verdict": "not testable", "reason": f"missing {row['missing']}"}
        return out, out
    r = row["R_G1"]
    if not usable(r):
        out = {
            "verdict": "not testable",
            "reason": "the bf16 gain is not positive in every resample",
            "R_G1": r,
        }
        return out, out
    if not r["ci"][2] < PRECONDITION_MAX_R:
        out = {
            "verdict": "not testable",
            "reason": f"precondition failed: upper bound of R_G1 {r['ci'][2]:.3f} is not below {PRECONDITION_MAX_R}",
        }
        return out, out
    tests = []
    for arm, reverse in (("G3", False), ("G2", True)):
        key = f"S_{arm}_minus_halfway"
        if key not in row:
            tests.append(
                {
                    "verdict": "not testable",
                    "reason": f"missing {row[f'{arm}_missing']}",
                }
            )
            continue
        ci = row[key]["ci"]
        if not usable(row[key]):
            tests.append(
                {"verdict": "not testable", "reason": "NaN resamples", key: row[key]}
            )
        elif reverse:
            v = (
                "supported"
                if ci[2] < 0
                else ("not supported" if ci[1] > 0 else "inconclusive")
            )
            tests.append({"verdict": v, key: ci, "R_G1": r["ci"]})
        else:
            tests.append({"verdict": above(ci), key: ci, "R_G1": r["ci"]})
    return tests[0], tests[1]


# ---- W6-H3: mlx-lm's defaults


def h3() -> dict:
    gains = {}
    for eng in ("mlx", "gguf"):
        for key in ("q06-def-lowlr", "q06-def"):
            try:
                ft, base = correct((key, f"{eng}-bf16"), ("q06-base", f"{eng}-bf16"))
                gains[f"{key} {eng}"] = float(ft.mean() - base.mean())
            except Missing as e:
                gains[f"{key} {eng}"] = f"missing {e}"
    if any(isinstance(g, str) for g in gains.values()):
        return {
            "verdict": "not testable",
            "reason": "a bf16 evaluation is missing",
            "gains": gains,
        }
    if min(round(g, 9) for g in gains.values()) < MIN_GAIN:
        return {
            "verdict": "not testable",
            "reason": f"a bf16 gain is below {MIN_GAIN}",
            "gains": gains,
        }
    tests = {}
    for q, fp in (("gguf-Q3_K_M", "gguf-bf16"), ("mlx-q4", "mlx-bf16")):
        try:
            arrs = correct(
                ("q06-def", q),
                ("q06-def-lowlr", q),
                ("q06-base", q),
                ("q06-def", fp),
                ("q06-def-lowlr", fp),
                ("q06-base", fp),
            )
        except Missing as e:
            tests[q] = {"verdict": "not testable", "reason": f"missing {e}"}
            continue
        b = boot(
            lambda s, g, z, sf, gf, zf: share(s, z, sf, zf) - share(g, z, gf, zf), arrs
        )
        tests[q] = {
            "R_strong_minus_R_gentle": b["ci"],
            "verdict": above(b["ci"]) if usable(b) else "not testable",
        }
    return {
        "verdict": overall([t["verdict"] for t in tests.values()]),
        "tests": tests,
        "gains": gains,
    }


# ---- W6-H4: importance matrix


def gap(s, g, b, sf, gf, bf) -> float:
    return share(s, b, sf, bf) - share(g, b, gf, bf)


def i_rows(prefix: str, fmt: str) -> dict:
    specs = []
    for im in ("", "-im"):
        specs += [
            (f"{prefix}-lora", f"gguf-{fmt}{im}"),
            (f"{prefix}-lora-lowlr", f"gguf-{fmt}{im}"),
            (f"{prefix}-base", f"gguf-{fmt}{im}"),
        ]
    specs += [
        (f"{prefix}-lora", "gguf-bf16"),
        (f"{prefix}-lora-lowlr", "gguf-bf16"),
        (f"{prefix}-base", "gguf-bf16"),
    ]
    try:
        s0, g0, b0, s1, g1, b1, sf, gf, bf = correct(*specs)
    except Missing as e:
        return {"missing": str(e)}
    return {
        "delta_no": boot(gap, [s0, g0, b0, sf, gf, bf]),
        "delta_im": boot(gap, [s1, g1, b1, sf, gf, bf]),
        "delta_im_minus_half_delta_no": boot(
            lambda *a: gap(*a[3:6], *a[6:]) - gap(*a[:3], *a[6:]) / 2,
            [s0, g0, b0, s1, g1, b1, sf, gf, bf],
        ),
        "R": {
            "strong_no": share(s0, b0, sf, bf),
            "gentle_no": share(g0, b0, gf, bf),
            "strong_im": share(s1, b1, sf, bf),
            "gentle_im": share(g1, b1, gf, bf),
        },
        "acc_base": {"no": float(b0.mean()), "im": float(b1.mean())},
    }


def h4(rows: dict) -> dict:
    tests = {}
    for prefix in ("q06", "olmo1"):
        row = rows[prefix]
        if "missing" in row:
            tests[prefix] = {
                "verdict": "not testable",
                "reason": f"missing {row['missing']}",
            }
        elif not (usable(row["delta_no"]) and row["delta_no"]["ci"][1] > 0):
            tests[prefix] = {
                "verdict": "not testable",
                "reason": "the lower bound of delta_no is not above 0",
                "delta_no": row["delta_no"],
            }
        else:
            b = row["delta_im_minus_half_delta_no"]
            tests[prefix] = {
                "verdict": above(b["ci"]) if usable(b) else "not testable",
                "delta_im_minus_half_delta_no": b["ci"],
                "delta_no": row["delta_no"]["ci"],
                "delta_im": row["delta_im"]["ci"],
            }
    return {"verdict": overall([t["verdict"] for t in tests.values()]), "tests": tests}


# ---- checks and exploratory


def agreement(a: Spec, b: Spec) -> float:
    x, y = aligned(a, b)
    return float(np.mean([p["pred"] == q["pred"] for p, q in zip(x, y, strict=True)]))


def acc_of(spec: Spec) -> float:
    return float(np.mean([r["correct"] for r in records(*spec)]))


def check_b() -> dict:
    out = {}
    try:
        for fmt in ("bf16", "Q8_0"):
            fused, loaded = (
                ("q06-lora-lowlr", f"gguf-{fmt}-valid"),
                ("q06-lora-lowlr", f"gguf-{fmt}-lora-valid"),
            )
            out[fmt] = {
                "acc_fused": acc_of(fused),
                "acc_load_time": acc_of(loaded),
                "agreement": agreement(fused, loaded),
            }
            out[fmt]["passed"] = (
                abs(out[fmt]["acc_fused"] - out[fmt]["acc_load_time"]) <= ACC_TOLERANCE
            )
        base, idle = (
            ("q06-base", "gguf-bf16-valid"),
            ("q06-base", "gguf-bf16-lora0-valid"),
        )
        a = agreement(base, idle)
        out["load_only"] = {
            "agreement_with_base": a,
            "acc_base": acc_of(base),
            "acc_load_only": acc_of(idle),
        }
        out["load_only"]["passed"] = (
            a >= LOAD_ONLY_AGREEMENT
            and abs(acc_of(base) - acc_of(idle)) <= ACC_TOLERANCE
        )
        out["passed"] = all(out[k]["passed"] for k in ("bf16", "Q8_0", "load_only"))
    except Missing as e:
        out = {"missing": str(e), "passed": False}
    return out


def val_loss(key: str) -> float | None:
    p = W6 / key / "train.log"
    hits = (
        re.findall(r"Iter \d+: Val loss ([^\s,]+)", p.read_text()) if p.exists() else []
    )
    try:
        return float(hits[-1]) if hits else None
    except ValueError:
        return None


def update_norm(key: str) -> float | None:
    p = W6 / key / "adapter" / "adapters.safetensors"
    if not p.exists():
        return None
    import mlx.core as mx

    scale = json.loads((p.parent / "adapter_config.json").read_text())[
        "lora_parameters"
    ]["scale"]
    w = mx.load(str(p))
    total = 0.0
    for k in w:
        if k.endswith(".lora_a"):
            a = np.array(w[k].astype(mx.float32), dtype=np.float64)
            b = np.array(w[k[:-1] + "b"].astype(mx.float32), dtype=np.float64)
            total += float(np.sum((scale * a @ b) ** 2))
    return total**0.5


def d_exploratory() -> dict:
    out = {
        "val_loss_note": "computed over every token (no prompt masking), not comparable with earlier runs"
    }
    for key in ("q06-def-lowlr", "q06-def"):
        row = {"val_loss": val_loss(key), "update_norm": update_norm(key)}
        for q, fp in FORMATS_A:
            try:
                x, f, b, bf = correct(
                    (key, q), (key, fp), ("q06-base", q), ("q06-base", fp)
                )
            except Missing:
                continue
            row[q] = {
                "R": share(x, b, f, bf),
                "A": float(x.mean() / f.mean()),
                "acc": float(x.mean()),
            }
        out[key] = row
    cells = {}
    for q, fp in FORMATS_A:
        try:
            s, sf, g, gf = correct(
                ("q06-def", q),
                ("q06-def", fp),
                ("q06-def-lowlr", q),
                ("q06-def-lowlr", fp),
            )
        except Missing:
            continue
        cells[q] = boot(
            lambda a, b, c, d: a.mean() / b.mean() - c.mean() / d.mean(), [s, sf, g, gf]
        )
    out["A_strong_minus_A_gentle"] = cells
    return out


def evaluations() -> dict:
    out = {}
    for p in sorted(W6.glob("*/eval/*.jsonl")):
        rs = [json.loads(line) for line in open(p)]
        out[f"{p.parent.parent.name}/{p.stem}"] = {
            "n": len(rs),
            "acc": float(np.mean([r["correct"] for r in rs])),
            "valid": float(np.mean([r["valid"] for r in rs])),
        }
    return out


def main() -> None:
    global W6
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(W6))
    W6 = Path(ap.parse_args().run)
    g = {
        f"{lr} {f}": g_rows(lr, f)
        for lr in ("lora-lowlr", "lora")
        for f in ("Q3_K_M", "Q2_K")
    }
    h1, h2 = h1_h2(g["lora-lowlr Q3_K_M"])
    i3 = {p: i_rows(p, "Q3_K_M") for p in ("q06", "olmo1")}
    out = {
        "W6-H1": h1,
        "W6-H2": h2,
        "W6-H3": h3(),
        "W6-H4": h4(i3),
        "check_b": check_b(),
        "adapter_checks": {
            p.parent.name: json.loads(p.read_text())
            for p in sorted(W6.glob("*/lora-check.json"))
        },
        "rebuild_checks": {
            p.stem: json.loads(p.read_text()) for p in sorted(W6.glob("rebuild-*.json"))
        },
        "exploratory": {
            "arm_g": g,
            "arm_g_val_loss": {
                k: val_loss(k)
                for k in (
                    "q06dq3-lora-lowlr",
                    "q06dq3-lora",
                    "q06dq2-lora-lowlr",
                    "q06dq2-lora",
                )
            },
            "arm_d": d_exploratory(),
            "arm_i_q2": {p: i_rows(p, "Q2_K") for p in ("q06", "olmo1")},
            "arm_i_q3": i3,
            "evaluations": evaluations(),
        },
    }
    (W6 / "verdicts.json").write_text(json.dumps(out, indent=1))
    for h in ("W6-H1", "W6-H2", "W6-H3", "W6-H4"):
        print(h, out[h]["verdict"], out[h].get("reason", ""))
    print("check (b) passed:", out["check_b"]["passed"])
    row = g["lora-lowlr Q3_K_M"]
    for k in (
        "R_G1",
        "S_G2",
        "S_G3",
        "S_G3_minus_halfway",
        "S_G2_minus_halfway",
        "S_G3_minus_S_G2",
    ):
        if k in row:
            p, lo, hi = row[k]["ci"]
            print(f"  gentle Q3_K_M {k} {p:+.3f} [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
