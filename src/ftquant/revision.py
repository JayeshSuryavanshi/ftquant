import json
import math
import re
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import yaml

from ftquant.predictor_v2 import (
    FIT_RUNS,
    fit,
    mech_lookup,
    metrics,
    points,
    sigmoid,
)

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
OUT = RUNS / "revision"
HARD = ["mlx-q4", "mlx-q3", "gguf-Q4_K_M", "gguf-Q3_K_M", "gguf-Q2_K"]
LADDER = [
    "mlx-q6",
    "mlx-q4",
    "mlx-q3",
    "gguf-Q6_K",
    "gguf-Q4_K_M",
    "gguf-Q3_K_M",
    "gguf-Q2_K",
]
REPS, SEED = 2000, 0
FAILED = {"q06-lora-lr3e-4"}

PAIRS = [
    (
        "Qwen3-0.6B, banking77",
        ("week1", "q06-lora-lowlr"),
        ("week1", "q06-lora"),
        ("week1", "q06-base"),
    ),
    (
        "Qwen3-0.6B, banking77, seed 1",
        ("week2", "q06-lora-lowlr-s1"),
        ("week2", "q06-lora-s1"),
        ("week1", "q06-base"),
    ),
    (
        "Qwen3-1.7B, banking77",
        ("week2", "q17-lora-lowlr"),
        ("week1", "q17-lora"),
        ("week1", "q17-base"),
    ),
    (
        "Qwen3-4B, banking77",
        ("week4", "q4b-lora-lowlr"),
        ("week4", "q4b-lora"),
        ("week4", "q4b-base"),
    ),
    (
        "OLMo-2 1B, banking77",
        ("week3", "olmo1-lora-lowlr"),
        ("week3", "olmo1-lora"),
        ("week3", "olmo1-base"),
    ),
    (
        "OLMo-2 1B, banking77, seed 1",
        ("week4", "olmo1-lora-lowlr-s1"),
        ("week4", "olmo1-lora-s1"),
        ("week3", "olmo1-base"),
    ),
    (
        "Qwen3-0.6B, MASSIVE",
        ("week3", "mas06-lora-lowlr"),
        ("week3", "mas06-lora"),
        ("week3", "mas06-base"),
    ),
    (
        "Qwen3-0.6B, MASSIVE, seed 1",
        ("week4", "mas06-lora-lowlr-s1"),
        ("week4", "mas06-lora-s1"),
        ("week3", "mas06-base"),
    ),
]
CONFIRMATORY = {
    "W3-H1": ["OLMo-2 1B, banking77", "Qwen3-0.6B, MASSIVE"],
    "W4-H1": ["OLMo-2 1B, banking77, seed 1", "Qwen3-0.6B, MASSIVE, seed 1"],
    "W4-H2": ["Qwen3-4B, banking77"],
}
MODELS = {
    "q06": "Qwen/Qwen3-0.6B",
    "q17": "Qwen/Qwen3-1.7B",
    "q4b": "Qwen/Qwen3-4B",
    "olmo1": "allenai/OLMo-2-0425-1B-Instruct",
    "mas06": "Qwen/Qwen3-0.6B",
}

Ref = tuple[str, str]


def records(ref: Ref, variant: str) -> list[dict] | None:
    p = RUNS / ref[0] / ref[1] / "eval" / f"{variant}.jsonl"
    return [json.loads(line) for line in open(p)] if p.exists() else None


def col(rs: list[dict], key: str = "correct") -> np.ndarray:
    return np.array([r[key] for r in rs], dtype=float)


def aligned(*rss: list[dict]) -> None:
    texts = [r["text"] for r in rss[0]]
    for rs in rss[1:]:
        if [r["text"] for r in rs] != texts:
            raise SystemExit("evaluation files are not item-aligned")


def fp_of(variant: str) -> str:
    return "gguf-bf16" if variant.startswith("gguf") else "mlx-bf16"


def resample_values(
    fn: Callable[..., float], arrays: list[np.ndarray]
) -> tuple[float, np.ndarray]:
    rng = np.random.default_rng(SEED)
    n = len(arrays[0])
    vals = np.array(
        [
            fn(*(a[i] for a in arrays))
            for i in (rng.integers(0, n, n) for _ in range(REPS))
        ]
    )
    return float(fn(*arrays)), vals


def interval(
    fn: Callable[..., float], arrays: list[np.ndarray], level: float = 95.0
) -> list[float]:
    point, vals = resample_values(fn, arrays)
    lo, hi = np.nanpercentile(vals, [(100 - level) / 2, 100 - (100 - level) / 2])
    return [point, float(lo), float(hi)]


def share(q, base_q, fp, base_fp) -> float:
    gain = fp.mean() - base_fp.mean()
    return float((q.mean() - base_q.mean()) / gain) if gain > 0 else float("nan")


def ratio(q, fp) -> float:
    return float(q.mean() / fp.mean())


def pairs_table() -> dict:
    out = {}
    for name, gentle, strong, _ in PAIRS:
        row: dict = {"bf16": {}, "formats": {}}
        for fp in ("mlx-bf16", "gguf-bf16"):
            g, s = records(gentle, fp), records(strong, fp)
            row["bf16"][fp] = {
                "gentle": float(col(g).mean()),
                "strong": float(col(s).mean()),
            }
        for v in HARD:
            gq, sq, gf, sf = (
                records(gentle, v),
                records(strong, v),
                records(gentle, fp_of(v)),
                records(strong, fp_of(v)),
            )
            aligned(gq, sq, gf, sf)
            a = [col(gq), col(gf), col(sq), col(sf)]
            row["formats"][v] = {
                "acc_gentle": float(a[0].mean()),
                "acc_strong": float(a[2].mean()),
                "A_gentle": ratio(a[0], a[1]),
                "A_strong": ratio(a[2], a[3]),
                "A_gentle_minus_strong": interval(
                    lambda gq, gf, sq, sf: ratio(gq, gf) - ratio(sq, sf), a
                ),
                "valid_gentle": float(col(gq, "valid").mean()),
                "valid_strong": float(col(sq, "valid").mean()),
                "acc_among_valid_gentle": float(col(gq)[col(gq, "valid") == 1].mean()),
                "acc_among_valid_strong": float(col(sq)[col(sq, "valid") == 1].mean()),
            }
        out[name] = row
    cells = [(n, v, r["formats"][v]) for n, r in out.items() for v in HARD]
    summary = {
        "gentle_higher_bf16_mlx": sum(
            r["bf16"]["mlx-bf16"]["gentle"] > r["bf16"]["mlx-bf16"]["strong"]
            for r in out.values()
        ),
        "bf16_cost_of_strong_pts_mlx": [
            100
            * min(
                r["bf16"]["mlx-bf16"]["gentle"] - r["bf16"]["mlx-bf16"]["strong"]
                for r in out.values()
            ),
            100
            * max(
                r["bf16"]["mlx-bf16"]["gentle"] - r["bf16"]["mlx-bf16"]["strong"]
                for r in out.values()
            ),
        ],
        "A_gentle_lower": f"{sum(c['A_gentle'] < c['A_strong'] for _, _, c in cells)} of {len(cells)}",
        "A_gap_interval_includes_zero": [
            f"{n} {v}"
            for n, v, c in cells
            if c["A_gentle_minus_strong"][1] <= 0 <= c["A_gentle_minus_strong"][2]
        ],
        "strong_more_accurate": {
            v: sum(
                out[n]["formats"][v]["acc_strong"] > out[n]["formats"][v]["acc_gentle"]
                for n in out
            )
            for v in HARD
        },
    }
    return {"pairs": out, "summary": summary}


def confirmatory_robustness() -> dict:
    out = {}
    by_name = {n: (g, s, b) for n, g, s, b in PAIRS}
    for test, names in CONFIRMATORY.items():
        for name in names:
            gentle, strong, base = by_name[name]
            for v in ("mlx-q3", "gguf-Q3_K_M"):
                fp = fp_of(v)
                rs = [
                    records(strong, v),
                    records(base, v),
                    records(strong, fp),
                    records(base, fp),
                    records(gentle, v),
                    records(gentle, fp),
                ]
                aligned(*rs)
                sq, bq, sf, bf, gq, gf = (col(r) for r in rs)
                arrs = [sq, bq, sf, bf, gq, gf]

                def r_diff(sq, bq, sf, bf, gq, gf) -> float:
                    return share(sq, bq, sf, bf) - share(gq, bq, gf, bf)

                def drop_diff(sq, bq, sf, bf, gq, gf) -> float:
                    return float(
                        100 * ((gf.mean() - gq.mean()) - (sf.mean() - sq.mean()))
                    )

                gain_s, gain_g = sf.mean() - bf.mean(), gf.mean() - bf.mean()
                out[f"{test} {name} {v}"] = {
                    "R_strong_minus_gentle_95": interval(r_diff, arrs),
                    "R_strong_minus_gentle_99_5": interval(r_diff, arrs, 99.5),
                    "accuracy_drop_gentle_minus_strong_pts": interval(drop_diff, arrs),
                    "A_strong_minus_gentle": interval(
                        lambda sq, bq, sf, bf, gq, gf: ratio(sq, sf) - ratio(gq, gf),
                        arrs,
                    ),
                    "base_drop_term_of_R_diff": float(
                        (bf.mean() - bq.mean()) * (1 / gain_s - 1 / gain_g)
                    ),
                }
    return out


def recorded_norms() -> dict[str, float]:
    out = {}
    for f in sorted(RUNS.glob("week*/mechanism-*-mlx.json")):
        for r in json.loads(f.read_text()):
            if isinstance(r, dict) and "delta_norm" in r:
                out[r["config"]] = float(r["delta_norm"])
    return out


def base_tensor_reader(model_id: str) -> Callable[[str], object]:
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    snap = Path(snapshot_download(model_id, local_files_only=True))
    index = snap / "model.safetensors.index.json"
    where = json.loads(index.read_text())["weight_map"] if index.exists() else None
    handles: dict[str, object] = {}

    def read(key: str):
        shard = where[key] if where else "model.safetensors"
        if shard not in handles:
            handles[shard] = safe_open(str(snap / shard), framework="pt")
        return handles[shard].get_tensor(key)

    return read


def computed_norm(run: str, cfg: str) -> tuple[float, float]:
    import torch
    from safetensors import safe_open

    d = RUNS / run / cfg
    train = yaml.safe_load((d / "train.yaml").read_text())
    scale = float(train["lora_parameters"]["scale"])
    read = base_tensor_reader(MODELS[cfg.split("-")[0]])
    rounded = exact = 0.0
    with safe_open(str(d / "adapter" / "adapters.safetensors"), framework="pt") as ad:
        for k in ad.keys():
            if not k.endswith(".lora_a"):
                continue
            stem = k[: -len(".lora_a")]
            a = ad.get_tensor(k).float()
            b = ad.get_tensor(stem + ".lora_b").float()
            delta = scale * (b.T @ a.T)
            w = read(stem + ".weight")
            fused = (w.float() + delta).to(w.dtype)
            diff = fused.float() - w.float()
            rounded += float(torch.sum(diff * diff))
            exact += float(torch.sum(delta * delta))
    return math.sqrt(rounded), math.sqrt(exact)


def update_norms() -> dict:
    norms = recorded_norms()
    cache = OUT / "week4-norms.json"
    week4 = (
        "olmo1-lora-lowlr-s1",
        "olmo1-lora-s1",
        "mas06-lora-lowlr-s1",
        "mas06-lora-s1",
        "q4b-lora-lowlr",
        "q4b-lora",
    )
    try:
        if not (RUNS / "week3" / "olmo1-lora-lowlr" / "adapter" / "adapters.safetensors").exists():
            raise FileNotFoundError("adapter weights not present")
        check_rounded, check_exact = computed_norm("week3", "olmo1-lora-lowlr")
        computed = {}
        for cfg in week4:
            rounded, exact = computed_norm("week4", cfg)
            computed[cfg] = {"bf16_rounded": rounded, "exact": exact}
        cache.write_text(
            json.dumps(
                {"check": [check_rounded, check_exact], "computed": computed}, indent=1
            )
        )
    except Exception as err:  # adapters or base weights absent, as in the supplement
        print(f"update norms read from {cache.name} ({type(err).__name__})", file=sys.stderr)
        cached = json.loads(cache.read_text())
        (check_rounded, check_exact), computed = cached["check"], cached["computed"]
    for cfg, v in computed.items():
        norms[cfg] = v["bf16_rounded"]
    return {
        "norms": norms,
        "week4_computed": computed,
        "method_check_olmo1-lora-lowlr": {
            "recorded": norms["olmo1-lora-lowlr"],
            "recomputed_bf16_rounded": check_rounded,
            "recomputed_exact": check_exact,
        },
    }


def cluster_metric_ci(pts: list[dict], keys: tuple[str, ...]) -> list[float]:
    y = np.clip(np.array([p["retention"] for p in pts]), 0.0, 1.0)
    cfgs = sorted({p["config"] for p in pts})
    idx = [np.array([p["config"] == c for p in pts]) for c in cfgs]
    errs = [np.abs(np.array([p[k] for p in pts]) - y) for k in keys]
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(REPS):
        pick = rng.integers(0, len(cfgs), len(cfgs))
        mask = [idx[i] for i in pick]
        m = [np.concatenate([e[x] for x in mask]).mean() for e in errs]
        vals.append(m[0] if len(m) == 1 else m[0] - m[1])
    point = float(errs[0].mean() if len(errs) == 1 else errs[0].mean() - errs[1].mean())
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [point, float(lo), float(hi)]


def norm_form(p: dict) -> list[float]:
    return [1.0, math.log(max(p["norm"], 1e-6)), math.log(max(p["kld"], 1e-6))]


def predictor_baselines(norms: dict[str, float]) -> dict:
    train = points(RUNS / "analysis-all.json", FIT_RUNS)
    for p in train:
        p["norm"] = norms[p["config"]]
    y = np.array([p["retention"] for p in train])
    X = np.array([norm_form(p) for p in train])
    theta = fit(X, y)
    lofo = np.zeros(len(train))
    for c in sorted({p["config"] for p in train}):
        test = np.array([p["config"] == c for p in train])
        lofo[test] = sigmoid(X[test] @ fit(X[~test], y[~test]))
    v2_lofo = json.loads((RUNS / "predictor" / "predictor-v2.json").read_text())["loco"]
    out = {
        "fit_points": len(train),
        "norm_kld_theta": theta.tolist(),
        "lofo": {"norm+kld": metrics(y, lofo), "v2": v2_lofo},
        "rounds": {},
    }
    for label, run in (("round 3", "week3"), ("round 4", "week4")):
        pts = json.loads((RUNS / run / "verdicts.json").read_text())["points"]
        for p in pts:
            p["norm"] = norms[p["config"]]
            p["pred_const"] = 1.0
            p["pred_norm_kld"] = float(sigmoid(np.array(norm_form(p)) @ theta))
        res = {}
        for subset, sel in (
            ("all", pts),
            (
                "without 6-bit",
                [p for p in pts if p["variant"] not in ("mlx-q6", "gguf-Q6_K")],
            ),
        ):
            yy = np.array([p["retention"] for p in sel])
            res[subset] = {
                k: {
                    **metrics(yy, np.array([p[k] for p in sel])),
                    "mae_ci": cluster_metric_ci(sel, (k,))[1:],
                }
                for k in (
                    "pred_v2",
                    "pred_v1",
                    "pred_kld_only",
                    "pred_bits_only",
                    "pred_const",
                    "pred_norm_kld",
                )
            }
            res[subset]["paired_mae_diff"] = {
                f"{a} minus {b}": cluster_metric_ci(sel, (a, b))
                for a, b in (
                    ("pred_kld_only", "pred_v2"),
                    ("pred_const", "pred_v2"),
                    ("pred_v2", "pred_norm_kld"),
                    ("pred_v2", "pred_v1"),
                )
            }
        out["rounds"][label] = res
    return out


def nsr_norm_correlation(norms: dict[str, float]) -> dict:
    pts = points(RUNS / "analysis-all.json", FIT_RUNS)
    by: dict[str, list[tuple[float, float]]] = {}
    for p in pts:
        by.setdefault(p["variant"], []).append((math.log(p["nsr"]), math.log(norms[p["config"]])))
    out = {}
    for v, xs in sorted(by.items()):
        a = np.array(xs)
        out[v] = {"n": len(xs), "r": float(np.corrcoef(a[:, 0], a[:, 1])[0, 1])}
    return out


def bands() -> dict:
    pts = points(RUNS / "analysis-all.json", FIT_RUNS)
    spec = [
        (
            "base damage < 1, kept >= 60%, NSR < 3",
            lambda p: p["kld"] < 1 and p["kept"] >= 0.6 and p["nsr"] < 3,
        ),
        (
            "base damage < 1, kept >= 60%, NSR 3 to 5",
            lambda p: p["kld"] < 1 and p["kept"] >= 0.6 and 3 <= p["nsr"] < 5,
        ),
        (
            "base damage < 1, kept >= 60%, NSR >= 5",
            lambda p: p["kld"] < 1 and p["kept"] >= 0.6 and p["nsr"] >= 5,
        ),
        ("kept < 60%", lambda p: p["kept"] < 0.6),
        ("kept >= 60%, base damage >= 1", lambda p: p["kept"] >= 0.6 and p["kld"] >= 1),
    ]
    out = {}
    for label, cond in spec:
        sel = [p for p in pts if cond(p)]
        r = [min(max(p["retention"], 0.0), 1.0) for p in sel]
        out[label] = {
            "pairs": len(sel),
            "kept_ge_90": sum(x >= 0.9 for x in r),
            "lowest": min(r) if r else None,
        }
    return {"n": len(pts), "bands": out}


def kept_count() -> dict:
    prefixes = {
        "week1": ("q06", "q17"),
        "week2": ("q06", "q17"),
        "week3": ("olmo1", "mas06"),
        "week4": ("olmo1", "mas06", "q4b"),
    }
    pairs = []
    for run, prefs in prefixes.items():
        for prefix in prefs:
            for (cfg, v), (_, kept) in mech_lookup(run, prefix).items():
                if cfg not in FAILED:
                    pairs.append((run, cfg, v, kept))
    return {
        "pairs": len(pairs),
        "kept_ge_98": sum(k >= 0.98 for *_, k in pairs),
        "fine_tunes": len({(r, c) for r, c, _, _ in pairs}),
        "lowest_ten": sorted(((k, f"{r}/{c} {v}") for r, c, v, k in pairs))[:10],
    }


def ranks(x: list[float]) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[order] = np.arange(1, len(x) + 1)
    for val in set(x):
        idx = [i for i, xi in enumerate(x) if xi == val]
        r[idx] = np.mean(r[idx])
    return r


def spearman(x: list[float], y: list[float]) -> float:
    rx, ry = ranks(x), ranks(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def lr_spearman() -> dict:
    cfgs = [
        c
        for c in json.loads((RUNS / "week2" / "summary.json").read_text())["configs"]
        if c["model"] == "0.6B" and c["seed"] == 0 and c["config"] not in FAILED
    ]
    out = {}
    for kind in sorted({c["kind"] for c in cfgs}):
        for v in ("mlx-q4", "mlx-q3", "gguf-Q4_K_M", "gguf-Q3_K_M"):
            sel = sorted((c for c in cfgs if c["kind"] == kind), key=lambda c: c["lr"])
            lrs, rs = [c["lr"] for c in sel], [c["gain_retention"][v] for c in sel]
            out[f"{kind} {v}"] = {"lrs": lrs, "R": rs, "spearman": spearman(lrs, rs)}
    return out


def analysis_rows() -> dict[tuple[str, str, str], dict]:
    rows = {}
    for f in [
        RUNS / "analysis-all.json",
        RUNS / "week3" / "analysis.json",
        RUNS / "week4" / "analysis.json",
    ]:
        for r in json.loads(f.read_text())["retention"]:
            run = r.get("run", f.parent.name)
            rows[(run, r["config"], r["variant"])] = r
    return rows


def drift() -> dict:
    rows = analysis_rows()
    cases, out = [], {}
    for name, gentle, strong, _ in PAIRS:
        for v in HARD:
            g, s = rows.get((*gentle, v)), rows.get((*strong, v))
            if g is None or s is None:
                continue
            dg = 100 * (g["q_coverage"] - g["fp_coverage"])
            ds = 100 * (s["q_coverage"] - s["fp_coverage"])
            cases.append(dg < ds)
            out[f"{name} {v}"] = {
                "coverage_change_gentle": dg,
                "coverage_change_strong": ds,
                "error_accepted_gentle": [
                    100 * g["fp_error_accepted"],
                    100 * g["q_error_accepted"],
                ],
                "error_accepted_strong": [
                    100 * s["fp_error_accepted"],
                    100 * s["q_error_accepted"],
                ],
            }
    return {"gentle_fell_more": f"{sum(cases)} of {len(cases)}", "cases": out}


def noise_floor() -> dict:
    out = {}
    for run in ("week1", "week2", "week3", "week4"):
        for d in sorted((RUNS / run).iterdir()):
            m, g = (
                records((run, d.name), "mlx-bf16"),
                records((run, d.name), "gguf-bf16"),
            )
            if not d.is_dir() or d.is_symlink() or m is None or g is None:
                continue
            aligned(m, g)
            out[f"{run}/{d.name}"] = {
                "agreement": float(
                    np.mean([a["pred"] == b["pred"] for a, b in zip(m, g)])
                ),
                "acc_mlx": float(col(m).mean()),
                "acc_gguf": float(col(g).mean()),
            }
    agree = [v["agreement"] for v in out.values()]
    working = [v["agreement"] for v in out.values() if v["acc_mlx"] > 0]
    tuned = [
        v["agreement"]
        for k, v in out.items()
        if v["acc_mlx"] > 0 and not k.endswith("-base")
    ]
    return {
        "range_all": [min(agree), max(agree)],
        "range_nonzero_accuracy": [min(working), max(working)],
        "range_fine_tunes": [min(tuned), max(tuned)],
        "configs": out,
    }


def confidence_strata() -> dict:
    out = {}
    for name, gentle, strong, _ in PAIRS:
        for v in ("mlx-q3", "gguf-Q3_K_M"):
            row = {}
            for label, ref in (("gentle", gentle), ("strong", strong)):
                fp, q = records(ref, fp_of(v)), records(ref, v)
                aligned(fp, q)
                ok = (col(fp) == 1) & (col(fp, "confidence") >= 0.9)
                band = (
                    (col(fp) == 1)
                    & (col(fp, "confidence") >= 0.95)
                    & (col(fp, "confidence") < 0.99)
                )
                row[label] = {
                    "n_confident_correct": int(ok.sum()),
                    "still_correct": float(col(q)[ok].mean()),
                    "n_band_95_99": int(band.sum()),
                    "flip_rate_band_95_99": float(1 - col(q)[band].mean())
                    if band.sum()
                    else None,
                }
            out[f"{name} {v}"] = row
    return out


def final_val_loss(ref: Ref) -> float | None:
    p = RUNS / ref[0] / ref[1] / "train.log"
    if not p.exists():
        return None
    hits = re.findall(r"Iter (\d+): Val loss ([\d.]+)", p.read_text())
    return float(hits[-1][1]) if hits else None


def val_losses() -> dict:
    return {
        name: {"gentle": final_val_loss(g), "strong": final_val_loss(s)}
        for name, g, s, _ in PAIRS
    }


def run_grid(norms: dict[str, float]) -> dict:
    rows, evals = [], 0
    for run in ("week1", "week2", "week3", "week4"):
        for d in sorted((RUNS / run).iterdir()):
            if (
                not d.is_dir()
                or d.is_symlink()
                or d.name.startswith("_")
                or not (d / "eval").exists()
            ):
                continue
            files = sorted((d / "eval").glob("*.jsonl"))
            evals += len(files)
            if d.name.endswith("-base"):
                continue
            train = yaml.safe_load((d / "train.yaml").read_text())
            mlx_bf16 = records((run, d.name), "mlx-bf16")
            rows.append(
                {
                    "round": int(run[-1]),
                    "config": d.name,
                    "kind": train["fine_tune_type"],
                    "lr": train["learning_rate"],
                    "seed": train["seed"],
                    "model": MODELS[d.name.split("-")[0]],
                    "task": Path(train["data"]).name,
                    "acc_mlx_bf16": float(col(mlx_bf16).mean()) if mlx_bf16 else None,
                    "norm": norms.get(d.name),
                    "formats_evaluated": len(files),
                    "failed": d.name in FAILED,
                }
            )
    return {"fine_tunes": len(rows), "evaluation_files": evals, "rows": rows}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    norms = update_norms()
    res = {
        "pairs": pairs_table(),
        "confirmatory_robustness": confirmatory_robustness(),
        "update_norms": norms,
        "predictor_baselines": predictor_baselines(norms["norms"]),
        "nsr_norm_correlation_fit_points": nsr_norm_correlation(norms["norms"]),
        "bands": bands(),
        "kept_count": kept_count(),
        "lr_spearman": lr_spearman(),
        "drift": drift(),
        "noise_floor": noise_floor(),
        "confidence_strata": confidence_strata(),
        "val_loss": val_losses(),
        "run_grid": run_grid(norms["norms"]),
    }
    (OUT / "revision.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("bands", "kept_count")}, indent=1)[:3000])
    print("pairs summary", json.dumps(res["pairs"]["summary"], indent=1))


if __name__ == "__main__":
    main()
