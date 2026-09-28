import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "runs"
TARGET_ERROR = 0.05


def load(run: Path, config: str, variant: str) -> dict[str, np.ndarray] | None:
    p = run / config / "eval" / f"{variant}.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(line) for line in open(p)]
    return {
        "correct": np.array([r["correct"] for r in rows], dtype=float),
        "conf": np.array([r["confidence"] for r in rows], dtype=float),
        "valid": np.array([r["valid"] for r in rows], dtype=float),
    }


def base_run(run: Path, prefix: str) -> Path:
    return run if (run / f"{prefix}-base").exists() else ROOT / "week1"


def reference(variant: str) -> str:
    return variant.removesuffix("-unfused")


def full_precision(variant: str) -> str:
    return "gguf-bf16" if variant.startswith("gguf") else "mlx-bf16"


def retention(ft_q, base_q, ft_fp, base_fp) -> float:
    gain = ft_fp.mean() - base_fp.mean()
    return float((ft_q.mean() - base_q.mean()) / gain) if gain > 0 else float("nan")


def bootstrap(
    ft_q, base_q, ft_fp, base_fp, reps: int = 2000, seed: int = 0
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(ft_q)
    vals = [
        retention(ft_q[i], base_q[i], ft_fp[i], base_fp[i])
        for i in (rng.integers(0, n, n) for _ in range(reps))
    ]
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def fit_tau(
    conf: np.ndarray, correct: np.ndarray, target: float = TARGET_ERROR
) -> float:
    order = np.argsort(-conf)
    errs = np.cumsum(1 - correct[order]) / np.arange(1, len(order) + 1)
    ok = np.where(errs <= target)[0]
    return float(conf[order][ok[-1]]) if len(ok) else float(np.inf)


def gate(conf: np.ndarray, correct: np.ndarray, tau: float) -> tuple[float, float]:
    acc = conf >= tau
    cov = float(acc.mean())
    err = float(1 - correct[acc].mean()) if acc.any() else float("nan")
    return cov, err


def drift(fp: dict, q: dict) -> dict[str, float]:
    n = len(fp["correct"])
    cal = np.arange(n) % 2 == 0
    test = ~cal
    tau = fit_tau(fp["conf"][cal], fp["correct"][cal])
    fp_cov, fp_err = gate(fp["conf"][test], fp["correct"][test], tau)
    q_cov, q_err = gate(q["conf"][test], q["correct"][test], tau)
    oracle_tau = fit_tau(q["conf"][cal], q["correct"][cal])
    o_cov, o_err = gate(q["conf"][test], q["correct"][test], oracle_tau)
    return {
        "tau": tau,
        "fp_coverage": fp_cov,
        "fp_error_accepted": fp_err,
        "q_coverage": q_cov,
        "q_error_accepted": q_err,
        "refit_coverage": o_cov,
        "refit_error_accepted": o_err,
    }


def recovery(
    q: dict, sizes=(25, 50, 100, 200, 400), reps: int = 300, seed: int = 0
) -> dict[int, dict[str, float]]:
    rng = np.random.default_rng(seed)
    n = len(q["correct"])
    cal_idx = np.where(np.arange(n) % 2 == 0)[0]
    test = np.arange(n) % 2 == 1
    out = {}
    for k in sizes:
        covs, errs, within = [], [], []
        for _ in range(reps):
            s = rng.choice(cal_idx, size=k, replace=False)
            tau = fit_tau(q["conf"][s], q["correct"][s])
            cov, err = gate(q["conf"][test], q["correct"][test], tau)
            covs.append(cov)
            errs.append(err if err == err else 0.0)
            within.append(err <= TARGET_ERROR + 0.01 if err == err else True)
        out[k] = {
            "coverage": float(np.mean(covs)),
            "error_accepted": float(np.mean(errs)),
            "p_error_within_1pt": float(np.mean(within)),
        }
    return out


def table(runs: list[Path]) -> list[dict]:
    rows = []
    for run in runs:
        for cfg_dir in sorted(
            p for p in run.iterdir() if p.is_dir() and (p / "eval").exists()
        ):
            cfg = cfg_dir.name
            if cfg.endswith("-base"):
                continue
            prefix = cfg.split("-")[0]
            brun = base_run(run, prefix)
            base_cfg = f"{prefix}-base"
            for f in sorted((cfg_dir / "eval").glob("*.jsonl")):
                v = f.stem
                fpv = full_precision(v)
                ft_q, ft_fp = load(run, cfg, v), load(run, cfg, fpv)
                base_q, base_fp = (
                    load(brun, base_cfg, reference(v)),
                    load(brun, base_cfg, fpv),
                )
                if any(x is None for x in (ft_q, ft_fp, base_q, base_fp)):
                    continue
                r = retention(
                    ft_q["correct"],
                    base_q["correct"],
                    ft_fp["correct"],
                    base_fp["correct"],
                )
                lo, hi = bootstrap(
                    ft_q["correct"],
                    base_q["correct"],
                    ft_fp["correct"],
                    base_fp["correct"],
                )
                row = {
                    "run": run.name,
                    "config": cfg,
                    "variant": v,
                    "acc_ft": ft_q["correct"].mean(),
                    "acc_base": base_q["correct"].mean(),
                    "valid_ft": ft_q["valid"].mean(),
                    "retention": r,
                    "ci": [lo, hi],
                    **drift(ft_fp, ft_q),
                }
                if not v.endswith("bf16"):
                    row["recovery"] = recovery(ft_q)
                rows.append(row)
    return rows


def noise_floor(runs: list[Path]) -> list[dict]:
    out = []
    for run in runs:
        for cfg_dir in sorted(p for p in run.iterdir() if p.is_dir()):
            a, b = (
                cfg_dir / "eval" / "mlx-bf16.jsonl",
                cfg_dir / "eval" / "gguf-bf16.jsonl",
            )
            if a.exists() and b.exists():
                ra = [json.loads(line) for line in open(a)]
                rb = [json.loads(line) for line in open(b)]
                out.append(
                    {
                        "run": run.name,
                        "config": cfg_dir.name,
                        "pred_agreement": float(
                            np.mean([x["pred"] == y["pred"] for x, y in zip(ra, rb)])
                        ),
                        "acc_mlx": float(np.mean([x["correct"] for x in ra])),
                        "acc_gguf": float(np.mean([x["correct"] for x in rb])),
                    }
                )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["week1"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    runs = [ROOT / r for r in args.runs if (ROOT / r).exists()]
    res = {"noise_floor": noise_floor(runs), "retention": table(runs)}
    Path(args.out).write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote {args.out}: {len(res['retention'])} rows")


if __name__ == "__main__":
    main()
