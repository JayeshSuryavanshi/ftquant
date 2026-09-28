import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

RUNS = Path(__file__).resolve().parents[2] / "runs"
FIT_RUNS = [
    ("week1", "q06", "Qwen3-0.6B"),
    ("week1", "q17", "Qwen3-1.7B"),
    ("week2", "q06", "Qwen3-0.6B"),
    ("week2", "q17", "Qwen3-1.7B"),
]


def mech_lookup(run: str, prefix: str) -> dict[tuple[str, str], tuple[float, float]]:
    out: dict[tuple[str, str], tuple[float, float]] = {}
    mlx = RUNS / run / f"mechanism-{prefix}-mlx.json"
    if mlx.exists():
        for r in json.loads(mlx.read_text()):
            for b, v in r["noise_to_signal"].items():
                out[(r["config"], f"mlx-q{b}")] = (v, r["signal_retained"][b])
    gg = RUNS / run / f"mechanism-{prefix}-gguf.json"
    if gg.exists():
        for cfg, row in json.loads(gg.read_text())["configs"].items():
            for t, v in row.items():
                if isinstance(v, dict):
                    out[(cfg, f"gguf-{t}")] = (
                        v["noise_to_signal"],
                        v["signal_retained"],
                    )
    return out


def kld_lookup(model_tag: str) -> dict[str, float]:
    d = json.loads((RUNS / "kld" / f"{model_tag}.json").read_text())
    return {**d["mlx"], **d["gguf"]}


def points(
    analysis: Path, runs: list[tuple[str, str, str]], variants: set[str] | None = None
) -> list[dict]:
    rows = json.loads(analysis.read_text())["retention"]
    pts = []
    for run, prefix, tag in runs:
        mech = mech_lookup(run, prefix)
        kld = kld_lookup(tag)
        for r in rows:
            v = r["variant"]
            if r["run"] != run or not r["config"].startswith(prefix + "-"):
                continue
            if (
                v.endswith("bf16")
                or v.endswith("unfused")
                or math.isnan(r["retention"])
            ):
                continue
            if variants is not None and v not in variants:
                continue
            key = (r["config"], v)
            if key in mech and v in kld:
                nsr, kept = mech[key]
                pts.append(
                    {
                        "config": r["config"],
                        "variant": v,
                        "nsr": nsr,
                        "kept": kept,
                        "kld": kld[v],
                        "retention": r["retention"],
                    }
                )
    return pts


def ln(x: float) -> float:
    return math.log(max(x, 1e-6))


FORMS = {
    "nsr+kld": lambda p: [1.0, ln(p["nsr"]), ln(p["kld"])],
    "snr+kld": lambda p: [1.0, ln(p["kept"]) - ln(p["nsr"]), ln(p["kld"])],
    "nsr+kept+kld": lambda p: [1.0, ln(p["nsr"]), ln(p["kept"]), ln(p["kld"])],
    "snr*kld": lambda p: [
        1.0,
        ln(p["kept"]) - ln(p["nsr"]),
        ln(p["kld"]),
        (ln(p["kept"]) - ln(p["nsr"])) * ln(p["kld"]),
    ],
    "kld-only": lambda p: [1.0, ln(p["kld"])],
}


def design(form: str, pts: list[dict]) -> np.ndarray:
    return np.array([FORMS[form](p) for p in pts])


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def fit(
    X: np.ndarray, y: np.ndarray, steps: int = 20000, lr: float = 0.02
) -> np.ndarray:
    y = np.clip(y, 0.0, 1.0)
    theta = np.zeros(X.shape[1])
    theta[0] = 2.0
    theta[1:] = -1.0 if X.shape[1] <= 3 else 0.0
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    for t in range(1, steps + 1):
        yhat = sigmoid(X @ theta)
        g = X.T @ (2 * (yhat - y) * yhat * (1 - yhat)) / len(y)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        theta -= lr * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)
    return theta


def variant_means(pts: list[dict]) -> dict[str, float]:
    groups: dict[str, list[float]] = {}
    for p in pts:
        groups.setdefault(p["variant"], []).append(min(max(p["retention"], 0.0), 1.0))
    return {k: float(np.mean(v)) for k, v in groups.items()}


def metrics(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    y = np.clip(y, 0.0, 1.0)
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(yhat - y))),
        "safe_agreement": float(np.mean((y >= 0.9) == (yhat >= 0.9))),
    }


def loco(form: str, pts: list[dict]) -> dict[str, float]:
    configs = sorted({p["config"] for p in pts})
    y = np.array([p["retention"] for p in pts])
    yhat = np.zeros(len(pts))
    for c in configs:
        test = np.array([p["config"] == c for p in pts])
        if form == "bits-only":
            means = variant_means([p for p, t in zip(pts, test) if not t])
            yhat[test] = [
                means.get(p["variant"], np.nan) for p, t in zip(pts, test) if t
            ]
            continue
        X = design(form, pts)
        theta = fit(X[~test], y[~test])
        yhat[test] = sigmoid(X[test] @ theta)
    keep = ~np.isnan(yhat)
    return metrics(y[keep], yhat[keep])


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--analysis", default=str(RUNS / "analysis-all.json"))
    f = sub.add_parser("fit")
    f.add_argument("--analysis", default=str(RUNS / "analysis-all.json"))
    f.add_argument("--form", required=True, choices=sorted(FORMS))
    f.add_argument("--out", required=True)
    args = ap.parse_args()
    pts = points(Path(args.analysis), FIT_RUNS)
    if args.cmd == "select":
        print(f"{len(pts)} points from {len({p['config'] for p in pts})} fine-tunes")
        for form in [*FORMS, "bits-only"]:
            m = loco(form, pts)
            print(
                f"{form:14s} leave-one-fine-tune-out MAE {m['mae']:.3f}  safe {m['safe_agreement']:.3f}"
            )
        return
    y = np.array([p["retention"] for p in pts])
    theta = fit(design(args.form, pts), y)
    theta_kld = fit(design("kld-only", pts), y)
    res = {
        "form": args.form,
        "theta": theta.tolist(),
        "theta_kld_only": theta_kld.tolist(),
        "variant_means": variant_means(pts),
        "fit_on": [list(r) for r in FIT_RUNS],
        "n_fit": len(pts),
        "in_sample": metrics(y, sigmoid(design(args.form, pts) @ theta)),
        "loco": loco(args.form, pts),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(
        json.dumps(res),
        "sha256",
        hashlib.sha256(Path(args.out).read_bytes()).hexdigest(),
    )


if __name__ == "__main__":
    main()
