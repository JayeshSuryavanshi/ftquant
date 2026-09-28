import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

RUNS = Path(__file__).resolve().parents[2] / "runs"


def nsr_lookup(run: str, prefix: str) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    mlx = RUNS / run / f"mechanism-{prefix}-mlx.json"
    if mlx.exists():
        for r in json.loads(mlx.read_text()):
            for bits, v in r["noise_to_signal"].items():
                out[(r["config"], f"mlx-q{bits}")] = v
    gg = RUNS / run / f"mechanism-{prefix}-gguf.json"
    if gg.exists():
        for cfg, row in json.loads(gg.read_text())["configs"].items():
            for t, v in row.items():
                if isinstance(v, dict):
                    out[(cfg, f"gguf-{t}")] = v["noise_to_signal"]
    return out


def kld_lookup(model_tag: str) -> dict[str, float]:
    d = json.loads((RUNS / "kld" / f"{model_tag}.json").read_text())
    return {**d["mlx"], **d["gguf"]}


def points(analysis: Path, run: str, prefix: str, model_tag: str) -> list[dict]:
    rows = json.loads(analysis.read_text())["retention"]
    nsr = nsr_lookup(run, prefix)
    kld = kld_lookup(model_tag)
    pts = []
    for r in rows:
        v = r["variant"]
        if r["run"] != run or not r["config"].startswith(prefix + "-"):
            continue
        if v.endswith("bf16") or v.endswith("unfused"):
            continue
        key = (r["config"], v)
        if key in nsr and v in kld:
            pts.append(
                {
                    "config": r["config"],
                    "variant": v,
                    "nsr": nsr[key],
                    "kld": kld[v],
                    "retention": r["retention"],
                }
            )
    return pts


def features(p: list[dict], use_nsr: bool = True) -> np.ndarray:
    rows = []
    for x in p:
        nsr = np.log(max(x["nsr"], 1e-6)) if use_nsr else 0.0
        rows.append([1.0, nsr, np.log(max(x["kld"], 1e-6))])
    return np.array(rows)


def predict(theta: np.ndarray, X: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(X @ theta)))


def fit(
    p: list[dict], use_nsr: bool = True, steps: int = 20000, lr: float = 0.02
) -> np.ndarray:
    X = features(p, use_nsr)
    y = np.clip(np.array([x["retention"] for x in p]), 0.0, 1.0)
    theta = np.array([2.0, -1.0 if use_nsr else 0.0, -1.0])
    mask = np.array([1.0, 1.0 if use_nsr else 0.0, 1.0])
    m = np.zeros(3)
    v = np.zeros(3)
    for t in range(1, steps + 1):
        yhat = predict(theta, X)
        g = mask * (X.T @ (2 * (yhat - y) * yhat * (1 - yhat)) / len(y))
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        theta -= lr * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)
    return theta


def metrics(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    y = np.clip(y, 0.0, 1.0)
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(yhat - y))),
        "safe_agreement": float(np.mean((y >= 0.9) == (yhat >= 0.9))),
    }


def variant_means(p: list[dict]) -> dict[str, float]:
    groups: dict[str, list[float]] = {}
    for x in p:
        groups.setdefault(x["variant"], []).append(min(max(x["retention"], 0.0), 1.0))
    return {k: float(np.mean(v)) for k, v in groups.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--analysis", required=True)
    f.add_argument("--run", default="week1")
    f.add_argument("--prefix", default="q06")
    f.add_argument("--model-tag", default="Qwen3-0.6B")
    f.add_argument("--out", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--predictor", required=True)
    e.add_argument("--analysis", required=True)
    e.add_argument("--run", required=True)
    e.add_argument("--prefix", required=True)
    e.add_argument("--model-tag", required=True)
    args = ap.parse_args()
    p = points(Path(args.analysis), args.run, args.prefix, args.model_tag)
    y = np.array([x["retention"] for x in p])
    if args.cmd == "fit":
        theta = fit(p)
        theta_kld = fit(p, use_nsr=False)
        res = {
            "form": "R = sigmoid(t0 + t1*ln(NSR) + t2*ln(KLD_base))",
            "theta": theta.tolist(),
            "theta_kld_only": theta_kld.tolist(),
            "variant_means": variant_means(p),
            "fit_on": f"{args.run}/{args.prefix}",
            "in_sample": metrics(y, predict(theta, features(p))),
            "in_sample_kld_only": metrics(
                y, predict(theta_kld, features(p, use_nsr=False))
            ),
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1))
        print(
            json.dumps(res),
            "sha256",
            hashlib.sha256(Path(args.out).read_bytes()).hexdigest(),
        )
        return
    pred = json.loads(Path(args.predictor).read_text())
    yhat = predict(np.array(pred["theta"]), features(p))
    ykld = predict(np.array(pred["theta_kld_only"]), features(p, use_nsr=False))
    ymean = np.array([pred["variant_means"].get(x["variant"], np.nan) for x in p])
    for x, a, b, c in zip(p, yhat, ykld, ymean):
        x.update(
            predicted=float(a),
            predicted_kld_only=float(b),
            predicted_bits_only=float(c),
        )
    keep = ~np.isnan(ymean)
    print(
        json.dumps(
            {
                "nsr_model": metrics(y, yhat),
                "kld_only": metrics(y, ykld),
                "bits_only": metrics(y[keep], ymean[keep]),
                "points": p,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
