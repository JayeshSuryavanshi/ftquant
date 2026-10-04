import json
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np

RUNS = Path(__file__).resolve().parents[2] / "runs"
W1, W3, W5 = RUNS / "week1", RUNS / "week3", RUNS / "week5"
MARGIN = 0.08
REPS, SEED = 2000, 0
OLMO_GENTLE = [("week3", "olmo1-lora-lowlr"), ("week4", "olmo1-lora-lowlr-s1")]
OLMO_STRONG = [("week3", "olmo1-lora"), ("week4", "olmo1-lora-s1")]

Spec = tuple[Path, str, str]


def records(run: Path, cfg: str, variant: str) -> list[dict]:
    p = run / cfg / "eval" / f"{variant}.jsonl"
    if not p.exists():
        raise SystemExit(f"missing {p.relative_to(RUNS)}")
    return [json.loads(line) for line in open(p)]


def correct(*specs: Spec) -> list[np.ndarray]:
    recs = [records(*s) for s in specs]
    texts = [r["text"] for r in recs[0]]
    for s, rs in zip(specs, recs):
        if [r["text"] for r in rs] != texts:
            raise SystemExit(f"item order differs: {s[0].name}/{s[1]}/{s[2]}")
    return [np.array([r["correct"] for r in rs], dtype=float) for rs in recs]


def accuracy(rs: list[dict]) -> float:
    return float(np.mean([r["correct"] for r in rs]))


def agreement(a: list[dict], b: list[dict]) -> float:
    return float(np.mean([x["pred"] == y["pred"] for x, y in zip(a, b, strict=True)]))


def share(x_q, base_q, ft_fp, base_fp) -> float:
    gain = ft_fp.mean() - base_fp.mean()
    return float((x_q.mean() - base_q.mean()) / gain) if gain > 0 else float("nan")


def boot(fn: Callable[..., float], arrays: list[np.ndarray]) -> list[float]:
    point = fn(*arrays)
    rng = np.random.default_rng(SEED)
    n = len(arrays[0])
    vals = []
    for _ in range(REPS):
        i = rng.integers(0, n, n)
        vals.append(fn(*(a[i] for a in arrays)))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return [float(point), float(lo), float(hi)]


def diff_shares(a, b, base_q, ft_fp, base_fp) -> float:
    return share(a, base_q, ft_fp, base_fp) - share(b, base_q, ft_fp, base_fp)


def base_reference(earlier: Path, cfg: str, bits: int) -> tuple[Spec, dict]:
    variant = f"mlx-q{bits}"
    old, new = records(earlier, cfg, variant), records(W5, cfg, variant)
    same = accuracy(old) == accuracy(new)
    check = {
        "acc_earlier": accuracy(old),
        "acc_week5": accuracy(new),
        "pred_agreement": agreement(old, new),
        "reference": "earlier evaluation" if same else "week-5 evaluation (deviation)",
    }
    return ((earlier if same else W5), cfg, variant), check


def unfused_rows() -> dict:
    rows = {}
    for bits in (3, 4):
        base_spec, check = base_reference(W3, "olmo1-base", bits)
        rows[f"olmo1-base mlx-q{bits} check"] = check
        for week, cfg in OLMO_GENTLE + OLMO_STRONG:
            run = RUNS / week
            arrs = correct(
                (W5, cfg, f"mlx-q{bits}-unfused"),
                (run, cfg, f"mlx-q{bits}"),
                base_spec,
                (run, cfg, "mlx-bf16"),
                (W3, "olmo1-base", "mlx-bf16"),
            )
            unf, fus, bq, ft, bb = arrs
            rows[f"{cfg} mlx-q{bits}"] = {
                "R_unfused": share(unf, bq, ft, bb),
                "R_fused": share(fus, bq, ft, bb),
                "acc_unfused": float(unf.mean()),
                "acc_fused": float(fus.mean()),
                "diff_unfused_minus_fused": boot(diff_shares, arrs),
            }
    return rows


def h1(rows: dict) -> dict:
    tests = {}
    for _, cfg in OLMO_GENTLE:
        point, lo, hi = rows[f"{cfg} mlx-q3"]["diff_unfused_minus_fused"]
        if hi < MARGIN:
            verdict = "supported"
        elif lo > MARGIN:
            verdict = "not supported"
        else:
            verdict = "inconclusive"
        tests[cfg] = {"diff_unfused_minus_fused": [point, lo, hi], "verdict": verdict}
    vs = [t["verdict"] for t in tests.values()]
    if all(v == "supported" for v in vs):
        overall = "supported"
    elif "not supported" in vs:
        overall = "not supported"
    else:
        overall = "inconclusive"
    return {"verdict": overall, "margin": MARGIN, "tests": tests}


def gap_change(gq, gf, gft, sq, sf, sft, bq, bb) -> float:
    qlora_gap = share(sq, bq, sft, bb) - share(gq, bq, gft, bb)
    fused_gap = share(sf, bq, sft, bb) - share(gf, bq, gft, bb)
    return qlora_gap - fused_gap


def val_loss(run: Path, cfg: str) -> float | None:
    p = run / cfg / "train.log"
    if not p.exists():
        return None
    hits = re.findall(r"Iter (\d+): Val loss ([\d.]+)", p.read_text())
    return float(hits[-1][1]) if hits else None


def qlora_rows() -> dict:
    rows = {}
    for bits in (4, 3):
        base_spec, check = base_reference(W1, "q06-base", bits)
        rows[f"q06-base mlx-q{bits} check"] = check
        for suffix in ("lora-lowlr", "lora"):
            name = f"q06q{bits}-{suffix}"
            arrs = correct(
                (W5, name, f"mlx-q{bits}-qlora"),
                (W1, f"q06-{suffix}", f"mlx-q{bits}"),
                base_spec,
                (W1, f"q06-{suffix}", "mlx-bf16"),
                (W1, "q06-base", "mlx-bf16"),
            )
            ql, fus, bq, ft, bb = arrs
            rows[name] = {
                "S_qlora": share(ql, bq, ft, bb),
                "R_fused": share(fus, bq, ft, bb),
                "acc_qlora": float(ql.mean()),
                "acc_fused": float(fus.mean()),
                "acc_base_q": float(bq.mean()),
                "diff_S_minus_R": boot(diff_shares, arrs),
                "val_loss_qlora": val_loss(W5, name),
                "val_loss_bf16_trained": val_loss(W1, f"q06-{suffix}"),
            }
        arrs = correct(
            (W5, f"q06q{bits}-lora-lowlr", f"mlx-q{bits}-qlora"),
            (W1, "q06-lora-lowlr", f"mlx-q{bits}"),
            (W1, "q06-lora-lowlr", "mlx-bf16"),
            (W5, f"q06q{bits}-lora", f"mlx-q{bits}-qlora"),
            (W1, "q06-lora", f"mlx-q{bits}"),
            (W1, "q06-lora", "mlx-bf16"),
            base_spec,
            (W1, "q06-base", "mlx-bf16"),
        )
        rows[f"gap change mlx-q{bits} (strong minus gentle, QLoRA minus fused)"] = boot(
            gap_change, arrs
        )
    return rows


def h2(rows: dict) -> dict:
    point, lo, hi = rows["q06q4-lora-lowlr"]["diff_S_minus_R"]
    if lo > 0:
        verdict = "supported"
    elif hi < 0 or point <= 0:
        verdict = "not supported"
    else:
        verdict = "inconclusive"
    return {"verdict": verdict, "diff_S_minus_R": [point, lo, hi]}


def repeatability() -> dict:
    np4 = records(W5, "q06-lora-lowlr", "gguf-Q3_K_M-np4")
    np1 = records(W5, "q06-lora-lowlr", "gguf-Q3_K_M-np1")
    wk1 = records(W1, "q06-lora-lowlr", "gguf-Q3_K_M")
    return {
        "acc_np4": accuracy(np4),
        "acc_np1": accuracy(np1),
        "acc_week1_np4": accuracy(wk1),
        "agree_np4_np1": agreement(np4, np1),
        "agree_np4_week1": agreement(np4, wk1),
        "agree_np1_week1": agreement(np1, wk1),
    }


def constrained() -> dict:
    out = {}
    for cfg in ("q06-lora-lowlr", "q06-lora"):
        row = {}
        for t in ("bf16", "Q3_K_M", "Q2_K"):
            lab, unc = (
                records(W5, cfg, f"gguf-{t}-labels"),
                records(W1, cfg, f"gguf-{t}"),
            )
            row[t] = {
                "acc_labels_only": accuracy(lab),
                "acc_unconstrained": accuracy(unc),
                "valid_unconstrained": float(np.mean([r["valid"] for r in unc])),
            }
        for t in ("Q3_K_M", "Q2_K"):
            q, f = correct((W5, cfg, f"gguf-{t}-labels"), (W5, cfg, "gguf-bf16-labels"))
            uq, uf = correct((W1, cfg, f"gguf-{t}"), (W1, cfg, "gguf-bf16"))
            row[t]["A_labels_only"] = boot(
                lambda a, b: float(a.mean() / b.mean()), [q, f]
            )
            row[t]["A_unconstrained"] = boot(
                lambda a, b: float(a.mean() / b.mean()), [uq, uf]
            )
        out[cfg] = row
    return out


def new_evaluations() -> dict:
    out = {}
    for p in sorted(W5.glob("*/eval/*.jsonl")):
        rs = [json.loads(line) for line in open(p)]
        out[f"{p.parent.parent.name}/{p.stem}"] = {
            "n": len(rs),
            "acc": accuracy(rs),
            "valid": float(np.mean([r["valid"] for r in rs])),
        }
    return out


def main() -> None:
    u, q = unfused_rows(), qlora_rows()
    out = {
        "W5-H1": h1(u),
        "W5-H2": h2(q),
        "exploratory": {
            "unfused": u,
            "qlora": q,
            "repeatability": repeatability(),
            "labels_only": constrained(),
            "evaluations": new_evaluations(),
        },
    }
    (W5 / "verdicts.json").write_text(json.dumps(out, indent=1))
    for h in ("W5-H1", "W5-H2"):
        print(h, out[h]["verdict"])
    for cfg, t in out["W5-H1"]["tests"].items():
        p, lo, hi = t["diff_unfused_minus_fused"]
        print(
            f"  {cfg} mlx-q3 R(unfused)-R(fused) {p:+.3f} [{lo:+.3f}, {hi:+.3f}] {t['verdict']}"
        )
    p, lo, hi = out["W5-H2"]["diff_S_minus_R"]
    print(f"  q06q4-lora-lowlr S-R {p:+.3f} [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
