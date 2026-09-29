import json

import numpy as np

from ftquant.week3 import RUNS, VARIANTS, rows, summary

HARD = ["mlx-q4", "mlx-q3", "gguf-Q4_K_M", "gguf-Q3_K_M", "gguf-Q2_K"]
SETTINGS = {
    "OLMo-2 1B, banking77, seed 1": ("olmo1-lora-lowlr-s1", "olmo1-lora-s1"),
    "Qwen3-0.6B, MASSIVE, seed 1": ("mas06-lora-lowlr-s1", "mas06-lora-s1"),
    "Qwen3-4B, banking77": ("q4b-lora-lowlr", "q4b-lora"),
}
SEED0 = {
    "OLMo-2 1B, banking77, seed 1": ("OLMo-2 1B, banking77", "olmo1"),
    "Qwen3-0.6B, MASSIVE, seed 1": ("Qwen3-0.6B, MASSIVE", "mas06"),
}
ARMS = {"olmo1": "OLMo-2 1B", "mas06": "Qwen3-0.6B MASSIVE", "q4b": "Qwen3-4B"}
PREDICTORS = ("pred_v2", "pred_v1", "pred_kld_only", "pred_bits_only")


def base_accuracy(cfg: str) -> dict[str, float]:
    out = {}
    for f in sorted((RUNS / "week4" / cfg / "eval").glob("*.jsonl")):
        recs = [json.loads(line) for line in open(f)]
        out[f.stem] = sum(r["correct"] for r in recs) / len(recs)
    return out


def pattern(settings: dict) -> dict:
    comparisons, same = 0, 0
    bf16_gentle_higher = 0
    misses = []
    for name, s in settings.items():
        g, st = s["LoRA 1e-5"], s["LoRA 1e-4"]
        bf16_gentle_higher += g["acc_bf16"] > st["acc_bf16"]
        for v in HARD:
            comparisons += 1
            if g["accuracy"][v] < st["accuracy"][v]:
                same += 1
            else:
                misses.append([name, v, g["accuracy"][v], st["accuracy"][v]])
    return {
        "settings": len(settings),
        "bf16_gentle_higher": int(bf16_gentle_higher),
        "comparisons": comparisons,
        "gentle_kept_less": same,
        "exceptions": misses,
    }


def errors(pts: list[dict]) -> dict:
    def score(sel: list[dict]) -> dict:
        y = np.clip([p["retention"] for p in sel], 0.0, 1.0)
        out = {"n": len(sel)}
        for k in PREDICTORS:
            yhat = np.array([p[k] for p in sel])
            out[k] = {
                "mae": float(np.mean(np.abs(yhat - y))),
                "safe_agreement": float(np.mean((y >= 0.9) == (yhat >= 0.9))),
            }
        return out

    y = np.clip([p["retention"] for p in pts], 0.0, 1.0)
    err = np.abs(np.array([p["pred_v2"] for p in pts]) - y)
    order = np.argsort(-err)
    wrong_call = [
        {
            "config": p["config"],
            "variant": p["variant"],
            "pred_v2": p["pred_v2"],
            "measured_clipped": float(yt),
            "call": "said safe, was not"
            if p["pred_v2"] >= 0.9
            else "said unsafe, was safe",
        }
        for p, yt in zip(pts, y)
        if (yt >= 0.9) != (p["pred_v2"] >= 0.9)
    ]
    return {
        "all": score(pts),
        "hard_formats": score([p for p in pts if p["variant"] in HARD]),
        "by_arm": {
            arm: score([p for p in pts if p["config"].startswith(arm + "-")])
            for arm in ARMS
        },
        "largest_v2_errors": [
            {
                "config": pts[i]["config"],
                "variant": pts[i]["variant"],
                "pred_v2": pts[i]["pred_v2"],
                "measured": pts[i]["retention"],
                "measured_clipped": float(y[i]),
                "nsr": pts[i]["nsr"],
                "kept": pts[i]["kept"],
                "kld": pts[i]["kld"],
            }
            for i in order[:6]
        ],
        "wrong_safe_calls": wrong_call,
    }


def seeds(res: dict) -> dict:
    w3 = json.loads((RUNS / "week3" / "summary.json").read_text())
    t3 = json.loads((RUNS / "week3" / "verdicts.json").read_text())["W3-H1_tests"]
    t4 = json.loads((RUNS / "week4" / "verdicts.json").read_text())["W4-H1_tests"]
    out = {}
    for name, (w3name, prefix) in SEED0.items():
        cells = {}
        for label in ("LoRA 1e-5", "LoRA 1e-4"):
            s0, s1 = w3[w3name][label], res[name][label]
            cells[label] = {
                "acc_bf16": [s0["acc_bf16"], s1["acc_bf16"]],
                "gain": {v: [s0["gain"][v], s1["gain"][v]] for v in VARIANTS},
                "accuracy": {
                    v: [s0["accuracy"][v], s1["accuracy"][v]] for v in VARIANTS
                },
            }
        cells["test_diff"] = {
            v: [t3[f"{prefix} {v}"]["diff"], t4[f"{prefix} {v}"]["diff"]]
            for v in ("mlx-q3", "gguf-Q3_K_M")
        }
        out[name] = cells
    return out


def main() -> None:
    by = rows("week4/analysis.json")
    res = {
        name: {
            label: summary(by, cfg)
            for label, cfg in (("LoRA 1e-5", gentle), ("LoRA 1e-4", strong))
        }
        for name, (gentle, strong) in SETTINGS.items()
    }
    w3 = json.loads((RUNS / "week3" / "summary.json").read_text())
    everything = {**w3, **res}
    kld = {
        tag: json.loads((RUNS / "kld" / f"{tag}.json").read_text())
        for tag in ("Qwen3-0.6B", "Qwen3-1.7B", "OLMo-2-0425-1B-Instruct", "Qwen3-4B")
    }
    pts = json.loads((RUNS / "week4" / "verdicts.json").read_text())["points"]
    out = {
        "settings": res,
        "pattern_week4": pattern(res),
        "pattern_all": pattern(everything),
        "seeds": seeds(res),
        "q4b_base_accuracy": base_accuracy("q4b-base"),
        "base_kld": {
            tag: {
                v: {**d["mlx"], **d["gguf"]}.get(v)
                for v in ("mlx-q4", "mlx-q3", "gguf-Q3_K_M", "gguf-Q2_K")
            }
            for tag, d in kld.items()
        },
        "predictor": errors(pts),
    }
    (RUNS / "week4" / "summary.json").write_text(json.dumps(out, indent=1))

    head = " ".join(f"{v.split('-', 1)[1]:>7s}" for v in VARIANTS)
    for name, cfgs in res.items():
        print(f"\n{name}")
        for metric in ("gain", "accuracy", "coverage_pts"):
            print(f"  {metric:13s} {'bf16 acc':>8s} {head}")
            for label, s in cfgs.items():
                fmt = "{:7.1f}" if metric == "coverage_pts" else "{:7.3f}"
                vals = " ".join(fmt.format(s[metric][v]) for v in VARIANTS)
                print(f"  {label:13s} {s['acc_bf16']:8.3f} {vals}")
    for key in ("pattern_week4", "pattern_all"):
        p = out[key]
        print(
            f"\n{key}: gentle higher at bf16 in {p['bf16_gentle_higher']}/{p['settings']}, "
            f"kept less in {p['gentle_kept_less']}/{p['comparisons']}, exceptions {p['exceptions']}"
        )
    print("\nseeds (seed 0 -> seed 1)")
    for name, cells in out["seeds"].items():
        print(f"  {name}")
        for label in ("LoRA 1e-5", "LoRA 1e-4"):
            c = cells[label]
            print(
                f"    {label}: bf16 {c['acc_bf16'][0]:.3f} -> {c['acc_bf16'][1]:.3f}; "
                + "; ".join(
                    f"{v} R {c['gain'][v][0]:.3f} -> {c['gain'][v][1]:.3f}, acc kept {c['accuracy'][v][0]:.3f} -> {c['accuracy'][v][1]:.3f}"
                    for v in ("mlx-q3", "gguf-Q3_K_M")
                )
            )
        print(
            "    test diff: "
            + "; ".join(
                f"{v} {a:+.3f} -> {b:+.3f}" for v, (a, b) in cells["test_diff"].items()
            )
        )
    print(
        "\nq4b base accuracy:",
        {k: round(v, 4) for k, v in out["q4b_base_accuracy"].items()},
    )
    print("base KLD:", json.dumps(out["base_kld"]))
    e = out["predictor"]
    for scope in ("all", "hard_formats"):
        print(
            f"\n{scope} n={e[scope]['n']}: "
            + ", ".join(
                f"{k} {e[scope][k]['mae']:.3f}/{e[scope][k]['safe_agreement']:.3f}"
                for k in PREDICTORS
            )
        )
    for arm, s in e["by_arm"].items():
        print(
            f"{arm} n={s['n']}: "
            + ", ".join(
                f"{k} {s[k]['mae']:.3f}/{s[k]['safe_agreement']:.3f}"
                for k in PREDICTORS
            )
        )
    print("\nlargest v2 errors:")
    for r in e["largest_v2_errors"]:
        print(
            f"  {r['config']:20s} {r['variant']:12s} pred {r['pred_v2']:.3f} measured {r['measured']:.3f} (clipped {r['measured_clipped']:.3f}) nsr {r['nsr']:.2f} kept {r['kept']:.3f} kld {r['kld']:.3f}"
        )
    print("wrong safe calls:")
    for r in e["wrong_safe_calls"]:
        print(
            f"  {r['config']:20s} {r['variant']:12s} pred {r['pred_v2']:.3f} measured {r['measured_clipped']:.3f}: {r['call']}"
        )


if __name__ == "__main__":
    main()
