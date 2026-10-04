import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = Path(__file__).resolve().parent / "tables"
HARD = ["mlx-q4", "mlx-q3", "gguf-Q4_K_M", "gguf-Q3_K_M", "gguf-Q2_K"]


def load(path: str) -> dict:
    return json.loads((RUNS / path).read_text())


def num(x: float, digits: int = 3, sign: bool = False) -> str:
    s = f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}"
    return "$" + s.replace("-", "{-}") + "$" if s.startswith("-") else f"${s}$"


def ci(lo: float, hi: float, digits: int = 3, sign: bool = False) -> str:
    return f"[{num(lo, digits, sign)}, {num(hi, digits, sign)}]"


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}\\%"


def acc(path: Path) -> float:
    recs = [json.loads(line) for line in open(path)]
    return sum(r["correct"] for r in recs) / len(recs)


def write(name: str, rows: list[str]) -> None:
    (OUT / name).write_text("\n".join(rows) + "\n")


def bases() -> None:
    spec = [
        (
            "Qwen3-0.6B",
            "banking77",
            "week1/q06-base",
            "1, 2, 5",
            "LoRA $3\\times10^{-6}$ to $3\\times10^{-4}$, full $3\\times10^{-6}$ to $3\\times10^{-5}$; LoRA on a quantized base (round 5)",
        ),
        (
            "Qwen3-1.7B",
            "banking77",
            "week1/q17-base",
            "1, 2",
            "LoRA $10^{-5}$ and $10^{-4}$",
        ),
        (
            "OLMo-2 1B",
            "banking77",
            "week3/olmo1-base",
            "3, 4, 5",
            "LoRA $10^{-5}$ and $10^{-4}$",
        ),
        (
            "Qwen3-0.6B",
            "MASSIVE",
            "week3/mas06-base",
            "3, 4",
            "LoRA $10^{-5}$ and $10^{-4}$, full $10^{-5}$",
        ),
        (
            "Qwen3-4B",
            "banking77",
            "week4/q4b-base",
            "4",
            "LoRA $10^{-5}$ and $10^{-4}$",
        ),
    ]
    rows = []
    for model, task, run, rounds, fts in spec:
        a = acc(RUNS / run / "eval" / "mlx-bf16.jsonl")
        rows.append(f"{model} & {task} & {rounds} & {pct(a)} & {fts} \\\\")
    write("bases.tex", rows)


def formats() -> None:
    mlx = load("bpw/mlx-q06.json")["bpw"]
    g06 = load("week1/mechanism-q06-gguf.json")["bpw"]
    g17 = load("week1/mechanism-q17-gguf.json")["bpw"]
    rows = []
    for b, rounds in (
        ("q8", "1"),
        ("q6", "1 to 4"),
        ("q4", "1 to 5"),
        ("q3", "1 to 5"),
        ("q2", "1"),
    ):
        rows.append(
            f"MLX affine, group 64 & {b[1:]}-bit & {mlx[b]:.2f} & & {rounds} \\\\"
        )
    rows.append("\\addlinespace")
    for t, rounds in (
        ("Q8_0", "1"),
        ("Q6_K", "1 to 4"),
        ("Q4_K_M", "1 to 4"),
        ("Q3_K_M", "1 to 5"),
        ("Q2_K", "1 to 5"),
    ):
        rows.append(
            f"GGUF k-quants & {t.replace('_', chr(92) + '_')} & {g06[t]:.2f} & {g17[t]:.2f} & {rounds} \\\\"
        )
    write("formats.tex", rows)


def week1() -> None:
    c = load("week1/confirmatory.json")
    h1 = c["H1"]["detail"]
    h1s = "; ".join(
        f"{lab} {num(h1[k][0], 2)} {ci(*h1[k][1], 2)}"
        for k, lab in (
            ("mlx-q8", "MLX 8-bit"),
            ("mlx-q6", "MLX 6-bit"),
            ("gguf-Q8_0", "Q8\\_0"),
            ("gguf-Q6_K", "Q6\\_K"),
        )
    )
    h2 = c["H2"]["detail"]
    h2s = (
        f"3 vs 4 bits: MLX {num(h2['mlx-q3 vs mlx-q4']['R3'], 2)} vs {num(h2['mlx-q3 vs mlx-q4']['R4'], 2)}, "
        f"GGUF {num(h2['gguf-Q3_K_M vs gguf-Q4_K_M']['R3'], 2)} vs {num(h2['gguf-Q3_K_M vs gguf-Q4_K_M']['R4'], 2)}; "
        f"2 bits: MLX {num(h2['mlx-q2']['R'], 2)}, Q2\\_K {num(h2['gguf-Q2_K']['R'], 2)}"
    )
    h3 = c["H3"]["detail"]
    h3s = "; ".join(
        f"{b}-bit {num(h3[k]['paired_diff_unfused_minus_fused'][0], 3, True)} {ci(*h3[k]['paired_diff_unfused_minus_fused'][1:], 3, True)}"
        for k, b in (("mlx-q4", 4), ("mlx-q3", 3))
    )
    h4 = c["H4"]["detail"]
    h4s = "; ".join(
        f"{lab} coverage {num(h4[k]['coverage_change_pts'], 1, True)} pts"
        for k, lab in (("mlx-q4", "MLX 4-bit"), ("gguf-Q4_K_M", "Q4\\_K\\_M"))
    )
    rows = [
        f"H1 & $R \\ge 0.95$ at 8 and 6 bits, every lower bound $\\ge 0.90$ & {h1s} & {c['H1']['verdict']} \\\\",
        f"H2 & $R$ lower at 3 bits than at 4, and $R < 0.5$ at 2 bits & {h2s} & {c['H2']['verdict']} \\\\",
        f"H3 & unfused adapter beats fused at MLX 4 and 3 bits & unfused minus fused: {h3s} & {c['H3']['verdict']} \\\\",
        f"H4 & confidence gate drifts at both 4-bit formats (error $+1$ pt or coverage $\\pm 3$ pts) & {h4s} & {c['H4']['verdict']} \\\\",
    ]
    write("week1.tex", rows)


def lrsweep() -> None:
    cols = ["mlx-q4", "mlx-q3", "gguf-Q4_K_M", "gguf-Q3_K_M", "gguf-Q2_K"]
    names = {
        3e-06: "$3\\times10^{-6}$",
        1e-05: "$10^{-5}$",
        3e-05: "$3\\times10^{-5}$",
        1e-04: "$10^{-4}$",
    }
    rows = []
    for c in load("week2/summary.json")["configs"]:
        if c["model"] != "0.6B" or c["seed"] != 0 or c["lr"] > 1e-4:
            continue
        vals = " & ".join(num(c["gain_retention"][v], 2) for v in cols)
        rows.append(
            (
                c["kind"] != "LoRA",
                c["lr"],
                f"{c['kind']} & {names[c['lr']]} & {c['delta_norm']:.1f} & {pct(c['acc_ft_bf16'])} & {vals} \\\\",
            )
        )
    rows.sort()
    out = [r for _, _, r in rows]
    first_full = next(i for i, (full, _, _) in enumerate(rows) if full)
    out.insert(first_full, "\\addlinespace")
    write("lrsweep.tex", out)


def replication() -> None:
    w3 = load("week3/verdicts.json")["W3-H1_tests"]
    w4 = load("week4/verdicts.json")
    names = {
        "olmo1": "OLMo-2 1B, banking77",
        "mas06": "Qwen3-0.6B, MASSIVE",
        "q4b": "Qwen3-4B, banking77",
    }
    fmt = {"mlx-q3": "MLX 3-bit", "gguf-Q3_K_M": "Q3\\_K\\_M"}
    robust = revision()["confirmatory_robustness"]
    rows = []
    for test, tests, seed in (
        ("W3-H1", w3, "0"),
        ("W4-H1", w4["W4-H1_tests"], "1"),
        ("W4-H2", w4["W4-H2_tests"], "0"),
    ):
        for label, t in tests.items():
            prefix, v = label.split(" ")
            setting = names[prefix] + (", seed 1" if test == "W4-H1" else "")
            d, lo, hi = robust[f"{test} {setting} {v}"][
                "accuracy_drop_gentle_minus_strong_pts"
            ]
            rows.append(
                f"{test} & {setting} & {fmt[v]} & {num(t['R_strong'])} & {num(t['R_gentle'])} & "
                f"{num(t['diff'], 3, True)} {ci(*t['ci'], 3, True)} & {num(d, 1, True)} {ci(lo, hi, 1, True)} \\\\"
            )
        rows.append("\\addlinespace")
    write("replication.tex", rows[:-1])


def lr_tex(lr: float) -> str:
    mantissa, exponent = f"{lr:.0e}".split("e")
    power = f"10^{{{int(exponent)}}}"
    return f"${power}$" if mantissa == "1" else f"${mantissa}\\times{power}$"


PAIR_ORDER = [
    "Qwen3-0.6B, banking77",
    "Qwen3-0.6B, banking77, seed 1",
    "Qwen3-1.7B, banking77",
    "Qwen3-4B, banking77",
    "OLMo-2 1B, banking77",
    "OLMo-2 1B, banking77, seed 1",
    "Qwen3-0.6B, MASSIVE",
    "Qwen3-0.6B, MASSIVE, seed 1",
]


def revision() -> dict:
    return load("revision/revision.json")


def pattern() -> None:
    pairs = revision()["pairs"]["pairs"]
    rows = []
    for name in PAIR_ORDER:
        p = pairs[name]
        bf = p["bf16"]["mlx-bf16"]
        cells = []
        for v in HARD:
            c = p["formats"][v]
            _, lo, hi = c["A_gentle_minus_strong"]
            mark = "$^\\dagger$" if lo <= 0 <= hi else ""
            cells.append(f"{c['A_gentle']:.2f} / {c['A_strong']:.2f}{mark}")
        rows.append(
            f"{name} & {100 * bf['gentle']:.1f} / {100 * bf['strong']:.1f} & "
            + " & ".join(cells)
            + " \\\\"
        )
    write("pattern.tex", rows)


def absolute() -> None:
    pairs = revision()["pairs"]["pairs"]
    rows = []
    for name in PAIR_ORDER:
        p = pairs[name]
        cells = []
        for v in HARD:
            c = p["formats"][v]
            g, s = 100 * c["acc_gentle"], 100 * c["acc_strong"]
            cells.append(
                f"\\textbf{{{g:.1f}}} / {s:.1f}"
                if g > s
                else f"{g:.1f} / \\textbf{{{s:.1f}}}"
            )
        bf = p["bf16"]["mlx-bf16"]
        rows.append(
            f"{name} & {100 * bf['gentle']:.1f} / {100 * bf['strong']:.1f} & "
            + " & ".join(cells)
            + " \\\\"
        )
    write("absolute.tex", rows)


def bands() -> None:
    labels = {
        "base damage < 1, kept >= 60%, NSR < 3": "base damage $< 1$, update kept $\\ge 60\\%$, NSR $< 3$",
        "base damage < 1, kept >= 60%, NSR 3 to 5": "base damage $< 1$, update kept $\\ge 60\\%$, NSR 3 to 5",
        "base damage < 1, kept >= 60%, NSR >= 5": "base damage $< 1$, update kept $\\ge 60\\%$, NSR $\\ge 5$",
        "kept < 60%": "update kept $< 60\\%$",
        "kept >= 60%, base damage >= 1": "update kept $\\ge 60\\%$, base damage $\\ge 1$",
    }
    b = revision()["bands"]["bands"]
    write(
        "bands.tex",
        [
            f"{labels[k]} & {v['pairs']} & {v['kept_ge_90']} & {100 * v['lowest']:.0f}\\% \\\\"
            for k, v in b.items()
        ],
    )


def baselines() -> None:
    rounds = revision()["predictor_baselines"]["rounds"]
    names = [
        ("pred_v2", "Predictor v2 (pre-registered)"),
        ("pred_v1", "Predictor v1"),
        ("pred_kld_only", "Base damage only (pre-registered baseline)"),
        ("pred_bits_only", "Format mean (pre-registered baseline)"),
        ("pred_const", "Constant, $\\hat R = 1$ (post hoc)"),
        ("pred_norm_kld", "Update norm and base damage (post hoc)"),
    ]
    rows = []
    for key, label in names:
        cells = []
        for rd in ("round 3", "round 4"):
            m = rounds[rd]["all"][key]
            cells.append(
                f"{num(m['mae'])} {ci(*m['mae_ci'])} & {num(m['safe_agreement'], 2)}"
            )
        rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
    write("baselines.tex", rows)


def grid() -> None:
    g = revision()["run_grid"]["rows"]
    kinds = {"lora": "LoRA", "full": "full"}
    tasks = {"banking77": "banking77", "massive": "MASSIVE"}
    models = {
        "Qwen/Qwen3-0.6B": "Qwen3-0.6B",
        "Qwen/Qwen3-1.7B": "Qwen3-1.7B",
        "Qwen/Qwen3-4B": "Qwen3-4B",
        "allenai/OLMo-2-0425-1B-Instruct": "OLMo-2 1B",
    }
    rows = []
    for r in sorted(
        g,
        key=lambda r: (
            r["round"],
            r["model"],
            r["task"],
            r["kind"],
            r["lr"],
            r["seed"],
        ),
    ):
        lr = lr_tex(r["lr"])
        acc = pct(r["acc_mlx_bf16"]) if r["acc_mlx_bf16"] is not None else "--"
        norm = f"{r['norm']:.1f}" if r["norm"] is not None else "--"
        note = "failed to learn" if r["failed"] else ""
        rows.append(
            f"{r['round']} & {models[r['model']]} & {tasks[r['task']]} & {kinds[r['kind']]} & {lr} & {r['seed']} & "
            f"{acc} & {norm} & {r['formats_evaluated']} & {note} \\\\"
        )
    write("grid.tex", rows)


def predictor() -> None:
    w2 = load("predictor/verdicts.json")
    w3 = load("week3/verdicts.json")
    w4 = load("week4/verdicts.json")
    w4s = load("week4/summary.json")["predictor"]["by_arm"]["q4b"]

    def row(test: str, m: dict, kld: float, bits: float, verdict: str) -> str:
        interval = (
            " " + ci(*m["mae_ci_config_bootstrap"])
            if "mae_ci_config_bootstrap" in m
            else ""
        )
        return f"{test} & {m['n']} & {num(m['mae'])}{interval} & {num(m['safe_agreement'], 2)} & {num(kld)} & {num(bits)} & {verdict} \\\\"

    rows = [
        row(
            "Round 2, v1: new learning rates and seeds, 0.6B",
            w2["T2"]["nsr_model"],
            w2["T2"]["kld_only"]["mae"],
            w2["T2"]["bits_only"]["mae"],
            "P1 not met; P2 met",
        ),
        row(
            "Round 2, v1: Qwen3-1.7B",
            w2["T1+T3"]["nsr_model"],
            w2["T1+T3"]["kld_only"]["mae"],
            w2["T1+T3"]["bits_only"]["mae"],
            "P3 met",
        ),
        row(
            "Round 3, v2: OLMo-2 1B and MASSIVE",
            w3["models"]["pred_v2"],
            w3["models"]["pred_kld_only"]["mae"],
            w3["models"]["pred_bits_only"]["mae"],
            "W3-H2, W3-H3 supported",
        ),
        row(
            "Round 4, v2: second seeds and Qwen3-4B",
            w4["models"]["pred_v2"],
            w4["models"]["pred_kld_only"]["mae"],
            w4["models"]["pred_bits_only"]["mae"],
            "W4-H3, W4-H4 supported",
        ),
        row(
            "Round 4, v2: Qwen3-4B only$^\\ast$",
            {"n": w4s["n"], **w4s["pred_v2"]},
            w4s["pred_kld_only"]["mae"],
            w4s["pred_bits_only"]["mae"],
            "below both bars",
        ),
    ]
    write("predictor.tex", rows)


def round5() -> None:
    path = RUNS / "week5" / "verdicts.json"
    if not path.exists():
        return
    v = json.loads(path.read_text())
    unfused, qlora = v["exploratory"]["unfused"], v["exploratory"]["qlora"]
    rows = ["\\multicolumn{5}{l}{\\emph{Adapter loaded unfused on the quantized OLMo-2 1B base: $R$ fused / $R$ unfused}} \\\\"]
    for cfg, label in (
        ("olmo1-lora-lowlr", "$10^{-5}$, seed 0"),
        ("olmo1-lora-lowlr-s1", "$10^{-5}$, seed 1"),
        ("olmo1-lora", "$10^{-4}$, seed 0"),
        ("olmo1-lora-s1", "$10^{-4}$, seed 1"),
    ):
        for bits in (3,):
            r = unfused[f"{cfg} mlx-q{bits}"]
            confirmatory = bits == 3 and "lowlr" in cfg
            test = "W5-H1" if confirmatory else "exploratory"
            d, lo, hi = r["diff_unfused_minus_fused"]
            rows.append(
                f"{test} & {label} & MLX {bits}-bit & {num(r['R_fused'])} / {num(r['R_unfused'])} & "
                f"{num(d, 3, True)} {ci(lo, hi, 3, True)} \\\\"
            )
    rows.append("\\addlinespace")
    rows.append("\\multicolumn{5}{l}{\\emph{Qwen3-0.6B LoRA trained on the quantized base: $R$ fused / $S$ trained on the base}} \\\\")
    for bits in (4, 3):
        for suffix, label in (("lora-lowlr", "$10^{-5}$"), ("lora", "$10^{-4}$")):
            r = qlora[f"q06q{bits}-{suffix}"]
            confirmatory = bits == 4 and suffix == "lora-lowlr"
            test = "W5-H2" if confirmatory else "exploratory"
            d, lo, hi = r["diff_S_minus_R"]
            rows.append(
                f"{test} & {label} & MLX {bits}-bit & {num(r['R_fused'])} / {num(r['S_qlora'])} & "
                f"{num(d, 3, True)} {ci(lo, hi, 3, True)} \\\\"
            )
    write("round5.tex", rows)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    bases()
    formats()
    week1()
    lrsweep()
    replication()
    pattern()
    predictor()
    absolute()
    bands()
    baselines()
    grid()
    round5()
    for p in sorted(OUT.glob("*.tex")):
        print(f"--- {p.name}")
        print(p.read_text())


if __name__ == "__main__":
    main()
