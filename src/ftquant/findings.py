import argparse
import hashlib
import json
import re
import sys
from datetime import datetime

import numpy as np
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

from ftquant.predictor_v2 import FIT_RUNS, FORMS, kld_lookup, loco, mech_lookup, points
from ftquant.revision import (
    HARD,
    PAIRS,
    REPS,
    RUNS,
    SEED,
    col,
    final_val_loss,
    fp_of,
    records,
    share,
)

ROOT = RUNS.parent
TEX = ROOT / "paper" / "main.tex"
OUT_JSON = RUNS / "findings" / "findings.json"
OUT_TABLE = ROOT / "paper" / "tables" / "findings.tex"
WORDS = "zero one two three four five six seven eight nine ten".split()
LORA_PAIRS = [
    (n, g[1], s[1]) for n, g, s, _ in PAIRS
]  # (setting, gentle config, strong config)

V: dict[str, str] = {}
RAW: dict[str, float | int | str] = {}
PROBLEMS: list[str] = []


def load(rel: str) -> dict:
    return json.loads((RUNS / rel).read_text())


def put(key: str, raw: float | int | str, text: str) -> None:
    if key in V:
        raise SystemExit(f"duplicate value key {key}")
    V[key] = text
    RAW[key] = raw


def fx(x: float, n: int) -> str:
    s = f"{x:.{n}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s


def num(key: str, x: float, n: int) -> None:
    put(key, x, fx(x, n))


def sgn(key: str, x: float, n: int) -> None:
    s = f"{x:+.{n}f}"
    put(key, x, "+" + s[1:] if float(s) == 0 else s)


def directed(x: float, n: int, mode: str) -> str:
    return f"{Decimal(repr(x)).quantize(Decimal(1).scaleb(-n), rounding=mode):.{n}f}"


def at_least(key: str, x: float, n: int) -> None:
    put(key, x, directed(x, n, ROUND_FLOOR))


def at_most(key: str, x: float, n: int) -> None:
    put(key, x, directed(x, n, ROUND_CEILING))


def pct(key: str, x: float, n: int = 0) -> None:
    put(key, x, fx(100 * x, n))


def count(key: str, k: int) -> None:
    put(key, k, f"{k:,}")


def word(key: str, k: int) -> None:
    put(key, k, WORDS[k])


def expect(ok: bool, what: str) -> None:
    if not ok:
        PROBLEMS.append(what)


def recs(run: str, cfg: str, variant: str) -> list[dict]:
    r = records((run, cfg), variant)
    if r is None:
        raise SystemExit(f"missing evaluation {run}/{cfg}/{variant}")
    return r


def acc(run: str, cfg: str, variant: str) -> float:
    return float(col(recs(run, cfg, variant)).mean())


def valid(run: str, cfg: str, variant: str) -> float:
    return float(col(recs(run, cfg, variant), "valid").mean())


def gain_ret(ft: tuple[str, str], base: tuple[str, str], variant: str) -> float:
    fp = fp_of(variant)
    return share(
        col(recs(*ft, variant)),
        col(recs(*base, variant.replace("-unfused", ""))),
        col(recs(*ft, fp)),
        col(recs(*base, fp)),
    )


def acc_ret(ft: tuple[str, str], variant: str) -> float:
    return acc(*ft, variant) / acc(*ft, fp_of(variant))


def pair(name: str) -> tuple[tuple[str, str], tuple[str, str], tuple[str, str]]:
    for n, g, s, b in PAIRS:
        if n == name:
            return g, s, b
    raise KeyError(name)


def setup() -> None:
    norms = load("revision/revision.json")["update_norms"]["norms"]
    ratios = [norms[s] / norms[g] for _, g, s in LORA_PAIRS]
    num("setup.ratio_min", min(ratios), 1)
    num("setup.ratio_max", max(ratios), 1)
    num("setup.ratio_min_int", min(ratios), 0)
    num("setup.ratio_max_int", max(ratios), 0)
    test = recs("week1", "q06-base", "mlx-bf16")
    golds = [r["gold"] for r in test]
    per_intent = {golds.count(g) for g in set(golds)}
    count("data.b77.test", len(test))
    count("data.b77.labels", len(set(golds)))
    expect(len(per_intent) == 1, "banking77 test set is not balanced across intents")
    count("data.b77.per_intent", per_intent.pop())
    count("data.massive.test", len(recs("week3", "mas06-base", "mlx-bf16")))
    pct("data.b77.epoch_share", 2000 / 9504)
    pct("data.massive.epoch_share", 2000 / 11514)
    pct("mas.base_acc", acc("week3", "mas06-base", "mlx-bf16"))
    grid = load("revision/revision.json")["run_grid"]["rows"]
    losses = {}
    for r in grid:
        run = f"week{r['round']}"
        cfg = next(
            d.name
            for d in (RUNS / run).iterdir()
            if d.is_dir() and (d / "train.yaml").exists() and d.name == r["config"]
        )
        losses[cfg] = final_val_loss((run, cfg))
    failed = "q06-lora-lr3e-4"
    num("setup.failed_val", losses.pop(failed), 2)
    num("setup.val_min", min(losses.values()), 2)
    num("setup.val_max", max(losses.values()), 2)
    pct("setup.failed_acc", acc("week2", failed, "mlx-bf16"))
    mlx = load("bpw/mlx-q06.json")["bpw"]
    g06 = load("week1/mechanism-q06-gguf.json")["bpw"]
    g17 = load("week1/mechanism-q17-gguf.json")["bpw"]
    num("fmt.q3km_lo", min(g06["Q3_K_M"], g17["Q3_K_M"]), 1)
    num("fmt.q3km_hi", max(g06["Q3_K_M"], g17["Q3_K_M"]), 1)
    num("fmt.mlx4", mlx["q4"], 1)
    num("fmt.mlx3", mlx["q3"], 1)
    nf = load("revision/revision.json")["noise_floor"]
    pct("nf.ft_lo", nf["range_fine_tunes"][0], 1)
    pct("nf.ft_hi", nf["range_fine_tunes"][1], 1)
    tuned = {
        k: v
        for k, v in nf["configs"].items()
        if v["acc_mlx"] > 0 and not k.endswith("-base")
    }
    at_most(
        "nf.max_acc_diff",
        100 * max(abs(v["acc_mlx"] - v["acc_gguf"]) for v in tuned.values()), 2,
    )
    bases = [
        v["agreement"]
        for k, v in nf["configs"].items()
        if v["acc_mlx"] > 0 and k.endswith("-base")
    ]
    pct("nf.base_lo", min(bases), 1)
    pct("nf.base_hi", max(bases), 1)
    count("boot.reps", REPS)
    count("boot.seed", SEED)
    klds = [json.loads(p.read_text()) for p in sorted((RUNS / "kld").glob("*.json"))]
    expect(
        len({(k["chunks"], k["ctx"]) for k in klds}) == 1,
        "base-damage files differ in chunks or context",
    )
    count("kld.chunks", klds[0]["chunks"])
    count("kld.ctx", klds[0]["ctx"])


def round1() -> None:
    q17b, q17 = ("week1", "q17-base"), ("week1", "q17-lora")
    pct("r1.q17base.acc_bf16", acc(*q17b, "mlx-bf16"), 1)
    pct("r1.q17base.acc_q3", acc(*q17b, "mlx-q3"), 1)
    pct("r1.q17base.valid_bf16", valid(*q17b, "mlx-bf16"), 1)
    pct("r1.q17base.valid_q3", valid(*q17b, "mlx-q3"), 1)
    pct("r1.q17.A_q3", acc_ret(q17, "mlx-q3"))
    pct("r1.q17.A_Q3KM", acc_ret(q17, "gguf-Q3_K_M"))
    pct("r1.q17.A_Q2K", acc_ret(q17, "gguf-Q2_K"))
    expect(
        acc(*q17b, "mlx-q2") == 0 and acc(*q17, "mlx-q2") == 0,
        "Qwen3-1.7B base or LoRA is not at 0% at MLX 2-bit",
    )
    c = load("week1/confirmatory.json")
    for v, k in (("mlx-q4", "mlx"), ("gguf-Q4_K_M", "gguf")):
        sgn(f"r1.h4.{k}", c["H4"]["detail"][v]["coverage_change_pts"], 1)
    num(
        "r1.h4.gguf_abs",
        abs(c["H4"]["detail"]["gguf-Q4_K_M"]["coverage_change_pts"]),
        1,
    )
    for v, k in (("mlx-q4", "q4"), ("mlx-q3", "q3")):
        d, lo, hi = c["H3"]["detail"][v]["paired_diff_unfused_minus_fused"]
        sgn(f"r1.h3.{k}.d", d, 3)
        sgn(f"r1.h3.{k}.lo", lo, 3)
        sgn(f"r1.h3.{k}.hi", hi, 3)
    for v, k in (
        ("mlx-q8", "q8"),
        ("mlx-q6", "q6"),
        ("gguf-Q8_0", "Q8"),
        ("gguf-Q6_K", "Q6K"),
    ):
        num(f"r1.h1.{k}", c["H1"]["detail"][v][0], 2)
    num("r1.h1.min", min(c["H1"]["detail"][v][0] for v in c["H1"]["detail"]), 2)
    num("r1.h1.max", max(c["H1"]["detail"][v][0] for v in c["H1"]["detail"]), 2)
    h3 = c["H3"]["detail"]
    lo4, hi4 = h3["mlx-q4"]["paired_diff_unfused_minus_fused"][1:]
    expect(lo4 < 0 < hi4, "round-1 H3 interval at MLX 4-bit does not include zero")
    expect(
        h3["mlx-q3"]["paired_diff_unfused_minus_fused"][2] < 0,
        "round-1 H3 interval at MLX 3-bit is not below zero",
    )
    count("r1.q17.acc_q2", round(100 * acc(*q17, "mlx-q2")))
    base = ("week1", "q06-base")
    for name, cfg in (
        ("strong", "q06-lora"),
        ("gentle", "q06-lora-lowlr"),
        ("full", "q06-full"),
    ):
        ft = ("week1", cfg)
        pct(f"r1.q06.{name}.acc", acc(*ft, "mlx-bf16"), 1)
        pct(f"r1.q06.{name}.R_q3", gain_ret(ft, base, "mlx-q3"))
        pct(f"r1.q06.{name}.R_Q3KM", gain_ret(ft, base, "gguf-Q3_K_M"))
    pct("r1.q06base.acc_q3", acc(*base, "mlx-q3"))


def round2() -> None:
    cfgs = {c["config"]: c for c in load("week2/summary.json")["configs"]}
    for key, cfg in (("full3e5", "q06-full-lr3e-5"), ("lora3e5", "q06-lora-lr3e-5")):
        c = cfgs[cfg]
        num(f"r2.{key}.norm", c["delta_norm"], 1)
        num(f"r2.{key}.R_q3", c["gain_retention"]["mlx-q3"], 2)
        num(f"r2.{key}.R_Q2K", c["gain_retention"]["gguf-Q2_K"], 2)
        pct(f"r2.{key}.R_Q2K_pct", c["gain_retention"]["gguf-Q2_K"])
    sp = load("revision/revision.json")["lr_spearman"]
    rhos = {k: v["spearman"] for k, v in sp.items()}
    expect(
        sum(abs(r - 1) < 1e-9 for r in rhos.values()) == 7,
        "Spearman is not 1.0 in exactly seven combinations",
    )
    expect(
        all(r > 0 for r in rhos.values()), "a learning-rate correlation is not positive"
    )
    num("r2.rho_lora_q4km", rhos["LoRA gguf-Q4_K_M"], 1)
    q4 = sp["LoRA gguf-Q4_K_M"]
    tie = abs(q4["R"][q4["lrs"].index(3e-05)] - q4["R"][q4["lrs"].index(1e-04)])
    expect(tie < 0.001, f"LoRA 3e-5 and 1e-4 at Q4_K_M differ by {tie:.4f}")
    put("r2.tie", tie, "0.001")
    base = ("week1", "q06-base")
    for key, run, cfg in (
        ("strong_s0", "week1", "q06-lora"),
        ("strong_s1", "week2", "q06-lora-s1"),
        ("gentle_s0", "week1", "q06-lora-lowlr"),
        ("gentle_s1", "week2", "q06-lora-lowlr-s1"),
    ):
        num(f"r2.seed.{key}", gain_ret((run, cfg), base, "mlx-q3"), 2)
        pct(f"r2.seed.{key}.acc", acc(run, cfg, "mlx-bf16"), 1)
    for kind, a, b in (
        ("strong", ("week1", "q06-lora"), ("week2", "q06-lora-s1")),
        ("gentle", ("week1", "q06-lora-lowlr"), ("week2", "q06-lora-lowlr-s1")),
    ):
        num(
            f"r2.seed.{kind}.dacc",
            100 * abs(acc(*a, "mlx-bf16") - acc(*b, "mlx-bf16")),
            1,
        )
        num(
            f"r2.seed.{kind}.dR",
            abs(gain_ret(a, base, "mlx-q3") - gain_ret(b, base, "mlx-q3")),
            2,
        )
    dRs = [
        abs(gain_ret(a, base, "mlx-q3") - gain_ret(b, base, "mlx-q3"))
        for a, b in (
            (("week1", "q06-lora"), ("week2", "q06-lora-s1")),
            (("week1", "q06-lora-lowlr"), ("week2", "q06-lora-lowlr-s1")),
        )
    ]
    at_most("r2.seed.max_dR", max(dRs), 3)
    word("r2.n_rho1", sum(abs(r - 1) < 1e-9 for r in rhos.values()))
    q17b = ("week1", "q17-base")
    for kind, ft in (
        ("gentle", ("week2", "q17-lora-lowlr")),
        ("strong", ("week1", "q17-lora")),
    ):
        pct(f"r2.q17.{kind}.R_q3", gain_ret(ft, q17b, "mlx-q3"))
        num(f"r2.q17.{kind}.A_q3", acc_ret(ft, "mlx-q3"), 2)


def mechanism() -> None:
    rev = load("revision/revision.json")
    b = rev["bands"]
    count("bands.n", b["n"])
    names = {
        "low": "base damage < 1, kept >= 60%, NSR < 3",
        "mid": "base damage < 1, kept >= 60%, NSR 3 to 5",
        "high": "base damage < 1, kept >= 60%, NSR >= 5",
        "rounded": "kept < 60%",
        "broken": "kept >= 60%, base damage >= 1",
    }
    for k, n in names.items():
        count(f"bands.{k}.pairs", b["bands"][n]["pairs"])
        count(f"bands.{k}.ge90", b["bands"][n]["kept_ge_90"])
    expect(
        b["bands"][names["low"]]["pairs"] == b["bands"][names["low"]]["kept_ge_90"],
        "not every low-noise pair kept 90% of its gain",
    )
    mech2 = mech_lookup("week2", "q06")
    base = ("week1", "q06-base")
    ft = ("week2", "q06-full-lr3e-6")
    nsr3, kept3 = mech2[("q06-full-lr3e-6", "mlx-q3")]
    nsr4, kept4 = mech2[("q06-full-lr3e-6", "mlx-q4")]
    pct("mech.full3e6.kept_q3", kept3, 1)
    num("mech.full3e6.nsr_q3", nsr3, 2)
    pct("mech.full3e6.R_q3", gain_ret(ft, base, "mlx-q3"))
    pct("mech.full3e6.kept_q4", kept4)
    pct("mech.full3e6.R_q4", gain_ret(ft, base, "mlx-q4"))
    kc = rev["kept_count"]
    count("mech.pairs", kc["pairs"])
    count("mech.kept98", kc["kept_ge_98"])
    low = [(k, ref) for k, ref in kc["lowest_ten"] if k < 0.6]
    word("mech.n_lt60", len(low))
    expect(
        all("/q06-full" in ref for _, ref in low),
        "a pair below 60% update kept is not a Qwen3-0.6B full fine-tune",
    )
    word("mech.n_lt60_3e6", sum("full-lr3e-6" in ref for _, ref in low))
    expect(len(low) < len(kc["lowest_ten"]), "more pairs below 60% than listed")
    r34 = [k for k, ref in kc["lowest_ten"] if ref.startswith(("week3", "week4"))]
    pct("mech.min_kept_r34", min(r34))
    broken = set()
    for tag in ("Qwen3-0.6B", "Qwen3-1.7B"):
        broken |= {(tag, v) for v, k in kld_lookup(tag).items() if k >= 1}
    expect(
        broken
        == {
            (t, v)
            for t in ("Qwen3-0.6B", "Qwen3-1.7B")
            for v in ("mlx-q3", "mlx-q2", "gguf-Q2_K")
        },
        f"formats with base damage >= 1 are {sorted(broken)}",
    )
    for key, cfg in (("full", "q06-full-lr3e-5"), ("lora", "q06-lora-lr3e-5")):
        nsr, _ = mech2[(cfg, "gguf-Q2_K")]
        num(f"mech.q2k.{key}_nsr", nsr, 2)
        pct(f"mech.q2k.{key}_R", gain_ret(("week2", cfg), base, "gguf-Q2_K"))


def replication() -> None:
    w3 = load("week3/verdicts.json")
    w4 = load("week4/verdicts.json")
    tests = {}
    for hyp, src in (
        ("W3-H1", w3["W3-H1_tests"]),
        ("W4-H1", w4["W4-H1_tests"]),
        ("W4-H2", w4["W4-H2_tests"]),
    ):
        for label, t in src.items():
            tests[f"{hyp} {label}"] = t
    count("rep.n_tests", len(tests))
    word("rep.n_tests_word", len(tests))
    supported = sum(t["ci"][0] > 0 for t in tests.values())
    count("rep.n_supported", supported)
    expect(supported == len(tests), f"only {supported} of {len(tests)} tests supported")
    rob = load("revision/revision.json")["confirmatory_robustness"]
    expect(
        all(v["R_strong_minus_gentle_99_5"][1] > 0 for v in rob.values()),
        "a test's 99.5% interval includes zero",
    )
    diffs = {k: t["diff"] for k, t in tests.items()}
    sgn("rep.diff_min", min(diffs.values()), 2)
    sgn("rep.diff_max", max(diffs.values()), 2)
    lo_keys = sorted(
        k for k, d in diffs.items() if fx(d, 2) == fx(min(diffs.values()), 2)
    )
    expect(
        lo_keys == ["W3-H1 olmo1 gguf-Q3_K_M", "W4-H1 olmo1 gguf-Q3_K_M"],
        f"smallest differences are {lo_keys}",
    )
    expect(
        max(diffs, key=diffs.get) == "W4-H1 mas06 mlx-q3",
        f"largest difference is {max(diffs, key=diffs.get)}",
    )
    q4b = rob["W4-H2 Qwen3-4B, banking77 mlx-q3"]
    num("rep.q4b.base_term", q4b["base_drop_term_of_R_diff"], 3)
    num("rep.q4b.diff", q4b["R_strong_minus_gentle_95"][0], 3)
    drops = [v["accuracy_drop_gentle_minus_strong_pts"] for v in rob.values()]
    expect(all(d[1] > 0 for d in drops), "an accuracy-drop interval includes zero")
    num("rep.drop_min", min(d[0] for d in drops), 1)
    num("rep.drop_max", max(d[0] for d in drops), 1)


def pattern() -> None:
    rev = load("revision/revision.json")["pairs"]
    pairs, summary = rev["pairs"], rev["summary"]
    cells = [(n, v, pairs[n]["formats"][v]) for n in pairs for v in HARD]
    lower = sum(c["A_gentle"] < c["A_strong"] for _, _, c in cells)
    count("pat.lower", lower)
    count("pat.cells", len(cells))
    expect(lower == len(cells), "the gentle LoRA did not keep less accuracy everywhere")
    expect(
        summary["gentle_higher_bf16_mlx"] == len(pairs),
        "the gentle LoRA is not more accurate at bf16 in every setting",
    )

    def gap(v: str) -> list[float]:
        return [100 * (c["A_strong"] - c["A_gentle"]) for _, w, c in cells if w == v]

    at_least("pat.min_gap_q4", min(gap("mlx-q4")), 1)
    at_least(
        "pat.min_gap_low", min(gap("mlx-q3") + gap("gguf-Q3_K_M") + gap("gguf-Q2_K")), 1
    )
    under2 = sum(g < 2 for g in gap("gguf-Q4_K_M"))
    word("pat.q4km_under2", under2)
    count("pat.q4km_under2_n", under2)
    expect(
        all(
            pairs[n]["formats"]["gguf-Q4_K_M"]["acc_gentle"]
            > pairs[n]["formats"]["gguf-Q4_K_M"]["acc_strong"]
            for n in pairs
        ),
        "the gentle LoRA is not strictly more accurate at Q4_K_M in every setting",
    )
    olmo0 = pairs["OLMo-2 1B, banking77"]["formats"]["gguf-Q4_K_M"][
        "A_gentle_minus_strong"
    ][2]
    expect(-1e-5 < olmo0 < 0, f"OLMo seed-0 Q4_K_M gap upper end is {olmo0}")
    expect(
        sorted(summary["A_gap_interval_includes_zero"])
        == [
            "Qwen3-0.6B, MASSIVE gguf-Q4_K_M",
            "Qwen3-0.6B, MASSIVE, seed 1 gguf-Q4_K_M",
        ],
        f"gap intervals including zero: {summary['A_gap_interval_includes_zero']}",
    )
    olmo_hi = {
        n: pairs[n]["formats"]["gguf-Q4_K_M"]["A_gentle_minus_strong"][2]
        for n in pairs
        if n.startswith("OLMo")
    }
    num("pat.olmo_q4km_hi", max(olmo_hi.values()), 4)
    sm = summary["strong_more_accurate"]
    for v, k in (
        ("mlx-q4", "q4"),
        ("mlx-q3", "q3"),
        ("gguf-Q4_K_M", "Q4KM"),
        ("gguf-Q3_K_M", "Q3KM"),
        ("gguf-Q2_K", "Q2K"),
    ):
        word(f"abs.strong_more.{k}", sm[v])
        count(f"abs.n.{k}", sm[v])
    num("abs.cost_min", summary["bf16_cost_of_strong_pts_mlx"][0], 1)
    num("abs.cost_max", summary["bf16_cost_of_strong_pts_mlx"][1], 1)
    q06 = pairs["Qwen3-0.6B, banking77"]["formats"]
    g, s, _ = pair("Qwen3-0.6B, banking77")
    expect(
        acc(*g, "mlx-q2") == 0 and acc(*s, "mlx-q2") == 0,
        "a Qwen3-0.6B LoRA is not at 0% at MLX 2-bit",
    )
    pct("valid.q06g.q4", q06["mlx-q4"]["valid_gentle"], 1)
    pct("valid.q06g.Q3KM", q06["gguf-Q3_K_M"]["valid_gentle"], 1)
    pct("valid.q06g.q3", q06["mlx-q3"]["valid_gentle"])
    pct("valid.q06g.q3_invalid", 1 - q06["mlx-q3"]["valid_gentle"])
    pct("valid.q06g.Q2K", q06["gguf-Q2_K"]["valid_gentle"])
    wrong = 1 - q06["mlx-q3"]["acc_gentle"]
    pct(
        "valid.q06g.q3_invalid_share_of_errors",
        (1 - q06["mlx-q3"]["valid_gentle"]) / wrong,
    )
    q4b = pairs["Qwen3-4B, banking77"]["formats"]
    num("pat.q4b.A_g_q3", q4b["mlx-q3"]["A_gentle"], 2)
    num("pat.q4b.A_s_q3", q4b["mlx-q3"]["A_strong"], 2)
    num("pat.q06.A_g_q3", q06["mlx-q3"]["A_gentle"], 2)
    num("pat.q06.A_s_q3", q06["mlx-q3"]["A_strong"], 2)
    pct("pat.q4b.loss_Q2K", 1 - q4b["gguf-Q2_K"]["A_gentle"])
    for tag, k in (("Qwen3-4B", "q4b"), ("Qwen3-0.6B", "q06"), ("Qwen3-1.7B", "q17")):
        num(f"kld.{k}.q3", kld_lookup(tag)["mlx-q3"], 2)
    vl = load("revision/revision.json")["val_loss"]
    expect(
        all(v["gentle"] <= v["strong"] for v in vl.values()),
        "a gentle LoRA had higher final validation loss than its strong pair",
    )
    st = load("revision/revision.json")["confidence_strata"][
        "Qwen3-4B, banking77 mlx-q3"
    ]
    pct("strata.q4b.gentle", st["gentle"]["still_correct"], 1)
    pct("strata.q4b.strong", st["strong"]["still_correct"], 1)


def seeds() -> None:
    def rpair(name: str, v: str) -> tuple[float, float]:
        g, s, b = pair(name)
        return gain_ret(g, b, v), gain_ret(s, b, v)

    a, b = "OLMo-2 1B, banking77", "OLMo-2 1B, banking77, seed 1"
    hard = [
        abs(x - y)
        for v in ("mlx-q3", "gguf-Q3_K_M")
        for x, y in zip(rpair(a, v), rpair(b, v))
    ]
    other = [
        abs(x - y)
        for v in ("mlx-q4", "gguf-Q4_K_M", "gguf-Q2_K")
        for x, y in zip(rpair(a, v), rpair(b, v))
    ]
    at_most("seed.olmo.dR_hard", max(hard), 3)
    at_most("seed.olmo.dR_other", max(other), 3)
    ga, sa, _ = pair(a)
    gb, sb, _ = pair(b)
    at_most(
        "seed.olmo.dacc",
        100
        * max(
            abs(acc(*x, fp) - acc(*y, fp))
            for fp in ("mlx-bf16", "gguf-bf16")
            for x, y in ((ga, gb), (sa, sb))
        ),
        1,
    )
    m0, m1 = "Qwen3-0.6B, MASSIVE", "Qwen3-0.6B, MASSIVE, seed 1"
    num("seed.mas.gR_s0", rpair(m0, "mlx-q3")[0], 2)
    num("seed.mas.gR_s1", rpair(m1, "mlx-q3")[0], 2)
    _, s0, _ = pair(m0)
    _, s1, _ = pair(m1)
    num("seed.mas.s_dacc", 100 * (acc(*s0, "mlx-bf16") - acc(*s1, "mlx-bf16")), 1)


def round5() -> None:
    w5 = load("week5/verdicts.json")
    q06g, base = ("week1", "q06-lora-lowlr"), ("week1", "q06-base")
    for v, k in (("mlx-q4", "q4"), ("mlx-q3", "q3")):
        num(f"r5.r1.unf_{k}", gain_ret(q06g, base, f"{v}-unfused"), 2)
        num(f"r5.r1.fus_{k}", gain_ret(q06g, base, v), 2)
    smoke = (RUNS / "smoke5" / "week5.log").read_text()
    ns = {int(n) for n in re.findall(r"n=(\d+)", smoke)}
    expect(len(ns) == 1, f"plumbing-test item counts {ns}")
    count("r5.smoke_n", ns.pop())
    pct("r5.olmo_base.acc_bf16", acc("week3", "olmo1-base", "mlx-bf16"), 1)
    pct(
        "r5.olmo_base.acc_q3",
        w5["exploratory"]["evaluations"]["olmo1-base/mlx-q3"]["acc"],
        1,
    )
    unf = w5["exploratory"]["unfused"]
    h1 = w5["W5-H1"]
    sgn("r5.h1.margin", h1["margin"], 2)
    misses, shares, uppers = [], [], []
    for k, cfg in (("s0", "olmo1-lora-lowlr"), ("s1", "olmo1-lora-lowlr-s1")):
        r = unf[f"{cfg} mlx-q3"]
        d, lo, hi = r["diff_unfused_minus_fused"]
        expect(hi < h1["margin"], f"W5-H1 upper bound {hi:.3f} for {cfg}")
        num(f"r5.h1.{k}.R_fused", r["R_fused"], 2)
        sgn(f"r5.h1.{k}.d", d, 3)
        sgn(f"r5.h1.{k}.lo", lo, 3)
        sgn(f"r5.h1.{k}.hi", hi, 3)
        miss = 1 - r["R_fused"]
        misses.append(miss)
        pct(f"r5.h1.{k}.share", d / miss)
        pct(f"r5.h1.{k}.share_hi", hi / miss)
        shares.append(d / miss)
        uppers.append(hi / miss)
    num("r5.h1.miss_lo", min(misses), 2)
    num("r5.h1.miss_hi", max(misses), 2)
    expect(max(uppers) < 0.25, "an upper end recovers a quarter or more of the loss")
    others = [
        abs(v["diff_unfused_minus_fused"][0])
        for k, v in unf.items()
        if " mlx-q" in k
        and not k.endswith("check")
        and not (k.endswith("mlx-q3") and "lowlr" in k)
    ]
    expect(
        len(others) == 6, f"{len(others)} exploratory unfused comparisons, expected 6"
    )
    at_most("r5.unf.maxdiff", max(others), 3)
    q = w5["exploratory"]["qlora"]
    d, lo, hi = w5["W5-H2"]["diff_S_minus_R"]
    sgn("r5.h2.d", d, 3)
    sgn("r5.h2.lo", lo, 3)
    sgn("r5.h2.hi", hi, 3)
    pct("r5.h2.S_pct", q["q06q4-lora-lowlr"]["S_qlora"])
    pct("r5.h2.R_pct", q["q06q4-lora-lowlr"]["R_fused"])
    pct("r5.q4g.acc", q["q06q4-lora-lowlr"]["acc_qlora"], 1)
    expect(
        abs(
            q["q06q4-lora-lowlr"]["acc_qlora"]
            - acc("week1", "q06-lora-lowlr", "mlx-bf16")
        )
        < 0.01,
        "the 4-bit-trained gentle LoRA is not within a point of the bf16 LoRA at bf16",
    )
    num("r5.q4g.val", q["q06q4-lora-lowlr"]["val_loss_qlora"], 3)
    num("r5.bf16g.val", q["q06q4-lora-lowlr"]["val_loss_bf16_trained"], 3)
    pct("r5.q3base.acc", w5["exploratory"]["evaluations"]["q06-base/mlx-q3"]["acc"])
    for k, cfg in (
        ("q3g", "q06q3-lora-lowlr"),
        ("q4s", "q06q4-lora"),
        ("q3s", "q06q3-lora"),
    ):
        r = q[cfg]
        pct(f"r5.{k}.acc", r["acc_qlora"], 1)
        pct(f"r5.{k}.S", r["S_qlora"])
        pct(f"r5.{k}.R", r["R_fused"])
        num(f"r5.{k}.val", r["val_loss_qlora"], 3)
        dd, dlo, dhi = r["diff_S_minus_R"]
        sgn(f"r5.{k}.d", dd, 3)
        sgn(f"r5.{k}.lo", dlo, 3)
        sgn(f"r5.{k}.hi", dhi, 3)
    num("r5.bf16s.val", q["q06q4-lora"]["val_loss_bf16_trained"], 3)
    num("r5.q4s.d_abs", -q["q06q4-lora"]["diff_S_minus_R"][0], 3)
    for bits in (4, 3):
        best = max(
            q[f"q06q{bits}-lora-lowlr"]["acc_qlora"],
            q[f"q06q{bits}-lora"]["acc_qlora"],
            q[f"q06q{bits}-lora-lowlr"]["acc_fused"],
            q[f"q06q{bits}-lora"]["acc_fused"],
        )
        expect(
            q[f"q06q{bits}-lora-lowlr"]["acc_qlora"] == best,
            f"the gentle LoRA trained on the {bits}-bit base is not the most accurate",
        )
    rep = w5["exploratory"]["repeatability"]
    expect(
        fx(100 * rep["acc_np4"], 1) == fx(100 * rep["acc_np1"], 1),
        "one-slot and four-slot accuracies differ at one decimal",
    )
    pct("r5.rep.acc", rep["acc_np4"], 1)
    pct("r5.rep.agree_np", rep["agree_np4_np1"], 1)
    pct("r5.rep.agree_w1", rep["agree_np4_week1"], 1)
    n = len(recs("week5", "q06-lora-lowlr", "gguf-Q3_K_M-np4"))
    count("r5.rep.n", n)
    count("r5.rep.diff_np", round(n * (1 - rep["agree_np4_np1"])))
    count("r5.rep.diff_w1", round(n * (1 - rep["agree_np4_week1"])))
    expect(
        rep["agree_np4_week1"] == rep["agree_np1_week1"],
        "one-slot and four-slot runs differ from round 1 on different numbers of items",
    )
    lab = w5["exploratory"]["labels_only"]
    for k, cfg in (("g", "q06-lora-lowlr"), ("s", "q06-lora")):
        num(f"r5.lab.{k}.A_lab", lab[cfg]["Q3_K_M"]["A_labels_only"][0], 3)
        num(f"r5.lab.{k}.A_unc", lab[cfg]["Q3_K_M"]["A_unconstrained"][0], 3)
    pct("r5.lab.g.A_lab_Q2K", lab["q06-lora-lowlr"]["Q2_K"]["A_labels_only"][0])
    at_most(
        "r5.lab.max_change",
        max(
            abs(
                lab[c]["Q3_K_M"]["A_labels_only"][0]
                - lab[c]["Q3_K_M"]["A_unconstrained"][0]
            )
            for c in ("q06-lora-lowlr", "q06-lora")
        ),
        3,
    )


def predictor() -> None:
    w2 = load("predictor/verdicts.json")
    t2 = w2["T2"]["nsr_model"]
    num("pred.v1.p1_miss", t2["mae"] - 0.10, 3)
    num("pred.v1.t2.lo", t2["mae_ci_config_bootstrap"][0], 3)
    num("pred.v1.t2.hi", t2["mae_ci_config_bootstrap"][1], 3)
    v1 = load("predictor/predictor-v1.json")["theta"]
    num("pred.v1.t0", v1[0], 3)
    num("pred.v1.t1abs", -v1[1], 2)
    num("pred.v1.t2abs", -v1[2], 3)
    v2 = load("predictor/predictor-v2.json")
    num("pred.v2.t0", v2["theta"][0], 3)
    num("pred.v2.t1", v2["theta"][1], 3)
    num("pred.v2.t2abs", -v2["theta"][2], 3)
    count("pred.fit_n", v2["loco"]["n"])
    num("pred.lofo.v2", v2["loco"]["mae"], 3)
    pb = load("revision/revision.json")["predictor_baselines"]
    expect(pb["fit_points"] == v2["loco"]["n"], "fit point counts disagree")
    fit_pts = points(RUNS / "analysis-all.json", FIT_RUNS)
    lofo_v2 = loco("snr+kld", fit_pts)["mae"]
    expect(
        abs(lofo_v2 - v2["loco"]["mae"]) < 1e-9,
        "recomputed v2 leave-one-out error differs from the frozen record",
    )
    num("pred.lofo.v1form", loco("nsr+kld", fit_pts)["mae"], 3)
    word("pred.n_forms", len([f for f in FORMS if f != "kld-only"]))
    for rd, vfile in (("r3", "week3/verdicts.json"), ("r4", "week4/verdicts.json")):
        vd = load(vfile)
        m2, m1 = vd["models"]["pred_v2"], vd["models"]["pred_v1"]
        num(f"pred.v2.{rd}.mae", m2["mae"], 3)
        num(f"pred.v2.{rd}.lo", m2["mae_ci_config_bootstrap"][0], 3)
        num(f"pred.v2.{rd}.hi", m2["mae_ci_config_bootstrap"][1], 3)
        num(f"pred.v2.{rd}.agree", m2["safe_agreement"], 2)
        num(f"pred.v1.{rd}.mae", m1["mae"], 3)
        expect(
            m1["mae"] <= 0.10 and m1["safe_agreement"] >= 0.85,
            f"v1 would not have met the accuracy bar in {rd}",
        )
        expect(
            m2["mae"] <= 0.10 and m2["safe_agreement"] >= 0.85,
            f"v2 did not meet the accuracy bar in {rd}",
        )
        count(f"pred.{rd}.n", vd["n_points"])
        pts = vd["points"]
        word(f"pred.{rd}.n_ft", len({p["config"] for p in pts}))
        pct(f"pred.{rd}.min_kept", min(p["kept"] for p in pts), 1 if rd == "r4" else 0)
        rows = pb["rounds"]["round 3" if rd == "r3" else "round 4"]
        num(f"pb.const.{rd}", rows["all"]["pred_const"]["mae"], 3)
        num(f"pb.norm.{rd}", rows["all"]["pred_norm_kld"]["mae"], 3)
        num(f"pred.v2.{rd}.no6", rows["without 6-bit"]["pred_v2"]["mae"], 3)
        diffs = rows["all"]["paired_mae_diff"]
        for k, name in (
            ("v2const", "pred_const minus pred_v2"),
            ("v2norm", "pred_v2 minus pred_norm_kld"),
        ):
            dd, lo, hi = diffs[name]
            num(f"pb.{k}.{rd}.d", dd, 3)
            num(f"pb.{k}.{rd}.lo", lo, 3)
            num(f"pb.{k}.{rd}.hi", hi, 3)
        mas = [p for p in pts if p["config"].startswith("mas06")]
        y = np.clip([p["retention"] for p in mas], 0, 1)
        for m in ("pred_v1", "pred_v2"):
            num(
                f"pred.mas.{rd}.{m[-2:]}",
                float(np.mean(np.abs(np.array([p[m] for p in mas]) - y))),
                3,
            )
    num("pb.lofo.norm", pb["lofo"]["norm+kld"]["mae"], 3)
    num("pb.lofo.v2", pb["lofo"]["v2"]["mae"], 3)
    expect(
        fx(pb["rounds"]["round 3"]["all"]["pred_norm_kld"]["mae"], 3)
        == fx(pb["rounds"]["round 4"]["all"]["pred_norm_kld"]["mae"], 3),
        "the update-norm model's held-out errors differ between rounds",
    )
    corr = load("revision/revision.json")["nsr_norm_correlation_fit_points"]
    rs = {v: c["r"] for v, c in corr.items()}
    nine = [r for v, r in rs.items() if v != "mlx-q3"]
    num("corr.lo", max(nine), 2)
    num("corr.hi", min(nine), 2)
    num("corr.mlxq3", rs["mlx-q3"], 2)
    word("corr.n_four", sum(c["n"] == 4 for c in corr.values()))
    w3p = {
        (p["config"], p["variant"]): p for p in load("week3/verdicts.json")["points"]
    }
    w4p = {
        (p["config"], p["variant"]): p for p in load("week4/verdicts.json")["points"]
    }
    pct("miss.mas.pred", w3p[("mas06-lora", "mlx-q3")]["pred_v2"])
    pct("miss.mas.s0", w3p[("mas06-lora", "mlx-q3")]["retention"])
    pct("miss.mas.s1", w4p[("mas06-lora-s1", "mlx-q3")]["retention"])
    expect(
        fx(100 * w4p[("mas06-lora-s1", "mlx-q3")]["pred_v2"], 0)
        == fx(100 * w3p[("mas06-lora", "mlx-q3")]["pred_v2"], 0),
        "the two MASSIVE strong seeds have different predictions",
    )
    g = w4p[("q4b-lora-lowlr", "mlx-q3")]
    pct("miss.q4b.R_q3", g["retention"])
    num("miss.q4b.nsr_q3", g["nsr"], 2)
    pct("miss.q4b.pred_q3", g["pred_v2"])
    g2 = w4p[("q4b-lora-lowlr", "gguf-Q2_K")]
    num("miss.q4b.R_Q2K", g2["retention"], 2)
    pct("miss.q4b.pred_Q2K", g2["pred_v2"])
    pct("miss.q4b.A_Q2K", acc_ret(("week4", "q4b-lora-lowlr"), "gguf-Q2_K"))
    s4 = load("week4/summary.json")["predictor"]["by_arm"]["q4b"]
    count("pred.q4b.n", s4["n"])
    num("pred.q4b.mae", s4["pred_v2"]["mae"], 3)
    num("pred.q4b.agree", s4["pred_v2"]["safe_agreement"], 2)
    m0 = w3p[("mas06-lora-lowlr", "mlx-q3")]
    m1 = w4p[("mas06-lora-lowlr-s1", "mlx-q3")]
    expect(
        fx(100 * m0["pred_v2"], 0) == fx(100 * m1["pred_v2"], 0),
        "the two MASSIVE gentle seeds have different predictions",
    )
    pct("miss.masg.pred", m0["pred_v2"])
    pct("miss.masg.s0", m0["retention"])
    pct("miss.masg.s1", m1["retention"])
    pct("tool.mas.R_q4", w3p[("mas06-lora-lowlr", "mlx-q4")]["retention"])


def drift() -> None:
    dr = load("revision/revision.json")["drift"]
    cases = dr["cases"]
    fell = [
        k
        for k, c in cases.items()
        if c["coverage_change_gentle"] < c["coverage_change_strong"]
    ]
    count("drift.n_fell", len(fell))
    count("drift.n_cases", len(cases))
    exceptions = sorted(set(cases) - set(fell))
    expect(
        exceptions
        == sorted(
            [
                "Qwen3-1.7B, banking77 gguf-Q4_K_M",
                "Qwen3-0.6B, MASSIVE, seed 1 mlx-q3",
                "Qwen3-0.6B, MASSIVE, seed 1 gguf-Q4_K_M",
                "Qwen3-0.6B, MASSIVE, seed 1 gguf-Q2_K",
            ]
        ),
        f"drift exceptions are {exceptions}",
    )
    rows = {
        (r["run"], r["config"], r["variant"]): r
        for r in load("analysis-all.json")["retention"]
    }
    for v, k in (("mlx-q3", "q3"), ("gguf-Q2_K", "Q2K")):
        r = rows[("week1", "q17-lora", v)]
        pct(f"drift.q17.cov_fp_{k}", r["fp_coverage"], 1)
        pct(f"drift.q17.cov_{k}", r["q_coverage"], 1)
    pct("drift.q06g.A_q4", acc_ret(("week1", "q06-lora-lowlr"), "mlx-q4"), 1)
    num(
        "drift.q06g.dcov_q4",
        -cases["Qwen3-0.6B, banking77 mlx-q4"]["coverage_change_gentle"],
        1,
    )
    strong_end = []
    for name, k in (
        ("Qwen3-4B, banking77", "q4b"),
        ("OLMo-2 1B, banking77", "olmo"),
        ("Qwen3-1.7B, banking77", "q17"),
    ):
        c = cases[f"{name} mlx-q3"]
        num(f"drift.err.{k}.fp", c["error_accepted_gentle"][0], 1)
        num(f"drift.err.{k}.q", c["error_accepted_gentle"][1], 1)
        strong_end.append(c["error_accepted_strong"][1])
    num("drift.err.strong_lo", min(strong_end), 1)
    num("drift.err.strong_hi", max(strong_end), 1)


def limits_and_appendix() -> None:
    for k, cfg in (("g", "q4b-lora-lowlr"), ("s", "q4b-lora")):
        num(
            f"lim.q4b.gain_{k}",
            100
            * (acc("week4", cfg, "mlx-bf16") - acc("week4", "q4b-base", "mlx-bf16")),
            0,
        )
    rev = load("revision/revision.json")
    count("grid.n", rev["run_grid"]["fine_tunes"])
    count("grid.evals", rev["run_grid"]["evaluation_files"])
    chk = rev["update_norms"]["method_check_olmo1-lora-lowlr"]
    at_most(
        "norms.check_pct",
        100 * abs(chk["recomputed_bf16_rounded"] / chk["recorded"] - 1),
        1,
    )
    for k in ("v1", "v2"):
        digest = hashlib.sha256(
            (RUNS / "predictor" / f"predictor-{k}.json").read_bytes()
        )
        put(f"sha.{k}", k, digest.hexdigest()[:8])


def stamp(line: str) -> datetime:
    return datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")


def when(d: datetime) -> str:
    return f"{d.day} {d:%B}, {d:%H:%M}"


def timeline() -> None:
    sweeps = [
        (RUNS / f"week{w}" / "sweep.log").read_text().splitlines() for w in (1, 2, 3, 4)
    ]
    lines = [x for s in sweeps for x in s]
    start = stamp(sweeps[0][0])
    orch = (RUNS / "orchestrate.log").read_text().splitlines()
    end = stamp(next(x for x in orch if "done verdicts week4" in x))
    put("time.r14.start", 0, when(start))
    put("time.r14.end", 0, when(end))
    num("time.r14.hours", (end - start).total_seconds() / 3600, 1)
    w5 = (RUNS / "week5" / "week5.log").read_text().splitlines()
    s5 = stamp(next(x for x in w5 if "week-5 start" in x))
    e5 = stamp(next(x for x in w5 if "week-5 complete" in x))
    put("time.r5.start", 0, when(s5))
    put("time.r5.end", 0, when(e5))
    num("time.r5.hours", (e5 - s5).total_seconds() / 3600, 1)
    trains = {
        m.group(1): int(m.group(2))
        for x in lines
        if (m := re.search(r"trained (\S+) in (\d+)s", x))
    }
    group = {"q06": "q06", "mas06": "q06", "q17": "q17", "olmo1": "olmo1", "q4b": "q4b"}
    for k in ("q06", "q17", "olmo1", "q4b"):
        mins = [
            s / 60
            for c, s in trains.items()
            if "-full" not in c and group[c.split("-")[0]] == k
        ]
        num(f"time.train.{k}.lo", min(mins), 1)
        num(f"time.train.{k}.hi", max(mins), 1)
    evals = [
        (m.group(1), int(m.group(2)) / 60)
        for x in lines
        if (m := re.search(r"eval ([a-z0-9-]+)/\S+: .*\((\d+)s\)", x))
    ]
    small = [m for c, m in evals if not c.startswith("q4b")]
    big = [m for c, m in evals if c.startswith("q4b")]
    num("time.eval.small.lo", min(small), 1)
    num("time.eval.small.hi", max(small), 1)
    num("time.eval.q4b.lo", min(big), 1)
    num("time.eval.q4b.hi", max(big), 1)
    last4 = max(i for i, x in enumerate(orch) if "week-4 orchestrator start" in x)
    mech = {
        m.group(1): int(m.group(2)) / 60
        for x in orch[last4:]
        if (m := re.search(r"done mechanism (\S+) \((\d+)s\)", x))
    }
    expect(
        sorted(mech) == ["mas06", "olmo1", "q4b"],
        f"round-4 mechanism runs {sorted(mech)}",
    )
    num("time.mech.lo", min(mech["mas06"], mech["olmo1"]), 1)
    num("time.mech.hi", max(mech["mas06"], mech["olmo1"]), 1)
    num("time.mech.q4b", mech["q4b"], 1)
    r5 = {
        m.group(1): int(m.group(2)) / 60
        for x in w5
        if (m := re.search(r"trained (\S+) in (\d+)s", x))
    }
    expect(len(r5) == 4, f"{len(r5)} round-5 training runs logged")
    normal = [v for k, v in r5.items() if k != "q06q3-lora"]
    num("time.r5.train.lo", min(normal), 1)
    num("time.r5.train.hi", max(normal), 1)
    num("time.r5.train.stalled", r5["q06q3-lora"], 1)


ANCHORS: list[str] = [
    r"delivered {{r5.h2.S_pct}}\% of the bf16 gain at 4 bits, against {{r5.h2.R_pct}}\% when trained in bf16, fused and quantized",
    r"evaluation uses all {{data.b77.test}} test items, {{data.b77.per_intent}} per intent",
    r"all {{data.b77.labels}} intent names",
    r"{{data.massive.test}} test utterances",
    r"Base Qwen3-0.6B scores {{mas.base_acc}}\% on this task zero-shot",
    r"the strong LoRA's update was {{setup.ratio_min}} to {{setup.ratio_max}} times larger than the gentle one's",
    r"({{data.b77.epoch_share}}\% of banking77, {{data.massive.epoch_share}}\% of MASSIVE)",
    r"(final validation loss {{setup.failed_val}}, against {{setup.val_min}} to {{setup.val_max}} for every other run, and {{setup.failed_acc}}\% test accuracy)",
    r"Q3\_K\_M, at {{fmt.q3km_lo}} to {{fmt.q3km_hi}} bits per weight, is closer in size to MLX~4-bit ({{fmt.mlx4}}) than to MLX~3-bit ({{fmt.mlx3}})",
    r"the two engines agreed on {{nf.ft_lo}}\% to {{nf.ft_hi}}\% of the bf16 test predictions of every fine-tune that learned its task, and their accuracies differed by at most {{nf.max_acc_diff}} points",
    r"Base models with nonzero accuracy agreed on {{nf.base_lo}}\% to {{nf.base_hi}}\%.",
    r"({{boot.reps}} resamples, seed {{boot.seed}}, 95\% percentile intervals)",
    r"on {{kld.chunks}} chunks of {{kld.ctx}} tokens of WikiText-2 test text",
    r"At MLX~3-bit the base's accuracy fell from {{r1.q17base.acc_bf16}}\% to {{r1.q17base.acc_q3}}\% and its rate of valid labels from {{r1.q17base.valid_bf16}}\% to {{r1.q17base.valid_q3}}\%, while the fine-tune kept {{r1.q17.A_q3}}\% of its own accuracy ({{r1.q17.A_Q3KM}}\% at Q3\_K\_M and {{r1.q17.A_Q2K}}\% at Q2\_K)",
    r"The 4-bit operating point moved on MLX (coverage ${{r1.h4.mlx}}$ points) but not on GGUF~(${{r1.h4.gguf}}$)",
    r"The LoRA at $10^{-4}$ kept {{r1.q06.strong.R_q3}}\% of its gain at MLX~3-bit and {{r1.q06.strong.R_Q3KM}}\% at Q3\_K\_M.",
    r"full fine-tuning at $3\times10^{-5}$ ($\lVert\Delta\rVert = {{r2.full3e5.norm}}$) kept {{r2.full3e5.R_q3}} of its gain at MLX~3-bit and {{r2.full3e5.R_Q2K}} at Q2\_K, while LoRA at $3\times10^{-5}$ ({{r2.lora3e5.norm}}) kept {{r2.lora3e5.R_q3}} and {{r2.lora3e5.R_Q2K}}.",
    r"and ${{r2.rho_lora_q4km}}$ for LoRA at Q4\_K\_M, where $3\times10^{-5}$ and $10^{-4}$ are tied to within {{r2.tie}}.",
    r"the LoRA at $10^{-5}$, kept {{r1.q06.gentle.R_q3}}\% of its gain at MLX~3-bit.",
    r"On Qwen3-1.7B, the LoRA at $10^{-5}$ kept {{r2.q17.gentle.R_q3}}\% of its gain at MLX~3-bit and the LoRA at $10^{-4}$ kept {{r2.q17.strong.R_q3}}\%. In accuracy retention, which the collapsing base does not inflate, the pair is {{r2.q17.gentle.A_q3}} against {{r2.q17.strong.A_q3}}.",
    r"all {{bands.low.pairs}} pairs with NSR below 3 kept at least 90\% of their gain, against {{bands.mid.ge90}} of {{bands.mid.pairs}} between 3 and 5 and {{bands.high.ge90}} of {{bands.high.pairs}} at 5 or more",
    r"only {{mech.full3e6.kept_q3}}\% of its update's projection survives MLX~3-bit",
    r"its NSR ({{mech.full3e6.nsr_q3}}) looks benign, and its measured gain retention is {{mech.full3e6.R_q3}}\%",
    r"the same fine-tune keeps {{mech.full3e6.kept_q4}}\% of its update and {{mech.full3e6.R_q4}}\% of its gain",
    r"the {{mech.n_lt60}} pairs that kept less than 60\% of their update were all exploratory Qwen3-0.6B full fine-tunes, {{mech.n_lt60_3e6}} of them from the run at $3\times10^{-6}$",
    r"the lowest update kept in rounds 3 and 4 was {{mech.min_kept_r34}}\%",
    r"In {{mech.kept98}} of all {{mech.pairs}} measured pairs",
    r"only large updates held on: {{bands.broken.ge90}} of {{bands.broken.pairs}} such pairs kept 90\% of their gain",
    r"at MLX~3-bit the Qwen3-0.6B base scored {{r1.q06base.acc_q3}}\%, while its LoRA at $10^{-4}$ kept {{r1.q06.strong.R_q3}}\% of its gain",
    r"kept {{mech.q2k.full_R}}\% and {{mech.q2k.lora_R}}\% of their gain with almost the same NSR ({{mech.q2k.full_nsr}} and {{mech.q2k.lora_nsr}})",
    r"All {{rep.n_tests}} tests were supported",
    r"The differences ranged from ${{rep.diff_min}}$ (OLMo at Q3\_K\_M, with both seeds) to ${{rep.diff_max}}$ (MASSIVE at MLX~3-bit, seed 1)",
    r"For Qwen3-4B at MLX~3-bit, {{rep.q4b.base_term}} of the {{rep.q4b.diff}} difference comes from the base's own drop.",
    r"It was positive in all ten, by {{rep.drop_min}} to {{rep.drop_max}} points, with every interval above zero.",
    r"{{pat.cells}} comparisons in the same direction",
    r"left both Qwen3-0.6B LoRAs at 0\%",
    r"{{pat.q4km_under2}} of the eight are under 2 points",
    r"At MLX~4-bit every gap is at least {{pat.min_gap_q4}} points, and at MLX~3-bit, Q3\_K\_M and Q2\_K at least {{pat.min_gap_low}}.",
    r"at MLX~3-bit and Q2\_K in all {{abs.strong_more.q3}} settings and at Q3\_K\_M in {{abs.strong_more.Q3KM}}, but only in {{abs.strong_more.q4}} of eight at MLX~4-bit and in none at Q4\_K\_M",
    r"The strong LoRA's bf16 cost was {{abs.cost_min}} to {{abs.cost_max}} points.",
    r"the gentle LoRA still answered {{strata.q4b.gentle}}\% correctly after quantization and the strong LoRA {{strata.q4b.strong}}\%",
    r"(base damage {{kld.q4b.q3}}, against {{kld.q06.q3}} for Qwen3-0.6B and {{kld.q17.q3}} for 1.7B)",
    r"accuracy retention {{pat.q4b.A_g_q3}} against {{pat.q4b.A_s_q3}}, compared with {{pat.q06.A_g_q3}} against {{pat.q06.A_s_q3}} at 0.6B",
    r"the gentle 4B LoRA still loses {{pat.q4b.loss_Q2K}}\% of its accuracy",
    r"went from {{seed.mas.gR_s0}} with seed 0 to {{seed.mas.gR_s1}} with seed 1, and the strong LoRA's bf16 accuracy fell {{seed.mas.s_dacc}} points",
    r"$R$ was {{r5.r1.unf_q4}} unfused against {{r5.r1.fus_q4}} fused at MLX~4-bit, and {{r5.r1.unf_q3}} against {{r5.r1.fus_q3}} at 3-bit",
    r"A {{r5.smoke_n}}-item plumbing test",
    r"({{r5.olmo_base.acc_bf16}}\% at bf16, {{r5.olmo_base.acc_q3}}\% at 3-bit) while the fused gentle LoRAs kept only {{r5.h1.s0.R_fused}} and {{r5.h1.s1.R_fused}} of their gain",
    r"recover most of the missing {{r5.h1.miss_lo}} to {{r5.h1.miss_hi}}; the plan called the hypothesis supported if each seed's upper bound stayed below ${{r5.h1.margin}}$",
    r"Unfusing changed $R$ by ${{r5.h1.s0.d}}$ [${{r5.h1.s0.lo}}$, ${{r5.h1.s0.hi}}$] for seed~0 and ${{r5.h1.s1.d}}$ [${{r5.h1.s1.lo}}$, ${{r5.h1.s1.hi}}$] for seed~1",
    r"recovered {{r5.h1.s0.share}}\% and {{r5.h1.s1.share}}\% of the loss by point estimate",
    r"within {{r5.unf.maxdiff}} of each other in $R$",
    r"delivered {{r5.h2.S_pct}}\% of the bf16 gain, against {{r5.h2.R_pct}}\% for the same LoRA trained in bf16, fused and quantized: a difference of ${{r5.h2.d}}$ [${{r5.h2.lo}}$, ${{r5.h2.hi}}$]",
    r"(final validation loss {{r5.q4s.val}} at 4 bits and {{r5.q3s.val}} at 3 bits, against {{r5.bf16s.val}} in bf16) and delivered {{r5.q4s.S}}\% of the gain at 4 bits and {{r5.q3s.S}}\% at 3 bits, against {{r5.q4s.R}}\% and {{r5.q3s.R}}\% when trained in bf16, fused and quantized",
    r"restricting decoding to the {{data.b77.labels}} intent names",
    r"(the gentle LoRA at Q3\_K\_M kept {{r5.lab.g.A_lab}} of its accuracy with the restriction and {{r5.lab.g.A_unc}} without; the strong one {{r5.lab.s.A_lab}} and {{r5.lab.s.A_unc}})",
    r"At Q2\_K it kept {{r5.lab.g.A_lab_Q2K}}\% of its accuracy even when forced to name a valid intent.",
    r"on new learning rates and seeds, by {{pred.v1.p1_miss}}",
    r"\hat R = \sigma\left({{pred.v2.t0}} + {{pred.v2.t1}} \ln \frac{\mathrm{kept}}{\mathrm{NSR}} - {{pred.v2.t2abs}} \ln \mathrm{KLD}_{\mathrm{base}}\right)",
    r"chosen among {{pred.n_forms}} candidates by leave-one-fine-tune-out cross-validation on all {{pred.fit_n}} round-1 and round-2 points (error {{pred.lofo.v2}}, against {{pred.lofo.v1form}} for the v1 form)",
    r"Its MAE was {{pred.v2.r3.mae}} on round~3 (OLMo-2 1B and MASSIVE, {{pred.r3.n}} points) and {{pred.v2.r4.mae}} on round~4 (second seeds and Qwen3-4B, {{pred.r4.n}} points), where the interval reaches {{pred.v2.r4.hi}}",
    r"Predictor v1 would also have met the accuracy bar in both rounds ({{pred.v1.r3.mae}} and {{pred.v1.r4.mae}})",
    r"With {{pred.r3.n_ft}} or {{pred.r4.n_ft}} held-out fine-tunes per round",
    r"the lowest update kept was {{pred.r3.min_kept}}\% in round~3 and {{pred.r4.min_kept}}\% in round~4",
    r"scores {{pb.const.r3}} in round~3, where v2 beats it by {{pb.v2const.r3.d}} [{{pb.v2const.r3.lo}}, {{pb.v2const.r3.hi}}] (paired, resampling fine-tunes), but {{pb.const.r4}} in round~4, where the difference, {{pb.v2const.r4.d}}, has an interval from ${{pb.v2const.r4.lo}}$ to {{pb.v2const.r4.hi}}",
    r"its leave-one-fine-tune-out error is {{pb.lofo.norm}} against {{pb.lofo.v2}}, its held-out error is {{pb.norm.r3}} in both rounds, and in round~4 it beats v2 by {{pb.v2norm.r4.d}} [{{pb.v2norm.r4.lo}}, {{pb.v2norm.r4.hi}}]",
    r"correlate at ${{corr.lo}}$ to ${{corr.hi}}$ within nine of the ten formats (${{corr.mlxq3}}$ at MLX~3-bit; {{corr.n_four}} formats have only four points)",
    r"was predicted to keep {{miss.mas.pred}}\% of its gain and kept {{miss.mas.s0}}\% (seed 0) and {{miss.mas.s1}}\% (seed 1)",
    r"the gentle 4B LoRA kept {{miss.q4b.R_q3}}\% of its gain at MLX~3-bit with NSR {{miss.q4b.nsr_q3}}, where v2 predicted {{miss.q4b.pred_q3}}\%",
    r"has $R = {{miss.q4b.R_Q2K}}$ against a prediction of {{miss.q4b.pred_Q2K}}\%, but only because its base collapsed; the fine-tune itself kept {{miss.q4b.A_Q2K}}\% of its accuracy",
    r"On the {{pred.q4b.n}} Qwen3-4B points alone, v2's MAE was {{pred.q4b.mae}} and its agreement {{pred.q4b.agree}}, short of both bars",
    r"the same {{miss.masg.pred}}\% prediction at MLX~3-bit, but kept {{miss.masg.s0}}\% and {{miss.masg.s1}}\% of their gain",
    r"because GGUF~Q4\_K\_M moved coverage by only {{r1.h4.gguf_abs}} points",
    r"MLX~3-bit kept {{r1.q17.A_q3}}\% of the accuracy but cut the share of auto-accepted items from {{drift.q17.cov_fp_q3}}\% to {{drift.q17.cov_q3}}\%, and Q2\_K kept {{r1.q17.A_Q2K}}\% of the accuracy but cut coverage from {{drift.q17.cov_fp_Q2K}}\% to {{drift.q17.cov_Q2K}}\%",
    r"the gentle LoRA's coverage fell further than the strong LoRA's in {{drift.n_fell}} of {{drift.n_cases}} cases",
    r"the Qwen3-0.6B LoRA at $10^{-5}$ kept {{drift.q06g.A_q4}}\% of its accuracy but accepted {{drift.q06g.dcov_q4}} points fewer items",
    r"\hat R = \sigma\left({{pred.v1.t0}} - {{pred.v1.t1abs}} \ln \mathrm{NSR} - {{pred.v1.t2abs}} \ln \mathrm{KLD}_{\mathrm{base}}\right)",
    r"P1 was not, by {{pred.v1.p1_miss}}, inside an interval from {{pred.v1.t2.lo}} to {{pred.v1.t2.hi}}",
    r"v1 did almost as well as v2 ({{pred.v1.r4.mae}} against {{pred.v2.r4.mae}} in round~4), and better on round~4's MASSIVE arm ({{pred.mas.r4.v1}} against {{pred.mas.r4.v2}}; in round~3, {{pred.mas.r3.v1}} against {{pred.mas.r3.v2}})",
    r"Without the easy 6-bit formats, v2's round-4 error was {{pred.v2.r4.no6}}, against {{pred.v2.r3.no6}} in round~3.",
    r"lists the {{grid.n}} fine-tunes of rounds 1 to 4",
    r"account for {{grid.evals}} full-test-set evaluations",
    r"reproduces the recorded round-3 values to within {{norms.check_pct}}\%",
    r"Measured on its {{data.massive.test}} test items, that fine-tune kept {{tool.mas.R_q4}}\% of its gain at MLX~4-bit and {{miss.masg.s0}}\% at 3-bit.",
    r"had the same prediction and kept {{miss.masg.s1}}\%",
    r"v1 SHA-256 prefix \texttt{{{sha.v1}}}, v2 \texttt{{{sha.v2}}}",
    r"In all {{rep.n_tests}} pre-registered tests (five fine-tune pairs, each at MLX~3-bit and GGUF~Q3\_K\_M), a LoRA trained at learning rate $10^{-4}$ retained more of its gain than the same LoRA at $10^{-5}$, mlx-lm's default, by ${{rep.diff_min}}$ to ${{rep.diff_max}}$ in $R$.",
    r"In exploratory comparisons over eight settings and five formats, the $10^{-5}$ LoRA kept a smaller share of its own accuracy in {{pat.lower}} of {{pat.cells}} cases, yet remained the more accurate model at Q4\_K\_M in all eight settings, while the $10^{-4}$ LoRA was the more accurate one at MLX~3-bit and Q2\_K in all {{abs.strong_more.q3}}.",
    r"Carrying the $10^{-5}$ OLMo-2 1B update exactly on the 3-bit base recovered {{r5.h1.s0.share}}\% and {{r5.h1.s1.share}}\% of the lost gain for two seeds (upper 95\% bounds {{r5.h1.s0.share_hi}}\% and {{r5.h1.s1.share_hi}}\%), while a $10^{-5}$ Qwen3-0.6B LoRA trained on the 4-bit base delivered {{r5.h2.S_pct}}\% of the bf16 gain at 4 bits, against {{r5.h2.R_pct}}\% when trained in bf16, fused and quantized (difference ${{r5.h2.d}}$, 95\% interval $[{{r5.h2.lo}}, {{r5.h2.hi}}]$); at $10^{-4}$, training on the 4-bit base delivered {{r5.q4s.S}}\% against {{r5.q4s.R}}\%.",
    r"A three-parameter predictor built from the weights and a one-time measurement of the base model met its pre-registered error bar of 0.10 on held-out fine-tunes (mean absolute error {{pred.v2.r3.mae}} and {{pred.v2.r4.mae}}), but a post hoc model of update size and base damage reached {{pb.norm.r3}} in both rounds.",
    r"We call a LoRA trained at learning rate $10^{-5}$ gentle and one trained at $10^{-4}$ strong; the gentle setting is mlx-lm's default, and its update was {{setup.ratio_min}} to {{setup.ratio_max}} times smaller.",
    r"For one fine-tune at Q3\_K\_M, llama-server with one slot and with four gave different predictions on {{r5.rep.diff_np}} of {{r5.rep.n}} items (Section~\ref{sec:round5}).",
    r"At 8 and 6 bits, $R$ was {{r1.h1.min}} to {{r1.h1.max}} in both engines (H1).",
    r"the most accurate of the three at bf16 ({{r1.q06.gentle.acc}}\%, against {{r1.q06.full.acc}}\% for full fine-tuning and {{r1.q06.strong.acc}}\% for the LoRA at $10^{-4}$)",
    r"A second seed moved $R$ at MLX~3-bit by at most {{r2.seed.max_dR}} ({{r2.seed.strong_s0}} and {{r2.seed.strong_s1}} for the LoRA at $10^{-4}$, {{r2.seed.gentle_s0}} and {{r2.seed.gentle_s1}} at $10^{-5}$), while bf16 accuracy moved by {{r2.seed.strong.dacc}} and {{r2.seed.gentle.dacc}} points.",
    r"Figure~\ref{fig:mechanism} in Appendix~\ref{app:extra} plots gain retention against NSR for the round-1 Qwen3-0.6B fine-tunes, and Table~\ref{tab:bands} counts all {{bands.n}} fine-tune and format pairs from rounds 1 and 2.",
    r"The rounded-away mode is rare: {{mech.n_lt60}} of {{mech.pairs}} pairs.",
    r"Broken outputs were {{valid.q06g.q3_invalid_share_of_errors}}\% of its errors at MLX~3-bit and all of them at Q2\_K.",
    r"$R$ moved by at most {{seed.olmo.dR_hard}} and bf16 accuracy by at most {{seed.olmo.dacc}} points",
    r"and {{r5.h1.s0.share_hi}}\% and {{r5.h1.s1.share_hi}}\% at the upper ends of the intervals.",
    r"and its final validation loss was nearly the same ({{r5.q4g.val}} against {{r5.bf16g.val}})",
    r"against {{r5.q3g.R}}\% for the fuse-then-quantize one (a difference of ${{r5.q3g.d}}$ [${{r5.q3g.lo}}$, ${{r5.q3g.hi}}$]), although its final validation loss",
    r"against {{r5.q4s.R}}\% and {{r5.q3s.R}}\% when trained in bf16, fused and quantized: differences of ${{r5.q4s.d}}$ [${{r5.q4s.lo}}$, ${{r5.q4s.hi}}$] and ${{r5.q3s.d}}$ [${{r5.q3s.lo}}$, ${{r5.q3s.hi}}$].",
    r"scored {{r5.rep.acc}}\% with four llama-server slots and with one; the two runs differed on {{r5.rep.diff_np}} of {{r5.rep.n}} items, and each differed from the round-1 run on {{r5.rep.diff_w1}}.",
    r"changed accuracy retention by at most {{r5.lab.max_change}}",
    r"from {{drift.err.olmo.fp}}\% to {{drift.err.olmo.q}}\% on OLMo-2 1B (seed 0) and",
    r"The 4B fine-tunes' gains are small ({{lim.q4b.gain_g}} points for the gentle LoRA and {{lim.q4b.gain_s}} for the strong)",
    r"At MLX~3-bit and GGUF~Q3\_K\_M, the gentle LoRA kept less of its gain than the strong one in {{rep.n_supported}} of {{rep.n_tests}} pre-registered tests across three base models and two tasks (F1), and it kept a smaller share of its accuracy at all five formats compared, {{pat.lower}} of {{pat.cells}} (F2).",
    r"In MLX, the loss did not come from rounding the update: the exact update recovered {{r5.h1.s0.share}}\% and {{r5.h1.s1.share}}\% of it, with upper 95\% bounds of {{r5.h1.s0.share_hi}}\% and {{r5.h1.s1.share_hi}}\% (F4), while training the gentle LoRA on the quantized base delivered {{r5.h2.S_pct}}\% of its bf16 gain at 4 bits, against {{r5.h2.R_pct}}\% (F5).",
    r"In practice, select a fine-tune on the quantized artifact it will ship as: the gentle LoRA was the more accurate model at Q4\_K\_M in all eight settings, the strong one at MLX~3-bit and Q2\_K in all {{abs.strong_more.q3}} and at Q3\_K\_M in {{abs.strong_more.Q3KM}} (F3).",
    r"Keeping the adapter unfused does not rescue a gentle LoRA in MLX; training it on the quantized base does, and gave our most accurate 4-bit and 3-bit models, but it cost the strong LoRA {{r5.q4s.d_abs}} of $R$ at 4 bits and was not tested for GGUF.",
    r"rounds 1 to 4 in {{time.r14.hours}} hours of wall-clock time from the first sweep ({{time.r14.start}}) to the last verdicts ({{time.r14.end}}), and round~5 in {{time.r5.hours}} hours from {{time.r5.start}}, to {{time.r5.end}},",
    r"In rounds 1 to 4, a LoRA run of 500 steps took {{time.train.q06.lo}} to {{time.train.q06.hi}} minutes on Qwen3-0.6B, {{time.train.q17.lo}} to {{time.train.q17.hi}} on 1.7B, {{time.train.olmo1.lo}} to {{time.train.olmo1.hi}} on OLMo-2 1B and {{time.train.q4b.lo}} to {{time.train.q4b.hi}} on 4B.",
    r"One full-test-set evaluation took {{time.eval.small.lo}} to {{time.eval.small.hi}} minutes on the smaller models and {{time.eval.q4b.lo}} to {{time.eval.q4b.hi}} on 4B, and the round-4 mechanism measurements for two fine-tunes took {{time.mech.lo}} to {{time.mech.hi}} minutes on the 0.6B and 1B models and {{time.mech.q4b}} on 4B.",
    r"In round~5, three of the four LoRA runs on a quantized Qwen3-0.6B base took {{time.r5.train.lo}} to {{time.r5.train.hi}} minutes; the fourth, retrained after it stalled, took {{time.r5.train.stalled}} minutes of wall-clock time.",
    r"Spearman $\rho = 1.0$ in {{r2.n_rho1}}",
    r"At MLX~2-bit both models score {{r1.q17.acc_q2}}\%.",
    r"W5-H1 needed each upper bound below ${{r5.h1.margin}}$",
    r"), kept {{r1.q06.gentle.R_q3}}\% at MLX~3-bit, and full fine-tuning at $10^{-5}$ kept {{r1.q06.full.R_q3}}\%.",
    r"at other formats $R$ moved by up to {{seed.olmo.dR_other}}",
    r"Its accuracy at 4 bits, {{r5.q4g.acc}}\%, was within a point of what the bf16-trained LoRA reached at bf16 ({{r1.q06.gentle.acc}}\%)",
    r"which scores {{r5.q3base.acc}}\% on its own, a gentle LoRA trained there reached {{r5.q3g.acc}}\% and delivered {{r5.q3g.S}}\% of the bf16 gain",
    r"although its final validation loss was higher than in bf16 ({{r5.q3g.val}} against {{r5.bf16g.val}})",
    r"rose from {{drift.err.q4b.fp}}\% to {{drift.err.q4b.q}}\% on Qwen3-4B,",
    r"from {{drift.err.q17.fp}}\% to {{drift.err.q17.q}}\% on Qwen3-1.7B, while the strong LoRAs ended at {{drift.err.strong_lo}}\% to {{drift.err.strong_hi}}\%",
    r"produced a valid intent name for {{valid.q06g.q4}}\% of items at MLX~4-bit and {{valid.q06g.Q3KM}}\% at Q3\_K\_M.",
]


FINDINGS: list[dict[str, str]] = [
    {
        "id": "F1",
        "section": "sec:replication",
        "status": "pre-registered",
        "finding": r"At MLX~3-bit and Q3\_K\_M, the gentle LoRA keeps less of its gain than the strong one",
        "result": r"5 pairs on 3 base models and 2 tasks: $R_{\mathrm{strong}} - R_{\mathrm{gentle}}$ from ${{rep.diff_min}}$ to ${{rep.diff_max}}$, interval above 0 in {{rep.n_supported}} of {{rep.n_tests}} tests, also at the 99.5\% level",
        "evidence": r"Pre-registered, supported (W3-H1, W4-H1, W4-H2); \S\ref{sec:replication}",
    },
    {
        "id": "F2",
        "section": "sec:replication",
        "status": "exploratory",
        "finding": r"The gentle LoRA keeps a smaller share of its accuracy at every format compared",
        "result": r"$A_{\mathrm{strong}} - A_{\mathrm{gentle}} > 0$ in {{pat.lower}} of {{pat.cells}} cells: at least {{pat.min_gap_q4}} points at MLX~4-bit and {{pat.min_gap_low}} at MLX~3-bit, Q3\_K\_M and Q2\_K; under 2 points in {{pat.q4km_under2_n}} of 8 at Q4\_K\_M",
        "evidence": r"Exploratory; \S\ref{sec:replication}",
    },
    {
        "id": "F3",
        "section": "sec:replication",
        "status": "exploratory",
        "finding": r"Which LoRA is more accurate after quantization depends on the format",
        "result": r"Strong more accurate in {{abs.n.q3}} of 8 settings at MLX~3-bit, {{abs.n.Q2K}} at Q2\_K, {{abs.n.Q3KM}} at Q3\_K\_M, {{abs.n.q4}} at MLX~4-bit and {{abs.n.Q4KM}} at Q4\_K\_M; its bf16 cost: {{abs.cost_min}} to {{abs.cost_max}} points",
        "evidence": r"Exploratory; \S\ref{sec:replication}",
    },
    {
        "id": "F4",
        "section": "sec:round5",
        "status": "pre-registered",
        "finding": r"In MLX, keeping a gentle update exact does not restore its gain",
        "result": r"Unfused minus fused $R$, OLMo-2 1B at MLX~3-bit, 2 seeds: ${{r5.h1.s0.d}}$ [${{r5.h1.s0.lo}}$, ${{r5.h1.s0.hi}}$] and ${{r5.h1.s1.d}}$ [${{r5.h1.s1.lo}}$, ${{r5.h1.s1.hi}}$]: {{r5.h1.s0.share}}\% and {{r5.h1.s1.share}}\% of the lost gain, upper bounds {{r5.h1.s0.share_hi}}\% and {{r5.h1.s1.share_hi}}\%",
        "evidence": r"Pre-registered, supported (W5-H1, bound ${{r5.h1.margin}}$); \S\ref{sec:round5}",
    },
    {
        "id": "F5",
        "section": "sec:round5",
        "status": "pre-registered",
        "finding": r"In MLX, training a gentle LoRA on the quantized base recovers its gain",
        "result": r"Gain delivered minus fused $R$, Qwen3-0.6B, 1 seed: ${{r5.h2.d}}$ [${{r5.h2.lo}}$, ${{r5.h2.hi}}$] at 4-bit and ${{r5.q3g.d}}$ [${{r5.q3g.lo}}$, ${{r5.q3g.hi}}$] at 3-bit; for the strong LoRA, ${{r5.q4s.d}}$ [${{r5.q4s.lo}}$, ${{r5.q4s.hi}}$] and ${{r5.q3s.d}}$ [${{r5.q3s.lo}}$, ${{r5.q3s.hi}}$]",
        "evidence": r"Pre-registered at 4-bit for the gentle LoRA, supported (W5-H2); rest exploratory; \S\ref{sec:round5}",
    },
    {
        "id": "F6",
        "section": "sec:predictor",
        "status": "pre-registered",
        "finding": r"A weight-only predictor meets its pre-registered bar, but the size of the update predicts as well",
        "result": r"Held-out MAE of v2: {{pred.v2.r3.mae}} [{{pred.v2.r3.lo}}, {{pred.v2.r3.hi}}] and {{pred.v2.r4.mae}} [{{pred.v2.r4.lo}}, {{pred.v2.r4.hi}}], bar 0.10; update norm with base damage: {{pb.norm.r3}} and {{pb.norm.r4}}; v2 on Qwen3-4B alone: {{pred.q4b.mae}}",
        "evidence": r"Pre-registered, supported (W3-H2, W3-H3, W4-H3, W4-H4); norm model post hoc; \S\ref{sec:predictor}",
    },
    {
        "id": "F7",
        "section": "sec:drift",
        "status": "exploratory",
        "finding": r"At a confidence threshold fixed at bf16, the gentle LoRA loses more coverage",
        "result": r"Coverage fell further for the gentle LoRA in {{drift.n_fell}} of {{drift.n_cases}} cells; in the pre-registered 4-bit test on a Qwen3-1.7B LoRA, Q4\_K\_M moved coverage by ${{r1.h4.gguf}}$ points",
        "evidence": r"Exploratory; pre-registered H4 not supported; \S\ref{sec:drift}",
    },
    {
        "id": "C",
        "section": "sec:setup",
        "status": "control",
        "finding": r"Measurement repeatability",
        "result": r"MLX and GGUF agree on {{nf.ft_lo}}\% to {{nf.ft_hi}}\% of the bf16 predictions of each fine-tune that learned its task; one and four llama-server slots differ on {{r5.rep.diff_np}} of {{r5.rep.n}} items; decoding restricted to valid labels changes $A$ by at most {{r5.lab.max_change}}",
        "evidence": r"Controls; \S\ref{sec:setup}, \S\ref{sec:round5}",
    },
]

CONSTANTS: list[tuple[str, str]] = [
    (r"Qwen3-(?:0\.6|1\.7)B|\b(?:0\.6|1\.7)B\b", "model name"),
    (r"mlx-lm 0\.31\.3|MLX 0\.32\.2", "software version"),
    (r"width=0\.\d+\\linewidth|p\{\d\.\dcm\}", "layout"),
    (
        r"at most 0\.1[05]|at least 0\.8[05]|the 0\.10 bar|bar of 0\.10|bar 0\.10",
        "pre-registered threshold",
    ),
    (
        r"at least 9[08]\\%|less than 60\\%|at most 5\\%|\$\\ge 90\\%\$|at least 0\.9\b|side of 0\.9\b",
        "threshold defined in the text",
    ),
    (r"95\\%", "interval level"),
    (r"99\.5\\%", "Bonferroni level for ten tests"),
    (r"0\\% task success|1\.0 percentage point", "number from a cited work"),
    (r"above 125\\% are drawn at 125\\%|capped at 100\\%", "plotting convention"),
    (r"adds 0\.5 bits per weight", "32 bits of scale and bias per 64 weights"),
]


def rounded(x: object) -> object:
    return round(x, 6) if isinstance(x, float) else x


def delatex(s: str) -> str:
    s = s.replace(r"\%", "%").replace(r"\_", "_").replace("~", " ")
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\S\\ref\{([^}]*)\}", r"section \1", s)
    return s.replace("$", "")


def outputs() -> tuple[str, str]:
    rows = []
    for f in FINDINGS:
        if f["id"] == "C":
            rows.append(r"\addlinespace")
        cells = " & ".join(render(f[k]) for k in ("finding", "result", "evidence"))
        rows.append(f"{f['id']} & {cells} \\\\")
    doc = {
        "findings": [
            {
                "id": f["id"],
                "section": f["section"],
                "status": f["status"],
                **{k: delatex(render(f[k])) for k in ("finding", "result", "evidence")},
                "values": {
                    k: V[k]
                    for k in sorted(
                        set(re.findall(r"\{\{([^}]+)\}\}", f["result"] + f["evidence"]))
                    )
                },
            }
            for f in FINDINGS
        ],
        "values": {k: {"text": V[k], "value": rounded(RAW[k])} for k in sorted(V)},
        "quoted_in_text": [normalize(render(a)) for a in ANCHORS],
        "constants": [{"pattern": p, "reason": r} for p, r in CONSTANTS],
    }
    return json.dumps(
        doc, indent=1, sort_keys=True, ensure_ascii=False
    ) + "\n", "\n".join(rows) + "\n"


def prose() -> str:
    t = re.sub(r"(?<!\\)%.*", "", TEX.read_text())
    t = t[t.index("\\begin{abstract}") : t.index("\\end{document}")]
    t = re.sub(r"\\begin\{verbatim\}.*?\\end\{verbatim\}", " ", t, flags=re.S)
    return normalize(t)


def uncovered() -> list[str]:
    text = prose()
    covered = np.zeros(len(text), dtype=bool)
    for tpl in ANCHORS:
        s = normalize(render(tpl))
        start = text.find(s)
        while start >= 0:
            covered[start : start + len(s)] = True
            start = text.find(s, start + 1)
    for pattern, _ in CONSTANTS:
        for m in re.finditer(pattern, text):
            covered[m.start() : m.end()] = True
    out = []
    token = r"\d[\d,]*\.\d+|\d+(?:\.\d+)?\\%|\b\d+ of \d+\b|\b(?:one|two|three|four|five|six|seven|eight|nine|ten) of (?:the )?(?:eight|ten|\d+)\b"
    for m in re.finditer(token, text):
        if not covered[m.start() : m.end()].any():
            out.append(
                f"{m.group(0):>10s}  ...{text[max(0, m.start() - 70) : m.end() + 30]}..."
            )
    return out


def render(tpl: str) -> str:
    return re.sub(r"\{\{([A-Za-z0-9_.\-]+)\}\}", lambda m: V[m.group(1)], tpl)


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s)


def compute() -> None:
    for step in (
        setup,
        round1,
        round2,
        mechanism,
        replication,
        pattern,
        seeds,
        round5,
        predictor,
        drift,
        limits_and_appendix,
        timeline,
    ):
        step()


def check_text() -> list[str]:
    text = normalize(re.sub(r"(?<!\\)%.*", "", TEX.read_text()))
    missing = []
    for tpl in ANCHORS:
        try:
            s = normalize(render(tpl))
        except KeyError as e:
            missing.append(f"unknown key {e} in: {tpl[:80]}")
            continue
        if s not in text:
            parts = re.split(r"\{\{[^}]+\}\}", tpl)
            head = normalize(parts[0])
            i = text.find(head) if head.strip() else -1
            seen = text[i : i + len(s) + 40] if i >= 0 else "(context not found)"
            missing.append(f"expected: {s}\n     found: {seen}")
    return missing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--values", action="store_true")
    args = ap.parse_args()
    compute()
    if args.values:
        for k in sorted(V):
            print(f"{k:40s} {V[k]}")
    paper = TEX.exists()
    missing = check_text() if paper else []
    loose = uncovered() if paper else []
    for u in loose:
        print("NUM   " + u)
    for m in missing:
        print("TEXT  " + m)
    for p in PROBLEMS:
        print("DATA  " + p)
    print(
        f"{len(V)} values, {len(PROBLEMS)} data problems; "
        + (
            f"{len(ANCHORS)} text checks, {len(missing)} mismatches, {len(loose)} unchecked numbers"
            if paper
            else "paper source not present, text checks skipped"
        )
    )
    doc, table = outputs()
    targets = [(OUT_JSON, doc)] + ([(OUT_TABLE, table)] if paper else [])
    stale = [p.name for p, s in targets if not p.exists() or p.read_text() != s]
    if args.check:
        for name in stale:
            print(f"STALE {name} differs from a fresh computation")
        if missing or PROBLEMS or loose or stale:
            sys.exit(1)
        return
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    for path, content in targets:
        path.write_text(content)


if __name__ == "__main__":
    main()
