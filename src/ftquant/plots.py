import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
FIGS = ROOT / "figures"

GROUND, INK, INK2, GRID = "#f8f9fb", "#101720", "#424c5a", "#d7dde7"
ACCENT, NEUTRAL, LIGHT = "#2b62e4", "#8a94a6", "#b9c3d2"
CONFIG_STYLE = {
    "lora": ("LoRA, learning rate 1e-4", ACCENT),
    "lora-lowlr": ("LoRA, learning rate 1e-5", NEUTRAL),
    "full": ("Full fine-tune, learning rate 1e-5", INK2),
}
MLX_ORDER = ["mlx-bf16", "mlx-q8", "mlx-q6", "mlx-q4", "mlx-q3", "mlx-q2"]
GGUF_ORDER = [
    "gguf-bf16",
    "gguf-Q8_0",
    "gguf-Q6_K",
    "gguf-Q4_K_M",
    "gguf-Q3_K_M",
    "gguf-Q2_K",
]


def fonts() -> None:
    names = {f.name for f in font_manager.fontManager.ttflist}
    family = "Helvetica" if "Helvetica" in names else "Arial"
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 14,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK2,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "text.color": INK,
        }
    )


def bpw_map(model_tag: str) -> dict[str, float]:
    mlx = json.loads((RUNS / "bpw" / f"mlx-{model_tag}.json").read_text())["bpw"]
    out = {f"mlx-{k}": v for k, v in mlx.items()}
    g = json.loads((RUNS / "week1" / f"mechanism-{model_tag}-gguf.json").read_text())[
        "bpw"
    ]
    out.update({f"gguf-{k}": v for k, v in g.items()})
    return out


def rows(model_tag: str) -> list[dict]:
    data = json.loads((RUNS / "week1" / "analysis.json").read_text())["retention"]
    return [r for r in data if r["config"].startswith(model_tag + "-")]


def frame(ax) -> None:
    ax.set_facecolor(GROUND)
    ax.grid(True, color=GRID, linewidth=1, linestyle=":")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def retention_chart(model_tag: str, out: Path, dpi: int = 200) -> None:
    fonts()
    bpw = bpw_map(model_tag)
    data = rows(model_tag)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=dpi)
    fig.patch.set_facecolor(GROUND)
    frame(ax)
    for cfg_suffix, (label, color) in CONFIG_STYLE.items():
        cfg = f"{model_tag}-{cfg_suffix}"
        for order, ls in ((MLX_ORDER, "-"), (GGUF_ORDER, "--")):
            pts = [
                (bpw[v], r)
                for v in order
                for r in data
                if r["config"] == cfg and r["variant"] == v
            ]
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [100 * p[1]["retention"] for p in pts]
            lo = [100 * p[1]["ci"][0] for p in pts]
            hi = [100 * p[1]["ci"][1] for p in pts]
            ax.fill_between(xs, lo, hi, color=color, alpha=0.12, linewidth=0)
            ax.plot(
                xs,
                ys,
                ls,
                color=color,
                linewidth=2.6 if cfg_suffix == "lora" else 2.2,
                marker="o",
                markersize=5,
                label=label if ls == "-" else None,
            )
    ax.axhline(100, color=INK2, linewidth=1, alpha=0.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks([2.5, 3.5, 4.5, 6.5, 8.5, 16])
    ax.set_xticklabels(["2.5", "3.5", "4.5", "6.5", "8.5", "16"])
    ax.set_xlim(2.3, 17.5)
    ax.set_ylim(-5, 112)
    ax.set_xlabel("Bits per weight (measured), log scale", fontsize=13)
    ax.set_ylabel("Fine-tune gain retained (%)", fontsize=13)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=12)
    for t in leg.get_texts():
        t.set_color(INK)
    ax.text(
        0.01,
        1.02,
        "Solid: MLX affine   Dashed: GGUF k-quants   Bands: 95% CI",
        transform=ax.transAxes,
        fontsize=11,
        color=INK2,
    )
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=GROUND, metadata={"Software": None})
    plt.close(fig)


def drift_chart(model_tag: str, cfg_suffix: str, out: Path, dpi: int = 200) -> None:
    fonts()
    bpw = bpw_map(model_tag)
    data = [r for r in rows(model_tag) if r["config"] == f"{model_tag}-{cfg_suffix}"]
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=dpi)
    fig.patch.set_facecolor(GROUND)
    frame(ax)
    for order, ls, eng in ((MLX_ORDER, "-", "MLX"), (GGUF_ORDER, "--", "GGUF")):
        pts = [(bpw[v], r) for v in order for r in data if r["variant"] == v]
        xs = [p[0] for p in pts]
        ax.plot(
            xs,
            [100 * p[1]["retention"] for p in pts],
            ls,
            color=ACCENT,
            linewidth=2.4,
            marker="o",
            markersize=5,
            label=f"Accuracy gain retained ({eng})",
        )
        ax.plot(
            xs,
            [100 * p[1]["q_coverage"] / p[1]["fp_coverage"] for p in pts],
            ls,
            color=INK2,
            linewidth=2.2,
            marker="s",
            markersize=5,
            label=f"Items auto-accepted at the fixed threshold ({eng})",
        )
    ax.axhline(100, color=INK2, linewidth=1, alpha=0.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks([2.5, 3.5, 4.5, 6.5, 8.5, 16])
    ax.set_xticklabels(["2.5", "3.5", "4.5", "6.5", "8.5", "16"])
    ax.set_xlim(2.3, 17.5)
    ax.set_ylim(-5, 112)
    ax.set_xlabel("Bits per weight (measured), log scale", fontsize=13)
    ax.set_ylabel("% of full precision", fontsize=13)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=10.5)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    fig.savefig(out, facecolor=GROUND, metadata={"Software": None})
    plt.close(fig)


def mechanism_chart(
    model_tag: str,
    out: Path,
    dpi: int = 200,
    note: str = "Circles: MLX   Diamonds: GGUF   (2-bit excluded: base model breaks)",
    xlabel: str = "Quantization noise relative to the fine-tune update (log)",
) -> None:
    fonts()
    data = rows(model_tag)
    mlx_mech = {
        r["config"]: r
        for r in json.loads(
            (RUNS / "week1" / f"mechanism-{model_tag}-mlx.json").read_text()
        )
    }
    gguf_mech = json.loads(
        (RUNS / "week1" / f"mechanism-{model_tag}-gguf.json").read_text()
    )["configs"]
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=dpi)
    fig.patch.set_facecolor(GROUND)
    frame(ax)
    for cfg_suffix, (label, color) in CONFIG_STYLE.items():
        cfg = f"{model_tag}-{cfg_suffix}"
        pts = []
        for r in data:
            if (
                r["config"] != cfg
                or r["variant"].endswith("bf16")
                or r["variant"].endswith("unfused")
            ):
                continue
            eng, q = r["variant"].split("-", 1)
            if eng == "mlx" and q[1:].isdigit() and cfg in mlx_mech and q[1:] != "2":
                pts.append(
                    (mlx_mech[cfg]["noise_to_signal"][q[1:]], r["retention"], "o")
                )
            elif eng == "gguf" and cfg in gguf_mech and q != "Q2_K":
                pts.append((gguf_mech[cfg][q]["noise_to_signal"], r["retention"], "D"))
        for x, y, m in pts:
            ax.scatter(x, 100 * y, color=color, marker=m, s=60, zorder=3)
        ax.scatter([], [], color=color, marker="o", s=60, label=label)
    ax.set_xscale("log")
    ax.set_ylim(-5, 112)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("Fine-tune gain retained (%)", fontsize=13)
    leg = ax.legend(loc="lower left", frameon=False, fontsize=11)
    for t in leg.get_texts():
        t.set_color(INK)
    ax.text(
        0.99,
        1.02,
        note,
        transform=ax.transAxes,
        ha="right",
        fontsize=10.5,
        color=INK2,
    )
    fig.tight_layout()
    fig.savefig(out, facecolor=GROUND, metadata={"Software": None})
    plt.close(fig)


WARM = "#d9480f"


def lr_sweep_chart(out: Path, dpi: int = 200) -> None:
    fonts()
    table = json.loads((RUNS / "week2" / "summary.json").read_text())["configs"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=dpi, sharey=True)
    fig.patch.set_facecolor(GROUND)
    for ax, (variant, title) in zip(
        axes, (("mlx-q3", "MLX 3-bit"), ("gguf-Q3_K_M", "GGUF Q3_K_M"))
    ):
        frame(ax)
        for kind, color, label in (
            ("LoRA", ACCENT, "LoRA"),
            ("full", INK2, "Full fine-tune"),
        ):
            pts = sorted(
                (t["lr"], t["gain_retention"][variant])
                for t in table
                if t["model"] == "0.6B"
                and t["kind"] == kind
                and t["seed"] == 0
                and t["acc_ft_bf16"] > 0
            )
            ax.plot(
                [p[0] for p in pts],
                [100 * p[1] for p in pts],
                "-o",
                color=color,
                linewidth=2.4,
                markersize=6,
                label=f"{label}, Qwen3-0.6B",
            )
            s1 = [
                (t["lr"], t["gain_retention"][variant])
                for t in table
                if t["model"] == "0.6B" and t["kind"] == kind and t["seed"] == 1
            ]
            if s1:
                ax.scatter(
                    [p[0] for p in s1],
                    [100 * p[1] for p in s1],
                    s=70,
                    facecolors="none",
                    edgecolors=color,
                    linewidths=1.8,
                    zorder=4,
                    label=f"{label}, second seed",
                )
        big = sorted(
            (t["lr"], min(t["gain_retention"][variant], 1.25))
            for t in table
            if t["model"] == "1.7B"
        )
        ax.plot(
            [p[0] for p in big],
            [100 * p[1] for p in big],
            "--s",
            color=WARM,
            linewidth=2,
            markersize=6,
            label="LoRA, Qwen3-1.7B",
        )
        ax.axhline(100, color=INK2, linewidth=1, alpha=0.5)
        ax.set_xscale("log")
        ax.set_xticks([3e-6, 1e-5, 3e-5, 1e-4])
        ax.set_xticklabels(["3e-6", "1e-5", "3e-5", "1e-4"])
        ax.set_xlabel("Learning rate (500 steps)", fontsize=13)
        ax.set_title(title, fontsize=13, color=INK, loc="left")
    axes[0].set_ylim(-5, 130)
    axes[0].set_ylabel("Fine-tune gain retained (%)", fontsize=13)
    leg = axes[1].legend(loc="lower right", frameon=False, fontsize=10.5)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.text(
        0.99,
        0.005,
        "1.7B values above 125% are drawn at 125% (the base model collapses at 3-bit)",
        ha="right",
        fontsize=9.5,
        color=INK2,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=GROUND, metadata={"Software": None})
    plt.close(fig)


def signal_lookup() -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    for run in ("week1", "week2"):
        for tag in ("q06", "q17"):
            p = RUNS / run / f"mechanism-{tag}-mlx.json"
            if p.exists():
                for r in json.loads(p.read_text()):
                    for b, v in r["signal_retained"].items():
                        out[(r["config"], f"mlx-q{b}")] = v
            g = RUNS / run / f"mechanism-{tag}-gguf.json"
            if g.exists():
                for cfg, row in json.loads(g.read_text())["configs"].items():
                    for t, v in row.items():
                        if isinstance(v, dict):
                            out[(cfg, f"gguf-{t}")] = v["signal_retained"]
    return out


def predictor_chart(out: Path, dpi: int = 200) -> None:
    fonts()
    sig = signal_lookup()
    pts = []
    for test in ("T1", "T2", "T3"):
        for p in json.loads((RUNS / "predictor" / f"eval-{test}.json").read_text())[
            "points"
        ]:
            if p["retention"] == p["retention"]:
                pts.append((p, sig.get((p["config"], p["variant"]), 1.0)))
    fig, ax = plt.subplots(figsize=(6.2, 5.6), dpi=dpi)
    fig.patch.set_facecolor(GROUND)
    frame(ax)
    ax.fill_between([0, 100], [-10, 90], [10, 110], color=GRID, alpha=0.5, lw=0)
    ax.plot([0, 100], [0, 100], color=INK2, linewidth=1)
    for p, s in pts:
        y = 100 * min(max(p["retention"], 0.0), 1.0)
        x = 100 * p["predicted"]
        if s < 0.6:
            ax.scatter(x, y, s=80, color=WARM, marker="X", zorder=4)
        else:
            big = p["config"].startswith("q17")
            ax.scatter(
                x,
                y,
                s=46,
                color=NEUTRAL if big else ACCENT,
                marker="s" if big else "o",
                alpha=0.85,
                zorder=3,
            )
    ax.scatter([], [], s=46, color=ACCENT, marker="o", label="Qwen3-0.6B (T2)")
    ax.scatter([], [], s=46, color=NEUTRAL, marker="s", label="Qwen3-1.7B (T1, T3)")
    ax.scatter(
        [], [], s=80, color=WARM, marker="X", label="Update rounded away (<60% kept)"
    )
    ax.set_xlim(-3, 103)
    ax.set_ylim(-3, 103)
    ax.set_xlabel("Predicted before quantizing (%)", fontsize=13)
    ax.set_ylabel("Measured gain retained (%, capped at 100)", fontsize=13)
    leg = ax.legend(
        loc="lower left", bbox_to_anchor=(0.3, 0.07), frameon=False, fontsize=10.5
    )
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    fig.savefig(out, facecolor=GROUND, metadata={"Software": None})
    plt.close(fig)


def replication_chart(out: Path, data: dict | None = None, dpi: int = 200) -> None:
    fonts()
    if data is None:
        data = json.loads((RUNS / "week3" / "summary.json").read_text())
    names = list(data)
    fig, axes = plt.subplots(
        1, 2, figsize=(10, 1.2 + 0.85 * len(names)), dpi=dpi, sharey=True
    )
    fig.patch.set_facecolor(GROUND)
    for ax, (variant, title) in zip(
        axes, (("mlx-q3", "MLX 3-bit"), ("gguf-Q3_K_M", "GGUF Q3_K_M"))
    ):
        frame(ax)
        ax.grid(True, axis="x", color=GRID, linewidth=1, linestyle=":")
        ax.grid(False, axis="y")
        for i, name in enumerate(names):
            y = len(names) - 1 - i
            g = 100 * data[name]["LoRA 1e-5"]["accuracy"][variant]
            s = 100 * data[name]["LoRA 1e-4"]["accuracy"][variant]
            ax.plot([g, s], [y, y], color=LIGHT, linewidth=3, zorder=2)
            ax.scatter(g, y, s=90, color=WARM, zorder=3)
            ax.scatter(s, y, s=90, color=ACCENT, zorder=3)
        ax.set_xlim(-3, 105)
        ax.set_xlabel("Accuracy kept, % of its own bf16", fontsize=13)
        ax.set_title(title, fontsize=13, color=INK, loc="left")
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(
        [
            f"{n}\n(bf16: {100 * data[n]['LoRA 1e-5']['acc_bf16']:.1f}% vs {100 * data[n]['LoRA 1e-4']['acc_bf16']:.1f}%)"
            for n in reversed(names)
        ],
        fontsize=11,
    )
    axes[0].tick_params(axis="y", length=0)
    axes[1].scatter([], [], s=90, color=WARM, label="LoRA, learning rate 1e-5")
    axes[1].scatter([], [], s=90, color=ACCENT, label="LoRA, learning rate 1e-4")
    leg = axes[1].legend(loc="lower left", frameon=False, fontsize=10.5)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    fig.savefig(out, facecolor=GROUND, metadata={"Software": None})
    plt.close(fig)


PRED_ARMS_W3 = {
    "olmo1": ("OLMo-2 1B, banking77", NEUTRAL, "s"),
    "mas06": ("Qwen3-0.6B, MASSIVE", ACCENT, "o"),
}
PRED_ARMS_W4 = {
    "olmo1": ("OLMo-2 1B, banking77, seed 1", NEUTRAL, "s"),
    "mas06": ("Qwen3-0.6B, MASSIVE, seed 1", ACCENT, "o"),
    "q4b": ("Qwen3-4B, banking77", WARM, "^"),
}
REPLICATION_ORDER = [
    "Qwen3-0.6B, banking77",
    "Qwen3-1.7B, banking77",
    "Qwen3-4B, banking77",
    "OLMo-2 1B, banking77",
    "OLMo-2 1B, banking77, seed 1",
    "Qwen3-0.6B, MASSIVE",
    "Qwen3-0.6B, MASSIVE, seed 1",
]


def predictor_v2_chart(
    out: Path,
    run: str = "week3",
    arms: dict | None = None,
    dpi: int = 200,
    figsize: tuple[float, float] = (6.2, 5.6),
    legend: dict | None = None,
    xlabel: str = "Predicted before quantizing (%)",
    ylabel: str = "Measured gain retained (%, capped at 100)",
    show_legend: bool = True,
) -> None:
    fonts()
    arms = arms or PRED_ARMS_W3
    legend = legend or {
        "loc": "lower left",
        "bbox_to_anchor": (0.3, 0.07),
        "fontsize": 10.5,
    }
    pts = json.loads((RUNS / run / "verdicts.json").read_text())["points"]
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(GROUND)
    frame(ax)
    ax.fill_between([0, 100], [-10, 90], [10, 110], color=GRID, alpha=0.5, lw=0)
    ax.plot([0, 100], [0, 100], color=INK2, linewidth=1)
    for p in pts:
        _, color, marker = arms[p["config"].split("-")[0]]
        ax.scatter(
            100 * p["pred_v2"],
            100 * min(max(p["retention"], 0.0), 1.0),
            s=50,
            color=color,
            marker=marker,
            alpha=0.9,
            zorder=3,
        )
    for label, color, marker in arms.values():
        ax.scatter([], [], s=50, color=color, marker=marker, label=label)
    ax.set_xlim(-3, 103)
    ax.set_ylim(-3, 103)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    if show_legend:
        leg = ax.legend(frameon=False, **legend)
        for t in leg.get_texts():
            t.set_color(INK)
    fig.tight_layout()
    fig.savefig(out, facecolor=GROUND, metadata={"Software": None})
    plt.close(fig)


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["week3"]:
        replication_chart(FIGS / "replication.png")
        predictor_v2_chart(FIGS / "predictor-v2-week3.png")
        print("wrote replication.png predictor-v2-week3.png")
        raise SystemExit
    if sys.argv[1:] == ["week4"]:
        merged = {
            **json.loads((RUNS / "week3" / "summary.json").read_text()),
            **json.loads((RUNS / "week4" / "summary.json").read_text())["settings"],
        }
        replication_chart(
            FIGS / "replication-week4.png", {k: merged[k] for k in REPLICATION_ORDER}
        )
        predictor_v2_chart(FIGS / "predictor-v2-week4.png", "week4", PRED_ARMS_W4)
        print("wrote replication-week4.png predictor-v2-week4.png")
        raise SystemExit
    if sys.argv[1:] == ["week2"]:
        lr_sweep_chart(FIGS / "lr-sweep.png")
        predictor_chart(FIGS / "predictor-heldout.png")
        print("wrote lr-sweep.png predictor-heldout.png")
        raise SystemExit
    tag = sys.argv[1] if len(sys.argv) > 1 else "q06"
    retention_chart(tag, FIGS / f"retention-{tag}.png")
    drift_chart(tag, "lora", FIGS / f"drift-{tag}-lora.png")
    mechanism_chart(tag, FIGS / f"mechanism-{tag}.png")
    print("wrote", sorted(p.name for p in FIGS.glob(f"*{tag}*")))
