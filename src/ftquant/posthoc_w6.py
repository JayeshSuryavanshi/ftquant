from pathlib import Path

import ftquant.verdicts_w6 as v6


def use(run: Path) -> None:
    v6.W6 = run


def g_spec(arm: str, lr: str, fmt: str) -> v6.Spec:
    key = f"q06-{lr}" if arm == "G2" else f"q06dq{fmt[1]}-{lr}"
    return (key, f"gguf-{fmt}-lora")


def s_minus_r(arm: str, lr: str, fmt: str) -> dict:
    core = [
        (f"q06-{lr}", f"gguf-{fmt}"),
        ("q06-base", f"gguf-{fmt}"),
        (f"q06-{lr}", "gguf-bf16"),
        ("q06-base", "gguf-bf16"),
    ]
    return v6.boot(
        lambda x, g, q, f, z: v6.share(x, q, f, z) - v6.share(g, q, f, z),
        v6.correct(g_spec(arm, lr, fmt), *core),
    )


def d_gap(q: str) -> dict:
    fp = "gguf-bf16" if q.startswith("gguf") else "mlx-bf16"
    arrays = v6.correct(
        ("q06-def", q),
        ("q06-def-lowlr", q),
        ("q06-base", q),
        ("q06-def", fp),
        ("q06-def-lowlr", fp),
        ("q06-base", fp),
    )
    return v6.boot(
        lambda s, g, z, sf, gf, zf: v6.share(s, z, sf, zf) - v6.share(g, z, gf, zf),
        arrays,
    )
