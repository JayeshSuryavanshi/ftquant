import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import yaml

from ftquant.sweep import LORA_KEYS

ROOT = Path(__file__).resolve().parents[2]
CONVERT = ROOT / "vendor" / "llama.cpp" / "convert_hf_to_gguf.py"
Q06 = "Qwen/Qwen3-0.6B"
OLMO = "allenai/OLMo-2-0425-1B-Instruct"
OLMO_ADAPTERS = [
    ("week3", "olmo1-lora-lowlr"),
    ("week4", "olmo1-lora-lowlr-s1"),
    ("week3", "olmo1-lora"),
    ("week4", "olmo1-lora-s1"),
]
QLORA = [
    (4, "lora-lowlr", 1e-5),
    (4, "lora", 1e-4),
    (3, "lora-lowlr", 1e-5),
    (3, "lora", 1e-4),
]
W1_LORAS = ["q06-lora-lowlr", "q06-lora"]
ENV = {
    **os.environ,
    "PYTHONPATH": str(ROOT / "src"),
    "HF_HUB_OFFLINE": "1",
    "PATH": f"/opt/homebrew/bin:{os.environ.get('PATH', '')}",
}

OUT = ROOT / "runs" / "week5"
LIMIT: int | None = None
ITERS = 500


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(OUT / "week5.log", "a") as f:
        f.write(line + "\n")


def run(cmd: list[str]) -> str:
    t0 = time.time()
    p = subprocess.run(cmd, cwd=ROOT, env=ENV, capture_output=True, text=True)
    if p.returncode != 0:
        log(
            f"FAILED ({time.time() - t0:.0f}s): {' '.join(cmd)}\n{p.stdout[-2000:]}\n{p.stderr[-4000:]}"
        )
        raise SystemExit(1)
    return p.stdout


def atomic_dir(target: Path, make: Callable[[Path], object]) -> Path:
    if target.exists():
        return target
    tmp = target.with_name(target.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    make(tmp)
    tmp.rename(target)
    return target


def atomic_file(target: Path, make: Callable[[Path], object]) -> Path:
    if target.exists():
        return target
    tmp = target.with_suffix(".partial")
    tmp.unlink(missing_ok=True)
    make(tmp)
    tmp.rename(target)
    return target


def mlx_quant(model: str, bits: int) -> Path:
    target = OUT / "base-quants" / f"{model.split('/')[-1]}-mlx-q{bits}"
    target.parent.mkdir(parents=True, exist_ok=True)
    return atomic_dir(
        target,
        lambda tmp: run(
            [
                sys.executable,
                "-m",
                "mlx_lm",
                "convert",
                "--hf-path",
                model,
                "--mlx-path",
                str(tmp),
                "-q",
                "--q-bits",
                str(bits),
                "--q-group-size",
                "64",
            ]
        ),
    )


def evaluate(res: Path, cmd: list[str]) -> None:
    if res.exists():
        return
    res.parent.mkdir(parents=True, exist_ok=True)
    tmp = res.with_suffix(".partial")
    t0 = time.time()
    run(
        [
            sys.executable,
            "-m",
            *cmd,
            "--out",
            str(tmp),
            *(["--limit", str(LIMIT)] if LIMIT else []),
        ]
    )
    tmp.rename(res)
    recs = [json.loads(line) for line in open(res)]
    acc = sum(r["correct"] for r in recs) / len(recs)
    valid = sum(r["valid"] for r in recs) / len(recs)
    log(
        f"eval {res.parent.parent.name}/{res.stem}: acc {acc:.4f} valid {valid:.4f} n={len(recs)} ({time.time() - t0:.0f}s)"
    )


def train_qlora(name: str, base: Path, lr: float) -> Path:
    out = OUT / name
    adapter = out / "adapter"
    if (adapter / "adapters.safetensors").exists():
        return adapter
    cfg = {
        "model": str(base),
        "train": True,
        "data": "data/banking77",
        "fine_tune_type": "lora",
        "mask_prompt": True,
        "num_layers": -1,
        "batch_size": 4,
        "iters": ITERS,
        "learning_rate": lr,
        "steps_per_report": 50,
        "steps_per_eval": 100,
        "val_batches": 25,
        "max_seq_length": 512,
        "grad_checkpoint": False,
        "adapter_path": str(adapter),
        "seed": 0,
        "lora_parameters": {
            "keys": LORA_KEYS,
            "rank": 8,
            "scale": 20.0,
            "dropout": 0.0,
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.yaml").write_text(yaml.safe_dump(cfg))
    t0 = time.time()
    p = subprocess.run(
        [sys.executable, "-m", "mlx_lm", "lora", "--config", str(out / "train.yaml")],
        cwd=ROOT,
        env=ENV,
        capture_output=True,
        text=True,
    )
    (out / "train.log").write_text(p.stdout + p.stderr)
    if p.returncode != 0:
        log(f"TRAIN FAILED {name}\n{p.stderr[-4000:]}")
        raise SystemExit(1)
    log(f"trained {name} in {time.time() - t0:.0f}s")
    return adapter


def arm_u() -> None:
    for bits in (3, 4):
        qb = mlx_quant(OLMO, bits)
        evaluate(
            OUT / "olmo1-base" / "eval" / f"mlx-q{bits}.jsonl",
            ["ftquant.eval_task_mlx", "--task", "banking77", "--model", str(qb)],
        )
        for week, cfg in OLMO_ADAPTERS:
            adapter = ROOT / "runs" / week / cfg / "adapter"
            evaluate(
                OUT / cfg / "eval" / f"mlx-q{bits}-unfused.jsonl",
                [
                    "ftquant.eval_task_mlx",
                    "--task",
                    "banking77",
                    "--model",
                    str(qb),
                    "--adapter",
                    str(adapter),
                ],
            )


def arm_q() -> None:
    for bits, suffix, lr in QLORA:
        qb = mlx_quant(Q06, bits)
        evaluate(
            OUT / "q06-base" / "eval" / f"mlx-q{bits}.jsonl",
            ["ftquant.eval_mlx", "--model", str(qb)],
        )
        name = f"q06q{bits}-{suffix}"
        adapter = train_qlora(name, qb, lr)
        evaluate(
            OUT / name / "eval" / f"mlx-q{bits}-qlora.jsonl",
            ["ftquant.eval_mlx", "--model", str(qb), "--adapter", str(adapter)],
        )


def gguf_files(cfg: str, types: list[str]) -> dict[str, Path]:
    work = OUT / "work" / cfg
    work.mkdir(parents=True, exist_ok=True)
    adapter = ROOT / "runs" / "week1" / cfg / "adapter"
    fused = atomic_dir(
        work / "fused-bf16",
        lambda tmp: run(
            [
                sys.executable,
                "-m",
                "mlx_lm",
                "fuse",
                "--model",
                Q06,
                "--adapter-path",
                str(adapter),
                "--save-path",
                str(tmp),
            ]
        ),
    )
    files = {
        "bf16": atomic_file(
            work / "gguf-bf16.gguf",
            lambda tmp: run(
                [
                    sys.executable,
                    str(CONVERT),
                    str(fused),
                    "--outtype",
                    "bf16",
                    "--outfile",
                    str(tmp),
                ]
            ),
        )
    }
    for t in types:
        files[t] = atomic_file(
            work / f"gguf-{t}.gguf",
            lambda tmp, t=t: run(["llama-quantize", str(files["bf16"]), str(tmp), t]),
        )
    return files


def arm_c() -> None:
    for cfg in W1_LORAS:
        files = gguf_files(cfg, ["Q3_K_M", "Q2_K"])
        ev = OUT / cfg / "eval"
        if cfg == "q06-lora-lowlr":
            for n in (4, 1):
                evaluate(
                    ev / f"gguf-Q3_K_M-np{n}.jsonl",
                    [
                        "ftquant.eval_gguf",
                        "--gguf",
                        str(files["Q3_K_M"]),
                        "--tokenizer",
                        Q06,
                        "--parallel",
                        str(n),
                    ],
                )
        for t in ("bf16", "Q3_K_M", "Q2_K"):
            evaluate(
                ev / f"gguf-{t}-labels.jsonl",
                [
                    "ftquant.eval_gguf",
                    "--gguf",
                    str(files[t]),
                    "--tokenizer",
                    Q06,
                    "--labels-only",
                ],
            )
        shutil.rmtree(OUT / "work" / cfg, ignore_errors=True)


def main() -> None:
    global OUT, LIMIT, ITERS
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--arms", nargs="*", default=["U", "Q", "C"])
    args = ap.parse_args()
    if args.smoke:
        OUT, LIMIT, ITERS = ROOT / "runs" / "smoke5", 20, 10
    OUT.mkdir(parents=True, exist_ok=True)
    needed = [
        ROOT / "runs" / w / c / "adapter" / "adapters.safetensors"
        for w, c in OLMO_ADAPTERS
    ]
    needed += [
        ROOT / "runs" / "week1" / c / "adapter" / "adapters.safetensors"
        for c in W1_LORAS
    ]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        raise SystemExit(f"missing adapters: {missing}")
    log(f"week-5 start: arms {args.arms}{' (smoke)' if args.smoke else ''}")
    arms = {"U": arm_u, "Q": arm_q, "C": arm_c}
    for arm in args.arms:
        arms[arm]()
        log(f"arm {arm} done")
    if not args.smoke:
        (OUT / "verdicts.txt").write_text(
            run([sys.executable, "-m", "ftquant.verdicts_w5"])
        )
    shutil.rmtree(OUT / "base-quants", ignore_errors=True)
    shutil.rmtree(OUT / "work", ignore_errors=True)
    log("week-5 complete")


if __name__ == "__main__":
    main()
