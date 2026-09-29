import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs" / "week1"
FULL_MLX = [8, 6, 4, 3, 2]
FULL_GGUF = ["Q8_0", "Q6_K", "Q4_K_M", "Q3_K_M", "Q2_K"]
CONVERT = ROOT / "vendor" / "llama.cpp" / "convert_hf_to_gguf.py"
LORA_KEYS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
             "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
MLX_BITS = [8, 6, 4, 3, 2]
GGUF_TYPES = ["Q8_0", "Q6_K", "Q4_K_M", "Q3_K_M", "Q2_K"]
UNFUSED_BITS = [8, 4, 3]


@dataclass(frozen=True)
class Config:
    name: str
    base: str
    kind: str
    lr: float = 0.0
    rank: int = 8
    iters: int = 500
    seed: int = 0
    mlx_bits: tuple[int, ...] = (8, 6, 4, 3, 2)
    gguf_types: tuple[str, ...] = ("Q8_0", "Q6_K", "Q4_K_M", "Q3_K_M", "Q2_K")
    unfused_bits: tuple[int, ...] = (8, 4, 3)
    task: str = "banking77"
    generic_eval: bool = False
    grad_ckpt: bool = False
    limit: int | None = None


@dataclass
class Plan:
    configs: list[Config] = field(default_factory=lambda: [
        Config("q06-base", "Qwen/Qwen3-0.6B", "base"),
        Config("q06-lora", "Qwen/Qwen3-0.6B", "lora", lr=1e-4),
        Config("q06-lora-lowlr", "Qwen/Qwen3-0.6B", "lora", lr=1e-5),
        Config("q06-full", "Qwen/Qwen3-0.6B", "full", lr=1e-5),
        Config("q17-base", "Qwen/Qwen3-1.7B", "base"),
        Config("q17-lora", "Qwen/Qwen3-1.7B", "lora", lr=1e-4),
    ])


REDUCED = {"mlx_bits": (6, 4, 3), "gguf_types": ("Q6_K", "Q4_K_M", "Q3_K_M", "Q2_K"), "unfused_bits": ()}
WEEK2 = [
    Config("q06-lora-lr3e-6", "Qwen/Qwen3-0.6B", "lora", lr=3e-6, **REDUCED),
    Config("q06-lora-lr3e-5", "Qwen/Qwen3-0.6B", "lora", lr=3e-5, **REDUCED),
    Config("q06-lora-lr3e-4", "Qwen/Qwen3-0.6B", "lora", lr=3e-4, **REDUCED),
    Config("q06-full-lr3e-6", "Qwen/Qwen3-0.6B", "full", lr=3e-6, **REDUCED),
    Config("q06-full-lr3e-5", "Qwen/Qwen3-0.6B", "full", lr=3e-5, **REDUCED),
    Config("q06-lora-lowlr-s1", "Qwen/Qwen3-0.6B", "lora", lr=1e-5, seed=1, **REDUCED),
    Config("q06-lora-s1", "Qwen/Qwen3-0.6B", "lora", lr=1e-4, seed=1, **REDUCED),
    Config("q17-lora-lowlr", "Qwen/Qwen3-1.7B", "lora", lr=1e-5, **REDUCED),
]
W3 = {**REDUCED, "generic_eval": True}
OLMO = "allenai/OLMo-2-0425-1B-Instruct"
WEEK3 = [
    Config("olmo1-base", OLMO, "base", **W3),
    Config("olmo1-lora-lowlr", OLMO, "lora", lr=1e-5, grad_ckpt=True, **W3),
    Config("olmo1-lora", OLMO, "lora", lr=1e-4, grad_ckpt=True, **W3),
    Config("mas06-base", "Qwen/Qwen3-0.6B", "base", task="massive", **W3),
    Config("mas06-lora-lowlr", "Qwen/Qwen3-0.6B", "lora", lr=1e-5, task="massive", **W3),
    Config("mas06-lora", "Qwen/Qwen3-0.6B", "lora", lr=1e-4, task="massive", **W3),
    Config("mas06-full", "Qwen/Qwen3-0.6B", "full", lr=1e-5, task="massive", **W3),
]
SMOKE = [
    Config("olmo1-lora", OLMO, "lora", lr=1e-4, iters=30, grad_ckpt=True, limit=40, **W3),
    Config("mas06-lora", "Qwen/Qwen3-0.6B", "lora", lr=1e-4, iters=30, task="massive", limit=40, **W3),
]
Q4B = "Qwen/Qwen3-4B"
WEEK4 = [
    Config("olmo1-lora-lowlr-s1", OLMO, "lora", lr=1e-5, seed=1, grad_ckpt=True, **W3),
    Config("olmo1-lora-s1", OLMO, "lora", lr=1e-4, seed=1, grad_ckpt=True, **W3),
    Config("mas06-lora-lowlr-s1", "Qwen/Qwen3-0.6B", "lora", lr=1e-5, seed=1, task="massive", **W3),
    Config("mas06-lora-s1", "Qwen/Qwen3-0.6B", "lora", lr=1e-4, seed=1, task="massive", **W3),
    Config("q4b-base", Q4B, "base", **W3),
    Config("q4b-lora-lowlr", Q4B, "lora", lr=1e-5, grad_ckpt=True, **W3),
    Config("q4b-lora", Q4B, "lora", lr=1e-4, grad_ckpt=True, **W3),
]
SMOKE4B = [
    Config("q4b-lora", Q4B, "lora", lr=1e-4, iters=10, grad_ckpt=True, limit=20, mlx_bits=(3,),
           gguf_types=("Q3_K_M",), unfused_bits=(), generic_eval=True),
]
PLANS = {"week1": lambda: Plan().configs, "week2": lambda: WEEK2, "week3": lambda: WEEK3, "week4": lambda: WEEK4,
         "smoke": lambda: SMOKE, "smoke4b": lambda: SMOKE4B}


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(RUNS / "sweep.log", "a") as f:
        f.write(line + "\n")


def run(cmd: list[str], env_src: bool = False) -> None:
    import os
    env = dict(os.environ)
    if env_src:
        env["PYTHONPATH"] = str(ROOT / "src")
    t0 = time.time()
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        log(f"FAILED ({time.time() - t0:.0f}s): {' '.join(cmd)}\n{p.stdout[-2000:]}\n{p.stderr[-4000:]}")
        raise SystemExit(1)
    return None


def train(c: Config, out: Path) -> Path:
    adapter = out / "adapter"
    if (adapter / "adapters.safetensors").exists():
        return adapter
    cfg = {
        "model": c.base, "train": True, "data": f"data/{c.task}", "fine_tune_type": c.kind,
        "mask_prompt": True, "num_layers": -1, "batch_size": 4, "iters": c.iters, "learning_rate": c.lr,
        "steps_per_report": 50, "steps_per_eval": 100, "val_batches": 25, "max_seq_length": 512,
        "grad_checkpoint": c.kind == "full" or "1.7B" in c.base or c.grad_ckpt, "adapter_path": str(adapter), "seed": c.seed,
        "lora_parameters": {"keys": LORA_KEYS, "rank": c.rank, "scale": 20.0, "dropout": 0.0},
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.yaml").write_text(yaml.safe_dump(cfg))
    t0 = time.time()
    p = subprocess.run([sys.executable, "-m", "mlx_lm", "lora", "--config", str(out / "train.yaml")],
                       cwd=ROOT, capture_output=True, text=True)
    (out / "train.log").write_text(p.stdout + p.stderr)
    if p.returncode != 0:
        log(f"TRAIN FAILED {c.name}\n{p.stderr[-4000:]}")
        raise SystemExit(1)
    log(f"trained {c.name} in {time.time() - t0:.0f}s")
    if c.generic_eval:
        for ckpt in adapter.glob("0*_adapters.safetensors"):
            ckpt.unlink()
    return adapter


def fused_dir(c: Config, out: Path, adapter: Path | None) -> Path:
    if c.kind == "base":
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(c.base))
    d = out / "fused-bf16"
    if not (d / "config.json").exists():
        from huggingface_hub import snapshot_download
        snapshot_download(c.base)
        run([sys.executable, "-m", "mlx_lm", "fuse", "--model", c.base, "--adapter-path", str(adapter), "--save-path", str(d)])
    return d


def evaluate(kind: str, name: str, out: Path, c: Config | None = None, **kw: str) -> None:
    res = out / "eval" / f"{name}.jsonl"
    if res.exists():
        return
    t0 = time.time()
    if c is not None and c.generic_eval:
        mod = "ftquant.eval_task_mlx" if kind == "mlx" else "ftquant.eval_task_gguf"
        cmd = [sys.executable, "-m", mod, "--task", c.task, "--out", str(res)]
        cmd += ["--model", kw["model"]] if kind == "mlx" else ["--gguf", kw["gguf"], "--tokenizer", kw["tokenizer"]]
        if c.limit:
            cmd += ["--limit", str(c.limit)]
    elif kind == "mlx":
        cmd = [sys.executable, "-m", "ftquant.eval_mlx", "--model", kw["model"], "--out", str(res)]
        if kw.get("adapter"):
            cmd += ["--adapter", kw["adapter"]]
    else:
        cmd = [sys.executable, "-m", "ftquant.eval_gguf", "--gguf", kw["gguf"], "--tokenizer", kw["tokenizer"], "--out", str(res)]
    run(cmd, env_src=True)
    recs = [json.loads(line) for line in open(res)]
    acc = sum(r["correct"] for r in recs) / len(recs)
    log(f"eval {out.name}/{name}: acc {acc:.4f} n={len(recs)} ({time.time() - t0:.0f}s)")


def do_config(c: Config, keep: bool) -> None:
    out = RUNS / c.name
    adapter = train(c, out) if c.kind != "base" else None
    src = fused_dir(c, out, adapter)
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    evaluate("mlx", "mlx-bf16", out, c, model=str(src))
    for b in c.mlx_bits:
        q = work / f"mlx-q{b}"
        if not (out / "eval" / f"mlx-q{b}.jsonl").exists():
            if not q.exists():
                run([sys.executable, "-m", "mlx_lm", "convert", "--hf-path", str(src), "--mlx-path", str(q), "-q", "--q-bits", str(b), "--q-group-size", "64"])
            evaluate("mlx", f"mlx-q{b}", out, c, model=str(q))
            if not keep:
                shutil.rmtree(q, ignore_errors=True)
    if c.kind == "lora":
        for b in c.unfused_bits:
            name = f"mlx-q{b}-unfused"
            if (out / "eval" / f"{name}.jsonl").exists():
                continue
            qb = RUNS / "base-quants" / f"{c.base.split('/')[-1]}-mlx-q{b}"
            if not qb.exists():
                run([sys.executable, "-m", "mlx_lm", "convert", "--hf-path", c.base, "--mlx-path", str(qb), "-q", "--q-bits", str(b), "--q-group-size", "64"])
            evaluate("mlx", name, out, model=str(qb), adapter=str(adapter))
    bf16 = work / "gguf-bf16.gguf"
    if not all((out / "eval" / f"gguf-{t}.jsonl").exists() for t in ["bf16", *c.gguf_types]):
        if not bf16.exists():
            run([sys.executable, str(CONVERT), str(src), "--outtype", "bf16", "--outfile", str(bf16)])
        evaluate("gguf", "gguf-bf16", out, c, gguf=str(bf16), tokenizer=c.base)
        for t in c.gguf_types:
            g = work / f"gguf-{t}.gguf"
            if not (out / "eval" / f"gguf-{t}.jsonl").exists():
                if not g.exists():
                    run(["llama-quantize", str(bf16), str(g), t])
                evaluate("gguf", f"gguf-{t}", out, c, gguf=str(g), tokenizer=c.base)
                if not keep:
                    g.unlink(missing_ok=True)
    if not keep:
        shutil.rmtree(work, ignore_errors=True)
        if c.kind != "base":
            shutil.rmtree(out / "fused-bf16", ignore_errors=True)
    log(f"done {c.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--plan", default="week1", choices=sorted(PLANS))
    args = ap.parse_args()
    global RUNS
    RUNS = ROOT / "runs" / args.plan
    RUNS.mkdir(parents=True, exist_ok=True)
    plan = [c for c in PLANS[args.plan]() if not args.only or c.name in args.only]
    if args.list:
        for c in plan:
            print(c)
        return
    log(f"sweep start: {[c.name for c in plan]}")
    for c in plan:
        do_config(c, args.keep)
    log("sweep complete")


if __name__ == "__main__":
    main()
