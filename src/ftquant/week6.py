import os

os.environ["HF_HUB_OFFLINE"] = "1"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import yaml  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

from ftquant.sweep import LORA_KEYS  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CONVERT = ROOT / "vendor" / "llama.cpp" / "convert_hf_to_gguf.py"
Q06, Q06_REV = "Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"
OLMO, OLMO_REV = (
    "allenai/OLMo-2-0425-1B-Instruct",
    "48d788eca847d4d7548f375ad03d3c9312f6139e",
)
WIKITEXT_REV = "b08601e04326c79dfdd32d625aee71d232d685c3"
LLAMA_BUILD, MLX_LM_VERSION = "build 11146", "0.31.3"
W1, W3 = ROOT / "runs" / "week1", ROOT / "runs" / "week3"
ENV = {
    **os.environ,
    "PYTHONPATH": str(ROOT / "src"),
    "HF_HUB_OFFLINE": "1",
    "PYTHONUNBUFFERED": "1",
    "PATH": f"/opt/homebrew/bin:{os.environ.get('PATH', '')}",
}
GENTLE, STRONG = 1e-5, 1e-4
MIN_FREE_GB = 10
STALL_IT_PER_S, STALL_WINDOW_S = 0.1, 1800
MAX_VAL_LOSS = 1.0
D_KEYS = ("q06-def-lowlr", "q06-def")

OUT = ROOT / "runs" / "week6"
WORK = OUT / "work"
LIMIT: int | None = None
SPLIT: str | None = None
G3_ITERS = 500
D_ITERS: int | None = None
IM_CHUNKS = 128


def snapshot(repo: str) -> Path:
    return Path(snapshot_download(repo, revision=Q06_REV if repo == Q06 else OLMO_REV))


def adapter_of(key: str) -> Path | None:
    if key.endswith("-base"):
        return None
    if key in ("q06-lora-lowlr", "q06-lora"):
        return W1 / key / "adapter"
    if key in ("olmo1-lora-lowlr", "olmo1-lora"):
        return W3 / key / "adapter"
    return OUT / key / "adapter"


def base_of(key: str) -> str:
    return OLMO if key.startswith("olmo1") else Q06


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(OUT / "week6.log", "a") as f:
        f.write(line + "\n")


def stop(msg: str) -> None:
    log(f"STOP: {msg}")
    raise SystemExit(1)


def run(cmd: list[str]) -> str:
    t0 = time.time()
    p = subprocess.run(cmd, cwd=ROOT, env=ENV, capture_output=True, text=True)
    if p.returncode != 0:
        stop(
            f"command failed ({time.time() - t0:.0f}s): {' '.join(cmd)}\n{p.stdout[-2000:]}\n{p.stderr[-4000:]}"
        )
    return p.stdout


def guard() -> None:
    for path in (ROOT, WORK):
        free = shutil.disk_usage(path).free / 1e9
        if free < MIN_FREE_GB:
            stop(
                f"{free:.1f} GB free on the disk holding {path}, below the {MIN_FREE_GB} GB rule"
            )


def drop(*paths: Path) -> None:
    for p in paths:
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)


def make_file(target: Path, make) -> Path:
    if target.exists():
        return target
    guard()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.stem + ".partial" + target.suffix)
    drop(tmp)
    make(tmp)
    tmp.rename(target)
    return target


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def record(kind: str, name: str, p: Path) -> None:
    rec_path = OUT / f"{kind}-files.json"
    rec = json.loads(rec_path.read_text()) if rec_path.exists() else {}
    h = sha256(p)
    if name in rec and rec[name] != h:
        stop(
            f"rebuilt {name} has sha256 {h[:12]}, but {rec[name][:12]} was recorded earlier"
        )
    rec[name] = h
    rec_path.write_text(json.dumps(rec, indent=1, sort_keys=True))


def check_versions() -> None:
    server = subprocess.run(
        ["llama-server", "--version"], capture_output=True, text=True, env=ENV
    )
    if LLAMA_BUILD not in server.stdout + server.stderr:
        stop(f"llama.cpp is not {LLAMA_BUILD}")
    version = run(
        [sys.executable, "-c", "import mlx_lm; print(mlx_lm.__version__)"]
    ).strip()
    if version != MLX_LM_VERSION:
        stop(f"mlx-lm is {version}, not {MLX_LM_VERSION}")


# ---- model files


def src_dir(key: str) -> Path:
    if key == "q06-base":
        target = WORK / "q06-tied"
        if not (target / "model.safetensors").exists():
            guard()
            tmp = target.with_name("q06-tied.partial")
            drop(tmp)
            run(
                [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; from ftquant.embedding_check import tied_copy; "
                    f"tied_copy(Path('{snapshot(Q06)}'), Path('{tmp}'))",
                ]
            )
            tmp.rename(target)
        return target
    if key == "olmo1-base":
        return snapshot(OLMO)
    target = WORK / key / "fused-bf16"
    if not (target / "config.json").exists():
        guard()
        tmp = target.with_name("fused-bf16.partial")
        drop(tmp)
        run(
            [
                sys.executable,
                "-m",
                "mlx_lm",
                "fuse",
                "--model",
                str(snapshot(base_of(key))),
                "--adapter-path",
                str(adapter_of(key)),
                "--save-path",
                str(tmp),
            ]
        )
        tmp.rename(target)
    return target


def bf16(key: str) -> Path:
    target = WORK / key / "gguf-bf16.gguf"
    if target.exists():
        return target
    make_file(
        target,
        lambda tmp: run(
            [
                sys.executable,
                str(CONVERT),
                str(src_dir(key)),
                "--outtype",
                "bf16",
                "--outfile",
                str(tmp),
            ]
        ),
    )
    if key == "q06-base":
        record("base", target.name, target)
    drop(WORK / key / "fused-bf16")
    return target


def wikitext() -> Path:
    def make(tmp: Path) -> None:
        run(
            [
                sys.executable,
                "-c",
                "from datasets import load_dataset; from pathlib import Path; "
                f"ds = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='train', revision='{WIKITEXT_REV}'); "
                f"Path('{tmp}').write_text(''.join(ds['text']), encoding='utf-8')",
            ]
        )

    return make_file(WORK / "wikitext2-train.txt", make)


def imatrix(key: str) -> Path:
    target = OUT / key / f"imatrix-c{IM_CHUNKS}.gguf"
    if target.exists():
        return target
    make_file(
        target,
        lambda tmp: run(
            [
                "llama-imatrix",
                "-m",
                str(bf16(key)),
                "-f",
                str(wikitext()),
                "-c",
                "512",
                "--chunks",
                str(IM_CHUNKS),
                "-ngl",
                "99",
                "-o",
                str(tmp),
            ]
        ),
    )
    record("imatrix", f"{key}/{target.name}", target)
    return target


def quant(key: str, fmt: str, im: bool = False) -> Path:
    if fmt == "bf16":
        return bf16(key)
    target = WORK / key / f"gguf-{fmt}{'-im' if im else ''}.gguf"
    if target.exists():
        return target
    extra = ["--imatrix", str(imatrix(key))] if im else []
    make_file(
        target,
        lambda tmp: run(["llama-quantize", *extra, str(bf16(key)), str(tmp), fmt]),
    )
    if key == "q06-base":
        record("base", target.name, target)
    return target


def mlx_q(key: str, bits: int) -> Path:
    target = WORK / key / f"mlx-q{bits}"
    if not (target / "config.json").exists():
        guard()
        src = snapshot(Q06) if key == "q06-base" else src_dir(key)
        tmp = target.with_name(target.name + ".partial")
        drop(tmp)
        run(
            [
                sys.executable,
                "-m",
                "mlx_lm",
                "convert",
                "--hf-path",
                str(src),
                "--mlx-path",
                str(tmp),
                "-q",
                "--q-bits",
                str(bits),
                "--q-group-size",
                "64",
            ]
        )
        tmp.rename(target)
    return target


def lora(key: str) -> Path:
    target = OUT / key / "lora-f32.gguf"
    if target.exists():
        return target
    guard()
    tmp = target.with_name("lora-f32.partial.gguf")
    drop(tmp)
    out = run(
        [
            sys.executable,
            "-m",
            "ftquant.lora_gguf",
            "--adapter",
            str(adapter_of(key)),
            "--base",
            str(snapshot(Q06)),
            "--out",
            str(tmp),
        ]
    )
    res = json.loads(out.strip().splitlines()[-1])
    (OUT / key / "lora-check.json").write_text(json.dumps(res, indent=1))
    log(
        f"adapter check (a) {key}: passed={res['passed']} modules={res['modules']} max_rel_diff={res['max_relative_difference']:.1e}"
    )
    if not res["passed"]:
        stop("adapter conversion check (a) failed; fix it and log a deviation")
    drop(tmp.parent / "peft")
    tmp.rename(target)
    return target


def dequant(fmt: str) -> Path:
    target = WORK / f"q06-deq-{fmt}"
    if (target / "model.safetensors").exists():
        return target
    guard()
    tmp = target.with_name(target.name + ".partial")
    drop(tmp)
    out = run(
        [
            sys.executable,
            "-m",
            "ftquant.rebuild",
            "--quant",
            str(quant("q06-base", fmt)),
            "--ref",
            str(src_dir("q06-base")),
            "--out",
            str(tmp),
        ]
    )
    res = json.loads(out.strip().splitlines()[-1])
    (OUT / f"rebuild-{fmt}.json").write_text(json.dumps(res, indent=1))
    log(f"rebuild check {fmt}: passed={res['passed']} tensors={res['tensors']}")
    if not res["passed"]:
        stop(
            "the rebuilt training model does not equal the dequantized file; fix it first"
        )
    tmp.rename(target)
    return target


# ---- evaluation


def evaluate(res: Path, cmd: list[str], split: str) -> None:
    if res.exists():
        return
    guard()
    res.parent.mkdir(parents=True, exist_ok=True)
    tmp = res.with_suffix(".partial")
    t0 = time.time()
    run(
        [
            sys.executable,
            "-m",
            *cmd,
            "--split",
            SPLIT or split,
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


def ev_gguf(key: str, fmt: str, im: bool = False, split: str = "test") -> None:
    name = f"gguf-{fmt}{'-im' if im else ''}{'-valid' if split == 'valid' else ''}"
    res = OUT / key / "eval" / f"{name}.jsonl"
    if not res.exists():
        evaluate(
            res,
            [
                "ftquant.eval_gguf",
                "--gguf",
                str(quant(key, fmt, im)),
                "--tokenizer",
                str(snapshot(base_of(key))),
                "--parallel",
                "4",
                "--cache-ram",
                "0",
            ],
            split,
        )


def ev_lora(key: str, fmt: str, scale: float = 1.0, split: str = "test") -> None:
    tag = "lora" if scale == 1.0 else f"lora{scale:g}"
    owner = key if scale == 1.0 else "q06-base"
    res = (
        OUT
        / owner
        / "eval"
        / f"gguf-{fmt}-{tag}{'-valid' if split == 'valid' else ''}.jsonl"
    )
    if not res.exists():
        evaluate(
            res,
            [
                "ftquant.eval_gguf",
                "--gguf",
                str(quant("q06-base", fmt)),
                "--tokenizer",
                str(snapshot(Q06)),
                "--parallel",
                "4",
                "--cache-ram",
                "0",
                "--lora",
                str(lora(key)),
                "--lora-scale",
                str(scale),
            ],
            split,
        )


def ev_mlx(key: str, bits: int | None) -> None:
    res = OUT / key / "eval" / f"mlx-{'bf16' if bits is None else f'q{bits}'}.jsonl"
    if res.exists():
        return
    if bits is None:
        model = snapshot(Q06) if key == "q06-base" else src_dir(key)
    else:
        model = mlx_q(key, bits)
    evaluate(res, ["ftquant.eval_mlx", "--model", str(model), "--batch", "32"], "test")
    if bits is not None:
        drop(WORK / key / f"mlx-q{bits}")


# ---- training


def last_iter(path: Path) -> int:
    hits = (
        re.findall(r"Iter (\d+):", path.read_text(errors="ignore"))
        if path.exists()
        else []
    )
    return max(map(int, hits), default=0)


def final_val_loss(path: Path) -> float:
    hits = re.findall(r"Iter \d+: Val loss ([^\s,]+)", path.read_text(errors="ignore"))
    try:
        return float(hits[-1]) if hits else math.nan
    except ValueError:
        return math.nan


def train_once(out: Path) -> str:
    with open(out / "train.log", "w") as logf:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "mlx_lm",
                "lora",
                "--config",
                str(out / "train.yaml"),
            ],
            cwd=ROOT,
            env=ENV,
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            samples = [(time.time(), 0)]
            while proc.poll() is None:
                time.sleep(60)
                now, it = time.time(), last_iter(out / "train.log")
                samples.append((now, it))
                old = [s for s in samples if now - s[0] >= STALL_WINDOW_S]
                if old and (it - old[-1][1]) / (now - old[-1][0]) < STALL_IT_PER_S:
                    return f"stalled ({it - old[-1][1]} iterations in {(now - old[-1][0]) / 60:.0f} min)"
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    return "ok" if proc.returncode == 0 else f"failed (exit {proc.returncode})"


def train(key: str, cfg: dict, masked: bool) -> Path | None:
    out = OUT / key
    adapter = out / "adapter"
    if (out / "TRAINED.json").exists():
        return adapter
    if (out / "NOT_TESTABLE").exists():
        return None
    failures = sorted(OUT.glob(f"_failed-{key}-attempt*"))
    if out.exists():
        failed = OUT / f"_failed-{key}-attempt{len(failures) + 1}"
        log(f"TRAIN {key}: an unfinished attempt was found; kept as {failed.name}")
        shutil.move(str(out), str(failed))
        failures.append(failed)
    while len(failures) < 2:
        attempt = len(failures) + 1
        guard()
        out.mkdir(parents=True)
        (out / "train.yaml").write_text(
            yaml.safe_dump({**cfg, "adapter_path": str(adapter)})
        )
        t0 = time.time()
        status = train_once(out)
        vl = final_val_loss(out / "train.log")
        if status == "ok" and masked and not vl < MAX_VAL_LOSS:
            status = f"did not learn (final validation loss {vl})"
        if status == "ok":
            (out / "TRAINED.json").write_text(
                json.dumps(
                    {
                        "attempt": attempt,
                        "seconds": round(time.time() - t0),
                        "final_iter": last_iter(out / "train.log"),
                        "final_val_loss": vl,
                    }
                )
            )
            log(
                f"trained {key} in {time.time() - t0:.0f}s (attempt {attempt}, final validation loss {vl})"
            )
            return adapter
        failed = OUT / f"_failed-{key}-attempt{attempt}"
        log(f"TRAIN {key} attempt {attempt}: {status}; kept as {failed.name}")
        shutil.move(str(out), str(failed))
        failures.append(failed)
    out.mkdir(parents=True, exist_ok=True)
    (out / "NOT_TESTABLE").write_text("two failed attempts; see week6.log\n")
    return None


def g3_cfg(model: Path, lr: float) -> dict:
    return {
        "model": str(model),
        "train": True,
        "data": "data/banking77",
        "fine_tune_type": "lora",
        "mask_prompt": True,
        "num_layers": -1,
        "batch_size": 4,
        "iters": G3_ITERS,
        "learning_rate": lr,
        "steps_per_report": 50,
        "steps_per_eval": 100,
        "val_batches": 25,
        "max_seq_length": 512,
        "grad_checkpoint": False,
        "seed": 0,
        "lora_parameters": {
            "keys": LORA_KEYS,
            "rank": 8,
            "scale": 20.0,
            "dropout": 0.0,
        },
    }


def d_cfg(lr: float) -> dict:
    cfg = {
        "model": str(snapshot(Q06)),
        "train": True,
        "data": "data/banking77",
        "seed": 0,
        "learning_rate": lr,
    }
    if D_ITERS:
        cfg["iters"] = D_ITERS
    return cfg


def check_defaults(key: str) -> None:
    cfg = json.loads((adapter_of(key) / "adapter_config.json").read_text())
    want = {
        "num_layers": 16,
        "iters": D_ITERS or 1000,
        "mask_prompt": False,
        "batch_size": 4,
    }
    got = {k: cfg.get(k) for k in want}
    lp = cfg.get("lora_parameters", {})
    if got != want or lp.get("rank") != 8 or lp.get("scale") != 20.0:
        stop(f"{key} did not train with mlx-lm's defaults: {got}, lora {lp}")


def g3(fmt: str, suffix: str, lr: float) -> None:
    key = f"q06dq{fmt[1]}-{suffix}"
    done = (OUT / key / "TRAINED.json").exists()
    if not done and (OUT / key / "NOT_TESTABLE").exists():
        return
    if not done and train(key, g3_cfg(dequant(fmt), lr), masked=True) is None:
        return
    ev_lora(key, fmt)


# ---- arms


def g_confirmatory() -> None:
    for fmt in ("bf16", "Q3_K_M"):
        ev_gguf("q06-base", fmt)
        ev_gguf("q06-lora-lowlr", fmt)
    lora("q06-lora-lowlr")
    for fmt in ("bf16", "Q8_0"):
        ev_gguf("q06-base", fmt, split="valid")
        ev_gguf("q06-lora-lowlr", fmt, split="valid")
        ev_lora("q06-lora-lowlr", fmt, split="valid")
    ev_lora("q06-lora-lowlr", "bf16", scale=0.0, split="valid")
    ev_lora("q06-lora-lowlr", "Q3_K_M")
    g3("Q3_K_M", "lora-lowlr", GENTLE)
    drop(
        WORK / "q06-deq-Q3_K_M",
        WORK / "q06-lora-lowlr" / "gguf-Q8_0.gguf",
        WORK / "q06-base" / "gguf-Q8_0.gguf",
    )


def d_confirmatory() -> None:
    for key, lr in zip(D_KEYS, (GENTLE, STRONG), strict=True):
        if train(key, d_cfg(lr), masked=False) is None:
            continue
        check_defaults(key)
        ev_mlx(key, None)
        ev_mlx(key, 4)
        for fmt in ("bf16", "Q3_K_M"):
            ev_gguf(key, fmt)
        drop(WORK / key)
    ev_mlx("q06-base", None)
    ev_mlx("q06-base", 4)


def i_confirmatory() -> None:
    for key in (
        "q06-base",
        "q06-lora-lowlr",
        "q06-lora",
        "olmo1-base",
        "olmo1-lora-lowlr",
        "olmo1-lora",
    ):
        ev_gguf(key, "bf16")
        ev_gguf(key, "Q3_K_M")
        ev_gguf(key, "Q3_K_M", im=True)
        if key != "q06-base":
            drop(WORK / key)


def exploratory() -> None:
    ev_lora("q06-lora", "Q3_K_M")
    g3("Q3_K_M", "lora", STRONG)
    drop(WORK / "q06-deq-Q3_K_M")
    for key in ("q06-base", "q06-lora-lowlr", "q06-lora"):
        ev_gguf(key, "Q2_K")
        ev_gguf(key, "Q2_K", im=True)
        if key != "q06-base":
            drop(WORK / key)
    for key in ("q06-lora-lowlr", "q06-lora"):
        ev_lora(key, "Q2_K")
    g3("Q2_K", "lora-lowlr", GENTLE)
    g3("Q2_K", "lora", STRONG)
    drop(WORK / "q06-deq-Q2_K")
    for key in D_KEYS:
        if not (OUT / key / "TRAINED.json").exists():
            continue
        ev_mlx(key, 3)
        for fmt in ("Q4_K_M", "Q2_K"):
            ev_gguf(key, fmt)
        drop(WORK / key)
    ev_mlx("q06-base", 3)
    ev_gguf("q06-base", "Q4_K_M")
    for key in ("olmo1-base", "olmo1-lora-lowlr", "olmo1-lora"):
        ev_gguf(key, "Q2_K")
        ev_gguf(key, "Q2_K", im=True)
        drop(WORK / key)


def main() -> None:
    global OUT, WORK, LIMIT, SPLIT, G3_ITERS, D_ITERS, IM_CHUNKS
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--phases", nargs="*", default=["G", "D", "I", "X"])
    args = ap.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
    if args.smoke:
        OUT, LIMIT, SPLIT, G3_ITERS, D_ITERS, IM_CHUNKS = (
            ROOT / "runs" / "smoke6",
            20,
            "valid",
            10,
            10,
            4,
        )
    WORK = (
        Path(os.environ["FTQ_WORK"]) / OUT.name
        if os.environ.get("FTQ_WORK")
        else OUT / "work"
    )
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    check_versions()
    needed = [
        adapter_of(k) / "adapters.safetensors"
        for k in ("q06-lora-lowlr", "q06-lora", "olmo1-lora-lowlr", "olmo1-lora")
    ]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        stop(f"missing adapters: {missing}")
    log(
        f"week-6 start: phases {args.phases}{' (smoke)' if args.smoke else ''}; work in {WORK}"
    )
    phases = {
        "G": g_confirmatory,
        "D": d_confirmatory,
        "I": i_confirmatory,
        "X": exploratory,
    }
    for ph in args.phases:
        phases[ph]()
        log(f"phase {ph} done")
    if args.phases == ["G", "D", "I", "X"]:
        (OUT / "verdicts.txt").write_text(
            run([sys.executable, "-m", "ftquant.verdicts_w6", "--run", str(OUT)])
        )
        drop(WORK)
    log("week-6 complete")


if __name__ == "__main__":
    main()
