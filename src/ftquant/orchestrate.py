import hashlib
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
LOG = RUNS / "orchestrate.log"


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def step(name: str, done: Path, cmd: list[str], stdout: Path | None = None) -> None:
    if done.exists():
        log(f"skip {name} (exists)")
        return
    log(f"start {name}")
    t0 = time.time()
    p = subprocess.run(
        [sys.executable, "-m", *cmd],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": str(ROOT / "src"),
            "PATH": "/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin",
            "HOME": str(Path.home()),
        },
    )
    if stdout is not None and p.returncode == 0:
        stdout.parent.mkdir(parents=True, exist_ok=True)
        stdout.write_text(p.stdout)
    if p.returncode != 0:
        log(
            f"FAILED {name} after {time.time() - t0:.0f}s\n{p.stdout[-3000:]}\n{p.stderr[-5000:]}"
        )
        raise SystemExit(1)
    log(f"done {name} ({time.time() - t0:.0f}s)")


def wait_for_week1() -> None:
    sweep_log = RUNS / "week1" / "sweep.log"
    while True:
        text = sweep_log.read_text() if sweep_log.exists() else ""
        if "sweep complete" in text:
            return
        alive = (
            subprocess.run(
                ["pgrep", "-f", "ftquant.sweep"], capture_output=True
            ).returncode
            == 0
        )
        if not alive and "sweep complete" not in text:
            log("week-1 sweep is not running and did not complete; stopping")
            raise SystemExit(1)
        time.sleep(60)


def main() -> None:
    log("orchestrator waiting for week-1 sweep")
    wait_for_week1()
    log("week-1 sweep complete")
    for model in ("Qwen3-0.6B", "Qwen3-1.7B"):
        out = RUNS / "kld" / f"{model}.json"
        step(
            f"kld {model}",
            out,
            ["ftquant.kld", "--model", f"Qwen/{model}", "--out", str(out)],
        )
    w1 = RUNS / "week1"
    step(
        "mechanism mlx q17",
        w1 / "mechanism-q17-mlx.json",
        [
            "ftquant.mechanism",
            "--configs",
            "q17-lora",
            "--model",
            "Qwen/Qwen3-1.7B",
            "--out",
            str(w1 / "mechanism-q17-mlx.json"),
        ],
    )
    step(
        "mechanism gguf q17",
        w1 / "mechanism-q17-gguf.json",
        [
            "ftquant.mechanism_gguf",
            "--model",
            "Qwen/Qwen3-1.7B",
            "--base",
            "q17-base",
            "--configs",
            "q17-lora",
            "--out",
            str(w1 / "mechanism-q17-gguf.json"),
        ],
    )
    step(
        "analysis week1",
        w1 / "analysis-final.json",
        [
            "ftquant.analyze",
            "--runs",
            "week1",
            "--out",
            str(w1 / "analysis-final.json"),
        ],
    )
    pred = RUNS / "predictor" / "predictor-v1.json"
    step(
        "predictor fit (frozen)",
        pred,
        [
            "ftquant.predictor",
            "fit",
            "--analysis",
            str(w1 / "analysis-final.json"),
            "--run",
            "week1",
            "--prefix",
            "q06",
            "--model-tag",
            "Qwen3-0.6B",
            "--out",
            str(pred),
        ],
    )
    digest = hashlib.sha256(pred.read_bytes()).hexdigest()
    with open(RUNS / "predictor" / "predictor-v1.sha256", "a") as f:
        f.write(
            f"{digest}  predictor-v1.json  {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
        )
    log(f"predictor frozen sha256 {digest}")
    step(
        "week2 sweep",
        RUNS / "week2" / "SWEEP_DONE",
        ["ftquant.sweep", "--plan", "week2"],
    )
    (RUNS / "week2" / "SWEEP_DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    w2 = RUNS / "week2"
    q06_w2 = [
        "q06-lora-lr3e-6",
        "q06-lora-lr3e-5",
        "q06-lora-lr3e-4",
        "q06-full-lr3e-6",
        "q06-full-lr3e-5",
        "q06-lora-lowlr-s1",
        "q06-lora-s1",
    ]
    step(
        "mechanism mlx week2 q06",
        w2 / "mechanism-q06-mlx.json",
        [
            "ftquant.mechanism",
            "--run",
            "week2",
            "--configs",
            *q06_w2,
            "--model",
            "Qwen/Qwen3-0.6B",
            "--out",
            str(w2 / "mechanism-q06-mlx.json"),
        ],
    )
    step(
        "mechanism gguf week2 q06",
        w2 / "mechanism-q06-gguf.json",
        [
            "ftquant.mechanism_gguf",
            "--run",
            "week2",
            "--model",
            "Qwen/Qwen3-0.6B",
            "--base",
            "q06-base",
            "--configs",
            *q06_w2,
            "--out",
            str(w2 / "mechanism-q06-gguf.json"),
        ],
    )
    step(
        "mechanism mlx week2 q17",
        w2 / "mechanism-q17-mlx.json",
        [
            "ftquant.mechanism",
            "--run",
            "week2",
            "--configs",
            "q17-lora-lowlr",
            "--model",
            "Qwen/Qwen3-1.7B",
            "--out",
            str(w2 / "mechanism-q17-mlx.json"),
        ],
    )
    step(
        "mechanism gguf week2 q17",
        w2 / "mechanism-q17-gguf.json",
        [
            "ftquant.mechanism_gguf",
            "--run",
            "week2",
            "--model",
            "Qwen/Qwen3-1.7B",
            "--base",
            "q17-base",
            "--configs",
            "q17-lora-lowlr",
            "--out",
            str(w2 / "mechanism-q17-gguf.json"),
        ],
    )
    allan = RUNS / "analysis-all.json"
    step(
        "analysis all",
        allan,
        ["ftquant.analyze", "--runs", "week1", "week2", "--out", str(allan)],
    )
    for name, run, prefix, tag in (
        ("T1", "week1", "q17", "Qwen3-1.7B"),
        ("T2", "week2", "q06", "Qwen3-0.6B"),
        ("T3", "week2", "q17", "Qwen3-1.7B"),
    ):
        out = RUNS / "predictor" / f"eval-{name}.json"
        step(
            f"predictor eval {name}",
            out,
            [
                "ftquant.predictor",
                "eval",
                "--predictor",
                str(pred),
                "--analysis",
                str(allan),
                "--run",
                run,
                "--prefix",
                prefix,
                "--model-tag",
                tag,
            ],
            stdout=out,
        )
    log("orchestrator complete")


if __name__ == "__main__":
    main()
