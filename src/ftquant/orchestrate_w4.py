from ftquant.orchestrate import RUNS, log, step

OLMO = "allenai/OLMo-2-0425-1B-Instruct"
Q06 = "Qwen/Qwen3-0.6B"
Q4B = "Qwen/Qwen3-4B"
W4 = RUNS / "week4"
BITS = ["6", "4", "3"]
TYPES = ["Q6_K", "Q4_K_M", "Q3_K_M", "Q2_K"]


def mech(prefix: str, model: str, base: str, configs: list[str]) -> None:
    mlx, gguf = (
        W4 / f"mechanism-{prefix}-mlx.json",
        W4 / f"mechanism-{prefix}-gguf.json",
    )
    step(
        f"mechanism {prefix}",
        gguf,
        [
            "ftquant.mechanism_stream",
            "--model",
            model,
            "--configs",
            *configs,
            "--run",
            "week4",
            "--bits",
            *BITS,
            "--base",
            base,
            "--types",
            *TYPES,
            "--mlx-out",
            str(mlx),
            "--gguf-out",
            str(gguf),
        ],
    )


def main() -> None:
    log("week-4 orchestrator start")
    step(
        "kld qwen3-4b",
        RUNS / "kld" / "Qwen3-4B.json",
        [
            "ftquant.kld",
            "--model",
            Q4B,
            "--out",
            str(RUNS / "kld" / "Qwen3-4B.json"),
            "--bits",
            *BITS,
            "--types",
            *TYPES,
        ],
    )
    step(
        "sweep week4",
        W4 / "SWEEP_DONE",
        ["ftquant.sweep", "--plan", "week4"],
        stdout=W4 / "SWEEP_DONE",
    )
    mech("olmo1", OLMO, "olmo1-base", ["olmo1-lora-lowlr-s1", "olmo1-lora-s1"])
    mech("mas06", Q06, "q06-base", ["mas06-lora-lowlr-s1", "mas06-lora-s1"])
    mech("q4b", Q4B, "q4b-base", ["q4b-lora-lowlr", "q4b-lora"])
    step(
        "analysis week4",
        W4 / "analysis.json",
        ["ftquant.analyze", "--runs", "week4", "--out", str(W4 / "analysis.json")],
    )
    step(
        "verdicts week4",
        W4 / "verdicts.json",
        ["ftquant.verdicts_w4"],
        stdout=W4 / "verdicts.txt",
    )
    log("week-4 orchestrator complete")


if __name__ == "__main__":
    main()
