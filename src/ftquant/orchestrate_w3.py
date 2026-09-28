from ftquant.orchestrate import RUNS, log, step

OLMO = "allenai/OLMo-2-0425-1B-Instruct"
W3 = RUNS / "week3"
LADDER_BITS = ["6", "4", "3"]
LADDER_TYPES = ["Q6_K", "Q4_K_M", "Q3_K_M", "Q2_K"]


def main() -> None:
    log("week-3 orchestrator start")
    W3.mkdir(parents=True, exist_ok=True)
    step(
        "kld olmo",
        RUNS / "kld" / "OLMo-2-0425-1B-Instruct.json",
        [
            "ftquant.kld",
            "--model",
            OLMO,
            "--out",
            str(RUNS / "kld" / "OLMo-2-0425-1B-Instruct.json"),
            "--bits",
            *LADDER_BITS,
            "--types",
            *LADDER_TYPES,
        ],
    )
    step(
        "sweep week3",
        W3 / "SWEEP_DONE",
        ["ftquant.sweep", "--plan", "week3"],
        stdout=W3 / "SWEEP_DONE",
    )
    step(
        "mechanism mlx olmo1",
        W3 / "mechanism-olmo1-mlx.json",
        [
            "ftquant.mechanism",
            "--configs",
            "olmo1-lora-lowlr",
            "olmo1-lora",
            "--model",
            OLMO,
            "--bits",
            *LADDER_BITS,
            "--run",
            "week3",
            "--out",
            str(W3 / "mechanism-olmo1-mlx.json"),
        ],
    )
    step(
        "mechanism mlx mas06",
        W3 / "mechanism-mas06-mlx.json",
        [
            "ftquant.mechanism",
            "--configs",
            "mas06-lora-lowlr",
            "mas06-lora",
            "mas06-full",
            "--model",
            "Qwen/Qwen3-0.6B",
            "--bits",
            *LADDER_BITS,
            "--run",
            "week3",
            "--out",
            str(W3 / "mechanism-mas06-mlx.json"),
        ],
    )
    step(
        "mechanism gguf olmo1",
        W3 / "mechanism-olmo1-gguf.json",
        [
            "ftquant.mechanism_gguf",
            "--model",
            OLMO,
            "--base",
            "olmo1-base",
            "--configs",
            "olmo1-lora-lowlr",
            "olmo1-lora",
            "--run",
            "week3",
            "--out",
            str(W3 / "mechanism-olmo1-gguf.json"),
        ],
    )
    step(
        "mechanism gguf mas06",
        W3 / "mechanism-mas06-gguf.json",
        [
            "ftquant.mechanism_gguf",
            "--model",
            "Qwen/Qwen3-0.6B",
            "--base",
            "q06-base",
            "--configs",
            "mas06-lora-lowlr",
            "mas06-lora",
            "mas06-full",
            "--run",
            "week3",
            "--out",
            str(W3 / "mechanism-mas06-gguf.json"),
        ],
    )
    step(
        "analysis week3",
        W3 / "analysis.json",
        ["ftquant.analyze", "--runs", "week3", "--out", str(W3 / "analysis.json")],
    )
    step(
        "verdicts week3",
        W3 / "verdicts.json",
        ["ftquant.verdicts_w3"],
        stdout=W3 / "verdicts.txt",
    )
    log("week-3 orchestrator complete")


if __name__ == "__main__":
    main()
