import sys

USAGE = """usage: ftquant <command> [options]

commands:
  check        noise-to-update ratio and retained update for MLX and legacy GGUF formats
  check-gguf   the same for llama.cpp k-quants, from bf16 GGUF files (needs llama-quantize)

run 'ftquant <command> --help' for options"""


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(USAGE)
        raise SystemExit(0 if len(sys.argv) >= 2 else 2)
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "check":
        from ftquant.check import main as run
    elif cmd == "check-gguf":
        from ftquant.check_gguf import main as run
    else:
        print(USAGE)
        raise SystemExit(2)
    run(rest)


if __name__ == "__main__":
    main()
