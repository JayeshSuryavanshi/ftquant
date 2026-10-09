import gzip
import hashlib
import io
import os
import re
import subprocess
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "supplementary.zip"
TOP = "supplementary"
EXCLUDE = ("paper/", "records/eval-week5.tar.gz")
EXTRA_GLOBS = [
    "runs/smoke*/sweep.log",
    "runs/smoke*/week5.log",
    "runs/smoke*/*/train.yaml",
    "runs/smoke*/*/train.log",
    "runs/smoke*/*/eval/*.jsonl",
    "runs/dryrun/*.jsonl",
    "runs/week5/week5.log",
    "runs/week5/verdicts.json",
    "runs/week5/verdicts.txt",
    "runs/week5/*/train.yaml",
    "runs/week5/*/train.log",
    "runs/week5/*/eval/*.jsonl",
    "runs/revision/*.json",
    "src/ftquant/week5.py",
    "src/ftquant/verdicts_w5.py",
    "src/ftquant/revision.py",
    "src/ftquant/findings.py",
    "runs/findings/findings.json",
    "paper/make_tables.py",
    "paper/make_figures.py",
    "records/prereg-originals/*.md",
    "runs/revision/embedding/*",
    "src/ftquant/embedding_check.py",
]
STORED = {".gz", ".png", ".pdf"}
LEAK = re.compile(r"jayesh|suryavanshi|gmail|/users/|github-updates|ebay", re.I)
# Prediction files contain the public test sets, whose utterances mention "jayesh", "gmail" and
# "eBay"; for them only path and surname leaks are checked.
PREDICTION_LEAK = re.compile(r"suryavanshi|/users/|github-updates", re.I)
REPLACEMENTS = [
    (
        "committed as `ea72821` (now `70b23e4`, see 2026-10-07)",
        "committed as `70b23e4` (rewritten from an earlier ID, see 2026-10-07)",
    ),
    (str(ROOT), "."),
    (str(Path.home()), "~"),
    (
        "- Pushed to the private GitHub repo (85c147d, then 6ab599e with the week-3 results).",
        "- Committed the results to version control, first through week 2 and then with the week-3 results.",
    ),
    (
        "and the correction is in git history (10c24c4).",
        "and the correction is recorded in `runs/week1/PREREG-week4.sha256` (the 05:16:29 amendment).",
    ),
    (
        "- Before the repository went public, its history was rewritten to change the wording of two notebook lines (13:42 and 14:06 on 2026-09-29) and of the supplement builder's leak check. The 12 commits from 2026-09-29 14:09 onward got new IDs; their authors, dates and messages, and every other file, are unchanged. The round-5 plan commit `ea72821`, pushed at 19:50 on 2026-10-03, is now `70b23e4` with the same `PREREGISTRATION-week5.md`; GitHub's push log keeps the original ID.",
        "- The version history was rewritten to change the wording of two notebook lines (13:42 and 14:06 on 2026-09-29) and of the supplement builder's leak check. Commit dates and messages and every other file are unchanged, including the plans and their hash logs.",
    ),
]
MUST_STAY_IDENTICAL = [
    "PREREGISTRATION.md",
    "PREREGISTRATION-week2.md",
    "PREREGISTRATION-week3.md",
    "PREREGISTRATION-week4.md",
    "PREREGISTRATION-week5.md",
    "runs/week1/PREREG.sha256",
    "runs/week1/PREREG-week2.sha256",
    "runs/week1/PREREG-week3.sha256",
    "runs/week1/PREREG-week4.sha256",
    "runs/week1/PREREG-week5.sha256",
    "runs/predictor/predictor-v1.json",
    "runs/predictor/predictor-v1.sha256",
    "runs/predictor/predictor-v2.json",
    "records/prereg-originals/PREREGISTRATION.md",
    "records/prereg-originals/PREREGISTRATION-week2.md",
    "records/prereg-originals/PREREGISTRATION-week3.md",
    "records/prereg-originals/PREREGISTRATION-week4.md",
    "records/prereg-originals/PREREGISTRATION-week5.md",
]
PLAN_LOGS = {
    "PREREGISTRATION.md": "runs/week1/PREREG.sha256",
    "PREREGISTRATION-week2.md": "runs/week1/PREREG-week2.sha256",
    "PREREGISTRATION-week3.md": "runs/week1/PREREG-week3.sha256",
    "PREREGISTRATION-week4.md": "runs/week1/PREREG-week4.sha256",
    "PREREGISTRATION-week5.md": "runs/week1/PREREG-week5.sha256",
}

README = """# Supplementary material

Start here. `README.md` describes the `check` tool itself.

This is an anonymized copy of the project files behind the submission. Round n in the paper is "week n" here, with three exceptions: the hash logs of all five plans are in `runs/week1/`, round 1's plan is `PREREGISTRATION.md`, and the round-1 and round-2 retention analysis is `runs/analysis-all.json`.

## Contents

- `PREREGISTRATION.md` and `PREREGISTRATION-week2.md` to `-week5.md`: the five pre-registrations, with their logged deviations.
- `runs/week1/PREREG*.sha256`: the hash logs of the five plans. Each line is the SHA-256 of one version of a plan and the time it was recorded; later lines are amendments that log a deviation.
- `records/prereg-originals/`: each plan as it was first hashed, before any deviation was logged. In every plan, the text before the deviation section is byte-identical to this first version.
- `runs/predictor/`: the frozen predictors (`predictor-v1.json`, `predictor-v2.json`), v1's hash log (`predictor-v1.sha256`), and the round-2 predictor tests (`eval-T1.json` to `eval-T3.json`, `verdicts.json`).
- `records/eval-weekN.tar.gz`: the per-item predictions of every full-test-set evaluation of rounds 1 to 4, 279 files in all. Unpacked, they give `runs/weekN/<config>/eval/<format>.jsonl`, one line per test item (text, gold, prediction, correct, valid, sequence log-probability, confidence). The week-4 archive also restores `runs/week4/olmo1-base` and `runs/week4/mas06-base`, links to the week-3 base evaluations that round 4 reuses. Round 5's per-item predictions are already unpacked in `runs/week5/<config>/eval/`.
- `runs/weekN/<config>/train.yaml` and `train.log`: the configuration and log of every training run, including the out-of-memory attempt in `runs/week4/_failed-oom-q4b-lora-lowlr/`, round 5's four LoRAs trained on a quantized base, and the stalled first attempt at one of them in `runs/week5/_stalled-q06q3-lora/`. The two Qwen3-4B configurations show `iters: 1000` because mlx-lm counts micro-batches; with 2-step accumulation that is 500 optimizer steps.
- `runs/weekN/sweep.log`, `runs/week5/week5.log` and `runs/orchestrate.log`: the time-stamped accuracy and wall time of every evaluation, and every pipeline step and failure.
- `runs/smoke*/` and `runs/dryrun/`: the pipeline smoke tests and dry runs (reduced test sets; not used for any result). `runs/smoke5/` is round 5's 20-item plumbing test.
- `runs/**/analysis*.json`, `runs/weekN/mechanism-*.json`, `runs/weekN/summary.json`, `runs/weekN/verdicts.json`, `runs/week1/confirmatory.json`, `runs/kld/`, `runs/bpw/`: the computed results.
- `runs/revision/revision.json`: every number the paper reports that is in none of the files above: absolute accuracy, accuracy-retention intervals, the base-free check of the ten replication tests, update norms, predictor baselines, the measurement bands, the 168-of-192 count, the Spearman values, drift counts, the noise floor, confidence strata, validation losses and the list of fine-tunes. `runs/revision/week4-norms.json` caches the round-4 update norms, which are computed from adapter weights that are not included.
- `RESULTS-week1.md` to `RESULTS-week4.md`: the results write-up after each round. `NOTEBOOK.md`: the lab notebook, written as the runs happened.
- `figures/`: figures made during the study. `drift-q06-lora.png`, `predictor-heldout.png` and `replication-week4.png` are not in the paper; the paper's own figures are made by `paper/make_figures.py` (below).
- `src/`, `tests/`, `scripts/`, `configs/`, `pyproject.toml`, `uv.lock`: the code, including the `check` tool.
- `paper/make_tables.py` and `paper/make_figures.py`: the scripts that write the paper's tables and figures.

## Names used in the files

- Configurations are `<base>-<kind>`. Bases: `q06` Qwen3-0.6B, `q17` Qwen3-1.7B, `q4b` Qwen3-4B and `olmo1` OLMo-2 1B, all on banking77, and `mas06` Qwen3-0.6B on MASSIVE. Kinds: `lora` is the LoRA at learning rate 1e-4 (strong), `lora-lowlr` the LoRA at 1e-5 (gentle), `full` full fine-tuning at 1e-5, and `-lrXeY` another learning rate; `-s1` marks seed 1 and `-base` the base model. In round 5, `q06q4-*` and `q06q3-*` are LoRAs trained on the Qwen3-0.6B base already quantized to MLX 4 and 3 bits.
- Formats: `mlx-bf16` and `mlx-qN` (MLX affine, group 64, N bits); `gguf-bf16` and `gguf-<TYPE>` (llama.cpp types such as `gguf-Q3_K_M`). Suffixes: `-unfused` is an adapter loaded on top of the quantized base, `-qlora` an adapter trained on that quantized base, `-labels` a GGUF run whose outputs were restricted to the 77 intent names, and `-npN` a llama-server run with N parallel slots.
- Fields: `retention` is gain retention R, `signal_retained` is update kept, `noise_to_signal` is NSR, `delta_norm` is the update norm and `kld` or `KLD_base` is base damage, in nats per token.

## Checking the pre-registrations

    shasum -a 256 PREREGISTRATION.md PREREGISTRATION-week2.md PREREGISTRATION-week3.md PREREGISTRATION-week4.md PREREGISTRATION-week5.md
    shasum -a 256 records/prereg-originals/*.md
    shasum -a 256 runs/predictor/predictor-v1.json runs/predictor/predictor-v2.json

Each plan's hash equals the last line of its log in `runs/week1/`, and each file in `records/prereg-originals/` equals the first line; the builder checks both. Predictor v1's hash and freeze time are logged in `runs/predictor/predictor-v1.sha256`; v2's full hash is quoted in `PREREGISTRATION-week3.md` and `PREREGISTRATION-week4.md`. Both match the prefixes in the paper (v1 `5df7d620`, v2 `6fd78627`).

## What was anonymized

Local file paths were replaced by `.` (the project root) or `~` (the home folder). Version-control commit identifiers were replaced by `[commit]`, and two notebook sentences that pointed to the private version history now point to the hash logs instead. One notebook entry about rewriting the version history was shortened to leave out where the history is hosted. Two notebook sentences that named the local tooling in use were generalized. The per-item archives were re-packed with anonymous file owners and without macOS metadata files. No result, number or timestamp was changed. The plans, their hash logs and the predictor files are byte-identical to the originals, so their hashes verify.

## Re-scoring from the archived predictions

This needs no model, no GPU and no llama.cpp: only Python 3 with numpy, plus PyYAML for the last two commands. From this folder:

    for w in 1 2 3 4; do tar -xzf records/eval-week$w.tar.gz; done
    export PYTHONPATH=src
    python3 -m ftquant.analyze --runs week1 --out runs/week1/analysis.json
    python3 -m ftquant.confirm
    python3 -m ftquant.analyze --runs week1 week2 --out runs/analysis-all.json
    python3 -m ftquant.verdicts
    python3 -m ftquant.week2
    python3 -m ftquant.analyze --runs week3 --out runs/week3/analysis.json
    python3 -m ftquant.verdicts_w3
    python3 -m ftquant.week3
    python3 -m ftquant.analyze --runs week4 --out runs/week4/analysis.json
    python3 -m ftquant.verdicts_w4
    python3 -m ftquant.week4
    python3 -m ftquant.verdicts_w5
    python3 -m ftquant.revision
    python3 -m ftquant.findings --check

These overwrite the shipped result files with regenerated ones, which should match them. Without the adapter weights, `ftquant.revision` reads the round-4 update norms from `runs/revision/week4-norms.json` and says so. The last command recomputes every number quoted in the paper, from the per-item predictions, the run logs and the files above, and exits with an error unless `runs/findings/findings.json` matches the recomputation byte for byte.

## Rebuilding the paper's tables and figures

    python3 paper/make_tables.py
    python3 paper/make_figures.py   # needs matplotlib

| In the paper | Written by | From |
|---|---|---|
| Base models table | `make_tables.py`, `bases` | `runs/week*/<base>/eval/mlx-bf16.jsonl` |
| Formats table | `formats` | `runs/bpw/mlx-q06.json`, `runs/week1/mechanism-q06-gguf.json`, `runs/week1/mechanism-q17-gguf.json` |
| Round-1 table | `week1` | `runs/week1/confirmatory.json` |
| Learning-rate table | `lrsweep` | `runs/week2/summary.json` |
| Replication table | `replication` | `runs/week3/verdicts.json`, `runs/week4/verdicts.json`, `runs/revision/revision.json` |
| Accuracy-retention table | `pattern` | `runs/revision/revision.json` |
| Measurement bands table | `bands` | `runs/revision/revision.json` |
| Round-5 table | `round5` | `runs/week5/verdicts.json` |
| Predictor tests table | `predictor` | `runs/predictor/verdicts.json`, `runs/week3/verdicts.json`, `runs/week4/verdicts.json`, `runs/week4/summary.json` |
| Predictor baselines table | `baselines` | `runs/revision/revision.json` |
| Absolute accuracy table | `absolute` | `runs/revision/revision.json` |
| Table of every fine-tune | `grid` | `runs/revision/revision.json` |
| Findings table (Table 1) | `python3 -m ftquant.findings` | the per-item predictions, the run logs and the result files above |
| Retention against bits per weight | `make_figures.py`, `retention_chart` | `runs/week1/analysis.json`, `runs/bpw/mlx-q06.json`, `runs/week1/mechanism-q06-gguf.json` |
| Learning-rate figure | `lr_sweep_chart` | `runs/week2/summary.json` |
| Retention against NSR | `mechanism_chart` | `runs/week1/analysis.json`, `runs/week1/mechanism-q06-mlx.json`, `runs/week1/mechanism-q06-gguf.json` |
| Accuracy kept in eight settings | `replication_chart` | `runs/revision/revision.json` |
| Predictor v2 against measurements | `predictor_v2_chart` | `runs/week3/verdicts.json`, `runs/week4/verdicts.json` |

Every number quoted in the text is in `runs/findings/findings.json`: `values` gives each one with the unrounded value behind it, `quoted_in_text` gives each sentence fragment exactly as it appears in the paper, and `findings` gives Table 1 with the values each finding rests on.

## Re-running the study

Everything ran on one Apple silicon laptop with 16 GB of memory: about 51 hours for rounds 1 to 4 and 14.5 hours for round 5, including a stalled run that was restarted (see the deviation in `PREREGISTRATION-week5.md`).

1. `sh scripts/fetch_llama_cpp.sh` fetches llama.cpp's converter and `gguf-py` at commit 7fe450e19. Run it before `uv sync`, because `gguf-py` is a workspace member.
2. `uv sync` installs the environment from `uv.lock`.
3. `sh scripts/fetch_data.sh` downloads banking77 and MASSIVE from their original sources, checks their SHA-256 hashes and builds the splits.
4. Put `llama-server` and `llama-quantize` from llama.cpp build 11146 (commit 7fe450e19) on `PATH`, for example Homebrew's llama.cpp 0.5.0. The orchestrators set `PATH` to `/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin`; edit that line elsewhere.
5. Round 1 and 2: `PYTHONPATH=src uv run python -m ftquant.sweep --plan week1`, then `PYTHONPATH=src uv run python -m ftquant.orchestrate`, which waits for that sweep and runs the base-damage, mechanism, predictor and round-2 steps.
6. Rounds 3 and 4: `PYTHONPATH=src uv run python -m ftquant.orchestrate_w3`, then `PYTHONPATH=src uv run python -m ftquant.orchestrate_w4`.
7. Round 5: `PYTHONPATH=src uv run python -m ftquant.week5`. It needs the round-1, round-3 and round-4 adapters, which are not included here, so it re-runs only after steps 5 and 6.

Each step skips outputs that already exist, so an interrupted run can be resumed with the same command.

Model and data revisions used (each repository's main branch, unchanged since before the study): Qwen/Qwen3-0.6B `c1899de289a04d12100db370d81485cdf75e47ca`, Qwen/Qwen3-1.7B `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, Qwen/Qwen3-4B `1cfa9a7208912126459214e8b04321603b3df60c`, allenai/OLMo-2-0425-1B-Instruct `48d788eca847d4d7548f375ad03d3c9312f6139e`, and Salesforce/wikitext `b08601e04326c79dfdd32d625aee71d232d685c3` (the WikiText-2 test text used for base damage has SHA-256 `bbf94c53a05abe9ee670d3b6343608095822c85e26de37c70b24fc571964574a`).

## Data

banking77 (PolyAI) and MASSIVE (Amazon) are released under CC BY 4.0. The per-item predictions contain their test utterances.
"""


def tracked() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return [f for f in out.stdout.splitlines() if not f.startswith(EXCLUDE)]


def extras() -> list[str]:
    found = {
        str(p.relative_to(ROOT))
        for g in EXTRA_GLOBS
        for p in ROOT.glob(g)
        if p.is_file()
    }
    return sorted(found)


def commit_ids() -> list[str]:
    out = subprocess.run(
        ["git", "log", "--all", "--format=%H"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.split()


def scrub_text(text: str, commits: list[str]) -> str:
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)

    def redact(m: re.Match) -> str:
        tok = m.group(0)
        return "[commit]" if any(c.startswith(tok) for c in commits) else tok

    return re.sub(r"\b[0-9a-f]{7,40}\b", redact, text)


def week_links() -> dict[str, list[tuple[str, str]]]:
    links: dict[str, list[tuple[str, str]]] = {}
    for p in sorted((ROOT / "runs").glob("week*/*")):
        if p.is_symlink():
            week = p.parent.name
            links.setdefault(week, []).append(
                (str(p.relative_to(ROOT)), os.readlink(p))
            )
    return links


def repack(data: bytes, links: list[tuple[str, str]]) -> bytes:
    out = io.BytesIO()
    with (
        tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as src,
        gzip.GzipFile(fileobj=out, mode="wb", mtime=0) as gz,
        tarfile.open(fileobj=gz, mode="w", format=tarfile.USTAR_FORMAT) as dst,
    ):
        for m in src.getmembers():
            if Path(m.name).name.startswith("._"):
                continue
            info = tarfile.TarInfo(m.name)
            info.type, info.mode, info.mtime, info.size = (
                m.type,
                m.mode,
                int(m.mtime),
                m.size,
            )
            info.linkname = m.linkname
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            payload = src.extractfile(m).read() if m.isfile() else None
            if payload is not None and PREDICTION_LEAK.search(
                payload.decode("utf-8", "replace")
            ):
                raise SystemExit(f"identifying text inside {m.name}")
            dst.addfile(info, io.BytesIO(payload) if payload is not None else None)
        for name, target in links:
            info = tarfile.TarInfo(name)
            info.type, info.linkname, info.mode = tarfile.SYMTYPE, target, 0o755
            info.uid = info.gid = 0
            dst.addfile(info)
    return out.getvalue()


def check_plan_hashes() -> None:
    for plan, log in PLAN_LOGS.items():
        logged = [
            line.split()[0]
            for line in (ROOT / log).read_text().splitlines()
            if line.strip()
        ]
        first = hashlib.sha256(
            (ROOT / "records/prereg-originals" / plan).read_bytes()
        ).hexdigest()
        last = hashlib.sha256((ROOT / plan).read_bytes()).hexdigest()
        assert (
            first == logged[0]
        ), f"{plan}: first-hashed version does not match its log"
        assert last == logged[-1], f"{plan}: current version does not match its log"


def main() -> None:
    check_plan_hashes()
    commits = commit_ids()
    for keep in ("5df7d620", "6fd78627", "7fe450e1"):
        assert not any(c.startswith(keep) for c in commits)
    links = week_links()
    entries: list[tuple[str, bytes]] = [
        (f"{TOP}/README-SUPPLEMENT.md", README.encode())
    ]
    changed = 0
    files = tracked()
    seen = set(files)
    for rel in files + [e for e in extras() if e not in seen]:
        raw = (ROOT / rel).read_bytes()
        if rel.endswith(".tar.gz"):
            week = re.search(r"eval-(week\d)", rel).group(1)
            data = repack(raw, links.get(week, []))
        elif Path(rel).suffix in STORED:
            data = raw
        else:
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                data = raw
            else:
                data = scrub_text(text, commits).encode("utf-8")
        if rel in MUST_STAY_IDENTICAL:
            assert data == raw, f"{rel} changed"
        changed += data != raw
        entries.append((f"{TOP}/{rel}", data))
    names = [n for n, _ in entries]
    assert len(names) == len(set(names)), "duplicate names"
    for name, data in entries:
        if name.endswith(".tar.gz"):
            continue
        check = PREDICTION_LEAK if name.endswith(".jsonl") else LEAK
        if check.search(data.decode("utf-8", "replace")):
            raise SystemExit(f"identifying text left in {name}")
    with zipfile.ZipFile(OUT, "w") as z:
        for name, data in entries:
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 3, 0, 0, 0))
            info.external_attr = 0o644 << 16
            info.compress_type = (
                zipfile.ZIP_STORED
                if Path(name).suffix in STORED
                else zipfile.ZIP_DEFLATED
            )
            z.writestr(info, data)
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()[:16]
    print(
        f"{OUT.name}: {len(entries)} files, {changed} scrubbed or re-packed, "
        f"{OUT.stat().st_size / 1e6:.1f} MB, sha256 {digest}"
    )


if __name__ == "__main__":
    main()
