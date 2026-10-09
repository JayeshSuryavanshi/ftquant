import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
from gguf import GGUFReader
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "revision" / "embedding"
CONVERT = ROOT / "vendor" / "llama.cpp" / "convert_hf_to_gguf.py"
TEXT = ROOT / "runs" / "kld" / "wikitext2-test.txt"
MODEL, REVISION = "Qwen/Qwen3-0.6B", "c1899de2"
FORMATS = ["Q4_K_M", "Q3_K_M", "Q2_K"]


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def tied_copy(snap: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for f in snap.iterdir():
        if f.name != "model.safetensors":
            shutil.copy(f, dst / f.name)
    w = mx.load(str(snap / "model.safetensors"))
    assert mx.array_equal(w["lm_head.weight"], w["model.embed_tokens.weight"]).item()
    del w["lm_head.weight"]
    mx.save_safetensors(str(dst / "model.safetensors"), w, metadata={"format": "mlx"})


def main() -> None:
    snap = Path(snapshot_download(MODEL, revision=REVISION))
    OUT.mkdir(parents=True, exist_ok=True)
    common = ["-f", str(TEXT), "-c", "512", "--chunks", "8", "-ngl", "99"]
    summary: dict = {
        "model": MODEL,
        "revision": REVISION,
        "kld_chunks": 8,
        "kld_ctx": 512,
        "llama_cpp_build": 11146,
        "note": "untied = converted from the checkpoint as stored (separate lm_head copy); "
        "tied = lm_head copy removed, as in the fused fine-tunes",
        "formats": {},
    }
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        tied_copy(snap, work / "tied")
        base_logits = work / "base.kld"
        for recipe, src in (("untied", snap), ("tied", work / "tied")):
            bf16 = work / f"{recipe}-bf16.gguf"
            run(
                [
                    sys.executable,
                    str(CONVERT),
                    str(src),
                    "--outtype",
                    "bf16",
                    "--outfile",
                    str(bf16),
                ]
            )
            if not base_logits.exists():
                run(
                    [
                        "llama-perplexity",
                        "-m",
                        str(bf16),
                        "--kl-divergence-base",
                        str(base_logits),
                        *common,
                    ]
                )
            for t in FORMATS:
                q = work / f"{recipe}-{t}.gguf"
                run(["llama-quantize", str(bf16), str(q), t])
                types = {
                    x.name: x.tensor_type.name
                    for x in GGUFReader(str(q)).tensors
                    if x.name in ("token_embd.weight", "output.weight")
                }
                p = run(
                    [
                        "llama-perplexity",
                        "-m",
                        str(q),
                        "--kl-divergence-base",
                        str(base_logits),
                        "--kl-divergence",
                        *common,
                    ]
                )
                kld = float(
                    re.search(
                        r"Mean\s+KLD:\s+([0-9.eE+-]+)", p.stdout + p.stderr
                    ).group(1)
                )
                preds = OUT / f"{recipe}-{t}.jsonl"
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "ftquant.eval_gguf",
                        "--gguf",
                        str(q),
                        "--tokenizer",
                        MODEL,
                        "--cache-ram",
                        "0",
                        "--out",
                        str(preds),
                    ],
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                )
                summary["formats"].setdefault(t, {})[recipe] = {
                    "kld": kld,
                    "embedding_types": types,
                    "predictions": str(preds.relative_to(ROOT)),
                }
                q.unlink()
            bf16.unlink()
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
