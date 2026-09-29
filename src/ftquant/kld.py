import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
from datasets import load_dataset
from huggingface_hub import snapshot_download
from mlx_lm import load

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "runs" / "kld"
CONVERT = ROOT / "vendor" / "llama.cpp" / "convert_hf_to_gguf.py"


def text_file() -> Path:
    p = WORK / "wikitext2-test.txt"
    if not p.exists():
        WORK.mkdir(parents=True, exist_ok=True)
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        p.write_text("".join(ds["text"]))
    return p


def chunks(tokenizer, n_chunks: int, ctx: int) -> list[list[int]]:
    ids = tokenizer.encode(text_file().read_text())
    return [ids[i * ctx:(i + 1) * ctx] for i in range(n_chunks)]


def mlx_kld(model_id: str, bits_list: list[int], n_chunks: int, ctx: int) -> dict[str, float]:
    ref, tok = load(model_id)
    data = chunks(tok, n_chunks, ctx)
    ref_lp = []
    for c in data:
        logits = ref(mx.array(c)[None])[0].astype(mx.float32)
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        mx.eval(lp)
        ref_lp.append(lp)
    del ref
    mx.clear_cache()
    out = {}
    for bits in bits_list:
        qpath = WORK / f"{model_id.split('/')[-1]}-mlx-q{bits}"
        if not qpath.exists():
            subprocess.run([sys.executable, "-m", "mlx_lm", "convert", "--hf-path", model_id, "--mlx-path", str(qpath),
                            "-q", "--q-bits", str(bits), "--q-group-size", "64"], check=True, capture_output=True)
        q, _ = load(str(qpath))
        total = 0.0
        count = 0
        for c, lp in zip(data, ref_lp):
            logits = q(mx.array(c)[None])[0].astype(mx.float32)
            qlp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            kl = mx.sum(mx.exp(lp) * (lp - qlp), axis=-1)
            total += float(mx.sum(kl))
            count += kl.shape[0]
        out[f"mlx-q{bits}"] = total / count
        del q
        subprocess.run(["rm", "-rf", str(qpath)])
    return out


def gguf_kld(model_id: str, types: list[str], n_chunks: int, ctx: int) -> dict[str, float]:
    d = WORK / model_id.split("/")[-1]
    d.mkdir(parents=True, exist_ok=True)
    bf16 = d / "bf16.gguf"
    if not bf16.exists():
        subprocess.run([sys.executable, str(CONVERT), snapshot_download(model_id), "--outtype", "bf16", "--outfile", str(bf16)],
                       check=True, capture_output=True)
    base_logits = d / "base.kld"
    common = ["-f", str(text_file()), "-c", str(ctx), "--chunks", str(n_chunks), "-ngl", "99"]
    if not base_logits.exists():
        subprocess.run(["llama-perplexity", "-m", str(bf16), "--kl-divergence-base", str(base_logits), *common],
                       check=True, capture_output=True)
    out = {}
    for t in types:
        q = d / f"{t}.gguf"
        if not q.exists():
            subprocess.run(["llama-quantize", str(bf16), str(q), t], check=True, capture_output=True)
        p = subprocess.run(["llama-perplexity", "-m", str(q), "--kl-divergence-base", str(base_logits), "--kl-divergence", *common],
                           capture_output=True, text=True, check=True)
        m = re.search(r"Mean\s+KLD:\s+([0-9.eE+-]+)", p.stdout + p.stderr)
        out[f"gguf-{t}"] = float(m.group(1)) if m else float("nan")
        q.unlink(missing_ok=True)
    base_logits.unlink(missing_ok=True)
    bf16.unlink(missing_ok=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bits", nargs="+", type=int, default=[8, 6, 4, 3, 2])
    ap.add_argument("--types", nargs="+", default=["Q8_0", "Q6_K", "Q4_K_M", "Q3_K_M", "Q2_K"])
    args = ap.parse_args()
    res = {"model": args.model, "chunks": args.chunks, "ctx": args.ctx}
    res["mlx"] = mlx_kld(args.model, args.bits, args.chunks, args.ctx)
    res["gguf"] = gguf_kld(args.model, args.types, args.chunks, args.ctx)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
