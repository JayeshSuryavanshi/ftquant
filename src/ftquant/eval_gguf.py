import argparse
import json
import math
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from transformers import AutoTokenizer

from ftquant.data import labels, read_split
from ftquant.prompting import item_tokens


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server(
    gguf: str,
    port: int,
    parallel: int,
    ctx_per_slot: int = 640,
    cache_ram: int | None = None,
    lora: str | None = None,
    lora_scale: float = 1.0,
) -> subprocess.Popen:
    cmd = [
        "llama-server",
        "-m",
        gguf,
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
        "-np",
        str(parallel),
        "-c",
        str(parallel * ctx_per_slot),
        "-ngl",
        "99",
        "--no-warmup",
        "--log-disable",
    ]
    if cache_ram is not None:
        cmd += ["--cache-ram", str(cache_ram)]
    if lora is not None:
        cmd += (
            ["--lora", lora]
            if lora_scale == 1.0
            else ["--lora-scaled", f"{lora}:{lora_scale}"]
        )
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}/health"
    for _ in range(600):
        try:
            if requests.get(url, timeout=1).status_code == 200:
                return proc
        except requests.RequestException:
            pass
        if proc.poll() is not None:
            raise RuntimeError("llama-server exited during startup")
        time.sleep(0.5)
    proc.kill()
    raise TimeoutError("llama-server did not become healthy")


def label_grammar(label_set: list[str]) -> str:
    return "root ::= " + " | ".join(json.dumps(label) for label in label_set)


def complete(
    port: int, tokens: list[int], max_tokens: int, grammar: str | None = None
) -> tuple[str, float]:
    payload = {
        "prompt": tokens,
        "n_predict": max_tokens,
        "temperature": 0.0,
        "top_k": 1,
        "cache_prompt": True,
        "n_probs": 1,
        "post_sampling_probs": False,
        "stop": ["<|im_end|>"],
    }
    if grammar:
        payload["grammar"] = grammar
    r = requests.post(f"http://127.0.0.1:{port}/completion", json=payload, timeout=300)
    r.raise_for_status()
    body = r.json()
    logp = sum(p["logprob"] for p in body.get("completion_probabilities", []))
    return body["content"], logp


def evaluate(
    gguf: str,
    tokenizer_id: str,
    split: str,
    limit: int | None,
    parallel: int = 4,
    max_tokens: int = 16,
    labels_only: bool = False,
    cache_ram: int | None = None,
    lora: str | None = None,
    lora_scale: float = 1.0,
) -> list[dict]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    label_set = labels()
    grammar = label_grammar(label_set) if labels_only else None
    examples = read_split(split)[:limit] if limit else read_split(split)
    port = free_port()
    proc = start_server(
        gguf, port, parallel, cache_ram=cache_ram, lora=lora, lora_scale=lora_scale
    )
    try:
        prompts = [item_tokens(tokenizer, e.text, label_set) for e in examples]
        with ThreadPoolExecutor(parallel) as pool:
            outs = list(
                pool.map(lambda p: complete(port, p, max_tokens, grammar), prompts)
            )
    finally:
        proc.terminate()
        proc.wait(timeout=30)
    records = []
    for ex, (text, logp) in zip(examples, outs):
        pred = text.strip()
        records.append(
            {
                "text": ex.text,
                "gold": ex.label,
                "pred": pred,
                "correct": pred == ex.label,
                "valid": pred in label_set,
                "seq_logprob": logp,
                "confidence": math.exp(logp),
            }
        )
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--labels-only", action="store_true")
    ap.add_argument("--cache-ram", type=int)
    ap.add_argument("--lora")
    ap.add_argument("--lora-scale", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    recs = evaluate(
        args.gguf,
        args.tokenizer,
        args.split,
        args.limit,
        args.parallel,
        labels_only=args.labels_only,
        cache_ram=args.cache_ram,
        lora=args.lora,
        lora_scale=args.lora_scale,
    )
    dt = time.time() - t0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    acc = sum(r["correct"] for r in recs) / len(recs)
    valid = sum(r["valid"] for r in recs) / len(recs)
    print(
        json.dumps(
            {
                "n": len(recs),
                "acc": round(acc, 4),
                "valid": round(valid, 4),
                "seconds": round(dt, 1),
                "items_per_s": round(len(recs) / dt, 2),
            }
        )
    )


if __name__ == "__main__":
    main()
