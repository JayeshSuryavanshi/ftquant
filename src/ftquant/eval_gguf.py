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


def start_server(gguf: str, port: int, parallel: int, ctx_per_slot: int = 640) -> subprocess.Popen:
    proc = subprocess.Popen(
        ["llama-server", "-m", gguf, "--port", str(port), "--host", "127.0.0.1", "-np", str(parallel),
         "-c", str(parallel * ctx_per_slot), "-ngl", "99", "--no-warmup", "--log-disable"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
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


def complete(port: int, tokens: list[int], max_tokens: int) -> tuple[str, float]:
    r = requests.post(f"http://127.0.0.1:{port}/completion", json={
        "prompt": tokens, "n_predict": max_tokens, "temperature": 0.0, "top_k": 1, "cache_prompt": True,
        "n_probs": 1, "post_sampling_probs": False, "stop": ["<|im_end|>"],
    }, timeout=300)
    r.raise_for_status()
    body = r.json()
    logp = sum(p["logprob"] for p in body.get("completion_probabilities", []))
    return body["content"], logp


def evaluate(gguf: str, tokenizer_id: str, split: str, limit: int | None, parallel: int = 4, max_tokens: int = 16) -> list[dict]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    label_set = labels()
    examples = read_split(split)[:limit] if limit else read_split(split)
    port = free_port()
    proc = start_server(gguf, port, parallel)
    try:
        prompts = [item_tokens(tokenizer, e.text, label_set) for e in examples]
        with ThreadPoolExecutor(parallel) as pool:
            outs = list(pool.map(lambda p: complete(port, p, max_tokens), prompts))
    finally:
        proc.terminate()
        proc.wait(timeout=30)
    records = []
    for ex, (text, logp) in zip(examples, outs):
        pred = text.strip()
        records.append({"text": ex.text, "gold": ex.label, "pred": pred, "correct": pred == ex.label,
                        "valid": pred in label_set, "seq_logprob": logp, "confidence": math.exp(logp)})
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    recs = evaluate(args.gguf, args.tokenizer, args.split, args.limit, args.parallel)
    dt = time.time() - t0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    acc = sum(r["correct"] for r in recs) / len(recs)
    valid = sum(r["valid"] for r in recs) / len(recs)
    print(json.dumps({"n": len(recs), "acc": round(acc, 4), "valid": round(valid, 4), "seconds": round(dt, 1), "items_per_s": round(len(recs) / dt, 2)}))


if __name__ == "__main__":
    main()
