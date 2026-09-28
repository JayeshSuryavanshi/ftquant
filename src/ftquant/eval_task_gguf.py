import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from transformers import AutoTokenizer

from ftquant.eval_gguf import complete, free_port, start_server
from ftquant.prompting import encode, render
from ftquant.tasks import get


def evaluate(task_name: str, gguf: str, tokenizer_id: str, split: str, limit: int | None, parallel: int = 4) -> list[dict]:
    task = get(task_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    items = task.split(split)[:limit] if limit else task.split(split)
    port = free_port()
    proc = start_server(gguf, port, parallel, ctx_per_slot=512 if task.name == "massive" else 640)
    try:
        prompts = [encode(tokenizer, render(tokenizer, task.prompt(it.text))) for it in items]
        with ThreadPoolExecutor(parallel) as pool:
            outs = list(pool.map(lambda p: complete(port, p, task.max_tokens), prompts))
    finally:
        proc.terminate()
        proc.wait(timeout=30)
    records = []
    for it, (text, logp) in zip(items, outs):
        pred = text.strip()
        records.append({"text": it.text, "gold": it.target, "pred": pred, "correct": task.correct(pred, it.target),
                        "valid": task.valid(pred), "seq_logprob": logp, "confidence": math.exp(logp)})
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    recs = evaluate(args.task, args.gguf, args.tokenizer, args.split, args.limit, args.parallel)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    acc = sum(r["correct"] for r in recs) / len(recs)
    print(json.dumps({"n": len(recs), "acc": round(acc, 4), "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
