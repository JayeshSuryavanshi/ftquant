import argparse
import json
import math
import time
from pathlib import Path

from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

from ftquant.eval_mlx import greedy_decode, step
from ftquant.prompting import encode, render
from ftquant.tasks import get


def evaluate(task_name: str, model_path: str, adapter_path: str | None, split: str, limit: int | None) -> list[dict]:
    task = get(task_name)
    model, tokenizer = load(model_path, adapter_path=adapter_path)
    items = task.split(split)[:limit] if limit else task.split(split)
    prefix = encode(tokenizer, render(tokenizer, task.prefix().rstrip() + "\x00").split("\x00")[0])
    cache = make_prompt_cache(model)
    step(model, prefix, cache)
    stop = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}
    records = []
    for it in items:
        toks = encode(tokenizer, render(tokenizer, task.prompt(it.text)))
        if toks[: len(prefix)] != prefix:
            raise ValueError("prompt does not start with the cached prefix")
        out, logp, fed = greedy_decode(model, toks[len(prefix):], cache, stop, task.max_tokens)
        trim_prompt_cache(cache, fed)
        pred = tokenizer.decode(out).strip()
        records.append({"text": it.text, "gold": it.target, "pred": pred, "correct": task.correct(pred, it.target),
                        "valid": task.valid(pred), "seq_logprob": logp, "confidence": math.exp(logp)})
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    recs = evaluate(args.task, args.model, args.adapter, args.split, args.limit)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    acc = sum(r["correct"] for r in recs) / len(recs)
    print(json.dumps({"n": len(recs), "acc": round(acc, 4), "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
