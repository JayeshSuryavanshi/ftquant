import argparse
import json
import math
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

from ftquant.data import labels, read_split
from ftquant.prompting import item_tokens, shared_prefix


def step(model, tokens: list[int], cache) -> mx.array:
    logits = model(mx.array(tokens)[None], cache=cache)[0, -1].astype(mx.float32)
    return logits - mx.logsumexp(logits)


def greedy_decode(model, tokens: list[int], cache, stop: set[int], max_tokens: int) -> tuple[list[int], float, int]:
    logprobs = step(model, tokens, cache)
    out: list[int] = []
    total = 0.0
    fed = len(tokens)
    for _ in range(max_tokens):
        t = int(mx.argmax(logprobs))
        if t in stop:
            break
        out.append(t)
        total += float(logprobs[t])
        logprobs = step(model, [t], cache)
        fed += 1
    return out, total, fed


def evaluate(model_path: str, adapter_path: str | None, split: str, limit: int | None, max_tokens: int = 16) -> list[dict]:
    model, tokenizer = load(model_path, adapter_path=adapter_path)
    label_set = labels()
    examples = read_split(split)[:limit] if limit else read_split(split)
    prefix = shared_prefix(tokenizer, label_set)
    cache = make_prompt_cache(model)
    step(model, prefix, cache)
    stop = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}
    records = []
    for ex in examples:
        toks = item_tokens(tokenizer, ex.text, label_set)
        if toks[: len(prefix)] != prefix:
            raise ValueError("prompt does not start with the cached prefix")
        rest = toks[len(prefix):]
        out, logp, fed = greedy_decode(model, rest, cache, stop, max_tokens)
        trim_prompt_cache(cache, fed)
        pred = tokenizer.decode(out).strip()
        records.append({"text": ex.text, "gold": ex.label, "pred": pred, "correct": pred == ex.label,
                        "valid": pred in label_set, "seq_logprob": logp, "confidence": math.exp(logp)})
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    recs = evaluate(args.model, args.adapter, args.split, args.limit)
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
