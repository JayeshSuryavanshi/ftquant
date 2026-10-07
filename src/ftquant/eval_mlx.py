import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import KVCache, make_prompt_cache, trim_prompt_cache

from ftquant.data import labels, read_split
from ftquant.prompting import item_tokens, shared_prefix


def step(model, tokens: list[int], cache) -> mx.array:
    logits = model(mx.array(tokens)[None], cache=cache)[0, -1].astype(mx.float32)
    return logits - mx.logsumexp(logits)


def greedy_decode(
    model, tokens: list[int], cache, stop: set[int], max_tokens: int
) -> tuple[list[int], float, int]:
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


def batch_step(model, tokens: mx.array, cache) -> mx.array:
    logits = model(tokens, cache=cache)[:, -1].astype(mx.float32)
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def batch_cache(
    model, prefix_state: list[tuple[mx.array, mx.array]], n: int
) -> list[KVCache]:
    cache = make_prompt_cache(model)
    for c, (k, v) in zip(cache, prefix_state, strict=True):
        if not isinstance(c, KVCache):
            raise TypeError(
                f"batched evaluation needs a plain KVCache, got {type(c).__name__}"
            )
        c.state = (mx.repeat(k, n, axis=0), mx.repeat(v, n, axis=0))
    return cache


def greedy_decode_batch(
    model, rows: list[list[int]], cache, stop: set[int], max_tokens: int
) -> list[tuple[list[int], float]]:
    logprobs = batch_step(model, mx.array(rows), cache)
    outs: list[list[int]] = [[] for _ in rows]
    totals = [0.0] * len(rows)
    done = [False] * len(rows)
    for i in range(max_tokens):
        nxt = mx.argmax(logprobs, axis=-1)
        picked = mx.take_along_axis(logprobs, nxt[:, None], axis=-1)[:, 0]
        for r, (t, lp) in enumerate(zip(nxt.tolist(), picked.tolist(), strict=True)):
            if done[r]:
                continue
            if t in stop:
                done[r] = True
                continue
            outs[r].append(t)
            totals[r] += lp
        if all(done) or i == max_tokens - 1:
            break
        logprobs = batch_step(model, nxt[:, None], cache)
    return list(zip(outs, totals, strict=True))


def decode_all(
    model,
    prefix: list[int],
    rests: list[list[int]],
    stop: set[int],
    max_tokens: int,
    batch: int = 1,
) -> list[tuple[list[int], float]]:
    cache = make_prompt_cache(model)
    step(model, prefix, cache)
    if batch <= 1:
        results = []
        for rest in rests:
            out, logp, fed = greedy_decode(model, rest, cache, stop, max_tokens)
            trim_prompt_cache(cache, fed)
            results.append((out, logp))
        return results
    prefix_state = [
        (c.keys[..., : c.offset, :], c.values[..., : c.offset, :]) for c in cache
    ]
    by_len: dict[int, list[int]] = defaultdict(list)
    for i, rest in enumerate(rests):
        by_len[len(rest)].append(i)
    batched: list[tuple[list[int], float] | None] = [None] * len(rests)
    # Rows in one batch share a suffix length, so every token sits at the same position as in the per-item path.
    for idx in by_len.values():
        for s in range(0, len(idx), batch):
            chunk = idx[s : s + batch]
            rows = [rests[i] for i in chunk]
            outs = greedy_decode_batch(
                model,
                rows,
                batch_cache(model, prefix_state, len(chunk)),
                stop,
                max_tokens,
            )
            for i, r in zip(chunk, outs, strict=True):
                batched[i] = r
    return [r for r in batched if r is not None]


def evaluate(
    model_path: str,
    adapter_path: str | None,
    split: str,
    limit: int | None,
    max_tokens: int = 16,
    batch: int = 1,
) -> list[dict]:
    model, tokenizer = load(model_path, adapter_path=adapter_path)
    label_set = labels()
    examples = read_split(split)[:limit] if limit else read_split(split)
    prefix = shared_prefix(tokenizer, label_set)
    stop = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}
    rests = []
    for ex in examples:
        toks = item_tokens(tokenizer, ex.text, label_set)
        if toks[: len(prefix)] != prefix:
            raise ValueError("prompt does not start with the cached prefix")
        rests.append(toks[len(prefix) :])
    records = []
    for ex, (out, logp) in zip(
        examples, decode_all(model, prefix, rests, stop, max_tokens, batch), strict=True
    ):
        pred = tokenizer.decode(out).strip()
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
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    recs = evaluate(args.model, args.adapter, args.split, args.limit, batch=args.batch)
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
