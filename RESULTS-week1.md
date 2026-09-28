# Week 1 results (confirmatory: Qwen3-1.7B, standard LoRA)

Analysis run 2026-09-27 23:40 PDT, after the week-1 sweep completed (23:32). The hypotheses are exactly as pre-registered in `PREREGISTRATION.md` (hashed 17:39, before any 1.7B run). The verdict categories are supported, not supported and inconclusive. A multi-part hypothesis is supported only if every part holds. The rule for combining parts was fixed in `src/ftquant/confirm.py` before its output was seen.

| Hypothesis | Verdict | What happened |
|---|---|---|
| H1: little loss at high precision (R ≥ 0.95 at MLX q8/q6 and GGUF Q8_0/Q6_K, lower CI ≥ 0.90) | **supported** | R = 1.01 to 1.07, all lower bounds ≥ 0.99 |
| H2: material loss at 3 bits, R < 0.5 at 2 bits | **not supported** | R at 3 bits is *higher* than at 4 bits (MLX 1.63 vs 1.25; GGUF 1.75 vs 1.11), because the base model collapses faster than the fine-tune (see below). At 2 bits, MLX R = 0 (the model breaks entirely) but GGUF Q2_K R = 1.78. |
| H3: unfused adapter beats fused at MLX q4 and q3 | **inconclusive** | q4: +0.022 [−0.002, +0.044]; q3: −0.038 [−0.073, −0.004]. No consistent advantage either way. |
| H4: operating point moves at 4 bits (MLX q4 and GGUF Q4_K_M) | **not supported** | MLX q4: auto-accept coverage 38.1% → 33.9% (−4.2 pts, meets the threshold). GGUF Q4_K_M: 36.4% → 35.3% (−1.2 pts, does not). Both parts had to hold. |

## Why R exceeds 1 (important for interpretation)

R is the gain over the base *at the same precision*. On Qwen3-1.7B the base's own accuracy collapses under quantization, mostly because it stops producing valid labels. The fine-tuned model does not:

| Variant | Fine-tuned acc | Base acc | Base valid-label rate | Fine-tuned acc / its own bf16 |
|---|---|---|---|---|
| MLX bf16 | 73.4% | 37.9% | 83.9% | 1.00 |
| MLX q4 | 70.4% | 26.2% | 80.4% | 0.96 |
| MLX q3 | 68.3% | 10.4% | 36.3% | 0.93 |
| MLX q2 | 0.0% | 0.0% | 0.0% | 0.00 |
| GGUF Q4_K_M | 73.4% | 34.2% | 76.8% | 1.00 |
| GGUF Q3_K_M | 72.0% | 10.5% | 34.5% | 0.98 |
| GGUF Q2_K | 62.6% | 0.0% | 0.0% | 0.85 |

So R > 1 means "the fine-tuned model is further ahead of an equally quantized base than it was at full precision". It does not mean the fine-tune improved. For the practical question, the added exploratory metric *accuracy retention* (fine-tuned accuracy / its own bf16 accuracy) is the clearer read. It was not pre-registered and is reported as exploratory.

## Exploratory reading

- **With standard LoRA, a fine-tuned model is more robust to quantization than its base.** At 3 bits it keeps 93% (MLX) and 98% (GGUF Q3_K_M) of its accuracy, while the base falls to about 27% of its own. Fine-tuning locks in the output format that quantization breaks in the base. The same thing happened on 0.6B: standard LoRA kept 62.5% accuracy at MLX q3 while the base scored 0%.
- **The operating point still drifts well before accuracy does, at 3 bits and below.**
  - MLX q3: accuracy 93% of bf16, auto-accept coverage 38.1% → 27.8%.
  - GGUF Q3_K_M: 98% of accuracy, coverage 36.4% → 30.5%.
  - GGUF Q2_K: 85% of accuracy, coverage 36.4% → 10.1%.
  - At 4 bits the drift is small on GGUF (−1.2 pts) and moderate on MLX (−4.2 pts), so H4 as stated fails.
- **Noise floor:** MLX bf16 and GGUF bf16 give identical accuracy on the fine-tuned 1.7B model (73.38% both) and agree on 99.2% of predictions.
- Combined with the exploratory 0.6B data, the picture is **two regimes**:
  - Fine-tunes with large updates (standard LoRA) survive to 3 bits and outlast the base.
  - Fine-tunes with small updates (low learning rate, full fine-tuning at 1e-5) are buried in quantization noise at 3 to 4 bits.
  - Week 2 tests whether a weight-only noise-to-update ratio predicts which regime a fine-tune is in.
