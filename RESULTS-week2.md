# ftquant results, week 2

Written 2026-09-28. Design and criteria: `PREREGISTRATION-week2.md` (frozen before any week-2 run). Numbers come from `runs/predictor/verdicts.json` (`python -m ftquant.verdicts`) and `runs/week2/summary.json` (`python -m ftquant.week2`).

## One run failed to train

LoRA at learning rate 3e-4 never learned the task. Validation loss fell from 5.45 to only 1.95 in 500 steps (every other run ended between 0.05 and 0.11), and at bf16 it answers 0 of 3,080 test items, emitting fragments such as `</think>\n\ncard_not`. The base model scores 7.4%, so this fine-tune has no gain, and gain retention is undefined by the pre-registered definition (`retention()` returns NaN when the full-precision gain is not positive). Its 7 test points are left out of T2, which therefore has 42 points instead of 49. The pre-registration did not anticipate a run with no gain, so this is logged as a deviation there.

## Pre-registered predictor tests

The frozen model is R̂ = sigmoid(8.707 - 5.81 ln NSR - 2.794 ln KLD_base), fitted only on week-1 Qwen3-0.6B points (SHA-256 `5df7d620...`). MAE is against R clipped to [0, 1]. Brackets are 95% intervals from resampling whole fine-tunes (configs), since points from one fine-tune are not independent.

| test set | n | NSR model MAE | KLD-only MAE | bits-only MAE | NSR model safe-call agreement | baselines |
|---|---|---|---|---|---|---|
| T2 (new learning rates and seeds, 0.6B) | 42 | 0.107 [0.048, 0.182] | 0.170 | 0.173 | 0.881 | 0.786 / 0.786 |
| T1 ∪ T3 (Qwen3-1.7B) | 17 | 0.103 [0.067, 0.154] | 0.147 | 0.140 | 0.882 | 0.647 / 0.647 |

- **P1 (T2 MAE ≤ 0.10 and safe agreement ≥ 0.85): not met.** Safe agreement passes; MAE misses the bar by 0.007.
- **P2 (T2 MAE below both baselines): met.** 0.107 vs 0.170 and 0.173.
- **P3 (T1 ∪ T3 MAE ≤ 0.15 and safe agreement ≥ 0.80): met.**

Because P1 failed, the pre-registered consequence applies: `ftquant check` ships as a descriptive diagnostic that reports NSR and base KLD, without a retention prediction.

### Why it missed: a second failure mode

The largest error is Qwen3-0.6B full fine-tuning at learning rate 3e-6, MLX 3-bit: predicted 98%, measured 0%. Its NSR is a benign-looking 2.16, but only 5.5% of the update's projection survives quantization (`signal_retained` in `mechanism-q06-mlx.json`). The update is so small relative to the 3-bit step that the fine-tuned and base weights round to the same codes, so Δq is close to zero. Little noise is added because almost nothing is left. NSR only measures the noise, so it reads this case as safe. At MLX 4-bit the same run keeps 50% of its projection, NSR is 4.95, and the model predicts 96% against a measured 27%.

So there are two ways a fine-tune dies in quantization: buried in rounding noise (high NSR), or rounded away (low signal retained, NSR can look fine). The week-1 fit set contained almost no rounded-away points (the lowest was week-1 full fine-tuning at MLX 2-bit, 33%), so the frozen model never saw the second mode.

The other large misses sit where the base model breaks (KLD above 1): full fine-tuning at 3e-5 keeps 89% of its gain at GGUF Q2_K (predicted 2%), while LoRA at 3e-5, with almost the same NSR (5.16 vs 5.29) and the identical base, keeps 1%. Neither input distinguishes them. One untested explanation is that full fine-tuning also changes the RMSNorm weights, which neither format quantizes.

## Exploratory findings

### Gentler fine-tunes lose more (pre-registered as exploratory)

Gain retained, Qwen3-0.6B, seed 0:

| fine-tune | bf16 accuracy | MLX 4-bit | MLX 3-bit | Q4_K_M | Q3_K_M |
|---|---|---|---|---|---|
| LoRA 3e-6 | 74.7% | 0.75 | 0.00 | 0.88 | 0.45 |
| LoRA 1e-5 | 78.5% | 0.84 | 0.06 | 0.91 | 0.80 |
| LoRA 3e-5 | 76.4% | 0.91 | 0.63 | 0.96 | 0.91 |
| LoRA 1e-4 | 73.5% | 0.94 | 0.95 | 0.96 | 1.01 |
| Full 3e-6 | 70.9% | 0.27 | 0.00 | 0.88 | 0.20 |
| Full 1e-5 | 76.6% | 0.78 | 0.14 | 0.94 | 0.75 |
| Full 3e-5 | 75.4% | 0.93 | 0.78 | 0.96 | 0.98 |

Retention rises with learning rate in every one of the 8 kind × format cells (Spearman ρ = +1.0 in 7, +0.8 in 1; n = 3 or 4 per cell, so each cell alone is weak evidence, but the direction never reverses). The bf16 accuracies do not rank the same way: the most accurate model at full precision (LoRA 1e-5) keeps 6% of its gain at MLX 3-bit.

### Seeds

A second seed reproduces retention closely: LoRA 1e-4 keeps 0.94/0.92 (MLX 4-bit) and 0.95/0.92 (MLX 3-bit); LoRA 1e-5 keeps 0.84/0.89 and 0.06/0.00. bf16 accuracy is less stable than retention: LoRA 1e-4 scores 73.5% with seed 0 and 65.1% with seed 1.

### Qwen3-1.7B

LoRA at 1e-5 on 1.7B keeps 36% of its gain at MLX 3-bit, against 163% for LoRA at 1e-4 (above 100% because the 1.7B base collapses at 3-bit, from 37.9% to 10.4%). In accuracy retention, which does not depend on the base, the pair is 0.32 vs 0.93. The fragility of small updates is not a 0.6B artifact.

### Operating-point drift

At a confidence threshold fitted at full precision, the share of items auto-accepted falls much faster than accuracy. At MLX 4-bit, LoRA 1e-5 keeps 88.5% of its accuracy but accepts 19.4 points fewer items; full fine-tuning at 1e-5 keeps 82.8% and accepts 29.7 points fewer. GGUF Q4_K_M is gentler (−9.9 and −4.6 points).

## Figures

- `figures/lr-sweep.png`: gain retained at MLX 3-bit and GGUF Q3_K_M against learning rate.
- `figures/predictor-heldout.png`: the frozen model's held-out predictions against measurements, with the rounded-away points marked.

## What this means for the claims

Supported by the data so far: small fine-tune updates are the fragile ones, across learning rates, fine-tuning kind, seeds and two model sizes, on one task and one model family. A weights-only measurement taken before quantizing beats both baselines out of sample, including across model size. Not supported: that the frozen v1 model is accurate enough to publish as a predictor. A v2 that also uses the retained signal needs a fresh, pre-registered test on data it has never seen.
