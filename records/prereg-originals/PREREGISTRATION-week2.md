# ftquant pre-registration (week 2): a pre-quantization predictor

Written 2026-09-27, about 18:00 PDT. The SHA-256 of this file is in `runs/week1/PREREG-week2.sha256`.

## Status at the time of writing

These things had been observed:

- The week-1 Qwen3-0.6B results (exploratory): retention, operating-point drift and recovery cost for `q06-lora`, `q06-lora-lowlr` and `q06-full`.
- Noise-to-signal ratios (NSR) for those three fine-tunes, in both MLX affine and GGUF k-quants.

These things had NOT been observed, or did not exist yet:

- Any result from a Qwen3-1.7B fine-tune. Only `q17-base` evaluations were pending.
- Any base-model KL divergence (the KLD input to the predictor).
- Any week-2 training run or evaluation.

This means the predictor's parameters are still unknown when this is written. They are produced by the fixed procedure below.

## Quantities

**NSR** is computed per fine-tune and per quantization variant, from weights only:

1. Δ = W_ft − W_base, over all LoRA-targeted linear weights (q, k, v, o, gate, up, down in every layer).
2. Δq = deq(Q(W_ft)) − deq(Q(W_base)), using the exact quantizer: `mx.quantize` with group size 64 for MLX, and `llama-quantize` then gguf-py dequantization for GGUF.
3. The signal is the projection of Δq onto Δ. NSR = ‖Δq − proj_Δ(Δq)‖ / ‖Δ‖.
4. The code is `src/ftquant/mechanism.py` and `src/ftquant/mechanism_gguf.py`.

**KLD_base** is the mean per-token KL(p_bf16 ‖ p_quantized) of the BASE model on 8 × 512 tokens of WikiText-2-raw test (`src/ftquant/kld.py`). For GGUF it comes from `llama-perplexity --kl-divergence` against the bf16 GGUF.

**Retention R** is the gain retention defined in `PREREGISTRATION.md`.

## Predictor (procedure frozen now)

- Model: R̂ = sigmoid(t0 + t1·ln NSR + t2·ln KLD_base).
- Fit by least squares on clipped R ∈ [0, 1] (Adam, 20,000 steps, learning rate 0.02, initial θ = (2, −1, −1)). The code is `predictor.py fit`.
- **Fit set:** every non-bf16, fused Qwen3-0.6B week-1 point from `q06-lora`, `q06-lora-lowlr` and `q06-full`, covering MLX q8/q6/q4/q3/q2 and GGUF Q8_0/Q6_K/Q4_K_M/Q3_K_M/Q2_K.
- The fitted file `runs/predictor/predictor-v1.json` is hashed when it is written. It is fitted before any test point below is evaluated, and it is never refitted on test data.

Two baselines are fitted in the same run on the same fit set:

1. **bits-only**: predict the mean fit-set retention of that quantization variant, ignoring the fine-tune.
2. **KLD-only**: sigmoid(t0 + t2·ln KLD_base).

## Test sets (out of sample)

- **T1, a new model size:** Qwen3-1.7B `q17-lora` (week 1), all fused non-bf16 variants.
- **T2, new update magnitudes and seeds on 0.6B (week 2):**
  - LoRA at learning rate 3e-6, 3e-5 and 3e-4.
  - Full fine-tuning at 3e-6 and 3e-5.
  - Seed 1 of `lora` and of `lora-lowlr`.
  - Variants: MLX q6/q4/q3 and GGUF Q6_K/Q4_K_M/Q3_K_M/Q2_K.
- **T3:** Qwen3-1.7B `q17-lora-lowlr` (week 2), same variants.

## Success criteria (decided now)

- **P1, useful:** on T2, the NSR model's MAE ≤ 0.10, AND its agreement on the "safe" call (R ≥ 0.9) is ≥ 0.85.
- **P2, better than the baselines:** on T2, the NSR model's MAE is lower than both the bits-only and the KLD-only baselines.
- **P3, transfers across model size:** on T1 ∪ T3, the NSR model's MAE ≤ 0.15 AND safe agreement ≥ 0.80.

Each criterion is reported as met, not met, or inconclusive, whatever the outcome. If P1 or P2 fails, `ftquant check` ships as a descriptive diagnostic (it reports NSR and KLD) without a retention prediction.

## Exploratory (week 2)

- At MLX q4/q3 and GGUF Q4_K_M/Q3_K_M, retention is expected to rise with learning rate, meaning with the size of the update, for both LoRA and full fine-tuning.
- Seed-to-seed variation in retention at matched configurations, reported as an estimate of noise.
- Operating-point drift and the recovery-cost curve (the labels needed to re-fit τ) across all the new configurations.

## Deviations

None yet.
