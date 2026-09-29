# ftquant pre-registration (week 4): second seeds and a larger model

Written 2026-09-28, about 18:10 PDT, before any week-4 fine-tune was trained at full length or evaluated on a full test set. The SHA-256 of this file is in `runs/week1/PREREG-week4.sha256`.

## Why

Week 3 supported every pre-registered hypothesis (`RESULTS-week3.md`), but it had two stated limits:

- one seed per configuration;
- no model larger than 1.7B parameters.

Week 4 tests the same claims against both limits, with every rule and the predictor left unchanged.

## Status at the time of writing

Observed:

- All results from weeks 1 to 3.
- A 4B pipeline smoke test (plan `smoke4b`: 10 training steps, 20 test items), still running when this file was hashed. It checks memory and plumbing only, and its accuracies are not used.

Not observed:

- Any week-4 fine-tune trained for 500 steps.
- Any full-test-set week-4 evaluation.
- The Qwen3-4B base-model KL divergence, which is computed after this file is hashed.

## Design

Training settings are identical to week 3: 500 steps, batch size 4, LoRA rank 8, scale 20, all attention and MLP projections. The format ladder is also identical: bf16, MLX affine group 64 at 6/4/3 bits, and GGUF Q6_K/Q4_K_M/Q3_K_M/Q2_K. So are the evaluation code and the full test sets.

- **Arm S (second seeds):** seed 1 of the four week-3 LoRA configurations.
  - Configs: `olmo1-lora-lowlr-s1`, `olmo1-lora-s1` (OLMo-2 1B, banking77), `mas06-lora-lowlr-s1`, `mas06-lora-s1` (Qwen3-0.6B, MASSIVE).
  - The base-model evaluations are reused from week 3. They are deterministic and belong to the same base model on the same test set; `runs/week4/*-base` are links to them.
- **Arm L (larger model):** Qwen3-4B (Apache-2.0) on banking77.
  - Configs: `q4b-base`, `q4b-lora-lowlr` (LoRA, learning rate 1e-5), `q4b-lora` (LoRA, 1e-4).
  - Gradient checkpointing is on, as it was for 1.7B and OLMo.

**Retention R** follows the definition in `PREREGISTRATION.md`, with the week-2 rule for undefined R: if a fine-tune has no bf16 gain, its points are excluded and reported.

**Mechanism inputs:** NSR and kept come from `src/ftquant/mechanism_stream.py`. It streams tensors because the 4B model does not fit in memory twice. It uses the `ftquant check` implementations:

- GGUF values were bit-identical to the study scripts on a 0.6B check.
- MLX values agreed within 0.0004 in kept and 0.001 in NSR, a difference that comes from how the LoRA update is added in floating point.

KLD_base for Qwen3-4B comes from `kld.py` with the same settings as before. The only change is that the bf16 reference model is freed from memory before the quantized copies load; values are unaffected.

## Hypotheses and criteria (confirmatory)

The predictor is **v2, unchanged**: `runs/predictor/predictor-v2.json`, SHA-256 `6fd78627c73120629847fae574fc2942ef01a81fc37c2e7565371261fa027580`. It is not refitted on week-3 or week-4 data.

**W4-H1 (seed replication):** the same four tests as W3-H1, run on the seed-1 fine-tunes. In each arm, R(LoRA 1e-4) − R(LoRA 1e-5) is tested at MLX 3-bit and at GGUF Q3_K_M.

- Each test is a paired bootstrap over test items (2,000 resamples, seed 0, 95% percentile interval).
- Verdict per test:
  - supported if the interval is above 0;
  - not supported if the interval is below 0 or the point estimate is ≤ 0;
  - inconclusive otherwise, or if either R is undefined.
- Supported overall only if all four tests are.

**W4-H2 (scale):** the same test for Qwen3-4B at MLX 3-bit and at GGUF Q3_K_M. Supported only if both tests are.

**W4-H3 (v2 accuracy):** pool all defined week-4 fine-tune points, 6 fine-tunes × 7 formats. v2 MAE ≤ 0.10 against R clipped to [0, 1], and safe-call agreement ≥ 0.85.

**W4-H4 (v2 beats the baselines):** on the same points, v2's MAE is below both the bits-only and the KLD-only baselines stored in the same frozen file.

Each is reported as supported, not supported, or inconclusive, whatever the outcome. MAE intervals come from resampling whole fine-tunes. v1 is scored as a secondary result.

**Consequence:** if W4-H3 or W4-H4 fails, the README and `ftquant check` say so next to the prediction, and the tested scope is stated with the failing setting. If W4-H2 is not supported, the headline claim is limited to models of 1.7B parameters or fewer.

## Exploratory

- Seed-to-seed spread of R and of bf16 accuracy for each week-3 configuration.
- Whether the gentle fine-tune is again the more accurate one at bf16 (it was in 4 of 4 settings in week 3).
- Accuracy retention and operating-point drift for Qwen3-4B.
- Whether Qwen3-4B's base breaks at MLX 3-bit, as the 0.6B and 1.7B bases did.

## Deviations

- **2026-09-29 05:40 PDT: the Qwen3-4B fine-tunes use micro-batches of 2 with 2-step gradient accumulation.** The first attempt at `q4b-lora-lowlr`, with batch size 4, ran out of GPU memory between steps 350 and 400. Its peak was 11.87 GB against the M1 Pro's 11.84 GB recommended working set, and the log is kept in `runs/week4/_failed-oom-q4b-lora-lowlr/`. Both 4B fine-tunes are therefore trained with micro-batch 2 and 2 accumulation steps: an effective batch of 4 over the same 500 optimizer steps and the same 2,000 training examples. The evaluation and reporting intervals are scaled to match (1,000 micro-iterations, validation every 200, 100 validation items). No evaluation of the failed attempt was run, and its adapter is not used. The two 4B fine-tunes share the same settings, so W4-H2 still compares like with like. All other week-4 runs, and all earlier weeks, used batch size 4 without accumulation. Nothing else changed.
