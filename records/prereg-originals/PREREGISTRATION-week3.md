# ftquant pre-registration (week 3): a new model family, a new task, and predictor v2

Written 2026-09-28, about 08:40 PDT, before any week-3 fine-tune was trained at full length or evaluated on a full test set. The SHA-256 of this file is in `runs/week1/PREREG-week3.sha256`.

## Why

Weeks 1 and 2 used one model family (Qwen3) and one task (banking77). Two things need a test on data that did not shape them:

1. The finding that small-update fine-tunes lose most of their gain at 3-bit.
2. A predictor. The frozen v1 predictor missed its accuracy bar (P1, `RESULTS-week2.md`), and the analysis of why suggested a better input. v2 is that input, chosen and frozen below, before any week-3 result exists.

## Status at the time of writing

These things had been observed:

- All week-1 and week-2 results (`RESULTS-week1.md`, `RESULTS-week2.md`).
- A pipeline smoke test (plan `smoke` in `src/ftquant/sweep.py`, output in `runs/smoke/`). It trains 30-step fine-tunes and evaluates them on 40 test items, only to check that training, fusing, conversion and both engines run. Its accuracies are not used for anything.
- Zero-shot OLMo-2 1B on the first 60 banking77 test items: 15% on MLX and on GGUF, with 56 of 60 predictions identical across engines.
- Zero-shot Qwen3-0.6B on the first 40 MASSIVE test items: 0 exact matches.

These things had NOT been observed:

- Any week-3 fine-tune trained for 500 steps, or any full-test-set week-3 evaluation.
- The OLMo-2 base-model KL divergence (computed after this file is hashed; it describes only the base model).

## Design

Two arms, 500 training steps each, batch size 4, LoRA rank 8, scale 20, on all attention and MLP projections, with the same training code as weeks 1 and 2.

- **Arm F (new family):** `allenai/OLMo-2-0425-1B-Instruct` (Apache-2.0) on banking77, with the same splits and prompt as before.
  - Configs: `olmo1-base`, `olmo1-lora-lowlr` (LoRA, learning rate 1e-5), `olmo1-lora` (LoRA, 1e-4).
  - Full fine-tuning is left out because it does not fit in 16 GB at this size.
- **Arm T (new task):** Qwen3-0.6B on MASSIVE 1.1 en-US slot annotation (CC BY 4.0).
  - Source: Amazon's original tarball, sha256 `4cba5faa...`.
  - Splits: train 11,514, dev 2,033 (used for validation), test 2,974.
  - The model rewrites the request with every slot marked `[slot_type : words]`, and is scored by whitespace-normalized exact match (`src/ftquant/tasks.py`).
  - Configs: `mas06-base`, `mas06-lora-lowlr` (LoRA 1e-5), `mas06-lora` (LoRA 1e-4), `mas06-full` (full fine-tuning 1e-5).

**Formats (the week-2 ladder):** bf16, MLX affine group 64 at 6, 4 and 3 bits, and GGUF Q6_K, Q4_K_M, Q3_K_M and Q2_K. Engines and versions are unchanged. Evaluation code is `eval_task_mlx.py` / `eval_task_gguf.py`, which matched the week-1 evaluator on 40 of 40 banking77 items. Every configuration, including the base, is evaluated on the full test set in every format.

**Retention R** is the gain retention defined in `PREREGISTRATION.md`: the quantized fine-tune's gain over the equally quantized base, divided by the bf16 gain, with the same engine. If a fine-tune's bf16 gain is not positive, its R is undefined. Its points are then excluded from every test and reported, as in week 2.

## Predictor v2 (frozen before this file was hashed)

- **Form:** R̂ = sigmoid(t0 + t1·ln(kept / NSR) + t2·ln KLD_base).
  - `kept` is the share of the update's projection that survives quantization (`signal_retained`).
  - NSR is the noise-to-update ratio from week 2.
  - kept / NSR is the signal-to-noise ratio of the quantized update. It catches the "rounded away" failure that NSR alone misses.
- **Selection:** four forms were compared by leave-one-fine-tune-out error on all 89 week-1 and week-2 points, using `python -m ftquant.predictor_v2 select`.

  | form | MAE | safe agreement |
  |---|---|---|
  | v1 form (NSR, KLD) | 0.095 | 0.899 |
  | **kept/NSR, KLD** | **0.071** | **0.933** |
  | NSR, kept, KLD | 0.072 | 0.921 |
  | kept/NSR × KLD interaction | 0.075 | 0.921 |
  | KLD only | 0.160 | 0.798 |
  | bits only | 0.157 | 0.798 |

  The simplest form with the lowest error was chosen. On the week-3 formats only, its leave-one-out MAE was 0.107.
- **Fit:** on all 89 points, same least-squares procedure as v1.
  - θ = (5.612, 3.296, −1.555).
  - File `runs/predictor/predictor-v2.json`, frozen 2026-09-28 08:39:25 PDT, SHA-256 `6fd78627c73120629847fae574fc2942ef01a81fc37c2e7565371261fa027580`.
  - Baselines fitted on the same 89 points and stored in the same file: KLD-only, and bits-only (the mean retention per format).
- **Inputs for the tests:**
  - NSR and kept come from `mechanism.py` and `mechanism_gguf.py` on the week-3 fine-tunes.
  - KLD_base for OLMo-2 1B comes from `kld.py` with the same settings as weeks 1 and 2 (8 × 512 WikiText-2 tokens).
  - The MASSIVE arm uses the Qwen3-0.6B KLD already measured, because KLD describes the base model and not the task.

## Hypotheses and criteria (confirmatory)

**W3-H1: small updates are more fragile.** In each arm, the LoRA at 1e-4 retains more of its gain than the LoRA at 1e-5. This is tested separately at MLX 3-bit and at GGUF Q3_K_M, giving 4 tests.

- Each test is a paired bootstrap over test items (2,000 resamples, seed 0, 95% percentile interval) of R(1e-4) − R(1e-5).
- Verdict per test:
  - supported if the interval is above 0;
  - not supported if the interval is below 0, or if the point estimate is ≤ 0;
  - inconclusive otherwise, or if either R is undefined.
- W3-H1 as a whole is supported only if all 4 tests are.

**W3-H2: v2 is accurate.** Pool every defined week-3 point, both arms and all 7 quantized formats. The criterion is v2 MAE ≤ 0.10 (against R clipped to [0, 1]) and safe-call agreement (R ≥ 0.9) ≥ 0.85, the same bar as P1.

**W3-H3: v2 beats the baselines.** On the same pooled points, v2's MAE is lower than both the bits-only and the KLD-only baselines.

Each is reported as supported, not supported, or inconclusive, whatever the outcome. MAE intervals are reported from resampling whole fine-tunes. The frozen v1 predictor is also scored on the same points, as a secondary result.

**Consequence:** if W3-H2 and W3-H3 are both supported, `ftquant check` may show v2's predicted retention, labelled with the models, tasks and formats it was tested on. Otherwise it stays descriptive.

## Exploratory

- Accuracy retention (acc_ft(q) / acc_ft(bf16)) for every configuration.
- Operating-point drift at the full-precision confidence threshold, as in weeks 1 and 2.
- Full fine-tuning vs LoRA on MASSIVE.
- Whether any week-3 fine-tune enters the "rounded away" regime (kept < 60%).
- Whether a generation task (slot annotation, about 20 output tokens) degrades faster than a single-label answer at the same format.
