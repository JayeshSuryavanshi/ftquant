# ftquant pre-registration (week 1)

Written 2026-09-27, about 17:45 PDT. The SHA-256 of this file is recorded in `runs/week1/PREREG.sha256` at the time of writing. Any later change is logged under Deviations with a date and a reason.

## Status at the time of writing

What had been observed:

- Dry runs on 200 test items, used only to test the pipeline: base Qwen3-1.7B at bf16 scored 25.5%; base Qwen3-0.6B scored 5.5%; adapters trained for 20 steps scored near 0%. On the same 20-step fused model, MLX bf16 and GGUF bf16 agreed on 196 of 200 predictions.
- Base Qwen3-0.6B on the full 3,080-item test set, read from the sweep log: MLX bf16 0.0744, q8 0.0854, q6 0.0630, q2 0.0000; GGUF bf16 0.0744, Q8_0 0.0731, Q6_K 0.0653, Q4_K_M 0.0977, Q3_K_M 0.0555.

What existed on disk but had NOT been read:

- Fine-tuned Qwen3-0.6B results (`q06-lora`, `q06-lora-lowlr`, and `q06-full` in progress). Only file names and completion counts were looked at.

Because those results already existed, **all Qwen3-0.6B analyses are exploratory**. The **Qwen3-1.7B runs (`q17-base`, `q17-lora`) had not started** when this was written, and they are the **confirmatory** test of H1 to H4 below.

## Setup (frozen for week 1)

- Task: banking77, the original PolyAI CSVs (train sha256 `b06e26ac...`, test sha256 `d12d6e3b...`). The model is fine-tuned on 9,504 training items; 499 are held out for validation (stratified, seed 0). Evaluation uses all 3,080 test items, 40 per class.
- Prompt: an instruction, all 77 intent names, then the message. The Qwen3 chat template is applied with thinking disabled. Decoding is greedy with at most 16 new tokens. The prediction is the decoded text, stripped; it is correct only if it exactly equals the gold intent name.
- Confidence: exp(sum of float32 log-probabilities of the greedy-decoded label tokens).
- Fine-tuning (mlx-lm 0.31.3): 500 iterations, batch size 4, mask prompt, seed 0, all layers.
  - LoRA: rank 8, scale 20, on q/k/v/o/gate/up/down.
  - Configs: `lora` (learning rate 1e-4), `lora-lowlr` (1e-5), `full` (1e-5). The 1.7B model is trained with `lora` only.
- Quantization:
  - MLX affine with group size 64 at 8, 6, 4, 3 and 2 bits (fused, then quantized).
  - LoRA configs also get the unfused adapter over an MLX-quantized base at 8, 4 and 3 bits.
  - GGUF (llama.cpp build 11146, commit 7fe450e19): bf16, then Q8_0, Q6_K, Q4_K_M, Q3_K_M and Q2_K, quantized from the bf16 GGUF with no imatrix.
- Engines: MLX through mlx-lm, and GGUF through llama-server (n_parallel 4, prompt cache on). Both engines receive identical token IDs.

## Metrics

- **Gain retention** for variant v:
  R(v) = (acc_ft(v) − acc_base(v)) / (acc_ft(fp) − acc_base(fp))
  - fp is bf16 in the same engine as v.
  - acc_base(v) is the base model at the same quantization. For unfused variants, the reference is the base model quantized to the same bits.
- **Uncertainty**: a paired bootstrap over the 3,080 test items (2,000 resamples, seed 0) gives a 95% percentile CI. Differences between two variants use the same paired resamples.
- **Noise floor**: the MLX bf16 vs GGUF bf16 prediction agreement and accuracy difference on the same fused model. A difference between variants smaller than this noise floor is not interpreted as an effect.
- **Operating-point drift**:
  1. A confidence gate accepts a prediction when confidence ≥ τ.
  2. τ is the lowest threshold at which the full-precision fine-tuned model's error among accepted items is ≤ 5%, fitted on the even-indexed test items.
  3. On the odd-indexed items, error among accepted and coverage are reported at full precision and for each quantized variant, keeping the same τ.
  4. Drift is the change in error among accepted (percentage points) and the change in coverage.
  5. In week 1, τ is fitted on half of the test set; later weeks fit it on the validation split.

## Confirmatory hypotheses (tested on Qwen3-1.7B `lora` only)

- **H1: little loss at high precision.** R ≥ 0.95 at MLX q8, MLX q6, GGUF Q8_0 and GGUF Q6_K, with every lower 95% CI bound ≥ 0.90.
- **H2: where it breaks.** At 3 bits (MLX q3 and GGUF Q3_K_M), R is lower than at 4 bits (MLX q4 and GGUF Q4_K_M respectively), with the upper 95% CI bound at 3 bits < 0.95. At 2 bits (MLX q2, GGUF Q2_K), R < 0.5.
- **H3: fused vs unfused.** At MLX q4 and q3, R(unfused adapter over the quantized base) > R(fused, then quantized), and the 95% CI of the paired difference excludes 0.
- **H4: the operating point moves before accuracy does.** At 4 bits (MLX q4 and GGUF Q4_K_M), the fixed-τ gate's error among accepted rises by ≥ 1.0 percentage point, or its coverage changes by ≥ 3 percentage points, relative to full precision. This holds even for a variant where R ≥ 0.95.

Each hypothesis is reported as supported, not supported, or inconclusive (a CI straddles the decision boundary). Every result is reported whichever way it falls, including nulls. For example, "LoRA gains survive to 3 bits" would be a finding.

## Exploratory (Qwen3-0.6B, and anything not listed above)

- The effect of fine-tune type at matched precision: `lora` vs `lora-lowlr` vs `full`. The prior expectation is that smaller weight updates (low learning rate, full fine-tuning at 1e-5) lose more at 4 bits and below, following the delta-vs-quantization-step argument in mlx-lm PR #1564 and arXiv 2609.04526.
- MLX affine vs GGUF K-quants at matched effective bits per weight.
- The pre-quantization predictor (week 3): the per-layer ratio of fine-tune delta to quantization step, against a margin-shrinkage baseline (arXiv 2608.06564).

## Known limitations

- Week 1 uses a single training seed and a single task. A second task, more seeds and a validation-fitted τ come in later weeks.
- Greedy exact match rewards format adherence, so invalid outputs count as wrong. The rate of invalid outputs is reported alongside accuracy.
- GGUF is quantized without an imatrix. Q4_K_M and Q3_K_M mix bit widths across tensors, so bits-per-weight comparisons use measured file sizes.

## Deviations

- **2026-09-27 ~18:00 PDT: analysis code clarified, before any 1.7B fine-tune result existed.** The first analysis draft fitted τ by stepping down from the most confident item and stopping at the first threshold where error exceeded 5%. The written definition above ("the lowest threshold at which ... error among accepted items is ≤ 5%") is the largest accepted set whose error is ≤ 5%. `analyze.py` now implements the written definition exactly. The metric definition is unchanged; only the code now matches it. The only results seen under either version were the exploratory Qwen3-0.6B ones.
- **2026-09-29 16:29 PDT, found in the week-4 audit: τ was never moved to the validation split.** Item 5 of the operating-point definition says later weeks fit τ on the validation split. Weeks 2 to 4 kept the week-1 rule and fitted τ on the even-indexed test items, and their evaluations were run on the test split only. This affects only the operating-point numbers (coverage and error among accepted items), which were exploratory in every week after week 1. They are reported as computed, with this note, rather than recomputed. Week 1's confirmatory H4 used the week-1 rule as written and is not affected.
