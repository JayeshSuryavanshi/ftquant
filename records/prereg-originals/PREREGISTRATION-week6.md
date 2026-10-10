# ftquant pre-registration (week 6): the GGUF remedy, mlx-lm's own defaults, and an importance matrix

Written 2026-10-08, about 19:43 PDT, before any week-6 model was trained or evaluated; the plumbing checks and smoke runs that preceded it are listed under Status. The SHA-256 hashes of this file and of the analysis code, `src/ftquant/verdicts_w6.py`, are in `runs/week1/PREREG-week6.sha256`.

## Why

Round 5 found that, in MLX, a gentle LoRA loses its gain because the weights it was trained against move under quantization, not because its update is rounded, and that training on the quantized base removes most of the loss. The paper says GGUF "has no such training path in our pipeline and was not tested". GGUF is the format most local deployments use, and llama.cpp can apply a LoRA adapter at load time on top of a quantized base, so the remedy can be tested there: train the adapter against the exact weights the quantized file ships, and serve it unmerged.

Two further gaps are tested at the same time. The paper calls the gentle learning rate "mlx-lm's default", but mlx-lm's other defaults differ from the study's setup: 1,000 steps instead of 500, LoRA on the last 16 layers instead of all, and no prompt masking. And every GGUF file so far was quantized without an importance matrix, which llama.cpp recommends at 3 bits and below.

The week-1 numbers this round builds on (banking77, GGUF, accuracy in %; the base files are the week-1 ones, see Design for how week 6 builds them):

| Qwen3-0.6B | bf16 | Q4_K_M | Q3_K_M | Q2_K | R at Q3_K_M | R at Q2_K |
|---|---|---|---|---|---|---|
| base | 7.4 | 9.8 | 5.6 | 0.0 | | |
| LoRA 1e-5 (gentle), fused | 78.5 | 74.6 | 62.1 | 0.0 | 0.80 | 0.00 |
| LoRA 1e-4 (strong), fused | 73.3 | 73.0 | 72.2 | 57.2 | 1.01 | 0.87 |

## Status at the time of writing

Observed:

- All results from weeks 1 to 5, and post hoc reanalyses of them.
- On 2026-10-07 the evaluation code gained batched MLX decoding (`--batch`) and a llama-server RAM-cache limit (`--cache-ram`), and on 2026-10-08 load-time adapters (`--lora`, `--lora-scale`) and the validation split. The first two were checked against stored predictions: on banking77, batched MLX evaluation agreed with the stored per-item runs on 97.5% (Qwen3-0.6B base, bf16), 99.0% (week-1 strong LoRA on the 4-bit base) and 98.6% (Qwen3-4B base, 4-bit) of items, and on 88.9% for the Qwen3-0.6B base on MASSIVE (0% accuracy either way); GGUF evaluation of the Qwen3-0.6B base at Q4_K_M with `--cache-ram 0` agreed with the stored run on 99.5% at 4 slots and at 16. Two of these runs used week-6 settings: the Qwen3-0.6B base at MLX bf16 with `--batch 32` (7.56%) and at Q4_K_M with 4 slots and `--cache-ram 0` (9.84%, week-1 base file).
- A review of this plan found that the Qwen3-0.6B and Qwen3-1.7B base checkpoints store a separate copy of the output head although their embeddings are tied, so the base GGUF files of weeks 1 to 5 quantized the input embedding at the file's base type (3-bit at Q3_K_M), while every fused fine-tune shares one embedding that llama-quantize keeps at 6 bits. On 2026-10-07 the Qwen3-0.6B base was rebuilt both ways and its accuracy and base damage measured at Q4_K_M, Q3_K_M and Q2_K (NOTEBOOK.md). Qwen3-4B and OLMo-2 1B are not affected.
- The review also converted a week-1 adapter to a GGUF LoRA (392 float32 tensors, alpha 160, maximum difference from mlx-lm's scaled update 0), quantized the base to Q4_K_M, Q3_K_M and Q2_K, and dequantized tensors. Nothing was evaluated and no adapter was loaded into a server.

- On 2026-10-08, before hashing, plumbing checks ran on validation items only, with no outputs read (NOTEBOOK.md): adapter conversion check (a) on the week-1 gentle adapter and on a 5-step smoke adapter (both passed), a tensor-mapping check (a model rebuilt from the lossless bf16 GGUF reproduced all 310 tensors bit for bit), the rebuild acceptance test on the base Q3_K_M file (passed), 5 training steps on that rebuilt model, and llama-server running the week-1 adapter on the base Q3_K_M at scales 1 and 0 on 20 validation items. Two smoke runs of the whole pipeline also ran (the first on code revised since), with 20 validation items per evaluation, 10 training steps and a 4-chunk importance matrix; their outputs are not used. Their logs print accuracies on those 20 items. These were not examined, except three exploratory Q2_K lines from the first run seen in passing: the strong LoRA with an importance matrix, and both week-1 LoRAs applied at load time on the base Q2_K file.

Not observed: any LoRA applied at load time in llama.cpp on test items; any fine-tuned model evaluated with week-6 settings; any LoRA trained against dequantized GGUF weights, with mlx-lm's default layer count, or without prompt masking; any GGUF file quantized with an importance matrix.

## Design

Unchanged from earlier weeks unless stated: banking77, the full test set (3,080 items), greedy exact match, the pinned model revisions (Qwen3-0.6B `c1899de2`, OLMo-2 1B `48d788ec`), llama.cpp build 11146, MLX affine quantization with group size 64, and paired bootstrap intervals over test items (2,000 resamples, seed 0, 95% percentile). Gain retention R is defined as in `PREREGISTRATION.md`. The analysis code, `src/ftquant/verdicts_w6.py`, is written and hashed with this plan.

Evaluation path. Every number used in a week-6 test comes from a week-6 evaluation, including the comparison models and the bf16 references, so compared conditions share one path: GGUF with llama-server at 4 slots and `--cache-ram 0`; MLX with `eval_mlx --batch 32`. A file used by several tests is evaluated once and that evaluation is shared.

Base files. Every Qwen3-0.6B GGUF file in week 6, base included, is converted from a checkpoint without the separate output head, so its one embedding serves input and output as in the fused fine-tunes, and llama-quantize stores it at Q6_K, or at Q8_0 in the Q8_0 file. One base file per format serves every arm; their SHA-256 hashes are recorded. To fit the laptop's disk, large intermediate files are deleted after use and rebuilt when needed; llama-quantize is deterministic, so a rebuilt base file must match its recorded hash, and a mismatch is logged. OLMo-2 1B does not tie its embeddings and is built as before.

**Arm G (GGUF remedy, Qwen3-0.6B).** Three ways to ship the week-1 gentle LoRA (`q06-lora-lowlr`) at Q3_K_M:

- G1, fused: the week-1 adapter, fused into the bf16 base, converted, quantized and evaluated in week 6, as in week 1.
- G2, load-time adapter: the same adapter converted to a GGUF LoRA and applied by llama-server (`--lora`, scale 1) on the base Q3_K_M file. Its update is stored in float32 and never quantized.
- G3, trained against the shipped weights: a new gentle LoRA trained with mlx-lm on a bf16 model whose weights are the dequantized tensors of the base Q3_K_M file (every tensor, including the shared embedding), then converted and applied at load time on that same file. Every other training setting matches week 1: learning rate 1e-5, 500 steps, batch size 4, rank 8, scale 20, all attention and MLP projections, prompt masked, seed 0.

As exploratory arms, the same three are run for the strong LoRA (learning rate 1e-4), and all six at Q2_K. G3 therefore trains four new LoRAs: gentle and strong, against the Q3_K_M and the Q2_K base.

For G2 and G3 the share of the bf16 gain delivered at format v is

S(v) = (acc_arm(v) − acc_base(v)) / (acc_ft(bf16) − acc_base(bf16)),

where acc_ft(bf16) is the week-1 LoRA of the same learning rate, fused and evaluated at GGUF bf16 in week 6. For G1 the same expression is R(v), so G1 to G3 share one base evaluation and one denominator, and S − R is an accuracy difference scaled by a common constant. The difference S_G3 − S_G2 is reported as a pre-specified contrast.

Adapter conversion. mlx-lm stores `lora_a` (in × r) and `lora_b` (r × out) and adds scale · (x · lora_a) · lora_b. The adapter is rewritten in PEFT form (lora_A = lora_aᵀ, lora_B = lora_bᵀ, lora_alpha = 20 × 8 = 160) and converted with `convert_lora_to_gguf.py --outtype f32`; llama.cpp applies alpha / rank = 20. Two checks:

- (a) Before any G2 or G3 evaluation: for all 196 adapted modules, (alpha / rank) · B · A read back from the GGUF LoRA equals 20 · lora_bᵀ · lora_aᵀ to float32 precision. A failure is fixed and logged as a deviation.
- (b) After hashing, on the 499 validation items: at GGUF bf16 and at Q8_0 (a quantized base, as a positive control), the load-time adapter's accuracy is within 1.0 point of the fused model's, and their agreement is reported. As a load-only check, the adapter loaded at scale 0 (`--lora-scaled <file>:0`, which llama.cpp loads but does not apply) on the GGUF bf16 base agrees with the base model's own run on at least 99% of items, with accuracy within 1.0 point. Check (b) passes if all three conditions hold. If (a) passes and (b) fails, the cause is logged and G2 and G3 still run.

Rebuilt training model. Before each G3 training run, every tensor of the rebuilt bf16 model must equal gguf-py's dequantization of the same quantized file after rounding to bf16. Otherwise the rebuild is fixed first.

**Arm D (mlx-lm's defaults, Qwen3-0.6B).** Two new LoRAs whose `train.yaml` sets only the model, data, `train`, adapter path, seed (0) and learning rate, so every other setting is an mlx-lm 0.31.3 default: 1,000 steps, batch size 4, LoRA on the last 16 layers (the same seven projections), rank 8, scale 20, no prompt masking. `q06-def-lowlr` uses 1e-5 (the default) and `q06-def` uses 1e-4. The training files are week 1's. Each is fused and evaluated at MLX bf16, 4-bit and 3-bit and at GGUF bf16, Q4_K_M, Q3_K_M and Q2_K. Without masking, the validation loss covers every prompt token, so it is not comparable with earlier runs.

**Arm I (importance matrix, Qwen3-0.6B and OLMo-2 1B).** For the base, the gentle and the strong LoRA of each model (week-1 `q06-lora-lowlr` and `q06-lora`; week-3 `olmo1-lora-lowlr` and `olmo1-lora`), an importance matrix is computed from the model's own GGUF bf16 file on WikiText-2 (`wikitext-2-raw-v1`, train split, revision `b08601e0`, rows joined with no separator as UTF-8):

    llama-imatrix -m <bf16.gguf> -f <text> -c 512 --chunks 128 -ngl 99 -o <model>.imatrix.gguf
    llama-quantize --imatrix <model>.imatrix.gguf <bf16.gguf> <out.gguf> Q3_K_M   (and Q2_K)

The matrix covers the transformer blocks' matrices, not the embedding or output head. The same models are quantized without a matrix in week 6, so its effect is measured within week 6.

## Hypotheses and criteria (confirmatory)

**Precondition for W6-H1 and W6-H2:** the upper 95% bound of the week-6 R_G1 at Q3_K_M is below 0.90, so there is a loss to recover. Otherwise both are reported as not testable.

**W6-H1: training against the shipped GGUF weights recovers most of the loss.** At Q3_K_M, for the gentle LoRA, the paired difference S_G3 − (1 + R_G1)/2, the margin by which G3 exceeds halfway between the fused result and full retention. Supported if its lower 95% bound is above 0; not supported if its upper bound is below 0; inconclusive otherwise.

**W6-H2: a load-time adapter recovers less than half of the loss.** At Q3_K_M, for the gentle LoRA, the paired difference S_G2 − (1 + R_G1)/2. Supported if its upper 95% bound is below 0; not supported if its lower bound is above 0; inconclusive otherwise.

**W6-H3: mlx-lm's defaults show the same pattern.** For the Arm D pair, the paired difference R(1e-4) − R(1e-5) at GGUF Q3_K_M and at MLX 4-bit. R is computed only if both LoRAs gain at least 35 points over the base (about half the week-1 gains of 71 and 66 points) at both MLX bf16 and GGUF bf16; otherwise W6-H3 is not testable and the gains are reported. Per test: supported if the 95% interval is above 0; not supported if it is below 0; inconclusive otherwise. Overall: not testable if neither test is; supported if both tests are; not supported if either is; inconclusive otherwise.

**W6-H4: an importance matrix removes less than half of the gap.** For each model, Δ_no and Δ_im are R(strong) − R(gentle) at Q3_K_M from the week-6 files without and with the matrix, resampled jointly. A model is testable only if the lower 95% bound of Δ_no is above 0. Per model: supported if the lower 95% bound of Δ_im − Δ_no/2 is above 0; not supported if its upper bound is below 0; inconclusive otherwise. Overall: not testable if neither model is; supported if both models are; not supported if either is; inconclusive otherwise.

## Rules

- A training run that fails, is interrupted, stalls (below 0.1 iterations per second over 30 minutes) or, for a run with the prompt masked (Arm G), ends with a validation loss of 1.0 or more or not a number is set aside and retrained once from scratch with identical settings; partial runs are not used, and attempts are counted across restarts of the runner. A second failure makes its test not testable. Arm D runs, whose loss covers the prompt, are judged by W6-H3's gain floor instead.
- A statistic whose point estimate or any bootstrap resample is undefined (a bf16 gain that is not positive) makes its test not testable.
- The runner stops if llama.cpp is not build 11146 or mlx-lm is not 0.31.3, if an Arm D adapter's saved configuration does not show mlx-lm's defaults, or if a rebuilt base file does not match its recorded hash.
- Confirmatory runs finish before exploratory ones.
- No step starts with less than 10 GB of free disk.

## Exploratory

- Arm G for the strong LoRA, and every Arm G file at Q2_K, where the base breaks.
- Arm D at MLX 3-bit and the other formats, the five accuracy-retention cells of the Arm D pair, their update norms, and the bf16 gains whether or not W6-H3 is testable.
- Arm I at Q2_K, and the change in each model's R with and without the matrix.
- Valid-output rates for every new evaluation, per-item agreement of G2 and G3 with G1, and wall-clock time per step.

## Known limitations

- One task, models of 1B parameters or fewer, and one seed per new LoRA.
- The G3 training model holds the dequantized weights in bf16. Rounding them moves the adapted projections by about 1.0 in Frobenius norm, about 11% of the gentle update's norm (9.42), so G3 trains against weights close to, but not exactly, those llama.cpp computes with.
- WikiText-2 is general text; an importance matrix from task text could behave differently.
- Week 6 uses the batched MLX path, `--cache-ram 0` and new base files, so week-6 numbers are compared only with each other.

## Deviations

None yet. Any change after hashing is logged here with a timestamp, as in earlier weeks.
