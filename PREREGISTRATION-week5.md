# ftquant pre-registration (week 5): is a small update rounded away, or outrun by the base?

Written 2026-10-03, about 19:50 PDT, before any week-5 model was trained or evaluated. The SHA-256 of this file is in `runs/week1/PREREG-week5.sha256`.

## Why

The paper draft explains why small updates lose more with weight-space measurements: quantization adds rounding noise that can be larger than a small update, so the update "drowns in the noise or is rounded away". A review of the draft pointed to evidence against that account in our own week-1 data, which the draft never reported. The gentle Qwen3-0.6B LoRA (learning rate 1e-5) was also evaluated unfused, with its update carried exactly in bf16 on top of a quantized base, and it lost about as much as the fused model:

| Qwen3-0.6B LoRA, lr 1e-5 | MLX q4 accuracy | R | MLX q3 accuracy | R |
|---|---|---|---|---|
| fused, then quantized | 69.4% | 0.84 | 4.3% | 0.06 |
| unfused, over the quantized base | 68.4% | 0.83 | 1.3% | 0.02 |

An unfused update is never rounded, so rounding of the update cannot explain this. The competing account is that a small update, trained against the bf16 weights, stops working when the weights it was trained against move by more than it does, whether or not the update itself is quantized. Week 1 cannot separate the two accounts cleanly: at MLX q3 the Qwen3-0.6B base collapses to 0% accuracy, and at q4 the loss is small.

Week 5 tests the competing account on a model whose base stays usable at 3 bits, and tests the remedy it implies: train against the quantized base you will ship.

## Status at the time of writing

Observed:

- All results from weeks 1 to 4, including the week-1 unfused evaluations above and the week-3 and week-4 fused OLMo-2 1B results used below to set the margin of W5-H1.
- Post hoc reanalyses of week-1 to week-4 outputs made while revising the paper. None uses week-5 data.

Not observed:

- Any unfused evaluation of an OLMo-2 1B adapter.
- Any LoRA trained on a quantized base.
- Any GGUF evaluation with a grammar, or with a single llama-server slot.

## Design

Unchanged from earlier weeks unless stated: banking77, the full test set (3,080 items), greedy exact match, MLX affine quantization with group size 64, llama.cpp build 11146 without an importance matrix, R as defined in `PREREGISTRATION.md`, and paired bootstrap intervals over test items (2,000 resamples, seed 0, 95% percentile). Models load offline from the same pinned Hugging Face revisions as earlier weeks (Qwen3-0.6B `c1899de2`, OLMo-2 1B `48d788ec`). The analysis code (`src/ftquant/verdicts_w5.py`) is written before any week-5 result exists.

**Arm U (unfused, OLMo-2 1B).** The four existing OLMo-2 1B banking77 adapters are evaluated unfused on the OLMo-2 1B base quantized with MLX at 4 and 3 bits (variants `mlx-q4-unfused` and `mlx-q3-unfused`), with `eval_task_mlx` loading the adapter on top of the quantized base:

- gentle (learning rate 1e-5): `olmo1-lora-lowlr` (week 3, seed 0) and `olmo1-lora-lowlr-s1` (week 4, seed 1);
- strong (learning rate 1e-4): `olmo1-lora` (week 3) and `olmo1-lora-s1` (week 4).

The quantized bases are made with the same `mlx_lm convert` command as weeks 3 and 4. They are evaluated again as a check. If either accuracy differs from the week-3 `olmo1-base` evaluation, the new evaluation is the reference and the difference is logged as a deviation. Fused results at the same formats exist from weeks 3 and 4 and are not rerun.

**Arm Q (training against the quantized base, Qwen3-0.6B).** Four new LoRAs are trained with mlx-lm on the Qwen3-0.6B base already quantized with MLX at 4 bits and at 3 bits (mlx-lm's QLoRA path): `q06q4-lora-lowlr` and `q06q4-lora` (4-bit base, learning rates 1e-5 and 1e-4), and `q06q3-lora-lowlr` and `q06q3-lora` (3-bit base). Every other training setting matches week 1: 500 steps, batch size 4, rank 8, scale 20, all attention and MLP projections, prompt masked, seed 0. Each adapter is evaluated unfused on the quantized base it was trained on, with `eval_mlx` as in week 1 (variant `mlx-q{b}-qlora`). The regenerated 4-bit and 3-bit Qwen3-0.6B bases are evaluated again as a check, with the same rule as in Arm U. Adapters are not merged, because merging into an already quantized base and re-quantizing is known to delete the adaptation (arXiv 2609.04526).

The share of the bf16 gain that a QLoRA adapter delivers at format v is

S(v) = (acc_qlora(v) − acc_base(v)) / (acc_ft(bf16) − acc_base(bf16)),

where acc_ft(bf16) is the week-1 bf16-trained LoRA with the same learning rate. For that bf16-trained LoRA fused and quantized to v, the same expression is R(v), so S and R share a denominator.

**Arm C (exploratory checks, Qwen3-0.6B, GGUF).**

- C1, repeatability: the week-1 gentle LoRA (`q06-lora-lowlr`) is fused and converted again, quantized to Q3_K_M, and evaluated with llama-server at 4 slots and at 1 slot. Reported: accuracy of each run, per-item agreement between the two, and agreement with the week-1 run.
- C2, label-constrained decoding: the week-1 gentle and strong LoRAs are evaluated at GGUF bf16, Q3_K_M and Q2_K with a GBNF grammar that allows only the 77 intent names. Reported: accuracy, accuracy retention A(v) = acc(v)/acc(bf16) under the grammar, and the change from unconstrained decoding. This separates wrong labels from invalid outputs.

## Hypotheses and criteria (confirmatory)

**W5-H1: an update that is never rounded does not rescue a gentle fine-tune.** For each gentle OLMo adapter (seeds 0 and 1), the paired difference R(unfused) − R(fused) at MLX q3.

MLX q3 is chosen because there the OLMo base stays usable (14.5% at bf16, 6.8% at q3) while the gentle LoRA loses much of its gain: fused R is 0.83 for seed 0 and 0.81 for seed 1, against 1.09 and 1.10 for the strong LoRA. If rounding of the update caused the loss, unfusing should recover most of the missing 0.17 to 0.19.

- Per test: supported if the upper 95% bound is below +0.08 (less than about half of the loss recovered); not supported if the lower bound is above +0.08; inconclusive otherwise.
- Supported overall only if both tests are.

**W5-H2: training against the shipped base removes the loss.** At MLX q4, the paired difference S(q4) − R(q4) between `q06q4-lora-lowlr` on its own 4-bit base and the week-1 `q06-lora-lowlr` fused and quantized to 4 bits (R = 0.84).

- Supported if the 95% interval is above 0; not supported if it is below 0 or the point estimate is ≤ 0; inconclusive otherwise.

MLX q4 is the confirmatory format because the Qwen3-0.6B base is intact there (9.7% at q4 against 7.4% at bf16). At q3 the base collapses to 0%, so the q3 runs are exploratory.

## Exploratory

- Arm U at MLX q4, and the strong OLMo adapters at both formats.
- Arm Q at MLX q3; S for the strong LoRAs; whether the gap between gentle and strong shrinks when both are trained against the quantized base; validation loss of the QLoRA runs against week 1.
- Arm C as described.
- Valid-output rates for every new evaluation.

## Known limitations

- One task (banking77), small models, and one seed in Arm Q.
- QLoRA here means mlx-lm's LoRA on an MLX-quantized base, evaluated unfused. GGUF has no equivalent training path in this study.
- C2 changes the decoding rule, so its accuracies are reported separately and are not compared with the main results.

## Deviations

Any change after hashing is logged here with a timestamp, as in earlier weeks.

- **2026-10-04 00:55 PDT: the Arm Q run `q06q3-lora` stalled and was restarted.** After its step-100 checkpoint (23:05) the training process slowed sharply: steps 100 to 200 took 92 minutes instead of about 5, and one validation pass took 17 minutes instead of about 20 seconds, with its memory footprint near 8 GB and the system swapping. It was stopped at 00:55, after its step-200 checkpoint (00:37). Its losses up to then were normal. Its partial outputs were moved to `runs/week5/_stalled-q06q3-lora/`, and the configuration is retrained from scratch with the same settings after Arm C. Only an exploratory run is affected; W5-H1 and W5-H2 were already computable and are unchanged.
