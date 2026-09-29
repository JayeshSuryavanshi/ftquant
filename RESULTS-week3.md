# ftquant results, week 3

Written 2026-09-28. Design and criteria: `PREREGISTRATION-week3.md`, hashed at 08:40:04 PDT, before any week-3 fine-tune was trained. Numbers come from `runs/week3/verdicts.json` (`python -m ftquant.verdicts_w3`) and `runs/week3/summary.json` (`python -m ftquant.week3`).

## What ran

- **Arm F (new model family):** OLMo-2 1B on banking77. Configs: base, LoRA at learning rate 1e-5, LoRA at 1e-4.
- **Arm T (new task):** Qwen3-0.6B on MASSIVE slot annotation. Configs: base, LoRA at 1e-5, LoRA at 1e-4, full fine-tuning at 1e-5.
- Every config was evaluated at bf16 and 7 quantized formats on the full test set: 63 evaluations, no failures.
- Every fine-tune learned its task, so no point was excluded.

## Pre-registered results

### W3-H1: small updates are more fragile. **Supported (4 of 4 tests).**

| arm | format | R, LoRA 1e-4 | R, LoRA 1e-5 | difference [95% CI] | verdict |
|---|---|---|---|---|---|
| OLMo-2 1B, banking77 | MLX 3-bit | 1.092 | 0.826 | +0.265 [+0.236, +0.294] | supported |
| OLMo-2 1B, banking77 | GGUF Q3_K_M | 1.076 | 1.018 | +0.058 [+0.037, +0.078] | supported |
| Qwen3-0.6B, MASSIVE | MLX 3-bit | 0.621 | 0.319 | +0.301 [+0.271, +0.332] | supported |
| Qwen3-0.6B, MASSIVE | GGUF Q3_K_M | 0.946 | 0.843 | +0.102 [+0.078, +0.127] | supported |

OLMo values above 1 come from base collapse, as at 1.7B in week 1: the OLMo base drops from 14.5% to 6.8% at MLX 3-bit. In accuracy retention, which does not depend on the base, the OLMo pairs are 0.97 vs 0.76 (MLX 3-bit) and 1.00 vs 0.95 (Q3_K_M). The OLMo gap at Q3_K_M is small but its interval excludes zero.

### W3-H2 and W3-H3: predictor v2. **Both supported.**

The model is frozen v2: R̂ = sigmoid(5.612 + 3.296 ln(kept / NSR) − 1.555 ln KLD_base), SHA-256 `6fd78627...`. All 35 week-3 points are pooled.

| model | MAE [95% CI, resampling fine-tunes] | safe-call agreement |
|---|---|---|
| **v2 (kept/NSR, base KLD)** | **0.059 [0.028, 0.090]** | **0.943** |
| v1 (NSR, base KLD), secondary | 0.071 [0.029, 0.107] | 0.943 |
| KLD-only baseline | 0.094 [0.063, 0.123] | 0.800 |
| bits-only baseline | 0.128 [0.081, 0.182] | 0.743 |

- **W3-H2 (MAE ≤ 0.10 and agreement ≥ 0.85): supported.**
- **W3-H3 (below both baselines): supported.**

Exploratory breakdowns:

- Without the easy 6-bit formats (25 points), v2's MAE is 0.080 against 0.097 (v1), 0.130 (KLD-only) and 0.177 (bits-only).
- By arm, v2's MAE is 0.053 on OLMo and 0.063 on MASSIVE.

The largest v2 errors:

| config | format | predicted | measured | note |
|---|---|---|---|---|
| OLMo, LoRA 1e-5 | Q2_K | 15% | 58% | |
| MASSIVE, LoRA 1e-4 | MLX 3-bit | 98% | 62% | low noise (1.60), but the base breaks (KLD 1.21) |
| MASSIVE, LoRA 1e-4 | Q2_K | 81% | 62% | |
| MASSIVE, full 1e-5 | MLX 3-bit | 17% | 35% | |

The pattern: when the base model breaks, the slot-annotation task loses more than a small noise ratio suggests. Classification on OLMo loses less than a large noise ratio suggests.

**Pre-registered consequence:** `ftquant check` and `check-gguf` now show v2's predicted retention, but only for the formats it was tested on and for bases with a measured base damage (Qwen3 0.6B and 1.7B, OLMo-2 1B). Other cells print n/a.

## Exploratory

### The same pattern in all four settings

Accuracy kept relative to each model's own bf16 accuracy, LoRA 1e-5 vs LoRA 1e-4:

| setting | bf16 accuracy | MLX 4-bit | MLX 3-bit | Q4_K_M | Q3_K_M | Q2_K |
|---|---|---|---|---|---|---|
| Qwen3-0.6B, banking77 | 78.5% vs 73.5% | 0.89 vs 0.98 | 0.06 vs 0.85 | 0.95 vs 1.00 | 0.79 vs 0.99 | 0.00 vs 0.78 |
| Qwen3-1.7B, banking77 | 77.2% vs 73.4% | 0.91 vs 0.96 | 0.32 vs 0.93 | 0.96 vs 1.00 | 0.71 vs 0.98 | 0.00 vs 0.85 |
| OLMo-2 1B, banking77 | 78.3% vs 76.8% | 0.94 vs 0.99 | 0.76 vs 0.97 | 0.99 vs 1.00 | 0.95 vs 1.00 | 0.55 vs 0.95 |
| Qwen3-0.6B, MASSIVE | 65.3% vs 62.6% | 0.93 vs 1.01 | 0.32 vs 0.62 | 0.98 vs 0.99 | 0.84 vs 0.95 | 0.00 vs 0.62 |

In every setting, the gentler LoRA had the higher bf16 accuracy and kept less at every format of 4 bits or fewer. That makes 20 of 20 comparisons in the same direction. On MASSIVE, full fine-tuning at 1e-5 was the most accurate model at bf16 (69.0%) and kept 35% of its accuracy at MLX 3-bit. The bf16 ranking is the week-3 point estimate from one seed. Week 2 showed that bf16 accuracy can move by several points between seeds, while retention barely moves.

How strongly this shows up depends on the family. OLMo-2's base loses much less at 3-bit than Qwen3's (base KLD at MLX 3-bit 0.66 vs 1.21), so its gentle LoRA keeps 76% where Qwen3-0.6B's keeps 6%.

### Other questions from the pre-registration

- **Generation vs classification:** mixed. On Qwen3-0.6B at MLX 3-bit, the strong LoRA kept less on MASSIVE than on banking77 (0.62 vs 0.85). The gentle LoRA kept more (0.32 vs 0.06). A multi-token output is not simply more fragile.
- **Rounded away:** no week-3 fine-tune entered the regime (kept < 60%). The closest was MASSIVE full fine-tuning at MLX 3-bit, with 74% of its update kept.
- **Operating-point drift:** coverage still falls faster than accuracy. OLMo LoRA 1e-5 at MLX 3-bit keeps 76% of its accuracy but auto-accepts 33.5 points fewer items at the full-precision threshold. MASSIVE full fine-tuning at MLX 4-bit keeps 89% and accepts 12.7 points fewer.

## Figures

- `figures/replication.png`: accuracy kept at MLX 3-bit and GGUF Q3_K_M, gentle vs strong LoRA, in all four settings.
- `figures/predictor-v2-week3.png`: v2's pre-quantization predictions against the week-3 measurements.

## Limits

- Five fine-tunes (35 points) in the week-3 test, one seed each. The upper end of v2's MAE interval (0.090) sits close to the 0.10 bar.
- Models of 1.7B parameters or fewer, LoRA rank 8, MLX group size 64 only, and GGUF without an importance matrix.
- Both tasks have short outputs.
- Base damage has to be measured once per base model (about 2 minutes for a 1B model). The tool ships values for three bases.
