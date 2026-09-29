# ftquant results, week 4

Written 2026-09-29. Design and criteria: `PREREGISTRATION-week4.md`, hashed at 18:11:07 PDT on 2026-09-28, before any week-4 fine-tune was trained at full length. Numbers come from `runs/week4/verdicts.json` (`python -m ftquant.verdicts_w4`) and `runs/week4/summary.json` (`python -m ftquant.week4`).

## What ran

- **Arm S (second seeds):** seed 1 of the four week-3 LoRAs. OLMo-2 1B on banking77 and Qwen3-0.6B on MASSIVE, each at learning rates 1e-5 and 1e-4. The base-model evaluations are the week-3 ones.
- **Arm L (larger model):** Qwen3-4B on banking77. Configs: base, LoRA at 1e-5, LoRA at 1e-4.
- Every config was evaluated at bf16 and 7 quantized formats on the full test set: 63 evaluations, no failures.
- Every fine-tune learned its task, so no point was excluded.

Deviation: the two 4B fine-tunes were trained with micro-batches of 2 and 2-step gradient accumulation, after the first attempt at batch size 4 ran out of GPU memory. The effective batch, the 500 optimizer steps and the 2,000 training examples are unchanged (details in `PREREGISTRATION-week4.md`). Two interruptions, a low-battery sleep and a stopped pipeline, cost time but changed no result (`NOTEBOOK.md`).

## Pre-registered results

### W4-H1: the week-3 result with a second seed. **Supported (4 of 4 tests).**

| arm | format | R, LoRA 1e-4 | R, LoRA 1e-5 | difference [95% CI] | verdict |
|---|---|---|---|---|---|
| OLMo-2 1B, banking77, seed 1 | MLX 3-bit | 1.105 | 0.813 | +0.292 [+0.263, +0.322] | supported |
| OLMo-2 1B, banking77, seed 1 | GGUF Q3_K_M | 1.064 | 1.002 | +0.061 [+0.042, +0.079] | supported |
| Qwen3-0.6B, MASSIVE, seed 1 | MLX 3-bit | 0.685 | 0.052 | +0.633 [+0.604, +0.659] | supported |
| Qwen3-0.6B, MASSIVE, seed 1 | GGUF Q3_K_M | 0.916 | 0.777 | +0.138 [+0.111, +0.168] | supported |

### W4-H2: the same test on Qwen3-4B. **Supported (2 of 2 tests).**

| arm | format | R, LoRA 1e-4 | R, LoRA 1e-5 | difference [95% CI] | verdict |
|---|---|---|---|---|---|
| Qwen3-4B, banking77 | MLX 3-bit | 1.464 | 0.925 | +0.539 [+0.451, +0.642] | supported |
| Qwen3-4B, banking77 | GGUF Q3_K_M | 1.075 | 0.871 | +0.204 [+0.153, +0.256] | supported |

R above 1 again comes from the base losing more than the fine-tune. The 4B base drops from 57.5% to 42.9% at MLX 3-bit while its LoRA at 1e-4 keeps 94% of its own accuracy. In accuracy retention, which does not depend on the base, the 4B pairs are 0.94 vs 0.80 (MLX 3-bit) and 0.99 vs 0.93 (Q3_K_M). The OLMo seed-1 pairs are 0.99 vs 0.75 and 0.99 vs 0.94.

### W4-H3 and W4-H4: predictor v2, unchanged. **Both supported.**

v2 is the model frozen before week 3 and fitted only on week-1 and week-2 points (SHA-256 `6fd78627...`). It was not refitted. All 42 week-4 points are pooled: 6 fine-tunes by 7 formats.

| model | MAE [95% CI, resampling fine-tunes] | safe-call agreement |
|---|---|---|
| **v2 (kept/NSR, base KLD)** | **0.084 [0.031, 0.146]** | **0.905** |
| v1 (NSR, base KLD), secondary | 0.090 [0.026, 0.174] | 0.905 |
| KLD-only baseline | 0.113 [0.083, 0.137] | 0.690 |
| bits-only baseline | 0.170 [0.131, 0.210] | 0.643 |

- **W4-H3 (MAE ≤ 0.10 and agreement ≥ 0.85): supported.**
- **W4-H4 (below both baselines): supported.**

Both verdicts use the point estimate, as pre-registered, and both are weaker than in week 3. The interval for v2's MAE reaches 0.146, above the 0.10 bar, and it overlaps the KLD-only interval. With six fine-tunes, this test cannot rule out an error above 0.10. v1 also came close to v2 this time (0.090 against 0.084; in week 3 it was 0.071 against 0.059), and on the MASSIVE arm it did better (0.072 against 0.088).

Exploratory breakdowns:

- Without the easy 6-bit formats (30 points), v2's MAE is 0.115 against 0.124 (v1), 0.156 (KLD-only) and 0.236 (bits-only). In week 3 the same cut gave 0.080.
- By arm, v2's MAE is 0.049 on OLMo (agreement 1.00), 0.088 on MASSIVE (0.93) and 0.115 on Qwen3-4B (0.79).
- v2 made the wrong "keeps at least 90%" call on 4 of 42 points. Two said safe when the fine-tune was not: MASSIVE LoRA 1e-4 at MLX 3-bit (98% predicted, 69% measured) and Qwen3-4B LoRA 1e-5 at MLX 4-bit (97% against 89%). Two said unsafe when R was at least 0.9, both on the gentle 4B LoRA (Q2_K and MLX 3-bit). At Q2_K, R is 1.28 only because the base collapsed; the fine-tune itself kept 60% of its accuracy, so in practice that "unsafe" call was fair.

The largest v2 errors:

| config | format | predicted | measured | note |
|---|---|---|---|---|
| Qwen3-4B, LoRA 1e-5 | Q2_K | 5% | 128% (capped at 100%) | the base falls to 15.8% while the fine-tune keeps 60% of its accuracy |
| Qwen3-4B, LoRA 1e-5 | MLX 3-bit | 48% | 92% | noise/update 6.85, but the 4B held on |
| OLMo, LoRA 1e-5, seed 1 | Q2_K | 16% | 55% | the same cell was the largest week-3 miss (15% vs 58%) |
| MASSIVE, LoRA 1e-4, seed 1 | Q2_K | 84% | 46% | the base breaks (KLD 2.77) |
| MASSIVE, LoRA 1e-5, seed 1 | MLX 3-bit | 41% | 5% | |
| MASSIVE, LoRA 1e-4, seed 1 | MLX 3-bit | 98% | 69% | low noise (1.52), but the base breaks (KLD 1.21); seed 0 was 98% vs 62% |

Two patterns repeat from week 3. On MASSIVE, when the base breaks, the slot-annotation task loses more than a small noise ratio suggests. On banking77, the OLMo-2 1B and Qwen3-4B fine-tunes kept more of their gain than a large noise ratio suggests. v2's two large 4B errors (0.95 and 0.44) both underestimate R. The larger one is at Q2_K, where R is 1.28 only because the 4B base collapsed; the fine-tune kept 60% of its accuracy.

**Consequences:** the pre-registered consequences apply only to failures, so none is triggered. Because W4-H2 is supported, the headline claim is not limited to models of 1.7B parameters or fewer. It has now been tested on one larger model, Qwen3-4B on banking77 with one seed, and on nothing above 4B.

One decision made after seeing the results: `ftquant check` keeps its tested scope and does not predict for Qwen3-4B. On the 14 Qwen3-4B points alone, v2's MAE was 0.115 and its safe-call agreement 0.79, both short of the bars. Qwen3-4B's base damage is therefore not bundled with the tool, and its predictions print n/a.

## Exploratory

### The same pattern in seven settings

Accuracy kept relative to each model's own bf16 accuracy, LoRA 1e-5 vs LoRA 1e-4, for the three new settings:

| setting | bf16 accuracy | MLX 4-bit | MLX 3-bit | Q4_K_M | Q3_K_M | Q2_K |
|---|---|---|---|---|---|---|
| Qwen3-4B, banking77 | 84.6% vs 78.6% | 0.94 vs 0.99 | 0.80 vs 0.94 | 0.98 vs 1.00 | 0.93 vs 0.99 | 0.60 vs 0.96 |
| OLMo-2 1B, banking77, seed 1 | 79.4% vs 77.3% | 0.93 vs 1.00 | 0.75 vs 0.99 | 0.97 vs 0.99 | 0.94 vs 0.99 | 0.52 vs 0.98 |
| Qwen3-0.6B, MASSIVE, seed 1 | 62.3% vs 54.9% | 0.96 vs 1.04 | 0.05 vs 0.69 | 0.985 vs 0.988 | 0.78 vs 0.92 | 0.00 vs 0.46 |

With the four week-3 settings, that makes seven. In all seven, the gentle LoRA had the higher bf16 accuracy and kept less at every format of 4 bits or fewer: 35 of 35 comparisons in the same direction.

Two limits on that count:

- Two of the seven settings are second seeds of week-3 settings, so there are five distinct model and task pairs.
- The margins are small at Q4_K_M. All five comparisons within 2 points are at that format, and one is a tie in practice (MASSIVE seed 1: 0.985 vs 0.988). At 3 bits and below, every gap is at least 4.5 points (the smallest is 4.57, OLMo seed 0 at Q3_K_M).

`figures/replication-week4.png` shows MLX 3-bit and Q3_K_M for all seven.

### Second seeds

| setting | fine-tune | bf16 accuracy, seed 0 → 1 | R at MLX 3-bit | R at Q3_K_M |
|---|---|---|---|---|
| OLMo-2 1B, banking77 | LoRA 1e-5 | 78.3% → 79.4% | 0.826 → 0.813 | 1.018 → 1.002 |
| OLMo-2 1B, banking77 | LoRA 1e-4 | 76.8% → 77.3% | 1.092 → 1.105 | 1.076 → 1.064 |
| Qwen3-0.6B, MASSIVE | LoRA 1e-5 | 65.3% → 62.3% | 0.319 → 0.052 | 0.843 → 0.777 |
| Qwen3-0.6B, MASSIVE | LoRA 1e-4 | 62.6% → 54.9% | 0.621 → 0.685 | 0.946 → 0.916 |

On OLMo, at MLX 3-bit and Q3_K_M, the second seed reproduces the first within 0.02 in R, and bf16 accuracy within 1.1 points. At the other formats R moves by up to 0.04. On MASSIVE, the direction holds but the size does not. The gentle LoRA kept 32% of its gain at MLX 3-bit with seed 0 and 5% with seed 1, so the test difference roughly doubled (+0.301 to +0.633). The strong LoRA's bf16 accuracy fell 7.7 points between seeds. Week 2 found that bf16 accuracy moves between seeds more than retention does. MASSIVE at MLX 3-bit is where retention moves too.

### Qwen3-4B

- **The base does not break at MLX 3-bit, by the tool's rule (base KLD below 1).** It keeps 74% of its accuracy there (57.5% to 42.9%), with base KLD 0.64 against 1.21 for Qwen3-0.6B and 1.08 for 1.7B. OLMo-2 1B has almost the same KLD (0.66), yet its weaker base still fell from 14.5% to 6.8%. The 4B base does break at Q2_K: 15.8% accuracy, KLD 1.03.
- **At 4B the gap is smaller than at Qwen3 0.6B and 1.7B at every format below, most of all at MLX 3-bit.** At Q2_K the gentle 4B LoRA still loses 40% of its accuracy. With one fine-tune pair per size, and learning rates that were not rescaled for size, this is not evidence of a trend with size. Accuracy kept by the gentle vs the strong LoRA on banking77:

  | model | MLX 3-bit | Q3_K_M | Q2_K |
  |---|---|---|---|
  | Qwen3-0.6B | 0.06 vs 0.85 | 0.79 vs 0.98 | 0.00 vs 0.78 |
  | Qwen3-1.7B | 0.32 vs 0.93 | 0.71 vs 0.98 | 0.00 vs 0.85 |
  | Qwen3-4B | 0.80 vs 0.94 | 0.93 vs 0.99 | 0.60 vs 0.96 |

- **The 4B fine-tune kept more than its noise ratio suggests.** The gentle 4B LoRA kept 92% of its gain at MLX 3-bit with noise/update 6.85. OLMo's gentle LoRA (seed 1), with almost the same base damage (KLD 0.66 against 0.64), kept 81% at 5.17. In accuracy retention the two are closer (0.80 against 0.75). v2 was fitted only on Qwen3 0.6B and 1.7B fine-tunes on banking77, which may be why it underestimated the 4B. A single 4B model cannot show whether this is an effect of size.
- **Operating-point drift is still larger than the accuracy loss.** The gentle 4B LoRA keeps 80% of its accuracy at MLX 3-bit, but it auto-accepts 17.5 points fewer items at its full-precision threshold. At Q2_K it keeps 60% and accepts 67.9 points fewer. For the strong LoRA the drops are 9.0 and 13.8 points. As in every week so far, the threshold is fitted on the even-indexed test items, not on the validation split that the week-1 pre-registration planned for later weeks (deviation logged in `PREREGISTRATION.md`).
- **Rounded away:** no week-4 fine-tune entered that regime. All six kept at least 98.5% of their update at every format.

## Figures

- `figures/replication-week4.png`: accuracy kept at MLX 3-bit and GGUF Q3_K_M, gentle vs strong LoRA, in all seven settings.
- `figures/predictor-v2-week4.png`: v2's pre-quantization predictions against the week-4 measurements.

## Limits

- Six fine-tunes (42 points) in the week-4 test. The upper end of v2's MAE interval (0.146) is above the 0.10 bar.
- One model above 1.7B (Qwen3-4B), one seed and one task on it, and its fine-tunes used gradient accumulation where the others did not.
- LoRA only in week 4, LoRA rank 8, MLX group size 64 only, and GGUF without an importance matrix.
- Both tasks have short outputs.
- The learning rates were not rescaled for model size. The 4B base already scores 57.5% zero-shot, so its fine-tunes' gains (21 and 27 points) are smaller and their R values noisier.
- Operating-point drift uses a threshold fitted on half of the test set, not on the validation split (see the deviation in `PREREGISTRATION.md`).
- R exceeds 1 whenever the base loses more than the fine-tune, which makes it hard to read at 3 bits and below. Accuracy retention is reported next to it for that reason.
