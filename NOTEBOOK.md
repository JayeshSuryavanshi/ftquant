# ftquant lab notebook

A chronological record: what ran and when, what was decided and why, what went wrong, and what was observed before the formal analysis. Times are PDT. Numbers are copied from the run logs (`runs/*/sweep.log`, `runs/orchestrate.log`). Formal results are in `RESULTS-week*.md`, and the pre-registered design and every deviation are in `PREREGISTRATION*.md`.

## Where everything lives

| what | where | in git |
|---|---|---|
| Per-item predictions for every evaluation (text, gold, prediction, correct, valid, sequence log-probability, confidence) | `runs/week*/<config>/eval/*.jsonl`; archived as `records/eval-week*.tar.gz` | archives yes |
| Training config and training log (loss, validation loss, speed, peak memory) | `runs/week*/<config>/train.yaml`, `train.log` | yes |
| Accuracy and wall time of every evaluation, as it happened | `runs/week*/sweep.log` | yes |
| Pipeline steps and failures | `runs/orchestrate.log` | yes |
| Retention with bootstrap intervals, operating-point drift, recovery curves | `runs/analysis-all.json` (weeks 1-2), `runs/week3/analysis.json`, `runs/week4/analysis.json` | yes |
| Noise-to-update ratio and update kept, per fine-tune and format | `runs/week*/mechanism-*.json` | yes |
| Base-model KL divergence | `runs/kld/*.json` | yes |
| Predictors, their tests and verdicts | `runs/predictor/`, `runs/week3/verdicts.*`, `runs/week4/verdicts.*` | yes |
| Pre-registration hashes | `runs/week1/PREREG*.sha256` | yes |
| Fine-tuned adapters and full fine-tuned weights | `runs/week*/<config>/adapter/` | no (weights stay local) |

## 2026-09-27

- Built the harness: banking77 from PolyAI's original CSVs, a float32 greedy decoder for MLX (mlx-lm returns bf16 log-probabilities), a llama-server evaluator for GGUF, and a resumable sweep driver.
- Noise floor between engines on the same bf16 fine-tune: 196 of 200 predictions identical. Median difference in sequence log-probability 0.023.
- 12:45: week-1 sweep started (Qwen3-0.6B base, LoRA 1e-4, LoRA 1e-5, full 1e-5; Qwen3-1.7B base, LoRA 1e-4; full format ladder).
- 17:39: `PREREGISTRATION.md` hashed. The 1.7B runs are confirmatory; the 0.6B runs are exploratory, because their results existed by then. 18:00: amended, because the threshold code was made to match the text.
- 18:00: `PREREGISTRATION-week2.md` hashed (predictor v1, test sets T1 to T3, criteria P1 to P3).
- Built `ftquant check`, a numpy-only implementation of the noise-to-update measurement, while the sweeps ran.
- 23:40: week-1 confirmatory analysis. H1 supported, H2 not supported (the 1.7B base collapses at 3-bit, so gain retention exceeds 1), H3 inconclusive, H4 not supported.
- 23:48: the predictor freeze crashed on a missing folder before writing anything. Re-run and frozen at 23:48:35 (deviation logged). Week-2 sweep started.

## 2026-09-28

- 08:22: week 2 finished.
  - LoRA at 3e-4 failed to learn: 0% accuracy, validation loss 1.95, and it emits `</think>` fragments. Its retention is undefined, so it was excluded (deviation logged).
  - Verdicts: P1 not met (MAE 0.107 against 0.10), P2 met, P3 met. As pre-registered, `ftquant check` became descriptive only.
- Found the second failure mode, "rounded away". Full fine-tuning at 3e-6 keeps 5.5% of its update at MLX 3-bit, while its noise ratio looks benign (2.16).
- 08:39:25: predictor v2 frozen: kept/NSR plus base KLD, chosen by leave-one-fine-tune-out error (0.071 against 0.095 for the v1 form).
- 08:40:04: `PREREGISTRATION-week3.md` hashed. Smoke tests of OLMo-2 1B and MASSIVE passed.
- 08:46 to 17:40: week 3. All 63 evaluations completed. W3-H1, W3-H2 and W3-H3 were all supported (v2 MAE 0.059).
- Pushed to the private GitHub repo (85c147d, then 6ab599e with the week-3 results).
- 18:11:07: `PREREGISTRATION-week4.md` hashed (second seeds, Qwen3-4B, v2 unchanged).
- A 4B smoke test found that `mlx_lm fuse` needs the full model snapshot, so the sweep now downloads it before fusing. It measured 11.6 GB peak memory at 10 steps.
- To make room for 4B, deleted regenerable leftovers: week-1 base quantizations, dry-run models, and bits-per-weight scratch copies.
- 18:25: week 4 started.

## 2026-09-29

- 22:26 on 09-28 to 00:25: the laptop was unplugged. Training drained the battery from 100% to 1% in about 1 h 45 min, and macOS forced a low-battery hibernate at 00:11. After it was woken on AC power at 00:25, all processes resumed.
  - No evaluation spanned the pause: `mas06-lora-s1/mlx-q3` finished just before it, and the next evaluation started after.
  - Every result file has the full line count.
- 00:57: all four seed-1 runs finished. Interim accuracy kept against each model's own bf16:

  | fine-tune | bf16 | MLX 3-bit | Q3_K_M |
  |---|---|---|---|
  | OLMo, LoRA 1e-5, seed 1 | 79.4% | 75% | 94% |
  | OLMo, LoRA 1e-4, seed 1 | 77.3% | 99% | 99% |
  | MASSIVE, LoRA 1e-5, seed 1 | 62.3% | 5% | 78% |
  | MASSIVE, LoRA 1e-4, seed 1 | 54.9% | 69% | 92% |

  The gentle LoRA is again more accurate at bf16 in both arms. Seed 0 on MASSIVE, LoRA 1e-5, MLX 3-bit, was 32%, so this cell varies a lot between seeds. That is recorded as an observation, not a test.
- 03:33: Qwen3-4B base evaluated. 57.5% zero-shot at bf16 and 42.9% at MLX 3-bit, so it does not collapse like the 0.6B and 1.7B bases. Its base KLD at MLX 3-bit is 0.64, against 1.21 for 0.6B.
- 05:14: the first 4B fine-tune (LoRA 1e-5) ran out of GPU memory between steps 350 and 400. Its peak was 11.87 GB against the M1 Pro's 11.84 GB recommended working set, and it had already been swapping at about 16 s per step. The log is kept in `runs/week4/_failed-oom-q4b-lora-lowlr/`.
  - Deviation: both 4B fine-tunes now use micro-batch 2 with 2-step gradient accumulation. That is the same effective batch, the same 500 optimizer steps and the same 2,000 examples.
  - Logged at 05:16 and restarted at 05:16. The first log entry said 05:40 in error, and the correction is in git history (10c24c4).
- 07:40: 4B LoRA 1e-5 trained in 2 h 24 min: 84.6% at bf16, 67.9% at MLX 3-bit (80% of its accuracy kept), 83.2% at Q4_K_M.
- About 10:00: free disk space fell to 2.3 GB. macOS was preparing a system update (`com.apple.os.update-MSUPrepareUpdate` local snapshot, about 13 GB); ftquant itself was using the expected 17 GB. Actions taken:
  - Freed about 5 GB: the Qwen3-1.7B download cache (no longer needed) and three byte-identical duplicate checkpoints.
  - Made the pipeline write fused, quantized and GGUF files under a temporary name and rename them when complete, so a disk-full failure cannot leave a truncated file that a resume would reuse.
  - Reordered the GGUF mechanism step to lower its disk peak. The reordered version was checked on a 0.6B configuration and was bit-identical.
- Archived the per-item predictions for weeks 1 to 3 (`records/`, about 30 MB), and added training configs and logs to git.

## Compute used

Everything ran on one Apple M1 Pro laptop (16 GB). Typical wall times:

| model | LoRA training, 500 steps | evaluations |
|---|---|---|
| Qwen3-0.6B | about 10 to 20 min (full fine-tuning 20 to 25 min) | 3 to 5 min each on banking77; 4 to 14 min on MASSIVE |
| Qwen3-1.7B | about 52 min | 6 to 9 min each |
| OLMo-2 1B | 42 to 53 min | 4 to 8 min each |
| Qwen3-4B | 2 h 24 min (micro-batch 2) | 14 to 25 min each |

Base KLD takes 2 to 7 minutes per base model.
