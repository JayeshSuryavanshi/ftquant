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
  - `mas06-lora-s1/mlx-q3` was logged at 00:11:34, one second after the sleep began. The GGUF conversion that followed completed after the wake. Evaluation is deterministic, so a pause cannot change a result.
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
- 10:33: 4B LoRA 1e-5 evaluations done. Accuracy kept against its own bf16: 80% at MLX 3-bit, 93% at Q3_K_M (79.1% vs 84.9%), 60% at Q2_K (51.0%).
- 12:41: 4B LoRA 1e-4 trained in 2 h 8 min (micro-batch 2). Its evaluations started, and 18 GB of disk was free.
- 13:41: its MLX evaluations finished. Accuracy kept against its own bf16 (78.6%): 99.8% at MLX 6-bit, 98.7% at 4-bit, 93.8% at 3-bit (73.7%). LoRA 1e-5 kept 99.9%, 93.8% and 80.2% from a higher bf16 of 84.6%. So at 4B the gentle LoRA is again the more accurate model at bf16 and keeps less at 4 and 3 bits. This is an observation; the pre-registered 4B test (W4-H2) is on gain retention and runs after the GGUF evaluations.
- 13:42: the pipeline stopped as the GGUF evaluations of LoRA 1e-4 were starting. Most likely cause: it ran as a background job of an interactive terminal session, and that session ended at about 13:43 (its session log was last written then). The kernel log shows no out-of-memory kill around that time, and 29 GB of disk was free at 14:03.
  - The sweep process had started at 05:16, before the 10:10 atomic-write change, so its GGUF bf16 file was checked before reuse: 398 tensors, and the tensor data ends exactly at the file size. It was complete and was reused.
  - The four MLX result files each have all 3,080 lines. The evaluators score every item before they open the output file, so an interrupted evaluation leaves no partial file that a resume could mistake for a finished one.
  - 14:06: restarted in its own process session, detached from any terminal session, so closing a window can no longer stop it. Finished configurations were skipped. About 25 minutes were lost.
- 14:08 to 15:30: 4B LoRA 1e-4 GGUF evaluations: 78.6% at bf16, 78.7% at Q6_K, 78.6% at Q4_K_M, 78.0% at Q3_K_M, 75.6% at Q2_K. The sweep finished at 15:30.
- About 14:30, before the verdicts were computed: `verdicts_w4.py` was checked against `PREREGISTRATION-week4.md` (bootstrap, verdict rules, thresholds, mechanism and KLD file layouts, base pairing through the week-3 links), and the analysis was dry-run on the partial data. By then the per-item results behind all four W4-H1 tests and the 4B MLX evaluations existed; the W4-H2 GGUF evaluations and the mechanism files did not. No change was needed, and the script was not modified.
- 15:30 to 16:12: mechanism measurements (OLMo 8 min, MASSIVE 5 min, 4B 30 min), analysis and verdicts. The week-4 pipeline finished at 16:12:44.
- Verdicts: W4-H1 supported (4 of 4), W4-H2 supported (2 of 2), W4-H3 supported (v2 MAE 0.084 [0.031, 0.146], safe-call agreement 0.905), W4-H4 supported (KLD-only 0.113, bits-only 0.170). The MAE interval reaches above 0.10, and on the 14 Qwen3-4B points alone v2 misses both bars (0.115, 0.79).
- Decision after seeing the results: `ftquant check` does not predict for Qwen3-4B, and its base damage is not bundled.
- Exploratory: across seven settings (two are second seeds), the gentle LoRA was more accurate at bf16 in all seven and kept less at every format of 4 bits or fewer, 35 of 35. The five smallest margins are all at Q4_K_M, under 2 points. At 4B the gap at MLX 3-bit shrinks to 80% against 94% of accuracy kept.
- Written: `RESULTS-week4.md`, `src/ftquant/week4.py` (`runs/week4/summary.json`), `figures/replication-week4.png` and `figures/predictor-v2-week4.png` (the chart functions now take their data; the week-3 figures re-render byte-identical), README updates, and `records/eval-week4.tar.gz`.
- 16:25: before committing, two independent agents audited the write-up. One recomputed 165 numbers from the raw per-item files with its own code, including all six bootstrap tests and the predictor metrics. The other checked 64 claims against the pre-registrations and the earlier results. Corrections applied:
  - Two rounding slips in `RESULTS-week3.md`, Qwen3-0.6B banking77 row: MLX 4-bit 0.89 → 0.88 (2,138/2,417 = 0.8846) and Q3_K_M 0.99 → 0.98 (2,224/2,259 = 0.9845). No comparison changes direction.
  - Overstated wording in `RESULTS-week4.md` and the README was rewritten: a trend with size claimed from one 4B model, an untested "which is why", "no longer limited to 1.7B", and confirmatory and exploratory results mixed in the README introduction.
  - A deviation not disclosed until now: the week-1 pre-registration planned to fit the drift threshold on the validation split from week 2 on, but weeks 2 to 4 kept fitting it on half of the test set. It is logged in `PREREGISTRATION.md` (re-hashed at 16:29) and in the week-4 limits. Only exploratory drift numbers are affected.
  - Smaller fixes: a stale footnote in `ftquant check` and the bundled predictor's test record now mention week 4.
  - One reported problem was rejected after checking: week 4 did run 63 new evaluations (7 configurations × 9), not 45.

## Compute used

Everything ran on one Apple M1 Pro laptop (16 GB). Typical wall times:

| model | LoRA training, 500 steps | evaluations |
|---|---|---|
| Qwen3-0.6B | 10 to 20 min (full fine-tuning 13 to 25 min) | 3 to 8 min each on banking77; 4 to 15 min on MASSIVE |
| Qwen3-1.7B | about 52 min | 5 to 9 min each |
| OLMo-2 1B | 40 to 53 min | 4 to 9 min each |
| Qwen3-4B | 2 h 8 min to 2 h 24 min (micro-batch 2) | 12 to 25 min each |

Base KLD takes 2 to 7 minutes per base model. The mechanism measurements for two fine-tunes take 5 to 8 minutes on 0.6B and 1B models and 30 minutes on 4B.
