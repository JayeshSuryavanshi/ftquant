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
- 05:14: the first 4B fine-tune (LoRA 1e-5) ran out of GPU memory between steps 350 and 400. Its peak was 11.87 GB against the M1 Pro's 11.84 GB recommended working set (corrected on 2026-10-07: the second figure is in GiB; in the same decimal units the working set is 12.71 GB, so the peak was 0.84 GB below it), and it had already been swapping at about 16 s per step. The log is kept in `runs/week4/_failed-oom-q4b-lora-lowlr/`.
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

## 2026-10-03

- Why a week 5: a mock review of the paper draft pointed out that the week-1 unfused evaluations of the gentle Qwen3-0.6B LoRA (exploratory, never reported) lost about as much as the fused model: R 0.83 against 0.84 at MLX q4, and 0.02 against 0.06 at q3. The "rounded away" account cannot explain that, because an unfused update is never rounded. Week 5 tests the alternative on OLMo-2 1B and tests a remedy (training against the quantized base).
- 19:49: `PREREGISTRATION-week5.md` hashed (`runs/week1/PREREG-week5.sha256`), committed as `ea72821` (now `70b23e4`, see 2026-10-07) and pushed at 19:50, before anything ran. It is the first plan with a hosted timestamp from before its runs started.
- Code for week 5: `src/ftquant/week5.py` (the run), `src/ftquant/verdicts_w5.py` (the pre-registered verdicts, written before any week-5 result existed), and a `--labels-only` option in `eval_gguf.py` that restricts outputs to the 77 intent names with a GBNF grammar.
- 19:53 to 20:03: a 20-item plumbing smoke test (`python -m ftquant.week5 --smoke`, outputs in `runs/smoke5/`, 10 training steps per QLoRA run). It checked that the OLMo adapters load over the quantized base, that mlx-lm trains LoRA on an MLX-quantized base, the GGUF rebuild, the single-slot server and the label grammar. Its accuracies were seen, but they cover 20 items of a single intent, are not used anywhere, and nothing in the plan changed after them.
- 20:03: full week-5 run started (`python -m ftquant.week5`, log `runs/week5/week5.log`).
- Also written: `src/ftquant/revision.py`, a reanalysis of weeks 1 to 4 for the paper revision, with no new runs. It covers absolute accuracy and accuracy-retention intervals for all eight LoRA pairs, update norms for every fine-tune (week 4 computed from the adapters; the method reproduces the recorded week-3 norm to within 0.2%), constant and update-norm baselines for the predictor, and the numbers the paper had typed by hand (the band table, 168 of 192, the Spearman values, the drift counts). Output: `runs/revision/revision.json`.

## 2026-10-04

- Week 5, Arm U (done 21:04): both base checks reproduced the week-3 evaluations exactly (6.79% at MLX q3, 14.61% at q4), so the week-3 base files stay the reference. Unfused and fused models agreed closely everywhere. **W5-H1 supported**: R(unfused) minus R(fused) at MLX q3 was +0.008 [-0.010, +0.027] for seed 0 and +0.023 [+0.002, +0.042] for seed 1, both upper bounds well under the +0.08 margin, against 0.17 to 0.19 of lost gain. The strong LoRAs and the 4-bit runs (exploratory) were within 0.014.
- Arm Q: the gentle LoRA trained on the 4-bit base reached 77.6% (S = 0.957) against 69.4% for the fuse-then-quantize LoRA (R = 0.841). **W5-H2 supported**: +0.116 [+0.097, +0.135]; final validation loss 0.057 against 0.056 for the bf16-trained one. Exploratory: the strong LoRA trained on the 4-bit base did worse (61.3% against 71.7% fused), and it also trained worse there (final validation loss 0.110 against 0.067), so at learning rate 1e-4 training on the quantized base hurt. On the 3-bit base, which scores 0%, the gentle LoRA trained there reached 67.5% against 4.3% for fuse-then-quantize.
- 23:05 to 00:55: the last Arm Q run (`q06q3-lora`, strong LoRA on the 3-bit base) stalled. After its step-100 checkpoint, steps 100 to 200 took 92 minutes instead of about 5 and one validation pass took 17 minutes, with the process near 8 GB and the system swapping (sampled: the main thread waiting on GPU completion and on GPU residency commits). Its losses were normal (validation 0.266 at step 200). It was stopped at 00:55, its files moved to `runs/week5/_stalled-q06q3-lora/`, and the deviation logged in `PREREGISTRATION-week5.md` (re-hashed 00:56). 00:56: restarted with `python -m ftquant.week5 --arms C Q`, so Arm C runs first and the stalled configuration is retrained from scratch afterwards.
- Afternoon: number audit for the paper. `src/ftquant/findings.py` recomputes every number quoted in the paper from the per-item predictions, the run logs and the result files (352 values), checks each against the sentence it appears in (126 checks, plus a scan that every remaining decimal, percentage and count is a stated constant), and writes `runs/findings/findings.json` and the paper's findings table; two runs give byte-identical files. Corrected in the paper as a result: at MLX q3 the gentle Qwen3-0.6B LoRA's invalid outputs were 43% of its errors, not the majority the text claimed; OLMo's seed-to-seed bf16 accuracy change is 1.5 points in GGUF (1.1 in MLX, the only engine the text had used); 'under a quarter' of the lost gain is now the upper bounds, 16% and 22%; '99.9%' agreement is now 2 of 3,080 items between one and four llama-server slots and 3 of 3,080 against round 1; and the restarted `q06q3-lora` run took 223.8 minutes of wall-clock time while its training steps ran at the usual 0.2 to 0.4 steps per second, so the paper no longer attributes that time to memory pressure. Compute times in the paper are now stated to 0.1 minute from the logs, which removes a rounding tie (4.5 minutes) the old text had rounded up. A second recomputation from the raw predictions, with separate code, matched every headline number to six decimals and caught 'at least 4.6' for a minimum of 4.566. Every bound in the paper now rounds in the safe direction, down for 'at least' and up for 'at most', which changed 0.6 to 0.61 (engine accuracy difference), 4.6 to 4.5, 0.06 to 0.061, 0.014 to 0.015 and 0.04 to 0.042; the engine-agreement range now says it covers the fine-tunes that learned their task, since the failed run agrees on only 80.6%.

## 2026-10-07

- 00:32 to 01:15: evaluation speed checks (scratch outputs, not used for any result). The per-item MLX path reproduced the stored predictions exactly (300 of 300 items, identical log-probabilities) for the week-1 Qwen3-0.6B base at bf16, the week-1 strong LoRA unfused on the 4-bit base, the week-4 Qwen3-4B base at 4 bits and the week-3 MASSIVE base. The new `--batch` path agreed with the stored full runs on 97.5%, 99.0%, 98.6% and 88.9% of items (the last at 0% accuracy), with accuracy within 0.1 point, and was 1.6 to 2.7 times faster. Profiling the 4B model explained the small gain: per item, 133 ms go to the item's own prompt tokens (compute-bound) and 28 ms to each generated token; at batch 16 these fall to 70 ms and 9.5 ms per item, and batch 32 swaps.
- 01:08: llama-server build 11146 keeps a host prompt cache of up to 8 GB by default (`--cache-ram 8192`). On this laptop it pushed swap to 12.4 GB, and a Qwen3-0.6B Q4_K_M evaluation that takes about 3 minutes ran past 8. With `--cache-ram 0`, the same file took 183 s at 4 slots and 156 s at 16, agreeing with the stored week-1 run on 99.5% of items. The default may have contributed to the memory-pressure stalls of week 5.
- Before the repository went public, its history was rewritten to change the wording of two notebook lines (13:42 and 14:06 on 2026-09-29) and of the supplement builder's leak check. The 12 commits from 2026-09-29 14:09 onward got new IDs; their authors, dates and messages, and every other file, are unchanged. The round-5 plan commit `ea72821`, pushed at 19:50 on 2026-10-03, is now `70b23e4` with the same `PREREGISTRATION-week5.md`; GitHub's push log keeps the original ID.
- 02:07: the 4B out-of-memory figures in the week-4 plan and in this notebook (2026-09-29, 05:14) compared decimal GB with GiB. The plan got a logged amendment; the deviation and all results are unchanged.
- 02:49 to 03:14: a review of the draft week-6 plan found that the Qwen3-0.6B and Qwen3-1.7B checkpoints store a separate copy of their tied output head, so their base GGUF files quantized the input embedding at the file's base type (Q4_K, Q3_K or Q2_K) while the fused fine-tunes share one embedding kept at Q6_K. Qwen3-4B (no stored copy) and OLMo-2 1B (untied) are not affected. `src/ftquant/embedding_check.py` rebuilt the Qwen3-0.6B base both ways (results in `runs/revision/embedding/`). The original recipe reproduced the stored base damage exactly (0.104, 0.441, 2.766 at Q4_K_M, Q3_K_M, Q2_K); with a shared embedding it was 0.102, 0.421 and 2.555, and base accuracy was 10.10% against 9.81%, 5.55% against 5.58%, and 0% at Q2_K either way. No conclusion changes; the paper's reproducibility appendix now states the asymmetry and these numbers.

## Compute used

Everything ran on one Apple M1 Pro laptop (16 GB). Typical wall times:

| model | LoRA training, 500 steps | evaluations |
|---|---|---|
| Qwen3-0.6B | 10 to 20 min (full fine-tuning 13 to 25 min) | 3 to 8 min each on banking77; 4 to 15 min on MASSIVE |
| Qwen3-1.7B | about 52 min | 5 to 9 min each |
| OLMo-2 1B | 40 to 53 min | 4 to 9 min each |
| Qwen3-4B | 2 h 8 min to 2 h 24 min (micro-batch 2) | 12 to 25 min each |

Base KLD takes 2 to 7 minutes per base model. The mechanism measurements for two fine-tunes take 5 to 8 minutes on 0.6B and 1B models and 30 minutes on 4B.
