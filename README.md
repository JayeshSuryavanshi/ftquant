# ftquant

Does your fine-tune survive quantization? `ftquant check` measures it from the weights, before you quantize.

You fine-tune a small model, then ship it as a 4-bit MLX model or a GGUF for llama.cpp and ollama. The usual assumption is that the gain you trained for comes along. Sometimes it does not. In our study the fine-tunes that lost the most were the gentle ones: small learning rates, small weight updates. At MLX 3-bit, a Qwen3-0.6B LoRA trained at learning rate 1e-5 kept 6% of its accuracy gain, while the same LoRA at 1e-4 kept 95%. The gentle one was the more accurate model before quantization. Pre-registered tests on a second model family (OLMo-2 1B), a second task (MASSIVE slot filling), second seeds and a larger model (Qwen3-4B) found the same direction in all 10 of their tests: at MLX 3-bit and GGUF Q3_K_M, the LoRA at 1e-4 kept more of its gain than the LoRA at 1e-5. In an exploratory count across seven settings, the LoRA that scored higher at bf16 kept less of its accuracy at every format of 4 bits or fewer. At 4B the gap is smaller at 3 bits (80% against 94% of accuracy kept at MLX 3-bit) and still large at Q2_K (60% against 96%).

`ftquant check` compares your fine-tuned weights with the base weights and runs both through the exact quantizer you plan to ship. It reports how much of the update survives, how much rounding noise lands on top of it, and, where it has been tested, how much of the gain it expects you to keep. It needs only numpy, runs on a CPU, and takes about 20 seconds for a 0.6B model.

## Install

Not on PyPI yet. From a clone:

```
pip install -e .
```

## Use

MLX adapter, PEFT adapter, or a full fine-tuned checkpoint:

```
ftquant check --base Qwen/Qwen3-0.6B --finetuned path/to/adapter
```

```
196 fine-tuned linear layers compared against Qwen/Qwen3-0.6B

format                     noise/update  update kept  base damage  predicted gain kept  reading
MLX 8-bit (group 64)               0.47         100%        0.004                  n/a  low noise
MLX 6-bit (group 64)               1.59         100%        0.021                 100%  low noise
MLX 4-bit (group 64)               3.80         100%        0.255                  97%  moderate noise
MLX 3-bit (group 64)               5.52          99%        1.207                  41%  high noise, base model breaks
```

That is real output for a Qwen3-0.6B LoRA trained on MASSIVE at learning rate 1e-5, a fine-tune the predictor had never seen. Measured on 2,974 test items, it kept 93% of its gain at MLX 4-bit and 32% at 3-bit.

A full fine-tune at learning rate 3e-6 fails differently:

```
format                     noise/update  update kept  base damage  predicted gain kept  reading
MLX 4-bit (group 64)               4.95          50%        0.255                  54%  update rounded away
MLX 3-bit (group 64)               2.16           5%        1.207                   0%  update rounded away
```

Its update is so small that the fine-tuned and base weights round to the same values, so there is little noise because almost nothing is left. Measured, it kept 27% of its gain at 4-bit and 0% at 3-bit.

Other formats:

```
ftquant check --base Qwen/Qwen3-0.6B --finetuned adapter --formats mlx:4 mlx:3g32 gguf:Q8_0 gguf:Q4_0
```

llama.cpp k-quants (Q6_K, Q4_K_M, Q3_K_M and the rest) come from llama.cpp's own quantizer. Point `check-gguf` at bf16 GGUF files of the base and the fine-tune:

```
ftquant check-gguf --base-gguf base-bf16.gguf --finetuned-gguf ft-bf16.gguf --types Q6_K Q4_K_M Q3_K_M
```

## What it measures

For every fine-tuned linear weight matrix, with base weights W and fine-tuned weights W' (rounded to the checkpoint's own precision):

1. The update is Δ = W' − W.
2. The quantized update is Δq = deq(Q(W')) − deq(Q(W)), where Q is the exact quantizer: MLX affine with its group size, or llama.cpp's.
3. **Update kept** is the projection of Δq onto Δ, as a share of Δ. **Noise/update** is what is left of Δq after that projection, relative to ‖Δ‖. Both are summed over all layers.
4. **Base damage** is the base model's own KL divergence from its 16-bit self on WikiText-2 at that format. It is shown for the bases bundled with the tool (Qwen3-0.6B and 1.7B, OLMo-2 1B). Qwen3-4B was measured in the study but is not bundled.

## How to read it

**Predicted gain kept** comes from a three-parameter model: sigmoid(t0 + t1·ln(update kept / noise) + t2·ln(base damage)). It was fitted on Qwen3 0.6B and 1.7B fine-tunes on banking77, frozen, and then tested on data it had never seen, in a pre-registered test: OLMo-2 1B on banking77 and Qwen3-0.6B on MASSIVE, 35 fine-tune and format pairs.

- Its mean absolute error was 0.059, and it made the right "keeps at least 90%" call 94% of the time.
- Both are better than guessing from the bit width alone (0.128, 74%) or from base damage alone (0.094, 80%).
- The first version of the model missed its own accuracy bar (0.107 against 0.10), and the reason is documented in `RESULTS-week2.md`.
- A second pre-registered test, on second seeds and Qwen3-4B (42 pairs), passed by a thinner margin. The error was 0.084, but its interval runs up to 0.146. It made the right call 90% of the time and again beat bit width alone (0.170) and base damage alone (0.113).
- On the 4B fine-tunes alone (14 pairs, a breakdown the pre-registration did not plan), the error was 0.115 and the right call came 79% of the time, both short of the bars. After seeing that, we chose not to predict for Qwen3-4B. Without the easy 6-bit formats, the week-4 error was also 0.115. Details in `RESULTS-week4.md`.

The prediction is shown only where it was tested: MLX 6, 4 and 3-bit with group size 64, GGUF Q6_K, Q4_K_M, Q3_K_M and Q2_K, and bases whose damage ships with the tool (Qwen3-0.6B, Qwen3-1.7B, OLMo-2-0425-1B-Instruct). Everywhere else it prints n/a, and you still get the measurements.

The **reading** column describes the measurement. Its bands were chosen after looking at weeks 1 and 2, across 89 pairs of fine-tune and format:

| measurement | cases | kept ≥ 90% of the gain | lowest kept |
|---|---|---|---|
| base damage < 1, update kept ≥ 60%, noise/update < 3 | 44 | 44 | 91% |
| base damage < 1, update kept ≥ 60%, noise/update 3 to 5 | 7 | 3 | 78% |
| base damage < 1, update kept ≥ 60%, noise/update ≥ 5 | 10 | 4 | 45% |
| update kept < 60% | 4 | 0 | 0% |
| update kept ≥ 60%, base damage ≥ 1 | 24 | 4 | 0% |

When the base model itself breaks (for Qwen3 0.6B and 1.7B: MLX 3-bit and 2-bit and GGUF Q2_K), only large updates held on.

## The study behind it

Five pre-registrations (`PREREGISTRATION*.md`, hashed before the runs they cover), four results files (`RESULTS-week1.md` to `RESULTS-week4.md`) and round 5's verdicts (`runs/week5/verdicts.txt`), including the criteria that were not met. `NOTEBOOK.md` logs every run, decision and incident as it happened.

- Qwen3 0.6B, 1.7B and 4B and OLMo-2 1B, fine-tuned on banking77 (intent classification) and MASSIVE (slot annotation), with LoRA and full fine-tuning, across learning rates from 3e-6 to 3e-4 and two seeds.
- Quantized to MLX 8, 6, 4, 3 and 2 bits and GGUF Q8_0 through Q2_K.
- Accuracy measured on the full test sets (3,080 and 2,974 items), with paired bootstrap confidence intervals.

## License

MIT
