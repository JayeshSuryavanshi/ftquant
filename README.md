# ftquant

Does your fine-tune survive quantization? `ftquant check` measures it from the weights, before you quantize.

You fine-tune a small model, then ship it as a 4-bit MLX model or a GGUF for llama.cpp and ollama. The usual assumption is that the gain you trained for comes along. Sometimes it does not. In our study, the fine-tunes that lost the most were the gentle ones: small learning rates, small weight updates. At MLX 3-bit, a Qwen3-0.6B LoRA trained at learning rate 1e-5 kept 6% of its accuracy gain, while the same LoRA at 1e-4 kept 95%. The gentle one was the more accurate model before quantization.

`ftquant check` compares your fine-tuned weights with the base weights, runs both through the exact quantizer you plan to ship, and reports how much of the update survives and how much rounding noise lands on top of it. It needs only numpy, runs on a CPU, and takes about 20 seconds for a 0.6B model.

## Install

```
pip install ftquant
```

## Use

MLX adapter, PEFT adapter, or a full fine-tuned checkpoint:

```
ftquant check --base Qwen/Qwen3-0.6B --finetuned path/to/adapter
```

```
196 fine-tuned linear layers compared against Qwen/Qwen3-0.6B

format                     noise/update  update kept  base damage  reading
MLX 8-bit (group 64)               0.48         100%        0.004  low noise
MLX 6-bit (group 64)               1.61         100%        0.021  low noise
MLX 4-bit (group 64)               3.84         100%        0.255  moderate noise
MLX 3-bit (group 64)               5.59          99%        1.207  high noise, base model breaks
```

That is real output for a LoRA fine-tune at learning rate 1e-5. Measured on 3,080 test items, it kept 84% of its accuracy gain at MLX 4-bit and 6% at 3-bit.

A full fine-tune at learning rate 3e-6 fails differently:

```
format                     noise/update  update kept  base damage  reading
MLX 4-bit (group 64)               4.95          50%        0.255  update rounded away
MLX 3-bit (group 64)               2.16           5%        1.207  update rounded away
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
4. **Base damage** is the base model's own KL divergence from its 16-bit self on WikiText-2 at that format. It is shown for bases measured in the study (Qwen3-0.6B and 1.7B).

## How to read it

The reading column describes the measurement. It is not a validated prediction. We pre-registered a model that turns these numbers into a predicted retention, and it missed its own accuracy bar on held-out fine-tunes (mean absolute error 0.107 against a target of 0.10, though it did beat the bit-width and base-damage baselines). So the tool reports what it measures and what we saw in the study, and leaves the call to you.

What we saw across 89 pairs of fine-tune and format (Qwen3 0.6B and 1.7B, banking77, LoRA and full fine-tuning, MLX and GGUF). The bands were chosen after looking at the data:

| measurement | cases | kept ≥ 90% of the gain | lowest kept |
|---|---|---|---|
| base damage < 1, update kept ≥ 60%, noise/update < 3 | 44 | 44 | 91% |
| base damage < 1, update kept ≥ 60%, noise/update 3 to 5 | 7 | 3 | 78% |
| base damage < 1, update kept ≥ 60%, noise/update ≥ 5 | 10 | 4 | 45% |
| update kept < 60% | 4 | 0 | 0% |
| update kept ≥ 60%, base damage ≥ 1 | 24 | 4 | 0% |

When the base model itself breaks (MLX 3-bit and 2-bit and GGUF Q2_K for these sizes), only large updates held on: all 4 cases that kept 90% were LoRA at learning rate 1e-4.

## The study behind it

`PREREGISTRATION.md` and `PREREGISTRATION-week2.md` hold the pre-registered design; `RESULTS-week1.md` and `RESULTS-week2.md` hold the results, including the criteria that were not met.

- Qwen3 0.6B and 1.7B fine-tuned on banking77 with LoRA and full fine-tuning, across learning rates from 3e-6 to 3e-4 and two seeds.
- Quantized to MLX 8, 6, 4, 3 and 2 bits and GGUF Q8_0 through Q2_K.
- Accuracy measured on all 3,080 test items, with paired bootstrap confidence intervals.

## License

MIT
