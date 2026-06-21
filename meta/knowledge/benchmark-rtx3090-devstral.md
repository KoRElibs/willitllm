# Benchmark — RTX 3090 + devstral-small-2:24b

Measured 2026-06-21 under real Cline agentic workload.

## Setup

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 (24 GB) |
| Model | devstral-small-2:24b (Q4_K_M, ~14.3 GB weights) |
| KV cache | q4_0 (OLLAMA_KV_CACHE_TYPE=q4_0, ~0.5 bpe) |
| num_ctx | ~200k (q4_0 allows this; VRAM showed 23,998 MiB) |
| Driver | 610.43.02 |

## Generation speed

~19.75 t/s at 37,287 tokens context (stable, measured across multiple chunks).

willitllm estimates 28–54 t/s for this GPU + quantization.
**Actual efficiency factor: ~0.30** (model assumes 0.43–0.82 for Q4_K_M).

Root cause candidates:
- KV cache bandwidth competes with weight bandwidth at long contexts
- gen_eff calibration is too optimistic for 24B models

## Prefill speed

Measured during deep codebase analysis (~85k token context):

| Tokens processed | Prefill t/s |
|---|---|
| 11,776 | 752 |
| 16,384 | 730 |
| 20,480 | 714 |
| 24,576 | 700 |

**Prefill degrades ~7% per 12k tokens** at this context scale. Current willitllm model
assumes fixed prefill efficiency — this is a modelling gap.

## Context in practice

With q4_0 KV cache at ~200k num_ctx, Cline processed a full codebase in sliding
chunks (512 tokens/chunk), with `cached n_tokens` reaching 85,000+ — only ~40% of the
allocated context. The "More context" mode on the coder page is validated as genuinely
useful for agentic coding at this scale.

## Multi-model benchmark (2026-06-21, --fill, RTX 3090, q4_0 KV cache)

| Model | Quant | Weights | Max ctx | Gen @ min ctx | Gen @ max ctx |
|---|---|---|---|---|---|
| devstral-small-2:24b | Q4_K_M | 15 GB | 200k | 50 t/s | 7 t/s |
| devstral:24b | Q4_K_M | 13.3 GB | 115k | 51 t/s | 19 t/s |
| gemma4:e4b (MoE) | Q8_0 | 10.8 GB | 115k | 100 t/s | 53 t/s |
| codellama:13b (MHA) | Q4_0 | 6.9 GB | 14k | 92 t/s | 58 t/s |

## Formula fix applied

Updated `calcSpeedEstimates` to include KV cache bandwidth in the denominator:
`gen_tps = bw × eff / (activeWeightsGB + kvCacheGB)`

Also added `params_b_active` for gemma4:e2b (2.0) and gemma4:e4b (4.0) in data.models.js.

**Error after fix at max context:**
- devstral-small-2 (200k): new lo is ~2.5× above measured (super-linear gap remains)
- devstral:24b (115k): new lo is ~19% above measured
- gemma4:e4b (115k): measured lands inside [44–79] range ✓
- codellama:13b (14k): measured lands inside [42–79] range ✓

## Willitllm implications

1. KV cache bandwidth is now accounted for — formula improved for moderate contexts
2. Super-linear degradation at extreme contexts (200k+) not yet modelled
3. MoE models must have `params_b_active` set for accurate estimates
4. Dense models still slightly overestimate gen speed at very large context

---

## Full-context sweeps + general decode model (2026-06-21)

Ran controlled `--fill` sweeps on two very different GPUs to derive a **general** decode-speed
formula (not per-model tuning). Raw run files in `meta/benchmarks/`.

**GPUs:** RTX 3090 (24 GB, 936 GB/s, flash, q4_0 KV) and GTX 1660 Super (6 GB, 336 GB/s, **no
flash**, f16 KV — virtualised host, restricted vCPU, direct GPU PCI passthrough).

**Models swept:** devstral:24b, gemma4:e4b (MoE), llama3.2:1b, llama3.2:3b, gemma3:4b, codellama:13b.

### Key findings

1. **Baseline decode efficiency varies more between models (≈0.55–0.76) than the context-driven
   decline within a model.** Those baselines all fall inside the existing `gen_eff` ranges, so the
   `[lo,hi]` band already brackets them — do **not** re-fit `gen_eff` upward (it breaks low context).

2. **Generation has a per-token attention-compute cost, not just memory streaming.** Full-attention
   models lose efficiency as context grows (devstral:24b effective gen_eff 0.76 → 0.37 over 1k →
   112k). The old weights+KV memory formula over-predicted and its range stopped bracketing the
   measurement at long context.

3. **Sliding-window attention is the generalising signal.** Gemma 2/3/4 attend to a fixed window
   (4096 / 1024 / 512) in most layers, so their decode speed stays nearly flat with context — the
   whole 99.5 → 53.6 t/s drop for gemma4 is explained by the KV term alone, efficiency ~flat
   (0.57 → 0.53). Capping the attention term's context at `sliding_window` makes one formula fit
   both flat (Gemma) and declining (Llama/Mistral/codellama) models. `sliding_window` is scraped
   from `{arch}.attention.sliding_window` (`/api/show`).

### Resulting formula (implemented in `app.calc.js`)

```
attn_ctx = min(maxCtx, sliding_window ?? ∞)
t_mem    = (active_weights_gb + kv_cache_gb) / (bandwidth × gen_eff)
t_attn   = 2 × attn_ctx × block_count × head_count_kv × (key+value) / (tflops × 1e12 × DECODE_ATTN_EFF)
gen      = 1 / (t_mem + t_attn)
DECODE_ATTN_EFF = 0.015   # calibrated; high-ctx bracketing RMS 32% → 15%
```

Validation: the user-facing `[lo,hi]` range brackets 8/9 sweep points across both GPUs (the miss is
llama3.2:1b at 27k, 78.5 vs range-top 78 — see BUG-18). Same `attn_ctx` cap applied to prefill.

### VRAM overhead

llama3.2:3b on the 6 GB GTX was told ~29k fit but **28k spilled to system RAM** (gen → 1.6 t/s).
Idle `nvidia-smi` showed only 5748 MiB free of 6 GB (≈4–6% driver/system reserve), and runtime
overhead exceeded the old 0.5 GB. Raised `OVERHEAD_GB` 0.5 → 0.8. On the smallest cards the unusable
fraction is proportionally largest, so a future refinement could model usable VRAM as a fraction of
rated rather than a fixed GB (BUG-11b).

### Residual gap

devstral-small-2:24b at 200k still measures ~7 t/s vs ~18–23 predicted — a super-linear collapse on
24B-class **dense** models beyond ~115k that the linear attention term does not fully capture (BUG-17).
