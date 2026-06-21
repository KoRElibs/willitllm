# RTX 3090 · gemma4:e4b-it-q8_0 · q4_0

Date: 2026-06-21  |  Fill: yes  |  Runs: 1

## Setup

| | |
|---|---|
| GPU | RTX 3090 (24 GB · 936 GB/s) |
| Model | gemma4:e4b-it-q8_0 (Q8_0, 10.8 GB) |
| KV cache | q4_0 (bpe=0.5) |
| Arch | 42 layers · 2 KV heads · key=512 · ctx_limit=128k |
| Max ctx | 115k tokens (willitllm formula) |

## Measured

| Ctx | Prompt tok | KV GB | Gen avg | Prefill | VRAM |
|---|---|---|---|---|---|
| default | 46 | 0.00 | 99.5 | 684 | 5.1 GB |
| 4k | 3,286 | 0.13 | 95.3 | 6365 | 4.9 GB |
| 16k | 13,400 | 0.54 | 87.7 | 6594 | 5.1 GB |
| 32k | 26,885 | 1.08 | 79.3 | 5948 | 5.1 GB |
| 64k | 53,854 | 2.16 | 66.5 | 4885 | 5.3 GB |
| 112k | 94,565 | 3.79 | 53.6 | 3707 | 5.4 GB |

## vs willitllm

gen_eff [0.5, 0.9] for Q8_0

| Ctx | Measured | Old est | New est | Old err | New err |
|---|---|---|---|---|---|
| 4k | 95.3 | 43–78 | 43–77 | -55% | -55% |
| 16k | 87.7 | 43–78 | 41–73 | -51% | -54% |
| 32k | 79.3 | 43–78 | 39–69 | -46% | -51% |
| 64k | 66.5 | 43–78 | 35–63 | -35% | -48% |
| 112k | 53.6 | 43–78 | 30–55 | -19% | -43% |
