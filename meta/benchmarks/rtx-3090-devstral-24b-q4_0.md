# RTX 3090 · devstral:24b · q4_0

Date: 2026-06-21  |  Fill: yes  |  Runs: 1

## Setup

| | |
|---|---|
| GPU | RTX 3090 (24 GB · 936 GB/s) |
| Model | devstral:24b (Q4_K_M, 14.1 GB) |
| KV cache | q4_0 (bpe=0.5) |
| Arch | 40 layers · 8 KV heads · key=128 · ctx_limit=128k |
| Max ctx | 115k tokens (willitllm formula) |

## Measured

| Ctx | Prompt tok | KV GB | Gen avg | Prefill | VRAM |
|---|---|---|---|---|---|
| default | 1,255 | 0.05 | 50.6 | 1677 | 14.6 GB |
| 4k | 4,095 | 0.16 | 46.9 | 1805 | 13.4 GB |
| 16k | 14,609 | 0.56 | 41.7 | 1719 | 14.2 GB |
| 32k | 28,094 | 1.07 | 35.3 | 1456 | 14.6 GB |
| 64k | 55,063 | 2.10 | 26.8 | 1179 | 16.2 GB |
| 112k | 95,774 | 3.65 | 19.6 | 913 | 18.5 GB |

## vs willitllm

gen_eff [0.43, 0.82] for Q4_K_M

| Ctx | Measured | Old est | New est | Old err | New err |
|---|---|---|---|---|---|
| 4k | 46.9 | 28–54 | 28–54 | -39% | -40% |
| 16k | 41.7 | 28–54 | 27–52 | -32% | -35% |
| 32k | 35.3 | 28–54 | 26–50 | -19% | -26% |
| 64k | 26.8 | 28–54 | 24–46 | +6% | -10% |
| 112k | 19.6 | 28–54 | 22–41 | +45% | +11% |
