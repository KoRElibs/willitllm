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
| default | 1,255 | 0.05 | 50.6 | 1694 | 14.6 GB |
| 4k | 4,095 | 0.16 | 46.6 | 1782 | 13.4 GB |
| 16k | 14,609 | 0.56 | 41.4 | 1697 | 14.2 GB |
| 32k | 28,094 | 1.07 | 35.1 | 1443 | 14.6 GB |
| 64k | 55,063 | 2.10 | 26.7 | 1172 | 16.2 GB |

## vs willitllm

gen_eff [0.43, 0.82] for Q4_K_M

| Ctx | Measured | Old est | New est | Old err | New err |
|---|---|---|---|---|---|
| 4k | 46.6 | 28–54 | 28–54 | -39% | -40% |
| 16k | 41.4 | 28–54 | 27–52 | -31% | -34% |
| 32k | 35.1 | 28–54 | 26–50 | -19% | -25% |
| 64k | 26.7 | 28–54 | 24–46 | +7% | -9% |
