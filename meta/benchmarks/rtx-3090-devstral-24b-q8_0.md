# RTX 3090 · devstral:24b · q8_0

Date: 2026-06-21  |  Fill: yes  |  Runs: 1

## Setup

| | |
|---|---|
| GPU | RTX 3090 (24 GB · 936 GB/s) |
| Model | devstral:24b (Q4_K_M, 14.1 GB) |
| KV cache | q8_0 (bpe=1.0) |
| Arch | 40 layers · 8 KV heads · key=128 · ctx_limit=128k |
| Max ctx | 107k tokens (willitllm formula) |

## Measured

| Ctx | Prompt tok | KV GB | Gen avg | Prefill | VRAM |
|---|---|---|---|---|---|
| default | 1,255 | 0.10 | 26.9 | 1742 | 15.9 GB |
| 4k | 4,095 | 0.31 | 46.9 | 1791 | 13.5 GB |
| 16k | 14,609 | 1.11 | 42.1 | 1706 | 14.8 GB |
| 32k | 28,094 | 2.14 | 35.8 | 1448 | 15.9 GB |
| 64k | 55,063 | 4.20 | 27.6 | 1176 | 18.7 GB |

## vs willitllm

gen_eff [0.43, 0.82] for Q4_K_M

| Ctx | Measured | Old est | New est | Old err | New err |
|---|---|---|---|---|---|
| 4k | 46.9 | 28–54 | 28–53 | -39% | -41% |
| 16k | 42.1 | 28–54 | 26–50 | -32% | -38% |
| 32k | 35.8 | 28–54 | 24–46 | -21% | -32% |
| 64k | 27.6 | 28–54 | 21–40 | +3% | -24% |
