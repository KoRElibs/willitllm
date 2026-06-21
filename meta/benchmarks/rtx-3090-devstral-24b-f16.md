# RTX 3090 · devstral:24b · f16

Date: 2026-06-21  |  Fill: yes  |  Runs: 1

## Setup

| | |
|---|---|
| GPU | RTX 3090 (24 GB · 936 GB/s) |
| Model | devstral:24b (Q4_K_M, 14.1 GB) |
| KV cache | f16 (bpe=2.0) |
| Arch | 40 layers · 8 KV heads · key=128 · ctx_limit=128k |
| Max ctx | 53k tokens (willitllm formula) |

## Measured

| Ctx | Prompt tok | KV GB | Gen avg | Prefill | VRAM |
|---|---|---|---|---|---|
| default | 1,255 | 0.19 | 52.2 | 1762 | 18.2 GB |
| 4k | 4,095 | 0.62 | 50.9 | 1824 | 13.8 GB |
| 16k | 14,609 | 2.23 | 46.2 | 1736 | 16.0 GB |
| 32k | 28,094 | 4.29 | 41.4 | 1485 | 18.2 GB |
| 48k | 41,579 | 6.34 | 37.5 | 1331 | 20.8 GB |

## vs willitllm

gen_eff [0.43, 0.82] for Q4_K_M

| Ctx | Measured | Old est | New est | Old err | New err |
|---|---|---|---|---|---|
| 4k | 50.9 | 28–54 | 27–52 | -44% | -46% |
| 16k | 46.2 | 28–54 | 24–46 | -38% | -48% |
| 32k | 41.4 | 28–54 | 21–40 | -31% | -49% |
| 48k | 37.5 | 28–54 | 19–35 | -24% | -50% |
