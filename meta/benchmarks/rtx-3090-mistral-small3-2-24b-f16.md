# RTX 3090 · mistral-small3.2:24b · f16

Date: 2026-06-21  |  Fill: yes  |  Runs: 1

## Setup

| | |
|---|---|
| GPU | RTX 3090 (24 GB · 936 GB/s) |
| Model | mistral-small3.2:24b (Q4_K_M, 14.1 GB) |
| KV cache | f16 (bpe=2.0) |
| Arch | 40 layers · 8 KV heads · key=128 · ctx_limit=128k |
| Max ctx | 53k tokens (willitllm formula) |

## Measured

| Ctx | Prompt tok | KV GB | Gen avg | Prefill | VRAM |
|---|---|---|---|---|---|
| default | 537 | 0.08 | 52.4 | 1610 | 18.3 GB |
| 4k | 3,777 | 0.58 | 50.6 | 1800 | 13.9 GB |
| 16k | 13,891 | 2.12 | 46.2 | 1733 | 16.0 GB |
| 32k | 27,376 | 4.18 | 41.4 | 1487 | 18.3 GB |
| 46k | 39,913 | 6.09 | 37.7 | 1343 | 20.7 GB |

## vs willitllm

gen_eff [0.43, 0.82] for Q4_K_M

| Ctx | Measured | Old est | New est | Old err | New err |
|---|---|---|---|---|---|
| 4k | 50.6 | 28–54 | 27–52 | -44% | -46% |
| 16k | 46.2 | 28–54 | 24–46 | -38% | -48% |
| 32k | 41.4 | 28–54 | 21–40 | -31% | -49% |
| 46k | 37.7 | 28–54 | 19–36 | -25% | -50% |
