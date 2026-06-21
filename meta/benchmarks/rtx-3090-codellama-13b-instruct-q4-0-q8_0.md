# RTX 3090 · codellama:13b-instruct-q4_0 · q8_0

Date: 2026-06-21  |  Fill: yes  |  Runs: 1

## Setup

| | |
|---|---|
| GPU | RTX 3090 (24 GB · 936 GB/s) |
| Model | codellama:13b-instruct-q4_0 (Q4_0, 6.9 GB) |
| KV cache | q8_0 (bpe=1.0) |
| Arch | 40 layers · 40 KV heads · key=128 · ctx_limit=16k |
| Max ctx | 14k tokens (willitllm formula) |

## Measured

| Ctx | Prompt tok | KV GB | Gen avg | Prefill | VRAM |
|---|---|---|---|---|---|
| default | 53 | 0.02 | 90.8 | 852 | 13.8 GB |
| 4k | 3,942 | 1.50 | 76.5 | 2731 | 8.6 GB |
| 8k | 7,988 | 3.05 | 65.6 | 2533 | 10.3 GB |
| 13k | 13,725 | 5.24 | 54.5 | 2112 | 12.8 GB |

## vs willitllm

gen_eff [0.43, 0.82] for Q4_0

| Ctx | Measured | Old est | New est | Old err | New err |
|---|---|---|---|---|---|
| 4k | 76.5 | 59–112 | 48–91 | -23% | -38% |
| 8k | 65.6 | 59–112 | 40–77 | -11% | -39% |
| 13k | 54.5 | 59–112 | 33–63 | +8% | -40% |
