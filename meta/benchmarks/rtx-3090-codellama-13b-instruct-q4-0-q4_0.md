# RTX 3090 · codellama:13b-instruct-q4_0 · q4_0

Date: 2026-06-21  |  Fill: yes  |  Runs: 1

## Setup

| | |
|---|---|
| GPU | RTX 3090 (24 GB · 936 GB/s) |
| Model | codellama:13b-instruct-q4_0 (Q4_0, 6.9 GB) |
| KV cache | q4_0 (bpe=0.5) |
| Arch | 40 layers · 40 KV heads · key=128 · ctx_limit=16k |
| Max ctx | 14k tokens (willitllm formula) |

## Measured

| Ctx | Prompt tok | KV GB | Gen avg | Prefill | VRAM |
|---|---|---|---|---|---|
| default | 53 | 0.01 | 92.2 | 1006 | 10.7 GB |
| 4k | 3,942 | 0.75 | 80.3 | 2785 | 7.8 GB |
| 8k | 7,988 | 1.52 | 70.4 | 2571 | 8.8 GB |
| 13k | 13,725 | 2.62 | 59.7 | 2139 | 10.1 GB |

## vs willitllm

gen_eff [0.43, 0.82] for Q4_0

| Ctx | Measured | Old est | New est | Old err | New err |
|---|---|---|---|---|---|
| 4k | 80.3 | 59–112 | 53–100 | -27% | -34% |
| 8k | 70.4 | 59–112 | 48–91 | -17% | -32% |
| 13k | 59.7 | 59–112 | 42–81 | -2% | -29% |
