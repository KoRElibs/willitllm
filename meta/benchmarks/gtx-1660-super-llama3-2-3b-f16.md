# GTX 1660 Super · llama3.2:3b · f16

Date: 2026-06-21  |  Fill: yes  |  Runs: 1

## Setup

| | |
|---|---|
| GPU | GTX 1660 Super (6 GB · 336 GB/s) |
| Model | llama3.2:3b (Q4_K_M, 1.9 GB) |
| KV cache | f16 (bpe=2.0) |
| Arch | 28 layers · 8 KV heads · key=128 · ctx_limit=128k |
| Max ctx | 29k tokens (willitllm formula) |

## Measured

| Ctx | Prompt tok | KV GB | Gen avg | Prefill | VRAM |
|---|---|---|---|---|---|
| default | 55 | 0.01 | 94.4 | 280 | 2.6 GB |
| 4k | 3,295 | 0.35 | 77.8 | 292 | 2.6 GB |
| 8k | 6,667 | 0.71 | 66.4 | 202 | 3.2 GB |
| 28k | 23,793 | 2.54 | 1.6 | 141 | 4.8 GB ⚠ SPILL |

## vs willitllm

gen_eff [0.43, 0.82] for Q4_K_M

| Ctx | Measured | Old est | New est | Old err | New err |
|---|---|---|---|---|---|
| 4k | 77.8 | 77–146 | 62–119 | -1% | -20% |
| 8k | 66.4 | 77–146 | 52–100 | +16% | -21% |
| 28k | 1.6 | 77–146 | 29–55 | +4764% | +1737% |
