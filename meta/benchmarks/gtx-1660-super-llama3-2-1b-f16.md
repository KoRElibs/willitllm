# GTX 1660 Super · llama3.2:1b · f16

Date: 2026-06-21  |  Fill: yes  |  Runs: 1

## Setup

| | |
|---|---|
| GPU | GTX 1660 Super (6 GB · 336 GB/s) |
| Model | llama3.2:1b (Q8_0, 1.9 GB) |
| KV cache | f16 (bpe=2.0) |
| Arch | 16 layers · 8 KV heads · key=64 · ctx_limit=128k |
| Max ctx | 104k tokens (willitllm formula) |

## Measured

| Ctx | Prompt tok | KV GB | Gen avg | Prefill | VRAM |
|---|---|---|---|---|---|
| default | 55 | 0.00 | 174.0 | 660 | 1.6 GB |
| 4k | 3,295 | 0.10 | 150.6 | 841 | 1.6 GB |
| 8k | 6,667 | 0.20 | 132.1 | 726 | 2.0 GB |
| 32k | 26,893 | 0.82 | 78.5 | 374 | 4.3 GB |

## vs willitllm

gen_eff [0.5, 0.9] for Q8_0

| Ctx | Measured | Old est | New est | Old err | New err |
|---|---|---|---|---|---|
| 4k | 150.6 | 89–161 | 84–151 | -41% | -44% |
| 8k | 132.1 | 89–161 | 79–142 | -32% | -40% |
| 32k | 78.5 | 89–161 | 58–105 | +14% | -26% |
