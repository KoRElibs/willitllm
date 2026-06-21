# GTX 1660 Super · gemma3:4b · f16

Date: 2026-06-21  |  Fill: yes  |  Runs: 1

## Setup

| | |
|---|---|
| GPU | GTX 1660 Super (6 GB · 336 GB/s) |
| Model | gemma3:4b (Q4_K_M, 3.1 GB) |
| KV cache | f16 (bpe=2.0) |
| Arch | 34 layers · 4 KV heads · key=256 · ctx_limit=128k |
| Max ctx | 16k tokens (willitllm formula) |

## Measured

| Ctx | Prompt tok | KV GB | Gen avg | Prefill | VRAM |
|---|---|---|---|---|---|
| default | 39 | 0.01 | 67.2 | 214 | 4.0 GB |
| 4k | 3,279 | 0.43 | 59.1 | 282 | 4.0 GB |
| 8k | 6,651 | 0.86 | 56.9 | 281 | 4.1 GB |
| 16k | 13,393 | 1.74 | 54.1 | 269 | 4.3 GB |

## vs willitllm

gen_eff [0.43, 0.82] for Q4_K_M

| Ctx | Measured | Old est | New est | Old err | New err |
|---|---|---|---|---|---|
| 4k | 59.1 | 46–89 | 40–76 | -21% | -33% |
| 8k | 56.9 | 46–89 | 35–66 | -18% | -39% |
| 16k | 54.1 | 46–89 | 28–53 | -14% | -49% |
