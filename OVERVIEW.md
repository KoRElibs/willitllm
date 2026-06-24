# will-it-llm — project overview

**willitllm.com** — pick a GPU, pick a model, find out if it runs locally and how much context fits.

Two pages, no backend, no build step, no dependencies. Pure static HTML/CSS/JS.

---

## What it does

**`index.html`** — fit checker
The user selects a GPU and an ollama model. The page instantly shows:
- Does the model fit in VRAM?
- How many context tokens fit in the remaining VRAM?
- The exact `ollama run` command at that context size
- Speed estimates (generation and processing) in human-friendly units
- A four-dimension scorecard (speed / quality / precision / context fit)

**`coder.html`** — local coding model picker
The user selects a GPU. The page shows all fitting coding models ranked for agentic performance, with ready-to-paste Cline and Continue configs for VS Code and JetBrains.

---

## Core formula

```
available_bytes = (vram_gb − 0.8 overhead − weights_gb) × 1024³
bytes_per_token = block_count × head_count_kv × (key_length + value_length) × bytes_per_element
raw_max_tokens  = available_bytes / bytes_per_token
max_ctx         = floor(raw_max_tokens × 0.9 / 128) × 128
```

0.8 GB is the fixed overhead reservation; 0.9 is a 10% safety factor; 128 is the rounding granularity.
Full derivation, all constants, and all edge cases: `SPEC.md §4`.

---

## File map

### Pages
| File | Purpose |
|---|---|
| `index.html` | Fit checker — structure only, no logic |
| `coder.html` | Coding model picker — structure only, no logic |
| `styles.css` | All styling — shared between both pages |

### Shared data (loaded by both pages)
| File | Global | Contents |
|---|---|---|
| `data.calc-constants.js` | `CALC_CONSTANTS` | Formula constants (overhead, safety factor, rounding, attn efficiency) — single source of truth for browser and Python tooling |
| `data.gpus.js` | `GPUS` | GPU specs: VRAM, bandwidth, TFLOPS, Flash Attention |
| `data.libraries.js` | `LIBRARIES` | Model library metadata: org, origin, coding_role, capabilities |
| `data.models.js` | `MODELS` | Model architecture parameters + quantization variants |
| `data.quantizations.js` | `QUANT_INFO` | Quantization speed/quality ratings and efficiency ranges |
| `data.flags.js` | `FLAGS`, `flagFor()` | Country → flag emoji map |
| `app.calc.js` | — | Pure calculation and formatting helpers (shared) |

### index.html only
| File | Purpose |
|---|---|
| `data.kv-cache.js` | KV cache precision options (f16 / q8_0 / q4_0) |
| `app.render.js` | DOM rendering functions |
| `app.ui.js` | Dropdown population, combobox, nudge buttons |
| `app.js` | Entry point: shared state, render orchestrator, init |

### coder.html only
| File | Purpose |
|---|---|
| `coder.js` | Ranking, config output, GPU selector, init |

### Maintenance
| File | Purpose |
|---|---|
| `meta/scripts/update_models.py` | Scrapes ollama.com — maintains `data.models.js` and `data.libraries.js` |
| `meta/scripts/benchmark.py` | Runs generation/prefill benchmarks against a live ollama instance |
| `meta/deploy/deploy.sh` | Deploys to production (requires `meta/deploy/deploy.cfg` — not in repo) |

---

## Running locally

No install needed — open `index.html` directly in a browser. Or with a local server:

```bash
python -m http.server
# or
npx serve .
```

---

## Document map

| Document | Audience | Contents |
|---|---|---|
| `README.md` | Anyone / GitHub | Public intro, stack, run, contributing, license |
| `OVERVIEW.md` | Quick read / low-context | This file — shape of the project, file map, document map |
| `SPEC.md` | Full read / AI reproducibility | Complete spec: data structures, formulas, all UI behaviour |
| `AGENTS.md` | AI agents | Read order, tools, maintenance rules, agent-specific setup |
| `meta/FEATURES.md` | Dev / AI | User story tracker (done / backlog) |
| `meta/BUGS.md` | Dev / AI | Bug tracker (open / fixed) |
| `meta/UX-FINDINGS.md` | Design | UX testing findings and improvement suggestions |
| `meta/knowledge/external-tools.md` | Dev / AI | Cline, Continue, Ollama, editor ecosystem reference |
| `meta/knowledge/nvidia-tflops-derived.md` | Dev / AI | Derived GPU FP16 TFLOPS values and conversion formulas |
| `meta/knowledge/nvidia-geforce-compare.md` | Read-only | NVIDIA compare page source data — never edit as side effect |
| `meta/scripts/update-models.md` | Dev / AI | Scraper workflow — how to add and refresh models |

---

## Maintenance rule

Every code change must update at least one of:
- `SPEC.md` — if any described behaviour, formula, data structure, or constant changed
- `meta/FEATURES.md` — if a feature was implemented or planned
- `meta/BUGS.md` — if a bug was found or fixed

Stale documentation actively misleads — if you changed the code, you changed at least one of these files.
