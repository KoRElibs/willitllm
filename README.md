# will-it-llm

Source code for **[willitllm.com](https://willitllm.com)** — pick a model, enter your GPU's VRAM, find out if it runs and how much context you actually get.

## What it does

Given a GPU's VRAM and an LLM model, will-it-llm calculates:

- Whether the model fits in your VRAM at all
- The maximum safe KV cache context window you can run
- How that context compares to the model's architectural limit
- The exact `ollama` command to run it at that context size

## How the calculation works

```
bytes_per_token = layers × kv_heads × head_dim × 2 (K+V) × 2 (fp16)
available_vram  = (total_vram − model_weights) × 0.92  ← 8% overhead reserve
max_context     = largest power-of-2 that fits in available_vram
```

The result is capped at the model's architectural `max_position_embeddings` limit where known.

## Stack

Pure static HTML/CSS/JS — no build step, no dependencies, no server.

| File | Purpose |
|---|---|
| `index.html` | Page structure and UI |
| `app.js` | Calculation logic and rendering |
| `styles.css` | Dark theme styling |
| `models.js` | Model database (architecture params, metadata) |
| `gpus.js` | GPU database (VRAM tiers, Flash Attention support) |
| `scripts/update_models.py` | Maintenance script — verifies and updates `models.js` |

## Running locally

```bash
git clone https://github.com/your-org/willitllm
cd willitllm
# open index.html in a browser — no server needed
```

Or with a local server:

```bash
npx serve .
# or
python -m http.server
```

## Updating model data

`scripts/update_models.py` cross-checks `models.js` against two live sources without downloading any model weights:

- **HuggingFace config.json** — architecture params (layers, heads, head_dim, max_position_embeddings)
- **Ollama registry manifest** — weight size and quantization

```bash
python scripts/update_models.py              # dry run — show diffs
python scripts/update_models.py --apply      # write changes to models.js
python scripts/update_models.py --tag llama3.2:1b   # one model only
```

Requires internet access and `ollama` installed (not necessarily running).

## Data provenance

Model data in `models.js` was compiled by AI (Claude) from training knowledge and cross-checked against HuggingFace `config.json` via the update script. It has not been manually verified. Architecture parameters sourced from HuggingFace; weight sizes and quantization from the Ollama registry. Curated metadata (organization, origin, specialty) is AI-generated and may contain errors.

## Contributing

To add or correct a model, edit `models.js` directly and run `scripts/update_models.py --apply` to verify the architecture parameters against HuggingFace. Each model entry documents all required fields in the file header comment.

## License

Licensed under the [European Union Public Licence v1.2 (EUPL-1.2)](https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12). Copyright © 2026 [KoRElibs.com](https://korelibs.com)

The EUPL is an open source licence created by the European Commission and legally reviewed for all EU jurisdictions. It is less widely known than MIT or Apache, but straightforward to understand in practice:

- **Using will-it-llm — including commercially — carries no obligations.** Run it, embed it, build on top of it however you like.
- **It is not viral.** Building a product or larger system that uses will-it-llm does not pull your code under the EUPL. Your surrounding code remains independently licensed.
- **Copyleft applies only if you distribute a modified version of will-it-llm itself.** In that case, your modifications must be released under the EUPL — but only those modifications, not your surrounding systems.

In short: use it freely, build on top of it freely, and only share back if you ship a modified will-it-llm to others.
