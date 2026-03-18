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
bytes_per_element = KV cache precision (2 = f16, 1 = q8_0, 0.5 = q4_0)
bytes_per_token   = block_count × head_count_kv × (key_length + value_length) × bytes_per_element
available_vram    = total_vram − model_weights − 0.5 GB overhead
max_context       = largest power-of-2 that fits in available_vram
```

The result is capped at the model's architectural `context_length` limit.

## Stack

Pure static HTML/CSS/JS — no build step, no dependencies, no server.

| File | Purpose |
|---|---|
| `index.html` | Page structure and UI |
| `app.js` | Calculation logic and rendering |
| `styles.css` | Dark theme styling |
| `models.js` | Model database (architecture params, variants) |
| `libraries.js` | Library metadata (organization, origin, multimodal flag) |
| `gpus.js` | GPU database (VRAM tiers, Flash Attention support) |
| `quantizations.js` | Quantization info (speed/quality ratings, summaries) |
| `scripts/update_models.py` | Maintenance script — scrapes and updates `models.js` |

## Running locally

```bash
git clone https://github.com/KoRElibs/willitllm
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

`scripts/update_models.py` reads architecture parameters and weight sizes directly from the Ollama registry — no model weights are downloaded.

```bash
python scripts/update_models.py                        # dry run — show what would change
python scripts/update_models.py --apply                # write changes to models.js
python scripts/update_models.py --discover --apply     # find and add new model sizes
python scripts/update_models.py --tag llama3.2:3b      # one model only
```

See `update_models_prompt.md` for full workflow documentation.

## Data provenance

All architecture parameters (`block_count`, `head_count_kv`, `key_length`, `value_length`, `context_length`) are read directly from the Ollama registry via the scraper. Weight sizes and quantization variants come from the same source. No data has been manually verified — treat all values as representative rather than authoritative. Organization and origin metadata is AI-generated and may contain errors.

## Contributing

To add a model or library, update `libraries.js` and run `scripts/update_models.py --discover --apply`. See `update_models_prompt.md` for step-by-step instructions.

## License

Licensed under the [European Union Public Licence v1.2 (EUPL-1.2)](https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12). Copyright © 2026 [KoRElibs.com](https://korelibs.com)

The EUPL is an open source licence created by the European Commission and legally reviewed for all EU jurisdictions. It is less widely known than MIT or Apache, but straightforward to understand in practice:

- **Using will-it-llm — including commercially — carries no obligations.** Run it, embed it, build on top of it however you like.
- **It is not viral.** Building a product or larger system that uses will-it-llm does not pull your code under the EUPL. Your surrounding code remains independently licensed.
- **Copyleft applies only if you distribute a modified version of will-it-llm itself.** In that case, your modifications must be released under the EUPL — but only those modifications, not your surrounding systems.

In short: use it freely, build on top of it freely, and only share back if you ship a modified will-it-llm to others.
