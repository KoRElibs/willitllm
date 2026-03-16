# willitllm — TODO

## Data quality

- [ ] **openhermes**: scraper skips it because canonical tags are version-based (`v2`, `v2.5`) not size-based. Either remove from `libraries.json` or fix `_VERSION_RE` in `update_models.py` to allow version-tagged canonicals through when no size tags exist.
- [ ] **organization + origin**: all 69 entries have `"organization": "TODO"` and `"origin": null`. Fill these in — see `update_models_prompt.md` for field definitions.

## Scraper improvements

- [ ] **Default variant via fetch_variants**: `fetch_variants` only fetches hyphenated quant variants. The canonical tag itself (e.g. `gemma3:270m`) has its own blob page with weights — this default variant is currently added only during `--discover`. If a model is re-scraped with `--variants` only, the default variant may be missing or stale.
- [ ] **Version tag canonicals**: `_VERSION_RE` filters out tags like `v2`, `v2.5`. For libraries that use version-based tags (openhermes), no canonical tags are found.

## Models to review

- [ ] **phi4**: only `14b` — check if `phi4-mini` or `phi4-reasoning` have been added to ollama
- [ ] **mistral-small**: has `22b` and `24b` — verify these are correct separate releases and not the same model

## UX / frontend

- [ ] **GPU picker redesign**: generic entries by VRAM size + individual card entries sorted alphabetically (discussed, not yet implemented)
- [ ] **KV cache hint**: add quality/precision loss note for q8_0 and q4_0 (discussed, not yet implemented)
- [ ] **bytes_per_element source label**: change from `ollama default` to `from kv cache type`
- [ ] **Remove organization/origin source column**: decided to remove `detail-src` column for those two rows since they are editorial data, not sourced from ollama.com
- [ ] **Absolute context color bands**: discuss whether to replace % of architectural max with absolute bands (< 4k / 4k–16k / > 16k)
