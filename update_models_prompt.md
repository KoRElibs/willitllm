# update_models — periodic run instructions

Use this prompt when asked to update the model database. Follow each step in order.

---

## 0. Prerequisites

```bash
cd c:/repos/willitllm
# activate venv if needed
source venv/Scripts/activate   # Windows bash
```

---

## 1. Discover new models

Run one library at a time with a few seconds between each to avoid rate limiting.
The script reads `libraries.json` for the authoritative library list.

```bash
python scripts/update_models.py --discover --apply --library <name>
```

Or run all at once (the script is polite by default):

```bash
python scripts/update_models.py --discover --apply
```

**What it does:**
- Scrapes `https://ollama.com/library/<name>` for canonical tags (e.g. `8b`, `70b`)
- Fetches architecture fields from the blob page for each canonical tag
- Fetches the default variant's weights_gb
- Inserts new entries into `models.js` — skips entries that already exist

**Watch the output for:**
- `partial — missing: <fields>` — architecture data not found on ollama, needs manual fill-in
- `no canonical tags found` — scraper could not identify size/name tags (version-tagged libraries like openhermes)
- `complete` — already in models.js, skipped

---

## 2. Refresh quantization variants

For entries already in `models.js`, fetch updated variant lists:

```bash
python scripts/update_models.py --variants --apply
```

Or for a single model:

```bash
python scripts/update_models.py --variants --apply --tag llama3.1:8b
```

**What it does:**
- Scrapes hyphenated quant tags for each existing model
- Adds new variants, updates weights_gb if changed beyond GB_TOLERANCE (0.15 GB)
- Does NOT remove variants that have disappeared

---

## 3. Verify the database

```bash
python scripts/update_models.py
```

All entries should show `OK`. Investigate any that don't.

---

## 4. Fix partial entries manually

If an entry is missing architecture fields, fill them in directly in `models.js`.

**Field reference:**

| field | source | notes |
|---|---|---|
| `ollama_tag` | ollama.com | `library:tag` format |
| `origin` | HuggingFace / paper | base model origin, e.g. `"meta-llama/Meta-Llama-3.1-8B"` |
| `organization` | public knowledge | org name, e.g. `"Meta"`, `"Mistral AI"`, `"Google"` |
| `max_context` | model card / blob page | architectural max context window in tokens |
| `params_b` | model card | total parameter count in billions |
| `params_b_active` | model card | active params for MoE models only |
| `moe` | model card | `true` for Mixture-of-Experts models |
| `layers` | model card / config.json | `num_hidden_layers` |
| `num_attention_heads` | model card / config.json | total attention heads |
| `num_key_value_heads` | model card / config.json | KV heads (equals `num_attention_heads` for MHA; less for GQA) |
| `hidden_size` | model card / config.json | embedding / model dimension |
| `head_dim` | derived or config | `hidden_size / num_attention_heads` unless overridden |
| `variants` | ollama.com | `[{tag, quantization, weights_gb}]` — first entry is the default |

**Finding missing fields:**
- Check `https://ollama.com/library/<name>:<tag>` — the blob page sometimes shows architecture
- Check the HuggingFace model card at `https://huggingface.co/<origin>`
- Check `config.json` on HuggingFace for `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `hidden_size`

---

## 5. Known scraper limitations

| issue | affected models | workaround |
|---|---|---|
| Version-tagged canonicals (`v2`, `v2.5`) filtered by `_VERSION_RE` | openhermes | manual entry or fix scraper |
| MB-sized weights (sub-1B models) | gemma3:270m fixed, others possible | `_parse_detail` now handles MB |
| Default variant not re-fetched by `--variants` | any | use `--discover` or add manually |

---

## 6. After updating

- Run `python scripts/update_models.py` to confirm all entries are `OK`
- Check `todo.md` for any quality issues to address
- Commit: `git add models.js && git commit -m "Update model database"`
