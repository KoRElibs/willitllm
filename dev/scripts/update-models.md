# update_models — workflow instructions

Run these steps in order when adding models or refreshing data. All commands run from the repo root with the project venv active.

---

## Rules

- Models must have an Ollama page before they can be added. If not on Ollama yet, stop.
- Never manually edit `data.models.js` — it is scraper-maintained and edits will be overwritten.
- Always add the library to `data.libraries.js` before running the scraper, or it will be skipped.

---

## Files

| file | edit? | purpose |
|---|---|---|
| `data.libraries.js` | yes | library metadata — organization, origin, multimodal flag. Dual-use: browser + scraper (all keys quoted). |
| `data.models.js` | no | model architecture and variants — maintained by scraper |
| `dev/scripts/update_models.py` | no | scraper |

Organization and origin are stored **only** in `data.libraries.js`, not in `data.models.js`.

---

## 1. Discover new models

```bash
python dev/scripts/update_models.py --discover --apply
```

Reads `data.libraries.js`, scrapes ollama.com, inserts missing entries into `data.models.js`.

Watch the output for:
- `partial — missing: <fields>` — architecture data not found, needs manual fill-in (see step 3)
- `no canonical tags found` — scraper could not identify size tags (see known limitations)
- `complete` — library already fully covered, skipped

---

## 2. Refresh quantization variants

```bash
python dev/scripts/update_models.py --variants --apply
# or for a single model:
python dev/scripts/update_models.py --variants --apply --tag llama3.1:8b
```

Adds new variants and updates `weights_gb` if changed beyond tolerance. Does not remove disappeared variants.

---

## 3. Fix partial entries manually

If an entry is missing architecture fields, fill them in directly in `data.models.js`.

| field | source | notes |
|---|---|---|
| `ollama_tag` | ollama.com | `library:tag` — library must exist in `data.libraries.js` |
| `moe` | model card | `true` for Mixture-of-Experts. Omit for dense models |
| `context_length` | blob page / model card | architectural max context window in tokens |
| `params_b` | blob page / model card | total parameters in billions |
| `params_b_active` | model card | active params per forward pass — MoE only |
| `block_count` | blob page | transformer decoder layer count |
| `head_count` | blob page | total query attention heads |
| `head_count_kv` | blob page | KV heads — critical for VRAM formula |
| `embedding_length` | blob page | embedding dimension |
| `key_length` | blob page | `attention.key_length` if explicit, else `embedding_length / head_count` |
| `variants[].group` | ollama.com | variant family for dropdown grouping — e.g. `"(default)"`, `"instruct"`, `"tools"` |

**Finding missing fields:**
- Blob page: `https://ollama.com/library/<name>:<tag>/blobs/<id>`
- HuggingFace `config.json` for `block_count`, `head_count`, `head_count_kv`, `embedding_length`

---

## 4. Known scraper limitations

| issue | affected | workaround |
|---|---|---|
| Version-tagged canonicals (`v2`, `v2.5`) filtered by `_VERSION_RE` | openhermes | manual entry or fix scraper |
| Sub-1B models with MB-sized weights | handled — `_parse_detail` converts MB→GB | — |
| Default variant not re-fetched by `--variants` | any | use `--discover` or patch manually |

---

## 5. Commit

Verify all entries before committing — read-only, prints status of every entry:

```bash
python dev/scripts/update_models.py --verify
git add data.models.js data.libraries.js
git commit -m "Update model database"
```

All entries should report `OK`. Investigate any that don't.
