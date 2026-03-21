# update_models — periodic run instructions

Use this prompt when asked to update the model database. Follow each step in order.

---

## File structure

| file | purpose | edit? |
|---|---|---|
| `data.libraries.js` | library metadata (org, origin, multimodal flag) — dual-use: loaded by browser and parsed by scraper (all keys are quoted strings) | yes — add/edit libraries here |
| `data.models.js` | one entry per canonical tag per library | no — maintained by scraper |
| `scripts/update_models.py` | scraper | no |

Organization and origin country are stored **only** in `data.libraries.js` — not in `data.models.js`.

---

## 0. Prerequisites

```bash
cd c:/repos/willitllm
source venv/Scripts/activate   # Windows bash
```

---

## 1. Discover new models

```bash
python scripts/update_models.py --discover --apply
```

The scraper reads `data.libraries.js`, then inserts missing model entries into `data.models.js`.

**Watch the output for:**
- `partial — missing: <fields>` — architecture data not found, needs manual fill-in (see step 4)
- `no canonical tags found` — scraper could not identify size/name tags (see known limitations)
- `complete` — library already fully covered, skipped

---

## 2. Refresh quantization variants

```bash
python scripts/update_models.py --variants --apply
# or for a single model:
python scripts/update_models.py --variants --apply --tag llama3.1:8b
```

Adds new variants and updates weights_gb if changed beyond tolerance. Does not remove disappeared variants.

---

## 3. Verify

```bash
python scripts/update_models.py
```

All entries should show `OK`. Investigate any that don't.

---

## 4. Fix partial entries manually

If an entry is missing architecture fields, fill them in directly in `data.models.js`.

| field | source | notes |
|---|---|---|
| `ollama_tag` | ollama.com | `library:tag` — library must exist in data.libraries.js |
| `moe` | model card | `true` for Mixture-of-Experts. Omit for dense models |
| `context_length` | blob page / model card | architectural max context window in tokens |
| `params_b` | blob page / model card | total parameters in billions |
| `params_b_active` | model card | active params per forward pass — MoE only |
| `block_count` | blob page | transformer decoder layer count |
| `head_count` | blob page | total query heads |
| `head_count_kv` | blob page | KV heads — critical for VRAM formula |
| `embedding_length` | blob page | embedding dimension |
| `key_length` | blob page | `attention.key_length` if explicit, else `embedding_length / head_count` |
| `variants` | ollama.com | `[{tag, quantization, weights_gb}]` — first entry is the default |

**Finding missing fields:**
- Blob page: `https://ollama.com/library/<name>:<tag>/blobs/<id>`
- HuggingFace config.json for `block_count`, `head_count`, `head_count_kv`, `embedding_length`

---

## 5. Known scraper limitations

| issue | affected | workaround |
|---|---|---|
| Version-tagged canonicals (`v2`, `v2.5`) filtered by `_VERSION_RE` | openhermes | manual entry or fix scraper |
| Sub-1B models with MB-sized weights | handled — `_parse_detail` converts MB→GB | — |
| Default variant not re-fetched by `--variants` | any | use `--discover` or patch manually |

---

## 6. After updating

```bash
python scripts/update_models.py   # confirm all OK
git add data.models.js data.libraries.js
git commit -m "Update model database"
```
