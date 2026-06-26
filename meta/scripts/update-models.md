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
| `meta/scripts/update_models.py` | no | scraper |

Organization and origin are stored **only** in `data.libraries.js`, not in `data.models.js`.

---

## 1. Discover new models

```bash
python meta/scripts/update_models.py --discover --apply
```

Reads `data.libraries.js`, scrapes ollama.com, inserts missing entries into `data.models.js`.

Watch the output for:
- `partial — missing: <fields>` — architecture data not found, needs manual fill-in (see step 3)
- `no canonical tags found` — scraper could not identify size tags (see known limitations)
- `complete` — library already fully covered, skipped

---

## 2. Refresh quantization variants

```bash
python meta/scripts/update_models.py --variants --apply
# or for a single model:
python meta/scripts/update_models.py --variants --apply --tag llama3.1:8b
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

## 5. Adding a new agent model

A model qualifies as an agent model if it was purpose-trained for agentic coding loops (tool
calling, multi-step planning, file edits). Example: devstral, devstral-small-2.

Two independent steps — both are required for the model to appear everywhere correctly:

**Step A — capability (automatic):** Run `--capabilities --apply`. If the model has the `tools`
badge on ollama.com, it will appear in the AGENT filter on index.html automatically.

**Step B — coding role (manual):** Open `data.libraries.js` and set `"coding_role": "agent"` on
the library entry. This makes the model appear in the AGENTS section on coder.html with Cline +
Continue agentic config. Do this only for models purpose-trained for coding agent loops — do not
set it for general tool-calling models (e.g. command-r) that happen to support function calling.

After both steps, verify:
- index.html AGENT filter shows the model
- coder.html shows the model in the top AGENTS section with ★ recommended eligibility
- Tooltip on the badge says "AGENT" (not CODE or FIM)

---

## 6. Capability scoring (researched & cited — do this for EVERY model)

`data.libraries.js` carries a capability benchmark per library that drives the
ranking on coder.html and the scorecard chip on index.html. **This must be done
each time a model is added or refreshed.** It is the one piece of data that is
neither scraped from ollama nor computed from hardware — it is *researched from a
citable source and recorded with that source*.

### Fields (in `data.libraries.js`, per library)

| field | meaning |
|---|---|
| `capability` | headline benchmark score, 0–100 |
| `capability_metric` | which benchmark (see canonical table below) |
| `capability_protocol` | **mandatory** when `capability` is set — `pass@1`, `5-shot`, etc. |
| `capability_ref` | the exact model the score is for (e.g. `"phi-4 (14B)"`) — scores are family-level |
| `capability_source` | **mandatory** URL the number came from |

The scraper preserves these across runs (`write_libraries_js`). Re-running
`--discover` / `--capabilities` / `--variants` will NOT erase them.

### Canonical {metric, protocol} per role — match it exactly or don't score

The ranking (`ROLE_CANONICAL` in `coder.rank.js`) uses a score **only** when its
`capability_metric` AND `capability_protocol` equal the role's canonical pair below.
This is what guarantees scores within a bucket are like-for-like (never pass@1 vs
pass@5, never SWE-bench vs HumanEval).

| role / type | canonical metric | canonical protocol | typical source |
|---|---|---|---|
| `coding_role: agent` | `swe-bench-verified` | `pass@1` | HF model card / vendor blog |
| `coding_role: code` | `humaneval` | `pass@1` | HF model card / paper |
| `coding_role: fim` | **— none —** | — | **never scored** (no benchmark measures autocomplete) |
| general (no `coding_role`) | `mmlu` | `5-shot` | HF model card / paper |

**If the only published number isn't the canonical pair — e.g. HumanEval+, or
pass@5, or pass@10 — leave the capability fields OUT entirely.** A model with no
score is ranked on speed + context + recency; that is honest. Recording a
non-comparable number (even with a caveat in `capability_ref`) is **not** an
option, because the ranking math can't see prose and would compare it as if it
were canonical. Unscored ≠ worst — it just leans on the deterministic signals.

### Procedure (must be reproducible, never from chat)

1. **Source of truth = the official HuggingFace model card** for the exact
   weights ollama serves, or the vendor's paper/blog. Open the page and read the
   benchmark table. The Ollama default tag maps to a specific HF repo — match it
   (e.g. `devstral-small-2:24b` → `mistralai/Devstral-Small-2-24B-Instruct-2512`;
   `starcoder2:15b` is the **base** model, not the `-instruct` finetune).
2. Record `capability`, `capability_metric`, `capability_ref` (the measured
   model), and `capability_source` (the URL).
3. **Cross-check a second source** when the card is ambiguous (e.g. a leaderboard
   such as the official SWE-bench or Aider polyglot board). Prefer official over
   third-party blogs.
4. If no score is published anywhere, **leave the fields out** and note it — the
   ranking falls back to the quant-quality proxy + release date for that model.
   Do not invent a number.

> **Hard rule:** the number must come from a citable web page recorded in
> `capability_source`. **Never** fill it from an LLM's memory, from this chat, or
> from "what sounds right." If you can't cite it, it doesn't go in.

### Cross-check every number against its own source (mandatory, EVERY update)

Recording a `capability_source` is not enough — the number you wrote must actually
appear on that page. **Before every commit that touches capability fields, open
each `capability_source` and read the figure back.** Adding a model and refreshing
an existing one both require this; a citation that no longer matches its source is
worse than no citation, because it looks trustworthy.

For each scored library, confirm all of:

- [ ] The **number** on the page equals `capability` (±0 — exact, not "about right").
- [ ] The page measures the **same weights** named in `capability_ref` (size,
      base-vs-instruct, version date).
- [ ] The recorded `capability_metric` and `capability_protocol` are the role's
      **canonical pair** (see table above). If the page only publishes something
      else — HumanEval **pass@5**, HumanEval+, etc. — **do not record it**; leave
      the capability fields out and let the model rank unscored. Pass@5 ≫ pass@1,
      so a non-canonical number silently inflates rank, and the ranking math can't
      see a caveat buried in `capability_ref`.
- [ ] If the source disagrees with what's recorded, **fix the data to match the
      source** (or change the source) — never leave them inconsistent.

This is reproducible web work: fetch the URL, find the row, compare. It is the
single check that keeps the whole feature honest. A full audit was run 2026-06
(all 12 sources opened); it caught `codegemma` recorded as 60.4 when its cited HF
card says **56.1** (corrected), and `mistral-small3.2`'s headline 92.9 being
HumanEval+ **pass@5** — not the canonical HumanEval pass@1 — so it was **dropped to
unscored** rather than left to lead its bucket on a non-comparable number.

### AI-reasoning guidance (when an assistant does this step)

An assistant asked to add/refresh models MUST, for each library:

- Use web tools (search + fetch) — not recall — to find the score. Search e.g.
  `"<model> SWE-bench Verified"` or `"<model> HumanEval pass@1"`, then **open the
  primary source** (HF card / paper) and read the number directly; don't trust a
  search snippet alone.
- Confirm you're reading the **same weights** ollama serves (size, base-vs-instruct,
  version date) — `capability_ref` should name it exactly.
- Record the URL in `capability_source`. Two independent sources should agree;
  if they conflict, prefer the official card and note the discrepancy.
- Caveats to honour for trust: HumanEval is **saturated** (top models cluster
  80–93%, so it barely separates strong coders — SWE-bench Verified is the better
  agentic signal); benchmark scores are **family-level**, so a library's smaller
  sizes really score lower than `capability_ref`.

### Release date (derived, no action needed)

Ranking breaks ties by release date, parsed automatically from the `YYMM` in
variant group/tags (`small-2505` → 2025-05) by `releasedRank()` in
`coder.rank.js`. Nothing to curate — just make sure the scraper captured the
dated variant tags.

---

## 7. Commit

Verify all entries before committing — read-only, prints status of every entry:

```bash
python meta/scripts/update_models.py --verify
git add data.models.js data.libraries.js
git commit -m "Update model database"
```

All entries should report `OK`. Investigate any that don't.
