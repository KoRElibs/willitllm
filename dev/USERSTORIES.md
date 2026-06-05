# will-it-llm — user stories

Status tags: `done` · `backlog` · `bug`

---

## Personas

| Persona | Question they bring | Page |
|---|---|---|
| **Checker** | Will this model run on my GPU? | `index.html` |
| **Explorer** | What can I actually run well on my GPU? | `index.html` |
| **Optimizer** | How do I squeeze the best config out of what I have? | `index.html` |
| **Buyer** | What GPU do I need for this model? | `buyer.html` |
| **Vibe Coder** | What's the best local coding model for my GPU, and how do I wire it into my editor? | `coder.html` |

---

## Checker — `index.html`

**US-01 — Fit check** `done`
As a Checker I want to know immediately whether a model fits in my GPU's VRAM so I don't waste time trying to run something that won't load.

**US-02 — Max context** `done`
As a Checker I want to see the maximum context window I can safely run so I can judge whether the model is useful for my task.

**US-03 — Ollama command** `done`
As a Checker I want a copy-paste ollama command with the correct `num_ctx` already set so I don't have to calculate or look up the parameter myself.

**US-04 — KV cache setup instructions** `done`
As a Checker I want OS-specific instructions for enabling non-default KV cache precision so I can apply the setting without guessing the env var syntax.

**US-05 — Shareable URL** `done`
As a Checker I want the current selection encoded in the URL so I can share a link that opens the exact same configuration.

---

## Explorer — `index.html`

**US-06 — GPU selection by VRAM** `done`
As an Explorer without a named GPU entry I want to select by VRAM size so I still get a useful (if wide) estimate across models.

**US-07 — Named GPU selection** `done`
As an Explorer with a specific GPU I want to select it by name so I get tight speed estimates rather than wide ranges.

**US-08 — Searchable model list** `done`
As an Explorer browsing 100+ models I want to type a name and filter in real time so I don't have to scroll a long list.

**US-09 — Model colour coding** `done`
As an Explorer I want models colour-coded by context fit (green / amber / orange / red) so I can spot good candidates at a glance.

**US-10 — Target context selector** `backlog`
As an Explorer I want to set a target context so model colour coding reflects whether a model meets my actual needs rather than what percentage of its architectural maximum fits.

Approach: replace % of max with a target context control. Rendered as a **pill row** below the 2×2 control grid, above the memory bar — lower visual weight than the main controls, all options visible at once, active pill in accent colour.

Presets:
```
8k   · a chat          ~25 pages
32k  · a document     ~100 pages    ← default
64k  · The Hobbit     ~200 pages
100k · Harry Potter   ~300 pages
200k · several books  ~600 pages
Max  · full model context            (restores current % of architectural limit behaviour)
```

Colour coding against target: green = fits, amber = within ~50% of target, orange = significantly below, red = doesn't fit at all.

**US-11 — "What fits?" discovery mode** `backlog`
As an Explorer I want to flip the interaction: given my GPU, show all models ranked by context fit %, with speed and quality visible, sortable by fit / speed / quality — so I can discover what to run rather than checking one model at a time.

---

## Optimizer — `index.html`

**US-12 — Scorecard** `done`
As an Optimizer I want an at-a-glance scorecard (speed / sharpness / memory clarity / attention span) so I can compare options without reading the full breakdown.

**US-13 — VRAM breakdown bar** `done`
As an Optimizer I want to see how VRAM is divided between model weights, KV cache, overhead, and free space so I understand the trade-offs.

**US-14 — Formula breakdown** `done`
As an Optimizer I want to see the full context calculation step by step so I can verify the result and understand what drives it.

**US-15 — Model info table** `done`
As an Optimizer I want to see the model's architecture parameters (block count, KV heads, key length, etc.) so I understand what drives the VRAM and context numbers.

**US-16 — Nudge buttons** `done`
As an Optimizer I want one-click nudge buttons that jump to the next faster/sharper/higher/longer variant so I can quickly explore trade-offs without cycling through every option manually.

**US-17 — Nudge button correctness** `bug`
The "faster" nudge disappears after nudging to higher sharpness even when lower-quality variants exist and fit in VRAM. Reproducible with `translategemma`. `updateNudgeButtons` / `groupVariantsSorted` should show "faster" whenever a lower-quality variant that fits exists, regardless of current selection.

**US-18 — Speed in human terms** `done`
As an Optimizer I want speed shown in words/second and pages of context rather than raw tokens/second so the numbers are meaningful without mental conversion.

---

## Buyer — `buyer.html` `backlog`

**US-19 — Minimum VRAM for a model**
As a Buyer I want to select a model and quantization and see the minimum VRAM required so I know what GPU tier to target.

**US-20 — GPU recommendations**
As a Buyer I want to see a ranked list of GPUs that can run my chosen model at my target context length, sorted by fit and speed, so I can make an informed purchase decision.

**US-21 — Context target input**
As a Buyer I want to specify a target context length (e.g. 32k) so the recommendations reflect my actual use case rather than a default.

---

## Vibe Coder — `coder.html` `backlog`

**US-22 — Best coding model for my GPU**
As a Vibe Coder I want to see which Mistral coding models (Devstral, Codestral, etc.) run on my GPU, ranked by the balance of context and speed relevant to agentic coding.

**US-23 — Cline config output**
As a Vibe Coder using Cline I want a ready-to-paste provider config (OpenAI-compatible base URL, model ID) so I can wire up my local model without digging through Cline's settings docs.

**US-24 — Context in coder terms**
As a Vibe Coder I want context window shown in practical coding units (approximate lines of code or files) so I know how much of my codebase the model can see at once.
