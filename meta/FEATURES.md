# will-it-llm — features

Status tags: `done` · `backlog`

Bugs live in `BUGS.md`. Keep this file updated on every change — see `SPEC.md §12`.

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
As a Checker I want copy-paste ollama commands with the correct context length already set so I don't have to calculate or look up the parameter myself.

The context length is set via `OLLAMA_CONTEXT_LENGTH` on the **server**, not `num_ctx` on the client —
see US-04 and BUG-26.

**US-04 — KV cache setup instructions** `done`
As a Checker I want OS-specific instructions for enabling non-default KV cache precision so I can apply the setting without guessing the env var syntax.

Implemented: a `▸ how to run it` toggle at the foot of the result card opens `#runSection` below it —
a `Show setup for` dropdown (Linux · quick start / Linux · systemd service / macOS / Windows) plus
the setup block. Every Ollama setting is emitted on whatever starts the *server* for that platform
(`ollama serve`, the systemd drop-in, System Properties) — `ollama run` is a client and ignores
env vars prefixed to it. `OLLAMA_FLASH_ATTENTION=1` ships alongside any non-f16 cache type, without
which Ollama silently ignores it (BUG-26). No option appends to a shell rc file (BUG-25).
Index-only: `coder.html` states the assumption and links back here rather than keeping a second copy.
See SPEC §7.5b.

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

**US-10 — Target context selector** `done`
As an Explorer I want to set a target context so model colour coding reflects whether a model meets my actual needs rather than what percentage of its architectural maximum fits.

Dropdown options:
```
as much as fits                     ← default (neutral, no explicit target)
8k   · a chat          ~25 pages
32k  · a document     ~100 pages
64k  · The Hobbit     ~200 pages
100k · Harry Potter   ~300 pages
200k · several books  ~600 pages
full model context
```
Colour: green = fits target, amber = within 50% of target, orange = below. The default
**`as much as fits`** sets no target — the model is shown at a sensible `min(32k, model max)` f16
default and its fit reads **neutral** (no green/amber/orange verdict on a target the user never set),
while the model list still colours models by that baseline. `full model context` maximises context
(q4_0) and grades against the model's whole trained window. See SPEC §6 (`effectiveTargetCtx`,
`contextFitColor`).

**Context gap display in aside:** When the selected model cannot fully meet the target, the aside context stat shows `~X / ~Y pages` (achieved / target) colored by fit quality (green / amber / orange). The sub-label appends `· Z% of target` to make the shortfall explicit. This gives the user an immediate, specific comparison rather than an abstract score.

**US-26 — Capability filter** `done`
As an Explorer I want to filter the model list by capability (tools / vision / thinking) so I only see models relevant to my use case without scrolling through the full list.

Pill row in bottom-left of the 2×2 control grid: `coding` · `vision` · `thinking`, plus a `clear`
pill. Multi-select with AND logic — selecting multiple pills shows only models that have **all**
selected capabilities. `coding` (synthetic — any `coding_role`) is styled identically to the others
(cyan accent, no special green). The `clear` pill is **hidden until a filter is active**, then shown
**last** as a reset (clicking it clears the filter and it hides again). Capabilities sourced from
ollama.com/library `x-test-capability` badges. Embedding models hidden from the list entirely. Pill
state resets on load; not encoded in URL hash (a UI filter, not a selection).

When GPU, target context, or capability pills change, the model dropdown auto-selects the first fitting (non-✗) model. If the currently selected model no longer fits or is filtered out, it is replaced by the best available option automatically.

The Variant selector is moved to the geek section (shown only in geek mode). Default variant (Q4_K_M) is auto-used in normal mode.

**US-11 — "What fits?" discovery mode** `backlog`
As an Explorer I want to flip the interaction: given my GPU, show all models ranked by context fit %, with speed and quality visible, sortable by fit / speed / quality — so I can discover what to run rather than checking one model at a time.

---

## Optimizer — `index.html`

**US-12 — Scorecard** `done`
As an Optimizer I want an at-a-glance scorecard (speed / sharpness / memory clarity / attention span) so I can compare options without reading the full breakdown.

**US-13 — VRAM breakdown bar** `done`
As an Optimizer I want to see how VRAM is divided between model weights, KV cache, overhead, and free space so I understand the trade-offs. Shown in geek mode (US-19).

**US-14 — Formula breakdown** `done`
As an Optimizer I want to see the full context calculation step by step so I can verify the result and understand what drives it. Shown in geek mode (US-19).

**US-15 — Model info table** `done`
As an Optimizer I want to see the model's architecture parameters (block count, KV heads, key length, etc.) so I understand what drives the VRAM and context numbers. Shown in geek mode (US-19).

**US-16 — Nudge buttons** `done`
As an Optimizer I want one-click nudge buttons that jump to the next faster/sharper/higher/longer variant so I can quickly explore trade-offs without cycling through every option manually.

**US-18 — Speed in human terms** `done`
As an Optimizer I want speed shown in words/second and pages of context rather than raw tokens/second so the numbers are meaningful without mental conversion.

**US-28 — Context-aware, architecture-aware speed estimates** `done`
As an Optimizer I want generation-speed estimates that account for how speed changes with context length and model architecture, so the numbers stay realistic at long context. Decode is modelled as memory streaming + a per-token attention-compute term, with the attended context capped at each model's `sliding_window` (Gemma 2/3/4 stay flat with context; full-attention models slow down). Calibrated against RTX 3090 + GTX 1660 Super benchmark sweeps in `meta/benchmarks/`. Known gaps tracked as BUG-17 (super-linear collapse on 24B-class dense models at extreme context) and BUG-18 (tight range for tiny Q8_0 models).

**US-19 — Geek mode** `done`
As an Optimizer I want to toggle detailed views (VRAM bar, model/GPU/formula tabs) on and off so the page stays clean for casual use but the full breakdown is one click away. Preference persists via localStorage. Default: off.

**US-30 — Ratings for non-GGUF variants** `done`
As an Optimizer I want mlx / nvfp4 / mxfp8 variants (which ollama publishes with no quant metadata)
to still show speed/quality/efficiency ratings so I can compare them against the GGUF quants instead
of seeing blanks. Implemented via `variantRatings()` (SPEC §5.5): GGUF variants key into `QUANT_INFO`
directly; null-quant variants are estimated by interpolating from the model's own labelled variants
by size, marked `~` in the variant dropdown and scorecard tooltips to flag the estimate. The 13 IQ
importance-matrix `QUANT_INFO` entries were removed — ollama ships none. Routed through every
consumer (index scorecard/dropdown/model-list colour, coder ranking, detail panel).

**US-31 — Restrained colour system** `done`
As any user I want the interface to use strong colour sparingly so the one signal that matters —
context fit — actually stands out. Status colour (green/amber/orange) is reserved for the context-fit
thread (model list, Context-fit meter, aside `~pages`), all driven by one shared `contextFitColor`
so they never disagree. The verdict is neutral (only the card border carries the tint; red kept for
OOM), Speed/Sharpness/Memory-clarity meters use a single neutral fill, the memory bar is pure
allocation (one blue ramp + greys, legend dots matching blocks), and the capability pills, the
`how to run it` toggle, and the coder page's OS tabs use the cyan accent. See SPEC §6 and §7.4.

---

## Buyer — `buyer.html` `backlog`

**US-20 — Minimum VRAM for a model**
As a Buyer I want to select a model and quantization and see the minimum VRAM required so I know what GPU tier to target.

**US-21 — GPU recommendations**
As a Buyer I want to see a ranked list of GPUs that can run my chosen model at my target context length, sorted by fit and speed, so I can make an informed purchase decision.

**US-22 — Context target input**
As a Buyer I want to specify a target context length (e.g. 32k) so the recommendations reflect my actual use case rather than a default.

---

## Vibe Coder — `coder.html`

**US-23 — Best coding model for my GPU** `done`
As a Vibe Coder I want to see which coding models run on my GPU, ranked by the balance of speed and context relevant to agentic coding — so I can pick without manual comparison.

Implemented: only curated coding models are shown — those with a `coding_role` (`agent`|`code`|`fim`).
The generic `tools` capability is no longer used for inclusion (it pulled in general chat models —
llama, mistral, mixtral … — while excluding real code models like codellama/codegemma/starcoder2).
Three ranked sections (Agents → Code → FIM), each by `speed×0.5 + context×0.3 + quality×0.2`. The
top agent is marked `★ recommended`. Sizes that don't fit the GPU are hidden; each row shows the
origin flag. See SPEC §13.3–13.5.

**US-24 — Editor config output** `done`
As a Vibe Coder I want a ready-to-paste provider config for my editor so I can wire up my local model without digging through settings docs.

Implemented: Cline (openai-compatible JSON block) and Continue (ollama JSON block) tabs, each with
a copy button. Ollama command also shown with optional KV cache note when non-f16 is selected.

**US-25 — Context in coder terms** `done`
As a Vibe Coder I want context window shown in practical coding units so I know how much of my codebase the model can see at once.

Implemented: `~N files` (≥5 files) or `~N lines` (<5 files) derived from `maxCtx / 1000` and
`maxCtx / 3` respectively. Tooltip shows exact tokens + lines + files.

**US-27 — FIM vs agent distinction** `done`
As a Vibe Coder I want to know whether a model is an autocomplete tool or an agent model so I don't try to use Codestral in an agentic loop where it won't work.

Implemented: AGENT / TOOLS / FIM badges on every row. FIM models (codestral) shown in a separate
labelled section below the ranked agent/tools list. `coding_role` field in `data.libraries.js`
carries this — set manually, not by the scraper.

**US-28 — OOM models visible** `done`
As a Vibe Coder I want to see models that don't fit my GPU listed at the bottom (not hidden) so I know what I'm missing and can consider an upgrade.

Implemented: OOM models are rendered muted with `✗ OOM` label, ranked below all fitting models,
and non-clickable (no config panel expands).

**US-29 — Capability-differentiated configs and curated model list** `done`
As a Vibe Coder I want configs appropriate to each model's actual capability so I'm not shown Cline agentic settings for a code-chat model, or chat settings for an autocomplete model.

Implemented: config output now branches on `coding_role`:

- `agent` → Cline + Continue (unchanged)
- `code` → Continue chat config only (Cline omitted — these models are not tuned for agent loops)
- `fim` → Continue `tabAutocompleteModel` block with an explanatory note (completely different format)

Section dividers updated: "Code chat & assistance — explanation, generation, review" and
"Autocomplete — fill-in-the-middle IDE completion".

Model list curated: added `phi4` and `mistral-small3.2` as `code`-role models (modern, non-CN,
top-tier coders for their size). Removed `coding_role` from `phind-codellama`, `wizardcoder`, and
`stable-code` (superseded, circa 2023). See SPEC §13.3 for updated curation principles.
