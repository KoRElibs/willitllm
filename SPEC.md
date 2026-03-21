# willitllm — design specification

> **Purpose of this document:** This is both a human-readable design spec and a machine-readable
> blueprint. An AI with sufficient capability should be able to reproduce the full codebase from
> this document alone, making no assumptions beyond what is written here.

---

## 1. What it is

**willitllm.com** is a single-page web tool that answers one question:

> *Can I run this LLM on my GPU?*

The user picks a GPU (by VRAM), picks an ollama model and quantization variant, optionally adjusts
the KV cache encoding, and the page instantly shows:

- Whether the model fits in VRAM at all
- How much context window fits in the remaining VRAM
- The ollama commands needed to run it at that context
- A scorecard rating speed, quality, precision, and context fit
- The full mathematical breakdown of the VRAM calculation

There is **no backend**. Everything runs in the browser. All data is static JS files loaded as
global variables.

---

## 2. File structure

```
index.html          — single HTML page, all structure, no logic
styles.css          — all styling
app.js              — all application logic
gpus.js             — GPU database (const GPUS)
libraries.js        — model library metadata (const LIBRARIES)
models.js           — model architecture + variant data (const MODELS)
quantizations.js    — quantization quality/speed ratings (const QUANT_INFO)
scripts/
  update_models.py  — Python scraper that maintains models.js and libraries.js
update_models_prompt.md  — instructions for running the scraper (AI-readable)
```

All JS files are loaded via `<script src="file.js?v=N">` tags. The `?v=N` query string is bumped
manually on each deploy to bust CDN caches. No build step, no bundler, no framework.

---

## 3. Data structures

### 3.1 `GPUS` — `gpus.js`

```js
const GPUS = [
  {
    vram:    24,        // VRAM in GB (number)
    flash:   'yes',    // Flash Attention support: 'yes' | 'no' | 'mixed'
    names:   ['RTX 4090'],  // GPU model names shown in dropdown
    default: true,     // optional — pre-selected on load
  },
  ...
]
```

`flash` values:
- `'yes'` — NVIDIA Ampere (sm_80+) or newer, Flash Attention fully supported in ollama
- `'no'`  — NVIDIA Turing (sm_75) or older, NOT supported
- `'mixed'` — AMD / other — support varies by build and driver

The GPU dropdown shows two sections: "Generic X GB" entries (one per unique VRAM size, with `flash`
set to `'mixed'` if GPUs of the same VRAM differ in flash support), followed by a disabled separator,
followed by individual named GPU entries sorted alphabetically.

### 3.2 `LIBRARIES` — `libraries.js`

```js
const LIBRARIES = [
  {
    "library":      "llama3.2",       // ollama library name (prefix before the colon)
    "organization": "Meta",           // company/group that made it
    "origin":       "USA",            // country/region — drives flag emoji
    "source":       null,             // optional URL to announcement/docs
    "multimodal":   true,             // optional — shows vision badge if true
  },
  ...
]
```

Keys are quoted (valid JSON) because this file is also parsed by the Python scraper.
`origin` drives flag emoji display. Supported values and their flags:

```
'USA' → 🇺🇸  'France' → 🇫🇷  'EU' → 🇪🇺  'Canada' → 🇨🇦
'UK' → 🇬🇧  'UAE' → 🇦🇪  'Switzerland' → 🇨🇭  'South Korea' → 🇰🇷
'Singapore' → 🇸🇬  'Portugal' → 🇵🇹  'International' → 🌍
```

If `origin` is null, the model is shown as "community project".

### 3.3 `MODELS` — `models.js`

One entry per canonical ollama tag (e.g. `llama3.1:8b`):

```js
{
  "ollama_tag":     "llama3.1:8b",  // library:size — the canonical ollama pull tag
  "context_length": 131072,         // max tokens the model was trained on (arch limit)
  "params_b":       8.0,            // total parameters in billions
  "params_b_active": 2.8,          // optional — active params per pass (MoE only)
  "moe":            true,           // optional — Mixture of Experts flag
  "block_count":    32,             // transformer decoder layer count
  "head_count":     32,             // total query attention heads
  "head_count_kv":  8,              // KV heads — drives VRAM formula (GQA)
  "embedding_length": 4096,         // embedding dimension
  "key_length":     128,            // key vector size per head
  "value_length":   128,            // optional — value vector size; defaults to key_length
  "variants": [
    { "tag": "8b",           "quantization": "Q4_K_M", "weights_gb": 4.7 },
    { "tag": "8b-q8_0",      "quantization": "Q8_0",   "weights_gb": 8.5 },
    { "tag": "8b-fp16",      "quantization": "F16",    "weights_gb": 16.1 },
  ]
}
```

`variants[0]` is the default. Each variant has a `tag` (the full ollama sub-tag after the colon),
a `quantization` key that must match a key in `QUANT_INFO`, and `weights_gb`.

The full runnable ollama tag is: `library:variant.tag`, e.g. `llama3.1:8b-q8_0`.

### 3.4 `QUANT_INFO` — `quantizations.js`

```js
const QUANT_INFO = {
  'Q4_K_M': {
    speed:   7,    // 1 (slowest) → 10 (fastest)
    quality: 6,    // 1 (worst)  → 10 (best / lossless)
    summary: 'The most popular choice — best size-to-quality ratio at 4-bit.'
  },
  ...
}
```

Covers all quantization formats used in ollama: IQ1_S through Q8_0, F16, FP16, BF16, F32.

---

## 4. Core VRAM calculation

This is the heart of the tool. Implemented in `calcMaxContext()`.

### 4.1 Constants

```js
const OVERHEAD_GB = 0.5;  // fixed reservation for CUDA context, driver, ollama runtime
```

### 4.2 KV cache encoding

The user selects a KV cache type which maps to bytes per element:

```
f16  → 2.0 bytes/element  (full precision, works on all GPUs)
q8_0 → 1.0 bytes/element  (half memory, requires Flash Attention)
q4_0 → 0.5 bytes/element  (quarter memory, requires Flash Attention)
```

### 4.3 Bytes per token

```
bytes_per_token = block_count × head_count_kv × (key_length + value_length) × bytes_per_element
```

`value_length` defaults to `key_length` if not specified in model data. This handles MLA
architectures (e.g. DeepSeek V2/V3) where value_length differs from key_length.

### 4.4 Available bytes

```
available_bytes = (vram_gb − OVERHEAD_GB − weights_gb) × 1024³
```

If `available_bytes ≤ 0`, the model does not fit. Show OOM state.

### 4.5 Maximum context window

```
raw_max_tokens = available_bytes / bytes_per_token
arch_max_raw   = min(raw_max_tokens, model.context_length)
max_ctx        = largest power of 2 ≤ arch_max_raw
```

Powers of 2 considered (largest first): 131072, 65536, 32768, 16384, 8192, 4096, 2048, 1024.
Context window labels: 128k, 64k, 32k, 16k, 8k, 4k, 2k, 1k.

If `raw_max_tokens > model.context_length`, the context is architecture-limited (not VRAM-limited)
and a note is shown in the formula breakdown.

### 4.6 KV cache memory used

```
kv_cache_gb = (max_ctx × bytes_per_token) / 1024³
```

### 4.7 Free VRAM

```
free_gb = vram_gb − weights_gb − kv_cache_gb
```

Note: `free_gb` includes the 0.5 GB overhead reservation (it is subtracted from available when
computing raw_max_tokens, but not added back to used). The "free" segment therefore includes the
overhead in its display.

---

## 5. Scoring system

Four dimensions, each scored 1–5 stars:

| Dimension | Scoring rule |
|-----------|-------------|
| **Speed** | `max(1, round((quant.speed / 10) × 5))` |
| **Quality** | `max(1, round((quant.quality / 10) × 5))` |
| **Precision** | f16 → 5, q8_0 → 3, q4_0 → 2 |
| **Context** | Based on `(max_ctx / model.context_length) × 100`: ≥90% → 5, ≥66% → 4, ≥40% → 3, ≥15% → 2, else → 1. Null if no context_length. |

`score_avg = (speed + quality + context + precision) / 4`

Headline class (drives border colour and verdict glow):
- `score_avg ≥ 4` → `score-high` (green)
- `score_avg ≥ 3` → `score-mid` (amber)
- `score_avg ≥ 2` → `score-low` (orange)
- else            → `score-poor` (red)
- OOM             → `error` (red)

---

## 6. UI layout and behaviour

### 6.1 Controls (2×2 grid)

| Position | Label | Element | Behaviour |
|----------|-------|---------|-----------|
| top-left | GPU VRAM | `<select id="vramInput">` | Two sections: generic sizes, then named GPUs |
| top-right | Model | `<select id="modelSelect">` | All models sorted alphabetically by ollama_tag |
| bottom-left | Variant | `<select id="variantSelect">` | Repopulated when model changes |
| bottom-right | KV Cache | `<select id="kvCacheType">` | Fixed: f16/q8_0/q4_0 with square-block visual indicators |

Each label shows the selected value inline after the label text in accent colour, e.g.
`GPU VRAM : 24 GB`. This is driven by `<span id="labelGpu" class="field-selected">`.

### 6.2 Model dropdown colouring

Each model option is coloured based on context fit percentage:
- ≥66% of max context fits → green
- ≥33% → amber
- <33% → orange
- Model doesn't fit (weights > usable VRAM) → red, prefixed with `✗  `

### 6.3 Variant dropdown

Variants are grouped by `getVariantGroup()`: anything between the size prefix (e.g. `8b-`) and the
quantization suffix (e.g. `-q4_k_m`) is the group name. If only one group exists, no `<optgroup>`
is used. Multiple groups produce labelled `<optgroup>` elements. The default group is named
`(default)`.

Each variant option displays: `speedRating qualityRating  X.X GB`
where speed uses `▶▷` chars (5 filled/empty) and quality uses `★☆` chars (5 filled/empty).
Both pairs are chosen from the same Unicode block to guarantee equal width in monospace fonts.

### 6.4 Memory bar

A horizontal bar divided into three segments:

```
[  model weights  ][      KV cache @ Xk ctx      ][ free ]
```

Segment widths are percentages of total VRAM. The KV cache segment label shows:
- `Xk · Y.YY GB` when the segment is wide enough (>20% of total bar)
- `Xk` when narrower (>8%)
- nothing when very narrow

Bar segment classes:
- `seg-model` — blue (`#1e3a5f`), accent text
- `seg-context` — dark green (`#1e3b2a`), green text
- `seg-free` — bg3, muted text
- `seg-overflow` — dark red (`#3b1e1e`), red text (used for both model and context when OOM)

The legend below shows: `Model weights (X.XX GB)` | `KV cache @ Xk ctx (X.XX GB)` | `Free (X.XX GB)`

### 6.5 Result headline

A card with a left border whose colour reflects the score class. Contains:

1. **Verdict** — `IT WILL LLM!` or `IT WON'T LLM!` — large monospace, animated pop-in on every render
2. **Scorecard** — four `score-row` lines with icon, label, and coloured star rating
3. **Selection info** — two lines in small muted monospace:
   - Line 1: `QUANTIZATION · summary of quantization`
   - Line 2: `kv-type · summary of kv type` + optional `ⓘ` flash attention warning tip
   - Line 3 (ctx hint): `Xk · Y% of max context · ~Z pages of typical English text [· images use tokens]`
4. **OOM label** — shown instead of scorecard when model doesn't fit
5. **Ollama command** — copy-paste ready: `ollama run library:tag\n>>> /set parameter num_ctx XXXXX`
6. **OS tabs + setup block** — only shown when KV cache type is not f16; toggleable Linux/Mac and Windows sections showing `OLLAMA_KV_CACHE_TYPE=TYPE ollama serve` instructions

Flash attention warning (`ⓘ` tooltip) when KV cache ≠ f16:
- GPU flash = `'no'`: warn that this GPU doesn't support it
- GPU flash = `'mixed'`: warn that AMD support varies
- GPU flash = `'yes'`: note that gains are modest below 8k context

### 6.6 Model info table

Two sections — "Model info" and "Architecture" — each a `<div class="details">`:

| Key | Value | Source label |
|-----|-------|-------------|
| organization | from LIBRARIES | — |
| origin | flag + country, or "community project" | — |
| architecture | "Mixture of Experts (MoE)" | hidden unless `model.moe` |
| modality | "Multimodal (text + vision)" | hidden unless `libInfo.multimodal` |
| context_length | N tokens | ollama.com link |
| block_count | N | ollama.com link |
| head_count_kv | N ← used (GQA) | ollama.com link |
| key_length | N | ollama.com link |
| value_length | N | ollama.com link |
| bytes_per_element | N (f16/q8_0/q4_0) | "kv cache selector" |
| model weights | X.XX GB (QUANTIZATION) | ollama.com link |

Each key cell has a `data-tip` attribute with a plain-English explanation shown on hover via a
fixed-position tooltip div.

### 6.7 Formula breakdown box

Shown below the details table when the model fits. Header: `Max context: Xk tokens`.

Displays the full calculation step by step:
1. `bytes_per_token = block_count × head_count_kv × (key_length + value_length) × bytes_per_element`
2. Computed value of bytes_per_token (with KB equivalent)
3. `available_vram = total − overhead − weights` (with GB values)
4. Available in bytes
5. `raw_max_tokens = available_vram ÷ bytes_per_token`
6. Final result: `→ largest power-of-2 that fits: NNNNN tokens (Xk)`
7. If arch-limited: muted note `↑ capped at model's architectural limit (context_length = N tokens)`

### 6.8 Tooltip system

A single `<div id="tooltip">` positioned fixed. On `mouseover`, any element with `data-tip` shows
the tooltip below the element. Position is clamped to `window.innerWidth - 276px` to keep it
on-screen.

---

## 7. Visual design

### 7.1 Colour palette (CSS variables)

```css
--bg:      #0d0f12   /* page background */
--bg2:     #141720   /* card backgrounds */
--bg3:     #1c2030   /* inset / darker areas */
--border:  #2a2f3f
--text:    #c8d0e0
--muted:   #5a6480
--accent:  #4fc3f7   /* cyan — primary highlight */
--green:   #56d88a
--amber:   #f5a623
--orange:  #f07418
--red:     #f06464
--purple:  #a78bfa   /* MoE badge */
```

### 7.2 Typography

- **Monospace** (`--mono`): Cascadia Code, Fira Code, JetBrains Mono, ui-monospace, Consolas
- **Sans** (`--sans`): system-ui, -apple-system, Segoe UI
- Labels and keys: 11px monospace, uppercase, muted
- Body text: 14px sans
- Verdict: 32px monospace bold, letter-spacing 0.06em

### 7.3 Page structure

Max-width 720px, centred. Padding 40px top/bottom on desktop. No sidebar. Single-column flow:
header → controls → bar → result headline → details → formula → disclaimer.

### 7.4 Mobile (≤600px)

- Body padding reduced to 20px/12px
- Header font size reduced
- Controls remain 2-column grid
- Detail table source column hidden
- Formula box font size reduced
- Bar legend gap reduced

---

## 8. Interactivity rules

- Every control change (`vramInput`, `modelSelect`, `variantSelect`, `kvCacheType`) immediately
  calls `render()` — no submit button
- Changing `modelSelect` also calls `populateVariants()` to rebuild the variant dropdown
- Changing `kvCacheType` or `vramInput` re-marks all model options (colours) via `markModelOptions()`
- The verdict text triggers a CSS animation (`verdict-pop` keyframes) on every render by removing
  and re-adding the animation class (forces reflow with `void el.offsetWidth`)
- OS setup tabs (Linux/Mac, Windows) are toggle — clicking the active tab collapses it

---

## 9. Data maintenance (model scraper)

`scripts/update_models.py` maintains `models.js` and `libraries.js`. It is never run
automatically — it is run manually by the developer (or by an AI following
`update_models_prompt.md`) when new models need to be added.

Workflow:
1. `--discover --apply` — reads `libraries.js`, scrapes ollama.com for new models, inserts entries
2. `--variants --apply` — refreshes quantization variant lists and weights
3. No args — verification pass, all entries should report `OK`

Architecture data (`block_count`, `head_count_kv`, `key_length`, etc.) is read from ollama blob
pages. When not available via scraping, it must be filled in manually from HuggingFace
`config.json` files.

`libraries.js` is dual-use: loaded as a browser `<script>` (exposes `LIBRARIES` global) and parsed
as JSON by the scraper (hence all keys must be quoted strings).

---

## 10. Key design decisions

**No framework, no build step** — the entire tool is vanilla HTML/CSS/JS. This keeps deployment
trivially simple (upload files to any static host) and eliminates dependency management.

**Powers of 2 for context** — ollama's `num_ctx` parameter must be set by the user; the tool
recommends the largest power-of-2 that fits because these are the conventional values and they
align with attention mask implementations.

**KV cache auto-maximises context** — when the user selects a more efficient KV encoding (e.g.
q4_0), the context window scales up to use the freed VRAM, up to the model's architectural limit.
The bar may look identical between encodings when the efficiency gain exactly fills to the arch
limit — this is correct behaviour, not a display bug.

**0.5 GB overhead** — a fixed reservation for CUDA context, driver overhead, and ollama runtime.
This is a conservative estimate; actual overhead varies by GPU and system but 0.5 GB is a
reasonable floor.

**Scores are heuristic** — the four-star ratings are opinionated approximations to guide
non-expert users. They are not benchmarks.

**AI notice** — the page includes a prominent disclaimer that model metadata was compiled using AI
and has not been manually verified. Users are directed to `ollama show <model>` and
`ollama.com/library` for authoritative data.
