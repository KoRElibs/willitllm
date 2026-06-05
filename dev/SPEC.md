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
index.html               — single HTML page, all structure, no logic
styles.css               — all styling
data.gpus.js             — GPU database (const GPUS)
data.libraries.js        — model library metadata (const LIBRARIES)
data.quantizations.js    — quantization quality/speed ratings (const QUANT_INFO)
data.kv-cache.js         — KV cache precision options (const KV_CACHE)
data.models.js           — model architecture + variant data (const MODELS)
app.calc.js              — pure calculation and formatting helpers
app.render.js            — DOM rendering functions
app.ui.js                — UI helpers (dropdowns, model option colouring)
app.js                   — entry point: shared state, render orchestrator, init
dev/
  SPEC.md                    — this file
  scripts/update-models.md   — scraper workflow instructions (AI-readable)
  todo.md                    — development todo list
  scripts/
    update_models.py         — Python scraper that maintains data.models.js and data.libraries.js
```

All JS files are loaded via `<script src="file.js?v=N">` tags in dependency order (data files
first, then app.calc/render/ui, then app.js last). The `?v=N` query string is bumped manually on
each deploy to bust CDN caches. No build step, no bundler, no framework.

---

## 3. Data structures

### 3.1 `GPUS` — `data.gpus.js`

```js
const GPUS = [
  {
    vram:        24,        // VRAM in GB (number)
    flash:       'yes',    // Flash Attention support: 'yes' | 'no' | 'mixed'
    bandwidth:   1008,     // Memory bandwidth in GB/s — primary input for generation speed
    tflops_fp16: 165.2,    // FP16 tensor TFLOPS, dense/no-sparsity — input for processing speed.
                           // For GPUs without tensor cores (Pascal, GTX 16xx) this equals
                           // the shader FP32 throughput. All values are approximate.
    names:       ['RTX 4090'],  // GPU model names shown in dropdown
    default:     true,     // optional — pre-selected on load
  },
  ...
]
```

`flash` values:
- `'yes'` — NVIDIA Ampere (sm_80+) or newer, Flash Attention fully supported in ollama
- `'no'`  — NVIDIA Turing (sm_75) or older, NOT supported
- `'mixed'` — AMD / other — support varies by build and driver

Each entry in `GPUS` has exactly one name (or a "series" name covering near-identical SKUs).
Entries that previously grouped very different GPUs (e.g. RTX 3070 and RTX 3070 Ti) are split
so each named card gets accurate bandwidth and TFLOPS values.

The GPU dropdown shows two sections: "Generic X GB" entries (one per unique VRAM size, with `flash`
set to `'mixed'` if GPUs of the same VRAM differ in flash support), followed by a disabled separator,
followed by individual named GPU entries sorted alphabetically.

Generic entries do not store bandwidth/tflops — these are derived at runtime as `[min, max]`
across all named entries at that VRAM tier, producing a wide speed estimate. A note prompts
the user to select their exact card for a tighter estimate.

#### Adding or updating GPU specs (procedure for AI assistant)

NVIDIA GeForce specs are maintained in `sources/nvidia-tflops-derived.md`, built from
screenshots of the official NVIDIA compare pages (`https://www.nvidia.com/nb-no/geforce/graphics-cards/compare/`).
The NVIDIA compare pages require JavaScript and cannot be fetched directly — the user provides
screenshots and the AI reads them, then updates both the source file and `data.gpus.js`.

**Deriving `tflops_fp16` from AI TOPS (shown on NVIDIA compare pages):**
The compare page shows AI TOPS (tensor core, sparse), not FP16 dense. Convert:
- NVIDIA Ada (RTX 40xx):       `FP16 dense = AI TOPS ÷ 8`   (Gen 4 tensor, INT8 sparse)
- NVIDIA Blackwell (RTX 50xx): `FP16 dense = AI TOPS ÷ 16`  (Gen 5 tensor, FP4 sparse)
- Cross-check: `FP16 dense ≈ CUDA cores × boost GHz × 4`

For older series (RTX 30/20, GTX 16/10), no AI TOPS is shown — use TechPowerUp "FP16 (half)".

**Deriving `bandwidth` (not shown on compare pages — separate source required):**
Read from TechPowerUp "Memory Bandwidth" or the NVIDIA nb-no compare page bandwidth row.
Record the source and date in `sources/nvidia-tflops-derived.md`.

**Laptop variants always get a separate entry:**
Laptop GPUs have lower bandwidth (narrower bus, lower TDP) than desktop — never share an entry.
Use `names: ['RTX XXXX Laptop']`. Laptop specs are not on the desktop compare page; source
from Notebookcheck or NVIDIA's laptop GPU compare page separately.

**Back-calculate from a real ollama benchmark when available:**
`bandwidth_implied = eval_rate_tps × weights_gb / gen_eff`
Use `gen_eff ≈ 0.5` for models between 3–8 GB weights. If implied bandwidth differs by >20%,
prefer the benchmark-derived value and note it in a comment.

### 3.2 `LIBRARIES` — `data.libraries.js`

```js
const LIBRARIES = [
  {
    "library":      "llama3.2",       // ollama library name (prefix before the colon)
    "organization": "Meta",           // company/group that made it
    "origin":       "USA",            // country/region — human-readable label
    "flag":         "🇺🇸",            // flag emoji — used directly in UI (model group headers)
    "source":       null,             // optional URL to announcement/docs
    "multimodal":   true,             // optional — shows vision badge if true
  },
  ...
]
```

Keys are quoted (valid JSON) because this file is also parsed by the Python scraper.
`flag` is an explicit emoji field used directly in model dropdown group headers and the detail
panel. `origin` is a human-readable label shown in the model info table. If `origin` is null,
the model is shown as "community project" in the detail panel and gets a 👥 group header.

### 3.3 `MODELS` — `data.models.js`

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
    { "tag": "8b",      "quantization": "Q4_K_M", "weights_gb": 4.9, "group": "(default)" },
    { "tag": "8b-q8_0", "quantization": "Q8_0",   "weights_gb": 8.5, "group": "(default)" },
    { "tag": "8b-fp16", "quantization": "F16",    "weights_gb": 16.1, "group": "(default)" },
  ]
}
```

`variants[0]` is the default (marked `← default` in the dropdown). Each variant has a `tag`
(the full ollama sub-tag after the colon), a `quantization` key that must match a key in
`QUANT_INFO`, `weights_gb`, and a `group` string used to separate variants into labelled
`<optgroup>` sections in the dropdown (e.g. `"(default)"`, `"instruct"`, `"tools"`). When all
variants share the same group, no optgroup is rendered.

The full runnable ollama tag is: `library:variant.tag`, e.g. `llama3.1:8b-q8_0`.

### 3.4 `QUANT_INFO` — `data.quantizations.js`

```js
const QUANT_INFO = {
  'Q4_K_M': {
    speed:       7,             // 1 (slowest) → 10 (fastest)
    quality:     6,             // 1 (worst)  → 10 (best / lossless)
    gen_eff:     [0.43, 0.82], // fraction of GPU bandwidth achieved during token generation
    prefill_eff: [0.07, 0.33], // fraction of GPU fp16 TFLOPS achieved during prompt processing
    summary: 'The most popular choice — best size-to-quality ratio at 4-bit.'
  },
  ...
}
```

Covers all quantization formats used in ollama: IQ1_S through Q8_0, F16, FP16, BF16, F32.

### 3.5 `KV_CACHE` — `data.kv-cache.js`

```js
const KV_CACHE = [
  { bytesPerElement: 2,   label: 'f16',  summary: '...' },
  { bytesPerElement: 1,   label: 'q8_0', summary: '...' },
  { bytesPerElement: 0.5, label: 'q4_0', summary: '...' },
];
```

Three fixed entries mapping KV cache precision to bytes per element. The `label` value matches
the `OLLAMA_KV_CACHE_TYPE` environment variable. The `bytesPerElement` value is used directly in
the KV cache VRAM formula. The KV cache selector in the UI is populated from this array.

`gen_eff` and `prefill_eff` are `[lo, hi]` efficiency ranges:
- Lower for heavily-quantised types due to dequantisation overhead
- `prefill_eff` is intentionally narrow and low (0.04–0.30) because single-user batch-1
  inference rarely saturates tensor cores — real efficiency varies widely with context length
- `gen_eff` is higher (0.35–0.70) because generation is a simpler memory streaming pattern

---

## 4. Core VRAM calculation

This is the heart of the tool. Implemented in `calcMaxContext()`.

### 4.1 Constants

```js
const OVERHEAD_GB   = 0.5;  // fixed reservation for CUDA context, driver, ollama runtime
const SAFETY_FACTOR = 0.9;  // 10% margin for overhead estimation uncertainty (0.5–1.0 GB in practice)
const CTX_ROUND     = 128;  // round down to nearest 128 (natural head-dimension granularity)
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
max_ctx        = floor(arch_max_raw × SAFETY_FACTOR / CTX_ROUND) × CTX_ROUND
```

`SAFETY_FACTOR = 0.9` applies a 10% reduction before rounding. `CTX_ROUND = 128` keeps
`num_ctx` a clean multiple of 128 while wasting less than 1% of available context.

The safety factor exists because `OVERHEAD_GB` is a rough estimate — actual GPU driver and
ollama runtime overhead can reach 0.7–1.0 GB. Setting `num_ctx` to the raw theoretical
maximum risks OOM; the 10% margin keeps users safely within budget.

Context is displayed via `fmtCtx(n)` which formats any integer: `n ÷ 1000`, rounded, with `k`
suffix (e.g. `55k`, `128k`). No fixed label set — the value is always exact.

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

## 5. Speed estimation

Two speed estimates are shown whenever the model fits: **processing speed** (prompt ingestion)
and **generation speed** (token output). Both are displayed as ranges when there is uncertainty.

### 5.1 GPU specs lookup — `getGpuSpecs(vramGB)`

The `vramInput` select stores `opt.dataset.gpuIdx` for named card entries. Generic entries
have no `gpuIdx`. The function returns:

```js
{ bwLo, bwHi, tflopsLo, tflopsHi, isExact }
```

- **Named card**: `bwLo === bwHi` and `tflopsLo === tflopsHi` (exact values from the GPUS entry).
  `isExact = true`.
- **Generic tier**: `[min, max]` across all `GPUS` entries at that `vramGB`. `isExact = false`.
  A note "Select your exact GPU for a tighter estimate" is shown.

### 5.2 Generation speed (decode) — memory-bandwidth bound

At batch size 1 (standard for ollama), every generated token requires loading all active model
weights from VRAM once. This makes generation almost entirely memory-bandwidth bound.

```
active_weights_gb = variant.weights_gb × (params_b_active / params_b)   // MoE only; 1.0 for dense
gen_lo = bandwidth_lo × gen_eff[0] / active_weights_gb   (tokens/sec)
gen_hi = bandwidth_hi × gen_eff[1] / active_weights_gb
```

`gen_eff` is taken from `QUANT_INFO[quantization].gen_eff`. Range: 0.35–0.70 depending on quant.

### 5.3 Processing speed (prefill) — compute bound

Prompt ingestion is dominated by large matrix multiplications (GEMM), making it compute-bound.
Only `tflops_fp16` is used (one field per GPU, not per-precision) because llama.cpp typically
dequantises to fp16 before GEMM regardless of weight quantisation. Per-quant efficiency factors
fold in any dequantisation overhead difference.

FLOPs per token has two components:

- **Linear** (`2 × params_active`): MLP and projection layers — constant regardless of context.
- **Quadratic** (`2 × context × block_count × head_count_kv × (key_length + value_length)`):
  attention QK^T and AV operations — grows with context length, dominates at 100k+ tokens.
  Uses KV head dimensions (same fields as the KV cache formula); slightly conservative for
  full MHA models, correct for GQA.

```
params_active    = (model.params_b_active || model.params_b) × 1e9
kv_dims_per_tok  = block_count × head_count_kv × (key_length + value_length)
flops_per_token  = 2 × params_active  +  2 × maxCtx × kv_dims_per_tok

prefill_lo = tflops_lo × 1e12 × prefill_eff[0] / flops_per_token   (tokens/sec)
prefill_hi = tflops_hi × 1e12 × prefill_eff[1] / flops_per_token
```

`prefill_eff` range: 0.04–0.30 (intentionally wide — single-user batch-1 rarely saturates tensor
cores; shorter prompts give lower utilisation than longer ones).

### 5.4 Display format

Speed is shown in the right aside panel using human-friendly units:

```
writing · ~45 words/s     ← label row: fmtSpeedHuman() converts tokens/s × 0.75
~18× speech pace          ← main display: fmtSpeechPace() computes round(avg_wps / 2.5)
reading · ~800 words/s
~320× speech pace
context · ≈30k words
~80 pages                 ← context as pages (~333 tokens/page, rounded to nearest 5)
```

Tooltips on each aside row show the full breakdown: speech-pace comparison, words/s, raw t/s, and a plain-English label.
`fmtSpeechPace(lo, hi)` computes `round((avg_wps) / 2.5)` where 2.5 words/s is average speech pace.

The GPU tab and Details tab also show raw `t/s` ranges via `fmtSpeed(lo, hi)`:
`~X–Y t/s` when lo ≠ hi, `~X t/s` when lo === hi. Values ≥ 1000 abbreviated: `~1.2k t/s`.

Both speed sections are hidden when the model does not fit (OOM state).

---

## 6. Scoring system


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

## 7. UI layout and behaviour

### 7.1 Controls (2×2 grid)

| Position | Label | Element | Behaviour |
|----------|-------|---------|-----------|
| top-left | GPU VRAM | `<select id="vramInput">` | Two sections: generic sizes, then named GPUs |
| top-right | Model | Custom combobox (`#modelComboWrap`) | Searchable: face button (`#modelFace`) opens a panel (`#modelPanel`) with a text filter (`#modelSearch`) and a scrollable list (`#modelList`). Models grouped by organization with flag emoji. A hidden `<select id="modelSelect">` is kept in sync for form compatibility. |
| bottom-left | Variant | `<select id="variantSelect">` | Repopulated when model changes |
| bottom-right | KV Cache | `<select id="kvCacheType">` | Fixed: f16/q8_0/q4_0 with square-block visual indicators |

A `<span id="selectionSummary">` above the memory bar shows the current selection summary (e.g. `VRAM allocation`).

### 7.2 Model dropdown colouring

Each model option is coloured based on context fit percentage:
- ≥66% of max context fits → green
- ≥33% → amber
- <33% → orange
- Model doesn't fit (weights > usable VRAM) → red, prefixed with `✗  `

### 7.3 Variant dropdown

Variants are grouped by the `group` field on each variant entry in `data.models.js` (e.g.
`"(default)"`, `"instruct"`, `"tools"`). If only one group exists, no `<optgroup>` is rendered.
Multiple groups produce labelled `<optgroup>` elements. The default group is named `"(default)"`.

Each variant option displays: `speedRating qualityRating  X.X GB  QUANTIZATION ← default`
where speed uses `▶▷` chars (5 filled/empty) and quality uses `★☆` chars (5 filled/empty).
Both pairs are chosen from the same Unicode block to guarantee equal width in monospace fonts.
The quantization name (e.g. `Q4_K_M`) follows the size, and ` ← default` is appended to the first variant.

### 7.4 Memory bar

A horizontal bar divided into five segments:

```
[ model weights ][ KV cache ][ overhead ][ safety ][ free ]
```

Segment widths are percentages of total VRAM. Each segment shows its GB value when wide enough
(threshold varies: >12% for model, >8% for context, >6% for others), otherwise empty.

Bar segment classes:
- `seg-model` — blue, or `seg-overflow` (dark red) when OOM
- `seg-context` — dark green, or `seg-overflow` when OOM
- `segOverhead` — muted (fixed ~0.5 GB overhead reservation)
- `segSafety` — muted (VRAM held back by the 10% safety factor)
- `segFree` — muted (genuinely free; only >0 when arch-limited)

The legend below the bar shows up to five items (hidden when negligible):
- `Model weights · X.X GB`
- `Xk context · KV cache X.X GB`
- `Overhead ~X.X GB`
- `Safety X.X GB`
- `Free X.X GB`

### 7.5 Result headline

A card with a left border whose colour reflects the score class. Uses a CSS grid with two columns:
left side (`result-main`), right aside (`result-aside`), and a full-width bottom row (`result-cmd`).

**Left side (`result-main`):**
1. **Verdict** — `IT WILL LLM!` or `IT WON'T LLM!` — large monospace, animated pop-in on every render
2. **Scorecard** — four `score-row` lines, each: icon · label (hoverable) · block bar `■■■■■■□□□□` · nudge button
   - ▶ Thinking speed — speed rating from `QUANT_INFO` (1–10 scale, full 10-step bar)
   - ★ Sharpness — quality rating from `QUANT_INFO` (1–10 scale)
   - ■ Memory clarity — KV precision: f16→10, q8_0→6, q4_0→4
   - ◎ Attention span — context fit: based on `contextFitPct`, 1–10 steps
   - Each row has a **nudge button** (`faster`/`better`/`higher`/`more`) that one-click adjusts the
     corresponding dropdown. Buttons are hidden when already at max/min or when the target doesn't fit.
3. **OOM label** — shown instead of scorecard when model doesn't fit

**Right aside (`result-aside`):**
- Writing speed: `~X words/s` (tooltip: speech-pace comparison + raw t/s)
- Reading speed: `~X words/s` (tooltip: speech-pace comparison + raw t/s)
- Divider
- Context: `~X pages` with label `in one go` (tooltip: token count + words)
  - `ⓘ` shown when `contextFitPct > 50%` — tooltip: "Like human memory — most models recall
    the start and end of a long text better than the middle."

**Bottom row (`result-cmd`):**
4. **Ollama command** — copy-paste ready: `ollama run library:tag\n>>> /set parameter num_ctx XXXXX`
5. **OS tabs + setup block** — only shown when KV cache type is not f16; toggleable Linux/Mac and Windows sections showing `OLLAMA_KV_CACHE_TYPE=TYPE ollama serve` instructions

Flash attention warning (`ⓘ` tooltip) when KV cache ≠ f16:
- GPU flash = `'no'`: warn that this GPU doesn't support it
- GPU flash = `'mixed'`: warn that AMD support varies
- GPU flash = `'yes'`: note that gains are modest below 8k context

### 7.6 Model info table

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

### 7.7 Formula breakdown box

Shown below the details table when the model fits. Header: `Context window: Xk`.

Displays the full calculation step by step:
1. `bytes_per_token = block_count × head_count_kv × (key_length + value_length) × bytes_per_element`
2. Computed value of bytes_per_token (with KB equivalent)
3. `available_vram = total − overhead − weights` (with GB values)
4. Available in bytes
5. `raw_max_tokens = available_vram ÷ bytes_per_token`
6. Inline safety step: `×0.9 safety, ÷128` (muted, between raw tokens and final result)
7. Final result: `→ NNNNN tokens (Xk)` — exact integer, formatted as `Xk`
8. If arch-limited: muted note `↑ capped at model's architectural limit (context_length = N tokens)`

### 7.8 Tooltip system

A single `<div id="tooltip">` positioned fixed. On `mouseover`, any element with `data-tip` shows
the tooltip below the element. Position is clamped to `window.innerWidth - 276px` to keep it
on-screen.

---

## 8. Visual design

### 8.1 Colour palette (CSS variables)

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

### 8.2 Typography

- **Monospace** (`--mono`): Cascadia Code, Fira Code, JetBrains Mono, ui-monospace, Consolas
- **Sans** (`--sans`): system-ui, -apple-system, Segoe UI
- Labels and keys: 11px monospace, uppercase, muted
- Body text: 14px sans
- Verdict: 32px monospace bold, letter-spacing 0.06em

### 8.3 Page structure

Max-width 720px, centred. Padding 40px top/bottom on desktop. No sidebar. Single-column flow:
header → controls → bar → result headline → details → formula → disclaimer.

### 8.4 Mobile (≤600px)

- Body padding reduced to 20px/12px
- Header font size reduced
- Controls remain 2-column grid
- Detail table source column hidden
- Formula box font size reduced
- Bar legend gap reduced

---

## 9. Interactivity rules

- Every control change (`vramInput`, `variantSelect`, `kvCacheType`) immediately
  calls `render()` — no submit button
- Selecting a model via the combobox (`#modelList`) updates the hidden `#modelSelect`, then calls
  `populateVariants()` to rebuild the variant dropdown, then `render()`
- Changing `vramInput` calls `updateKvOptions()` (hides q8_0/q4_0 for non-Flash GPUs) then `render()`
- Changing `kvCacheType` or `vramInput` re-marks all model options (colours) via `markModelOptions()`
- The verdict text triggers a CSS animation (`verdict-pop` keyframes) on every render by removing
  and re-adding the animation class (forces reflow with `void el.offsetWidth`)
- OS setup tabs (Linux/Mac, Windows) are toggle — clicking the active tab collapses it

---

## 10. Data maintenance (model scraper)

`dev/scripts/update_models.py` maintains `data.models.js` and `data.libraries.js`. It is never run
automatically — it is run manually by the developer (or by an AI following
`dev/scripts/update-models.md`) when new models need to be added.

Workflow:
1. `--discover --apply` — reads `data.libraries.js`, scrapes ollama.com for new models, inserts entries
2. `--variants --apply` — refreshes quantization variant lists and weights
3. `--verify` — read-only verification pass, all entries should report `OK`

Architecture data (`block_count`, `head_count_kv`, `key_length`, etc.) is read from ollama blob
pages. When not available via scraping, it must be filled in manually from HuggingFace
`config.json` files.

---

## 11. Key design decisions

**No framework, no build step** — the entire tool is vanilla HTML/CSS/JS. This keeps deployment
trivially simple (upload files to any static host) and eliminates dependency management.

**10% safety factor for context** — ollama's `num_ctx` accepts any integer; powers-of-2 are
not required. We apply `SAFETY_FACTOR = 0.9` before rounding down to the nearest `CTX_ROUND = 128`
tokens. The margin exists because `OVERHEAD_GB = 0.5` is a rough estimate; actual driver and
runtime overhead can reach 0.7–1.0 GB. Setting `num_ctx` to the raw theoretical maximum risks
OOM — the 10% margin keeps the recommended value safely within budget. Rounding to 128 keeps
`num_ctx` clean while wasting less than 1% of available context.

**KV cache auto-maximises context** — when the user selects a more efficient KV encoding (e.g.
q4_0), the context window scales up to use the freed VRAM, up to the model's architectural limit.
The bar may look identical between encodings when the efficiency gain exactly fills to the arch
limit — this is correct behaviour, not a display bug.

**0.5 GB overhead** — a fixed reservation for CUDA context, driver overhead, and ollama runtime.
This is a conservative estimate; actual overhead varies by GPU and system but 0.5 GB is a
reasonable floor.

**Scores are heuristic** — the four-star ratings are opinionated approximations to guide
non-expert users. They are not benchmarks.

**Context window ≠ quality throughout** — the calculator shows how much text *fits in VRAM*,
not how well the model attends to all of it. Most models degrade in the middle of long contexts
(the "lost in the middle" effect) — like human memory, recall is better at the start and end
than in the middle. A `ⓘ` caveat is shown in the aside and formula breakdown when `maxCtx`
exceeds 50% of the model's trained `context_length`, as degradation becomes practically
significant at that point.

**AI notice** — the page includes a prominent disclaimer that model metadata was compiled using AI
and has not been manually verified. Users are directed to `ollama show <model>` and
`ollama.com/library` for authoritative data.
