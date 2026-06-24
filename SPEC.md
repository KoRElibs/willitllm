# willitllm — design specification

> **Purpose of this document:** This is both a human-readable design spec and a machine-readable
> blueprint. An AI with sufficient capability should be able to reproduce the full codebase from
> this document alone, making no assumptions beyond what is written here.

---

## 1. What it is

**willitllm.com** is a static web tool with two pages.

**`index.html`** answers:

> *Can I run this LLM on my GPU?*

The user picks a GPU (by VRAM), picks an ollama model and quantization variant, optionally adjusts
the KV cache encoding, and the page instantly shows:

- Whether the model fits in VRAM at all
- How much context window fits in the remaining VRAM
- The ollama commands needed to run it at that context
- A scorecard rating speed, quality, precision, and context fit
- The full mathematical breakdown of the VRAM calculation

**`coder.html`** answers:

> *What's the best local coding model for my GPU, and how do I wire it into my editor?*

The user picks a GPU; the page shows all fitting coding models ranked for agentic performance,
with ready-to-paste configs for Cline and Continue.

There is **no backend**. Everything runs in the browser. All data is static JS files loaded as
global variables.

---

## 2. File structure

```
index.html               — fit checker page, all structure, no logic
coder.html               — vibe coder page, all structure, no logic
styles.css               — all styling (shared between both pages)
data.calc-constants.js   — formula constants shared between browser and Python tooling (const CALC_CONSTANTS)
data.gpus.js             — GPU database (const GPUS)
data.libraries.js        — model library metadata (const LIBRARIES)
data.quantizations.js    — quantization quality/speed ratings (const QUANT_INFO)
data.kv-cache.js         — KV cache precision options (const KV_CACHE)
data.flags.js            — origin → flag emoji map + flagFor() helper (const FLAGS)
data.models.js           — model architecture + variant data (const MODELS)
app.calc.js              — pure calculation and formatting helpers (shared)
app.render.js            — DOM rendering functions (index.html only)
app.ui.js                — UI helpers: dropdowns, combobox, nudge buttons (index.html only)
app.js                   — entry point for index.html: shared state, render orchestrator, init
coder.js                 — entry point for coder.html: ranking, config output, init
meta/
  FEATURES.md                — feature backlog and done list
  BUGS.md                    — bug tracker (open / fixed)
  UX-FINDINGS.md             — design and UX research findings
  knowledge/
    external-tools.md        — Cline, Continue, Ollama, editor ecosystem reference
    nvidia-geforce-compare.md — NVIDIA compare page verbatim source data (read-only)
    nvidia-tflops-derived.md  — derived FP16 TFLOPS values and conversion formulas
  benchmarks/                — empirical benchmark result files
  scripts/
    update-models.md         — scraper workflow instructions (AI-readable)
    update_models.py         — Python scraper that maintains data.models.js and data.libraries.js
    benchmark.py             — benchmark runner
  skills/
    browser-verifier.md      — Playwright + Firefox visual verification skill
  deploy/
    deploy.sh                — production deployment script
    deploy.cfg.example       — config template (deploy.cfg is gitignored)
```

All JS files are loaded via `<script src="file.js?v=N">` tags in dependency order (data files
first, then app.calc, then page-specific files). The `?v=N` query string is bumped manually on
each deploy to bust CDN caches. No build step, no bundler, no framework.

**Shared vs page-specific:** `data.*.js` and `app.calc.js` are loaded by both pages.
`app.render.js`, `app.ui.js`, and `app.js` are index.html only. `coder.js` is coder.html only.
`data.kv-cache.js` is index.html only (coder.html uses `autoKvBpe` from app.calc.js directly).

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

The GPU dropdown uses `<optgroup>` to group cards by vendor in this order: **NVIDIA GeForce** (consumer GTX/RTX), **NVIDIA Professional** (workstation and data-centre: RTX A-series, Ada generation, A100, H100, etc.), **AMD Radeon**, **Apple** (when entries exist), **Generic** (one entry per unique VRAM size). Within each named group cards are sorted alphabetically. Generic entries are sorted by VRAM descending.

Generic entries do not store bandwidth/tflops — these are derived at runtime as `[min, max]`
across all named entries at that VRAM tier, producing a wide speed estimate. A note prompts
the user to select their exact card for a tighter estimate.

#### Adding or updating GPU specs (procedure for AI assistant)

NVIDIA GeForce specs are maintained in `meta/knowledge/nvidia-tflops-derived.md`, built from
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
Record the source and date in `meta/knowledge/nvidia-tflops-derived.md`.

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
    "library":      "llama3.2",            // ollama library name (prefix before the colon)
    "organization": "Meta",                // company/group that made it
    "origin":       "USA",                 // country/region — human-readable label
    "source":       null,                  // optional URL to announcement/docs
    "capabilities": ["tools"],             // optional — capability badges from ollama.com/library
    "coding_role":  "agent",               // optional — agent | code | fim (coder.html, see §13.3)
    "pulls":        "71.6M",              // optional — download count from ollama.com/library
  },
  ...
]
```

Keys are quoted (valid JSON) because this file is also parsed by the Python scraper.
`origin` is a human-readable label shown in the model info table. If `origin` is null, the model is
shown as "community project" in the detail panel and gets a 👥 group header. The **flag emoji is not
stored here** — it is derived from `origin` via `FLAGS` / `flagFor()` in `data.flags.js` (single
source of truth; see §3.6), so adding a model needs only `origin`.

`capabilities` is sourced exclusively from the `x-test-capability` elements on `ollama.com/library`.
Possible values: `tools` | `vision` | `thinking` | `embedding` | `audio`. Omit the field when empty.
Do **not** set manually — run `python scripts/update_models.py --capabilities --apply` to refresh.

`pulls` is sourced from the `x-test-pull-count` element on `ollama.com/library`. Omit when not
available. Do **not** set manually — refreshed by `--capabilities`.

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
  "sliding_window": 1024,           // optional — sliding-window attention span (Gemma 2/3/4).
                                    //   Absent = full global attention. Caps the attended
                                    //   context in the decode/prefill attention-compute terms.
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

### 3.6 `FLAGS` — `data.flags.js`

```js
const FLAGS = { 'USA': '🇺🇸', 'France': '🇫🇷', 'Canada': '🇨🇦', /* … */ };
function flagFor(origin) { … }   // FLAGS[origin] | 🌍 (unmapped) | 👥 (null/empty)
```

Single source of truth for the flag emoji shown next to a model's origin. Keyed by the `origin`
string in `LIBRARIES`, so libraries store only `origin` and never a redundant per-entry emoji.
Loaded by both pages (after `data.libraries.js`). All consumers — the model combobox group rows
(`app.ui.js`), the detail-panel origin row (`app.render.js`), and the coder row (`coder.js`) — call
`flagFor(lib.origin)`.

---

## 4. Core VRAM calculation

This is the heart of the tool. Implemented in `calcMaxContext()`.

### 4.1 Constants

```js
const OVERHEAD_GB    = 0.8;  // fixed reservation for CUDA context, driver, ollama runtime + driver-reserved VRAM
const SAFETY_FACTOR  = 0.9;  // 10% margin for overhead estimation uncertainty (0.5–1.0 GB in practice)
const CTX_ROUND      = 128;  // round down to nearest 128 (natural head-dimension granularity)
const DECODE_ATTN_EFF = 0.015; // batch-1 decode attention: fraction of fp16 TFLOPS reached per token
```

`OVERHEAD_GB` was raised 0.5 → 0.8 after a GTX 1660 Super spilled to system RAM at a context
the 0.5 value predicted would fit: rated VRAM overstates usable VRAM (driver/system reserve
~4–6%) and runtime overhead alone can reach ~0.8 GB. See `meta/benchmarks/README.md`.

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

Note: `free_gb` includes the 0.8 GB overhead reservation (it is subtracted from available when
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

### 5.2 Generation speed (decode) — two serial costs per token

At batch size 1 (standard for ollama), each generated token incurs two costs in series:

1. **Memory streaming** (bandwidth-bound) — read all active weights once *and* read the full
   KV cache once for attention.
2. **KV-access slowdown** — a per-token cost that grows with the attended context, applied **only
   when KV access is not free-flowing**: quantized KV (dequantization on every read) or a GPU
   without flash attention (unfused attention). With **f16 KV on a flash GPU it is gated off** —
   decode then stays flat with context (measured ~0.80 effective gen_eff to ~48k on both
   llama-arch devstral:24b and mistral3 mistral-small3.2:24b). When it applies it reproduces the
   measured decline of quantized/no-flash setups (e.g. devstral:24b q4_0: ~0.76 → ~0.37 effective
   gen_eff from 1k → 112k).

```
active_weights_gb = variant.weights_gb × (params_b_active / params_b)   // MoE only; 1.0 for dense
attn_ctx          = min(maxCtx, sliding_window ?? ∞)                    // sliding-window aware
kv_slowdown       = (bytes_per_element < 2) OR (gpu.flash ≠ 'yes')      // quantized KV OR no flash
attn_flops        = kv_slowdown ? 2 × attn_ctx × block_count × head_count_kv × (key+value) : 0

t_mem  = (active_weights_gb + kv_cache_gb) / (bandwidth × gen_eff)
t_attn = attn_flops / (tflops_fp16 × 1e12 × DECODE_ATTN_EFF)
gen    = 1 / (t_mem + t_attn)                                           (tokens/sec)
```

The range is produced by pairing the fast/slow ends: `gen_hi` uses `bandwidth_hi`, `gen_eff[1]`,
`tflops_hi`; `gen_lo` uses `bandwidth_lo`, `gen_eff[0]`, `tflops_lo`. `kv_cache_gb` is passed in
from `calcMaxContext` (the KV memory at the recommended context). `gen_eff` is taken from
`QUANT_INFO[quantization].gen_eff`.

**Two things make the slowdown term general across architectures rather than a per-model fudge:**
- **Sliding-window attention** (`sliding_window` set, Gemma 2/3/4): most layers attend only to a
  fixed window (512–4096 tokens), so `attn_ctx` is capped and the term stays small — matching the
  measured near-flat generation speed of Gemma models across context.
- **The KV/flash gate**: full-attention models on quantized KV or no-flash GPUs decline with
  context; the same models on f16+flash do not. Measured per-token speed is f16 ≥ quantized at any
  context (q4_0 is smaller but dequant overhead cancels the bandwidth saving — its benefit is
  capacity, not speed), and the gate captures exactly that.

`DECODE_ATTN_EFF = 0.015` was calibrated against full-context sweeps on an RTX 3090 (flash; f16,
q8_0 and q4_0 KV) and a GTX 1660 Super (no-flash, f16 KV) covering devstral:24b, mistral-small3.2:24b,
gemma4:e4b, llama3.2:1b/3b, gemma3:4b, codellama:13b. Adding the (gated) term cut high-context
bracketing error from ~32% to ~15% RMS without regressing the f16+flash path.
A residual super-linear collapse remains for very large dense models at extreme context
(devstral-small-2:24b at 200k still over-predicts) — documented gap, not modelled.

### 5.3 Processing speed (prefill) — compute bound

Prompt ingestion is dominated by large matrix multiplications (GEMM), making it compute-bound.
Only `tflops_fp16` is used (one field per GPU, not per-precision) because llama.cpp typically
dequantises to fp16 before GEMM regardless of weight quantisation. Per-quant efficiency factors
fold in any dequantisation overhead difference.

FLOPs per token has two components:

- **Linear** (`2 × params_active`): MLP and projection layers — constant regardless of context.
- **Quadratic** (`2 × attn_ctx × block_count × head_count_kv × (key_length + value_length)`):
  attention QK^T and AV operations — grows with context length, dominates at 100k+ tokens.
  Uses `attn_ctx = min(maxCtx, sliding_window ?? ∞)` (same sliding-window cap as decode) and KV
  head dimensions (same fields as the KV cache formula); slightly conservative for full MHA
  models, correct for GQA.

```
params_active    = (model.params_b_active || model.params_b) × 1e9
attn_ctx         = min(maxCtx, sliding_window ?? ∞)
kv_dims_per_tok  = block_count × head_count_kv × (key_length + value_length)
flops_per_token  = 2 × params_active  +  2 × attn_ctx × kv_dims_per_tok

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
| top-left | Your GPU | `<select id="vramInput">` | Grouped by vendor: NVIDIA GeForce → NVIDIA Professional → AMD Radeon → Generic |
| top-right | Model | Custom combobox (`#modelComboWrap`) | Searchable: face button (`#modelFace`) opens a panel (`#modelPanel`) with a text filter (`#modelSearch`) and a scrollable list (`#modelList`). Models grouped by organization with flag emoji. A hidden `<select id="modelSelect">` is kept in sync for form compatibility. Embedding models are hidden from the list entirely. |
| bottom-left | Capability | `<div id="capFilter">` pill row | Pills: `any` · `tools` · `vision` · `thinking`. Multi-select AND — active pills highlighted; list shows only models whose library has **all** selected caps. `any` (`data-cap=""`) clears the filter. On change, auto-selects first fitting model. `embedding` has no pill — those models are hidden from the list entirely (not chat models). `audio` has no pill — no current library carries it. Cap membership is stored in `item.dataset.caps` at list-build time and checked as `[..._activeCaps].every(c => itemCaps.has(c))`. |
| bottom-right | Context | `<select id="targetCtx">` | Presets for common context sizes; drives model colour coding |

Variant (`<select id="variantSelect">`) and KV Cache are auto-managed and shown in geek mode only (see §7.3).

A `<span id="selectionSummary">` above the memory bar shows the current selection summary (e.g. `VRAM allocation`).

### 7.2 Model dropdown colouring

Each model option is coloured based on how well it serves the current target context.

**No target selected (`targetCtx = null`, i.e. "max"):** percentage of the model's architectural context limit that fits in VRAM:
- `maxCtx / context_length ≥ 66%` → green
- ≥ 33% → amber
- < 33% → orange

**Target context selected:** whether `maxCtx` (already bounded by VRAM and arch limit) meets the target:
- `maxCtx ≥ targetCtx` → green
- `maxCtx ≥ targetCtx × 0.5` → amber
- otherwise → orange

**OOM (weights don't fit):** red, prefixed with `✗  `

The arch limit is not used to cap the target — a model that can only do 32k tokens shows amber/orange when the user wants 200k, even if 32k is all that model can ever do.

### 7.3 Variant dropdown (geek mode only)

The variant selector (`<select id="variantSelect">`) lives in the geek section, shown only when
geek mode is enabled. In normal mode the default variant (index 0, always Q4_K_M) is used automatically.

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
- `segOverhead` — muted (fixed ~0.8 GB overhead reservation)
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
   - ◎ Context fit — how well the model meets the user's chosen context target, 1–10 steps
   - Each row has a **nudge button** (`faster`/`better`/`higher`/`more`) that one-click adjusts the
     corresponding dropdown. Buttons are hidden when already at max/min or when the target doesn't fit.
3. **OOM label** — shown instead of scorecard when model doesn't fit

**Right aside (`result-aside`):**
- Writing speed: `~X words/s` (tooltip: speech-pace comparison + raw t/s)
- Reading speed: `~X words/s` (tooltip: speech-pace comparison + raw t/s)
- Divider
- Context: `~X pages` (or `~X pages / ~Y pages` when target not fully met), colored by target fit
  - Green when target fully met (≥95%); amber at 50–94%; orange below 50%
  - Sub-label: `context · ≈N words · Z% of target` when a gap exists; plain `context · ≈N words` when met
  - `ⓘ` shown when `contextFitPct > 50%` (% of arch limit) — tooltip: "Like human memory — most models recall
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
header → controls → result headline → details → formula → footer.

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

`meta/scripts/update_models.py` maintains `data.models.js` and `data.libraries.js`. It is never run
automatically — it is run manually by the developer (or by an AI following
`meta/scripts/update-models.md`) when new models need to be added.

Workflow:
1. `--capabilities --apply` — scrapes `ollama.com/library` once for capability badges and pull counts, writes to `data.libraries.js`. Run whenever new libraries are added or capability data may have changed.
2. `--discover --apply` — reads `data.libraries.js`, scrapes ollama.com for new models, inserts entries
3. `--variants --apply` — refreshes quantization variant lists and weights
4. `--verify` — read-only verification pass, all entries should report `OK`

Architecture data (`block_count`, `head_count_kv`, `key_length`, etc.) is read from ollama blob
pages. When not available via scraping, it must be filled in manually from HuggingFace
`config.json` files.

---

## 11. Key design decisions

**No framework, no build step** — the entire tool is vanilla HTML/CSS/JS. This keeps deployment
trivially simple (upload files to any static host) and eliminates dependency management.

**10% safety factor for context** — ollama's `num_ctx` accepts any integer; powers-of-2 are
not required. We apply `SAFETY_FACTOR = 0.9` before rounding down to the nearest `CTX_ROUND = 128`
tokens. The margin exists because `OVERHEAD_GB = 0.8` is a rough estimate; actual driver and
runtime overhead plus driver-reserved VRAM can reach 0.8–1.2 GB. Setting `num_ctx` to the raw
theoretical maximum risks OOM — the 10% margin keeps the recommended value safely within budget.
Rounding to 128 keeps `num_ctx` clean while wasting less than 1% of available context.

**KV cache auto-maximises context** — when the user selects a more efficient KV encoding (e.g.
q4_0), the context window scales up to use the freed VRAM, up to the model's architectural limit.
The bar may look identical between encodings when the efficiency gain exactly fills to the arch
limit — this is correct behaviour, not a display bug.

**0.8 GB overhead** — a fixed reservation for CUDA context, driver overhead, ollama runtime, and
driver-reserved VRAM (rated VRAM is ~4–6% larger than what is actually addressable). Raised from
0.5 GB after a GTX 1660 Super spilled to system RAM at a context 0.5 GB predicted would fit. Actual
overhead varies by GPU/system; 0.8 GB is a safer floor. On small cards the unusable fraction is
proportionally larger, so a single fixed value still slightly under-protects the smallest GPUs —
a future refinement could model usable VRAM as a fraction of rated.

**Scores are heuristic** — the four-star ratings are opinionated approximations to guide
non-expert users. They are not benchmarks.

**Context window ≠ quality throughout** — the calculator shows how much text *fits in VRAM*,
not how well the model attends to all of it. Most models degrade in the middle of long contexts
(the "lost in the middle" effect) — like human memory, recall is better at the start and end
than in the middle. A `ⓘ` caveat is shown in the aside and formula breakdown when `maxCtx`
exceeds 50% of the model's trained `context_length`, as degradation becomes practically
significant at that point.

**Footer** — a single compact line at the bottom of the page: `AI-assisted data · source: ollama.com · treat as a guide · about & disclaimer`. Clicking "about & disclaimer" opens a bottom sheet (`#infoSheet`) with full EU AI Act Article 13 disclosure: data sources, AI-generated content notice, VRAM estimate methodology, and score explanation. The sheet closes on backdrop click or Escape key. The footer renders outside `#results` so it is always visible regardless of model selection state.

---

## 13. Coder page — `coder.html`

### 13.1 Purpose

`coder.html` answers a different question from `index.html`. The interaction is **ranked discovery**,
not single-model inspection. The user selects a GPU once; the page instantly shows all fitting coding
models ranked by agentic loop performance, with ready-to-paste editor configs.

### 13.2 Scripts loaded

```
data.gpus.js          — GPU selector (shared)
data.libraries.js     — coding_role field consumed here
data.flags.js         — flagFor(origin) for the row flag (shared)
data.quantizations.js — gen_eff, quality, speed ratings (shared)
data.models.js        — weights, context, architecture (shared)
app.calc.js           — calcMaxContext(), calcSpeedEstimates(), autoKvBpe() (shared)
coder.js              — all coder logic: GPU selector, ranking, row building, init
```

`coder.js` is one file (~250 lines) with clear `// ── Section ──` dividers:
- Pure helpers (data, formatting, ranking formula)
- GPU selector (mirrors app.js — same optgroup structure)
- Data/ranking (buildEntries)
- Config HTML (makeConfigHtml)
- Row building (makeRow + event wiring)
- List render (renderList)
- Render + init

### 13.3 `coding_role` field in `LIBRARIES`

One optional, hand-curated field in `data.libraries.js` — it is the **sole** signal for coder-page
inclusion (the generic `tools` capability is deliberately NOT used):

```js
"coding_role": "agent"   // "agent" | "code" | "fim" — omit for everything else
```

- `"agent"` — purpose-built for tool-calling agent loops (devstral, devstral-small-2). Shows Cline + Continue agentic config.
- `"code"` — excellent for code chat, explanation, review, and generation (codellama, codegemma, granite-code, phi4, mistral-small3.2). Shows Continue chat config only — Cline is omitted because these models are not tuned for autonomous agent loops. Includes both purpose-built coding models and general models that are among the best for coding specifically (non-CN only).
- `"fim"` — fill-in-the-middle autocomplete; uses FIM tokens for single-cursor completion (codestral, starcoder2). Shows Continue `tabAutocompleteModel` config — completely different format from chat/agent configs. Not for chat or agent loops.
- omitted — not shown on the coder page at all (includes general chat models that merely have a
  `tools` badge — llama, mistral, mixtral, command-r, … — they are not among the best for coding specifically)

This cannot be derived from `capabilities`: the `tools` badge lets in general chat models while
real code models (codellama, codegemma, starcoder2) carry no badge. Do **not** set automatically
with the scraper — it requires human judgement. The scraper **preserves** `coding_role` across
rewrites but never sets it.

**Curation principles for `"code"` role:**

- Prefer models that rank competitively on coding benchmarks (HumanEval, SWE-bench, etc.) for their size tier
- Exclude Chinese-origin models (Qwen-coder etc.) — the page is curated around non-CN models
- Exclude models superseded by newer entries in the same size tier (e.g. phind-codellama, wizardcoder, stable-code are retired in favour of phi4, mistral-small3.2, granite-code)

### 13.4 Model filtering

The coder page shows a model only when its library has a `coding_role` (`agent` | `code` | `fim`).
Sizes that do not fit the selected VRAM are **hidden entirely**; every fitting size is shown (no
family collapse). No variant selector — only the default variant (index 0) is used.

Rows are grouped into three sections in order, each its own ranked block:
1. **Agents** — top section. The top-ranked agent is flagged `★ recommended`.
2. **Code chat & assistance** — under a divider ("Code chat & assistance — explanation, generation, review").
3. **Autocomplete** — under a divider ("Autocomplete — fill-in-the-middle IDE completion").

Each row shows the origin **flag** (`flagFor(lib.origin)`, see §3.6) next to the model name.

### 13.5 Ranking

Models within each section are ranked by a weighted coding score:

```
speed_norm  = min(1, gen_lo / 30)       // normalised at 30 t/s
ctx_norm    = min(1, maxCtx / 65536)    // normalised at 64k; cap avoids outlier inflation
qual_norm   = QUANT_INFO[quant].quality / 10

coding_score = speed_norm × 0.5 + ctx_norm × 0.3 + qual_norm × 0.2
```

Speed weighted highest (0.5) because agentic coding sessions chain 30–100 sequential tool calls.
Context second (0.3) — seeing more of the codebase is the primary quality lever. Quality last (0.2)
— at Q4_K_M, differences are small for code generation.

`coding_score` is not displayed to the user. It drives sort order and the score bar width (0–100%).

Non-fitting sizes are not ranked — they are excluded from the list entirely (§13.4).

### 13.6 Context in coding units

Context is shown in developer units in the row and tooltip:

```
files = round(maxCtx / 1000)
if files >= 5: display "~N files"  (rounded to nearest 5)
else:          display "~N lines"  (maxCtx / 3, rounded to nearest 100)
```

Tooltip also shows: `X tokens · ~Y lines · ~Z avg files (est. ~1000 tokens/file)`.

### 13.7 KV cache for ranking

`autoKvBpe(model, vramGB, weightsGB, null, flashOk)` selects the best KV precision that maximises
context. `targetCtx = null` means "push to arch limit". The selected `bpe` drives the **More
context** mode tab; the baseline `bpe = 2` (f16) drives **Quick start**.

Two mode tabs are shown only when `bpe < 2` AND the optimised context exceeds the f16 context
(i.e. the model is not already arch-limited at f16). When both modes yield the same context, only
a single config section is shown with no tabs.

**Quality transparency** — the "More context" tab tooltip states the tradeoff explicitly:
- `q8_0`: "Quality: nearly lossless (~0.5% perplexity hit)"
- `q4_0`: "Quality: modest hit (~2–5% perplexity, degrades further at long contexts)"

### 13.8 Config output

Config blocks are templated from the model's default variant tag and the user-supplied Ollama URL.
The UI has an **Ollama URL** text input (defaults to `http://localhost:11434`; re-renders list on
`change`). Remote users set this to their server address (e.g. `http://192.168.1.10:11434`).

Note: remote ollama requires `OLLAMA_HOST=0.0.0.0` on the server — without it ollama binds only
to loopback and remote connections get ECONNREFUSED.

Config output differs by `coding_role`:

**`agent` models** — full two-mode config (Cline + Continue):

*Quick start* — f16 KV, no setup required:
```
ollama run devstral:24b
>>> /set parameter num_ctx 24576
```

*More context* (when `bpe < 2` and modes yield different context) — q8_0 or q4_0 KV, OS-specific
setup step shown first (see §13.11), then run command with the larger context value.

Cline — native Ollama provider, configured through the Cline UI (not a JSON file):
```
API Provider      Ollama
Base URL          <ollama-url>
Model             devstral:24b
Context Window    55296
```

Continue (`.continue/config.json` model entry):
```json
{
  "title": "devstral:24b",
  "provider": "ollama",
  "model": "devstral:24b",
  "apiBase": "<ollama-url>",
  "contextLength": 55296
}
```

**`code` models** — same two-mode layout but **Cline tab is omitted**. Only Continue chat config
is shown. These models are not designed for autonomous agent loops; showing Cline would mislead.

**`fim` models** — same two-mode Quick start / More context layout as `agent`/`code` (ollama hosting
is the same concern regardless of downstream use). The editor plugin section is replaced with a
Continue `tabAutocompleteModel` entry instead of Cline/Continue chat configs. Shows:

1. Ollama `run` + `num_ctx` command (with optional KV setup step in More context mode)
2. Continue `tabAutocompleteModel` entry (`.continue/config.json`) with `contextLength`:

```json
{
  "tabAutocompleteModel": {
    "title": "starcoder2:3b",
    "provider": "ollama",
    "model": "starcoder2:3b",
    "apiBase": "<ollama-url>"
  }
}
```

A note explains that FIM models use fill-in-the-middle tokens and belong in `tabAutocompleteModel`,
not in a chat session or agent loop.

Each sub-block has a copy button (`navigator.clipboard.writeText`). The KV setup section has OS
tabs (Linux / macOS / Windows) with separate copy buttons; the handler scopes to the active
`.kv-os-block` so only the visible command is copied.

### 13.9 UI layout

```
<nav>  ← fit checker
<header>  title + subtitle
<controls>  GPU selector + Ollama URL input (single column, max-width 480px)
<noGpu>  placeholder until GPU selected
<coderList>  ranked rows
  <coder-row>  (one per model)
    .coder-row-header  [BADGE] [name] [speed] [ctx] [score bar]
    .coder-config      (hidden until row clicked)
      [mode tabs: Quick start · More context]  ← only when context modes differ
        .mode-block[quick]   ollama cmd + Cline/Continue tabs
        .mode-block[optimized]
          [KV setup: Linux | macOS | Windows]  step 1
          ollama cmd                            step 2
          Cline/Continue tabs                   step 3
  <fim-divider>
  <coder-row>  (FIM models)
<footer>  links: ollama.com · fit checker · about & disclaimer
```

Clicking a row header expands its config panel and collapses any previously open row.
Clicking again collapses it. OOM rows are not clickable.

### 13.10 Navigation

### 13.11 External references — tools & docs

**Always verify these URLs and instructions before changing anything in `coder.js` that references
external tools.** Docs move; installers change. Keep this table current whenever a URL breaks or
a tool ships a config change that affects our instructions.

#### Editor extensions

| Tool | URL | Notes |
|---|---|---|
| Cline | [docs.cline.bot](https://docs.cline.bot) | VS Code extension for agentic coding loops; configured via its own settings UI (not a JSON file). Extension ID: `saoudrizwan.claude-dev` |
| Continue | [continue.dev](https://www.continue.dev) | VS Code / JetBrains extension for coding assistance and tab autocomplete |
| Continue config docs | [docs.continue.dev](https://docs.continue.dev) | Config file reference; `tabAutocompleteModel`, model entries, providers |

Links appear inline in `coder.js` config labels (Cline label → github.com/cline/cline; Continue
label → docs.continue.dev). Update them here and in the source together if URLs change.

#### Ollama platform setup

**Verify against these docs before changing platform-specific setup instructions**
in `kvSetupHtml` or anywhere else in `coder.js`. Our instructions must stay aligned with
what the official installer actually creates.

| Tool | URL | Notes |
|---|---|---|
| Ollama install | [ollama.com](https://ollama.com) | Download page; the `ollama` CLI becomes available after install |
| Ollama docs | [docs.ollama.com](https://docs.ollama.com) | General reference |

| Platform | Doc URL | Key facts |
|---|---|---|
| Linux | https://docs.ollama.com/linux | systemd service at `/etc/systemd/system/ollama.service`; customise via drop-in `…/ollama.service.d/override.conf` (same file `sudo systemctl edit ollama` creates); env vars as `Environment=` lines under `[Service]` |
| macOS | https://docs.ollama.com/macos | App stores data in `~/.ollama`; env var docs sparse — current approach: export in `~/.zshrc`, quit menu bar app, run `ollama serve` from terminal |
| Windows | https://docs.ollama.com/windows | Env vars via Settings → Environment Variables UI or `setx` from cmd/PowerShell; restart Ollama from system tray after |

#### Ollama command intent

The `ollama run <tag>` + `/set parameter num_ctx N` block shown in the config panels serves
two purposes: it pulls the model if not already installed, and lets the user verify it runs at
the target context in a local terminal session. It is **not** the live num_ctx setting for
Cline/Continue — those editors send `num_ctx` automatically via their own config fields
(Cline: Context Window; Continue: `contextLength`). A note in the config panel makes this clear.

**Linux implementation note**: use the drop-in (`override.conf`) rather than editing
`/etc/systemd/system/ollama.service` directly. The original service file is written by the
ollama installer and may be overwritten on `ollama update`. The drop-in survives updates.

Write the drop-in with:
```bash
sudo mkdir -p /etc/systemd/system/ollama.service.d && printf '[Service]\nEnvironment="OLLAMA_KV_CACHE_TYPE=q8_0"\n' | sudo tee /etc/systemd/system/ollama.service.d/override.conf
sudo systemctl daemon-reload && sudo systemctl restart ollama
```

(Split into write + apply in the UI so users don't bounce the service unnecessarily on re-runs.)

### 13.12 Navigation

- `index.html` footer: `· vibe coder` link to `coder.html`
- `coder.html` nav: `← fit checker` link to `index.html`

Both pages are independently linkable. No shared nav component.

---

## 12. Documentation maintenance

Every code change must keep the following three files in sync with the codebase. Stale documentation is worse than none — it actively misleads future contributors and AI assistants working from this spec.

| File | Update when |
|---|---|
| `meta/FEATURES.md` | A feature is implemented (`backlog` → `done`), a new feature is planned (add as `backlog`), or an existing feature's behaviour changes |
| `meta/BUGS.md` | A bug is discovered (add as `open`) or fixed (`open` → `fixed` with a description of the root cause and fix) |
| `SPEC.md` | Any described behaviour changes — data structures, formulas, UI layout, constants, file paths, or interaction rules |

The rule: if you changed the code, you changed at least one of these files.
