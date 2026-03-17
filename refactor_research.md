# Refactor Research — will-it-llm.com

Code reduction, naming readability, and responsive layout findings.

---

## 1. Code Reduction

### fmtCtx() — 8 chained if-statements (app.js ~64-74)
Replace with a lookup array:
```js
const CTX_LABELS = [
  [131072,'128k'],[65536,'64k'],[32768,'32k'],[16384,'16k'],
  [8192,'8k'],[4096,'4k'],[2048,'2k'],[1024,'1k']
];
function fmtCtx(n) {
  return (CTX_LABELS.find(([t]) => n >= t) || [0,'0'])[1];
}
```

### Repeated .textContent assignments in render() (app.js ~356-365)
Seven consecutive getElementById/textContent lines. Extract to a map:
```js
const detailValues = {
  detailLayers:    model.block_count,
  detailAttnHeads: model.head_count,
  detailKvHeads:   model.head_count_kv,
  detailHeadDim:   model.key_length,
  detailHiddenSize: model.embedding_length,
  detailBpe:       bytesPerElement,
  detailWeights:   fmtGB(weightsGB),
};
Object.entries(detailValues).forEach(([id, val]) => {
  document.getElementById(id).textContent = val;
});
```

### Memory bar segment update — duplicated pattern (app.js ~201-214)
Three near-identical blocks for segModel/segContext/segFree. Extract:
```js
function updateMembarSegment(id, pct, label, cls) {
  const el = document.getElementById(id);
  el.className = `membar-seg ${cls}`;
  el.style.flex = pct;
  el.textContent = label;
}
```

### Inline color styles in index.html (lines 102, 106)
```html
<!-- before -->
<div class="detail-val" style="color:var(--purple)">Mixture of Experts (MoE)</div>
<div class="detail-val" style="color:var(--accent)">Multimodal (text + vision)</div>

<!-- after -->
<div class="detail-val detail-moe">Mixture of Experts (MoE)</div>
<div class="detail-val detail-multimodal">Multimodal (text + vision)</div>
```
```css
.detail-moe       { color: var(--purple); }
.detail-multimodal { color: var(--accent); }
```

### Verdict text-shadow — repeated rgba pattern (styles.css ~217-221)
Move to CSS variables:
```css
:root {
  --shadow-green:  0 0 32px rgba(86, 216, 138, 0.45);
  --shadow-amber:  0 0 32px rgba(245, 166, 35, 0.45);
  --shadow-orange: 0 0 32px rgba(240, 116, 24, 0.45);
  --shadow-red:    0 0 32px rgba(240, 100, 100, 0.45);
}
.result-headline.score-high .verdict { text-shadow: var(--shadow-green); }
/* etc. */
```

### Dead element: `id="srcMaxCtx"` (index.html line 111)
Created in HTML, never populated in app.js. Remove or wire up.

---

## 2. Naming Readability

### Single-letter / abbreviated variables (app.js)

| current | suggested | where |
|---|---|---|
| `v` | `selectedVariant` | ~line 85 |
| `qi` | `quantInfo` | ~line 87 |
| `spd` | `speedRating` | ~line 106 |
| `qlt` | `qualityRating` | ~line 107 |
| `r` (calcMaxContext result) | `ctxResult` | ~line 191 |
| `pct` | `contextFitPct` | ~line 232 |
| `sSpeed/sQuality/sContext/sPrecision` | `scoreSpeed` etc. | ~line 230-234 |
| `avg` | `scoreAvg` | ~line 235 |

### Cryptic function names (app.js)

| current | suggested |
|---|---|
| `rating()` | `buildRatingBar()` |
| `libMeta()` | `getLibMeta()` |
| `modelLabel()` | `formatModelOption()` |
| `scoreColor()` | `colorForScore()` |

### Cryptic HTML element IDs

Formula box IDs use an unexplained `f` prefix — suggest `formula` prefix:

| current | suggested |
|---|---|
| `fLayers` | `formulaBlockCount` |
| `fKvHeads` | `formulaKvHeads` |
| `fHeadDim` | `formulaKeyLength` |
| `fPerToken` | `formulaPerToken` |
| `fBpeLabel` | `formulaBpeLabel` |
| `fAvailGB` | `formulaAvailGB` |
| `fRawTokens` | `formulaRawTokens` |
| `fMaxCtx` | `formulaMaxCtx` |
| `archLimitNote` | `formulaArchCapNote` |

Source cell IDs mix `src` and `detail` prefixes — standardise to `detailSrc`:

| current | suggested |
|---|---|
| `srcMaxCtx` | `detailSrcMaxCtx` |
| `srcAttnHeads` | `detailSrcAttnHeads` |
| `srcHeadDim` | `detailSrcKeyLength` |

---

## 3. Responsive / Mobile Layout

All suggestions use a single breakpoint: `@media (max-width: 600px)`.

### Controls grid — stacks 2-column on mobile (styles.css ~55-60)
```css
@media (max-width: 600px) {
  .controls { grid-template-columns: 1fr; gap: 12px; }
}
```

### Details table — 180px fixed column too wide on phones (styles.css ~270-272)
```css
@media (max-width: 600px) {
  .detail-row { grid-template-columns: 1fr; gap: 2px; }
  .detail-src { display: none; }          /* hide source column on mobile */
  .detail-key { color: var(--accent); }
}
```

### Ollama command block — overflows horizontally (styles.css ~349-361)
```css
/* change white-space: pre → pre-wrap globally (safe) */
.ollama-cmd { white-space: pre-wrap; word-break: break-all; overflow-x: auto; }
```

### Header font size — 32px too large on small screens (styles.css ~38-46)
```css
@media (max-width: 600px) {
  header h1 { font-size: 24px; }
  header p  { font-size: 13px; }
}
```

### Body padding — 40px excessive on mobile (styles.css ~21-28)
```css
@media (max-width: 600px) {
  body { padding: 20px 12px; }
}
```

### Formula box — line-height 2 wastes space on mobile (styles.css ~290-304)
```css
@media (max-width: 600px) {
  .formula-box { font-size: 11px; line-height: 1.6; padding: 12px 14px; }
}
```

### Tooltip — fixed 260px width may clip on narrow screens (styles.css ~393-407)
```css
.tooltip { max-width: 90vw; }
```
And in app.js tooltip positioning:
```js
const tooltipW = Math.min(260, window.innerWidth * 0.85);
tip.style.left = Math.max(8, Math.min(rect.left, window.innerWidth - tooltipW - 8)) + 'px';
```

### Scorecard — score-label fixed width may overflow (styles.css ~223-239)
```css
@media (max-width: 600px) {
  .score-label { width: 4.5em; font-size: 12px; }
  .score-row   { font-size: 12px; gap: 6px; }
}
```

---

## Summary

| priority | area | item |
|---|---|---|
| high | responsive | Controls grid: single column on mobile |
| high | responsive | Details table: stack + hide source col |
| high | naming | Single-letter vars: v, qi, r, pct, sSpeed etc. |
| medium | responsive | Ollama cmd: pre-wrap |
| medium | responsive | Header font scaling |
| medium | naming | Formula box IDs: `f` prefix → `formula` prefix |
| medium | code | fmtCtx lookup array |
| medium | code | textContent map in render() |
| low | responsive | Body padding, tooltip max-width, scorecard scaling |
| low | code | Inline color styles → CSS classes |
| low | code | Verdict text-shadow CSS variables |
| low | naming | Function renames (rating, libMeta, modelLabel) |
