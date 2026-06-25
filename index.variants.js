// ─── INDEX.VARIANTS — variant dropdown, rating bars, nudge buttons
//
// Depends on:  MODELS, QUANT_INFO (data files), app.calc.js (OVERHEAD_GB),
//              index.js (render — called at runtime only)
// Provides:    buildRatingBar, populateVariants, getSelectedVariantIdx,
//              getSelectedVariant, variantOllamaTag,
//              variantsSortedByQuality, nudgeVariant, updateNudgeButtons

function buildRatingBar(val, filled, empty, max = 5) {
  const n = Math.round((val / 10) * max);
  return filled.repeat(n) + empty.repeat(max - n);
}

function populateVariants(model) {
  const sel   = document.getElementById('variantSelect');
  const field = sel.closest('.field-secondary');
  sel.innerHTML = '';
  if (!model) {
    if (field) field.hidden = true;
    return;
  }
  if (field) field.hidden = false;
  if (!model.variants || model.variants.length === 0) {
    const opt = document.createElement('option');
    opt.textContent = 'no variants';
    sel.appendChild(opt);
    updateSelectionSummary(model);
    return;
  }

  const groups = new Map();
  model.variants.forEach((variant, i) => {
    const group = variant.group || '(default)';
    if (!groups.has(group)) groups.set(group, []);
    groups.get(group).push({ variant, i });
  });

  groups.forEach((items, groupName) => {
    const container = groups.size > 1 ? document.createElement('optgroup') : sel;
    if (groups.size > 1) {
      container.label = groupName;
      sel.appendChild(container);
    }
    items.forEach(({ variant, i }) => {
      const opt       = document.createElement('option');
      opt.value       = i;
      const quantInfo = QUANT_INFO[variant.quantization];
      const quant     = variant.quantization || '?';
      const gb        = variant.weights_gb.toFixed(1);
      const defMark   = i === 0 ? ' ← default' : '';
      if (window.innerWidth <= 600) {
        const s = quantInfo ? quantInfo.speed   : '?';
        const q = quantInfo ? quantInfo.quality : '?';
        opt.textContent = `${s}S ${q}Q  ${quant}${defMark}`;
      } else {
        const speedRating   = quantInfo ? buildRatingBar(quantInfo.speed,   '▶', '▷') : '▷▷▷▷▷';
        const qualityRating = quantInfo ? buildRatingBar(quantInfo.quality, '★', '☆') : '☆☆☆☆☆';
        opt.textContent = `${speedRating} ${qualityRating}  ${gb} GB  ${quant}${defMark}`;
      }
      container.appendChild(opt);
    });
  });

  sel.value = '0';  // always start on the default variant
  updateSelectionSummary(model);
}

function getSelectedVariantIdx(model) {
  if (!model || !model.variants || model.variants.length === 0) return 0;
  return Math.min(parseInt(document.getElementById('variantSelect').value) || 0, model.variants.length - 1);
}

function getSelectedVariant(model) {
  if (!model || !model.variants || model.variants.length === 0) return null;
  return model.variants[getSelectedVariantIdx(model)];
}

// Returns the full ollama tag for a specific variant, e.g. "llama3.2:3b-q4_K_M".
function variantOllamaTag(model, variantIdx) {
  const variant = model.variants[variantIdx];
  const library = model.ollama_tag.split(':')[0];
  return `${library}:${variant.tag}`;
}

// ── Nudge buttons ─────────────────────────────────────────────────────────────

// All variants sorted by quality ascending (lowest quality = fastest first).
function variantsSortedByQuality(model) {
  return model.variants
    .map((v, i) => ({ v, i, qi: QUANT_INFO[v.quantization] }))
    .sort((a, b) => (a.qi?.quality ?? 5) - (b.qi?.quality ?? 5) || a.v.weights_gb - b.v.weights_gb);
}

function nudgeVariant(direction) {
  const modelIdx = parseInt(document.getElementById('modelSelect').value);
  const model    = MODELS[modelIdx];
  if (!model || !model.variants) return;
  const idx    = getSelectedVariantIdx(model);
  const sorted = variantsSortedByQuality(model);
  const pos    = sorted.findIndex(({ i }) => i === idx);
  const target = direction === 'quality' ? pos + 1 : pos - 1;
  if (target < 0 || target >= sorted.length) return;
  document.getElementById('variantSelect').value = sorted[target].i;
  render();
}

function updateNudgeButtons(vramGB) {
  const modelIdx = parseInt(document.getElementById('modelSelect').value);
  const model    = MODELS[modelIdx];
  const show     = (id, visible) => { const el = document.getElementById(id); if (el) el.hidden = !visible; };

  if (!model || !model.variants) {
    ['nudge-speed', 'nudge-quality'].forEach(id => show(id, false));
    return;
  }
  const idx    = getSelectedVariantIdx(model);
  const sorted = variantsSortedByQuality(model);
  const pos    = sorted.findIndex(({ i }) => i === idx);
  const fits   = v => !vramGB || v.weights_gb < vramGB - OVERHEAD_GB;
  show('nudge-speed',   pos > 0                 && fits(sorted[pos - 1].v));
  show('nudge-quality', pos < sorted.length - 1 && fits(sorted[pos + 1].v));
}
