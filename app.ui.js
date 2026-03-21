// ─────────────────────────────────────────────────────────────────────────────
// APP.UI — dropdown population, variant selection, nudge buttons, OS tab toggle
//
// Stateful UI layer. Calls render() from app.js on user-driven input changes.
//
// Globals:     MODELS, QUANT_INFO                           (data files)
//              OVERHEAD_GB, calcMaxContext                  (app.calc.js)
//              activeOsTab, setupContent, render            (app.js, shared mutable state)
// ─────────────────────────────────────────────────────────────────────────────

// ── Variant dropdown ──────────────────────────────────────────────────────────

function buildRatingBar(val, filled, empty, max = 5) {
  const n = Math.round((val / 10) * max);
  return filled.repeat(n) + empty.repeat(max - n);
}

function updateSelectionSummary(model) {
  const el = document.getElementById('selectionSummary');
  if (!el) return;

  const modelSel = document.getElementById('modelSelect');
  const modelOpt = modelSel.selectedOptions[0];
  if (!modelOpt || modelOpt.value === '') { el.textContent = 'VRAM allocation'; return; }

  const library = model ? model.ollama_tag.split(':')[0] : modelOpt.textContent.trim().split(/[:\s]/)[0];
  const variant = model ? getSelectedVariant(model) : null;
  const fullTag = variant ? `${library}:${variant.tag}` : library;

  const kvOpt  = document.getElementById('kvCacheType').selectedOptions[0];
  const kvLabel = kvOpt ? kvOpt.textContent.trim().replace(/^[■□\s]+/, '') : '';

  const gpuOpt = document.getElementById('vramInput').selectedOptions[0];
  const gpuName = gpuOpt ? gpuOpt.textContent.trim() : '';

  const modelParts = [fullTag, kvLabel ? `KV ${kvLabel}` : ''].filter(Boolean);
  el.textContent = gpuName
    ? `${gpuName}: ${modelParts.join(' · ')}`
    : modelParts.join(' · ');
}

function populateVariants(model) {
  const sel = document.getElementById('variantSelect');
  sel.innerHTML = '';
  if (!model || !model.variants || model.variants.length === 0) {
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

// ── Model dropdown ────────────────────────────────────────────────────────────

// Marks OOM models red and colours in-VRAM models by context fit.
function markModelOptions(vramGB, bytesPerElement) {
  const sel = document.getElementById('modelSelect');
  Array.from(sel.options).forEach((opt) => {
    const m = MODELS[parseInt(opt.value)];
    if (!m) return;
    const weightsGB = m.variants && m.variants.length ? m.variants[0].weights_gb : 0;
    const fits      = weightsGB < vramGB - OVERHEAD_GB;
    if (!fits) {
      opt.textContent = `✗  ${m.ollama_tag}`;
      opt.style.color = '#f06464';
      return;
    }
    const ctxResult     = calcMaxContext(m, vramGB, bytesPerElement, weightsGB);
    const contextFitPct = m.context_length ? Math.round((ctxResult.maxCtx / m.context_length) * 100) : 100;
    opt.textContent = m.ollama_tag;
    opt.style.color = contextFitPct >= 66 ? '#56d88a' : contextFitPct >= 33 ? '#f5a623' : '#f07418';
  });
}

// ── KV cache dropdown ─────────────────────────────────────────────────────────

function updateKvOptions() {
  const gpuOpt  = document.getElementById('vramInput').selectedOptions[0];
  const flash   = gpuOpt ? gpuOpt.dataset.flash : 'yes';
  const flashOk = flash === 'yes' || flash === 'mixed';
  const kvSel   = document.getElementById('kvCacheType');
  Array.from(kvSel.options).forEach(opt => {
    const needsFlash = parseFloat(opt.value) < 2;
    opt.hidden   = needsFlash && !flashOk;
    opt.disabled = needsFlash && !flashOk;
  });
  if (parseFloat(kvSel.value) < 2 && !flashOk) kvSel.value = '2';
}

// ── Nudge buttons ─────────────────────────────────────────────────────────────

// Returns variants in the same group as variantIdx, sorted by quality ascending (fastest first).
// Falls back to all variants when the current group has only one member.
function groupVariantsSorted(model, variantIdx) {
  const v     = model.variants[variantIdx];
  const group   = v.group || '(default)';
  const all     = model.variants.map((v, i) => ({ v, i, qi: QUANT_INFO[v.quantization] }));
  const inGroup = all.filter(({ v }) => (v.group || '(default)') === group);
  const candidates = inGroup.length > 1 ? inGroup : all;
  return candidates.sort((a, b) => {
    const qa = a.qi ? a.qi.quality : 5, qb = b.qi ? b.qi.quality : 5;
    return qa !== qb ? qa - qb : a.v.weights_gb - b.v.weights_gb;
  });
}

function nudgeVariant(direction) {
  const modelIdx = parseInt(document.getElementById('modelSelect').value);
  const model    = MODELS[modelIdx];
  if (!model || !model.variants) return;
  const idx    = getSelectedVariantIdx(model);
  const sorted = groupVariantsSorted(model, idx);
  const pos    = sorted.findIndex(({ i }) => i === idx);
  const target = direction === 'quality' ? pos + 1 : pos - 1;
  if (target < 0 || target >= sorted.length) return;
  document.getElementById('variantSelect').value = sorted[target].i;
  render();
}

function nudgeKv(direction) {
  const sel     = document.getElementById('kvCacheType');
  const visible = Array.from(sel.options).filter(o => !o.hidden && !o.disabled);
  const curIdx  = visible.findIndex(o => o.value === sel.value);
  const target  = direction === 'quality' ? curIdx - 1 : curIdx + 1;
  if (target < 0 || target >= visible.length) return;
  sel.value = visible[target].value;
  render();
}

function updateNudgeButtons(ctxAtMax, vramGB) {
  const modelIdx = parseInt(document.getElementById('modelSelect').value);
  const model    = MODELS[modelIdx];
  const kvSel    = document.getElementById('kvCacheType');
  const show     = (id, visible) => { const el = document.getElementById(id); if (el) el.hidden = !visible; };

  if (!model || !model.variants) {
    ['nudge-speed','nudge-quality','nudge-ctx-quality','nudge-ctx-size'].forEach(id => show(id, false));
    return;
  }
  const idx    = getSelectedVariantIdx(model);
  const sorted = groupVariantsSorted(model, idx);
  const pos    = sorted.findIndex(({ i }) => i === idx);

  const fits = v => !vramGB || v.weights_gb < vramGB - OVERHEAD_GB;
  show('nudge-speed',   pos > 0                 && fits(sorted[pos - 1].v));
  show('nudge-quality', pos < sorted.length - 1 && fits(sorted[pos + 1].v));
  const kvVisible = Array.from(kvSel.options).filter(o => !o.hidden && !o.disabled);
  const kvCurIdx  = kvVisible.findIndex(o => o.value === kvSel.value);
  show('nudge-ctx-quality', kvCurIdx > 0);
  show('nudge-ctx-size',    !ctxAtMax && kvCurIdx < kvVisible.length - 1);
}

// ── OS tab toggle ─────────────────────────────────────────────────────────────

function setOsTab(os) {
  const setupEl    = document.getElementById('ollamaSetup');
  const tabLinux   = document.getElementById('tabLinux');
  const tabWindows = document.getElementById('tabWindows');

  if (activeOsTab === os) {
    activeOsTab = null;
    setupEl.hidden = true;
    tabLinux.textContent   = '▶ Linux / Mac';
    tabWindows.textContent = '▶ Windows';
    tabLinux.classList.remove('active');
    tabWindows.classList.remove('active');
  } else {
    activeOsTab = os;
    setupEl.innerHTML = setupContent[os];
    setupEl.hidden = false;
    tabLinux.textContent   = os === 'linux'   ? '▼ Linux / Mac' : '▶ Linux / Mac';
    tabWindows.textContent = os === 'windows' ? '▼ Windows'     : '▶ Windows';
    tabLinux.classList.toggle('active',   os === 'linux');
    tabWindows.classList.toggle('active', os === 'windows');
  }
}
