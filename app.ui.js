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

  const gpuOpt = document.getElementById('vramInput').selectedOptions[0];
  const gpuName = gpuOpt ? gpuOpt.textContent.trim() : '';

  el.textContent = gpuName ? `${gpuName}: ${fullTag}` : fullTag;
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

// ── Model dropdown ────────────────────────────────────────────────────────────

// Returns the colour for a model that fits in VRAM, based on target context.
// targetCtx=null means "model max" (legacy % behaviour).
function modelCtxColor(ctxResult, model, targetCtx) {
  if (targetCtx === null) {
    const pct = model.context_length ? Math.round((ctxResult.maxCtx / model.context_length) * 100) : 100;
    return pct >= 66 ? '#56d88a' : pct >= 33 ? '#f5a623' : '#f07418';
  }
  return ctxResult.maxCtx >= targetCtx ? '#56d88a' : ctxResult.maxCtx >= targetCtx * 0.5 ? '#f5a623' : '#f07418';
}

// Marks OOM models red and colours in-VRAM models by target context fit.
// Each model's optimal KV type is auto-selected before colouring.
function markModelOptions(vramGB, targetCtx, flashOk) {
  const sel = document.getElementById('modelSelect');
  Array.from(sel.options).forEach((opt) => {
    const m = MODELS[parseInt(opt.value)];
    if (!m) return;
    const weightsGB = m.variants && m.variants.length ? m.variants[0].weights_gb : 0;
    if (weightsGB >= vramGB - OVERHEAD_GB) {
      opt.textContent = `✗  ${m.ollama_tag}`;
      opt.style.color = '#f06464';
      return;
    }
    const bpe       = autoKvBpe(m, vramGB, weightsGB, targetCtx, flashOk);
    const ctxResult = calcMaxContext(m, vramGB, bpe, weightsGB);
    const color     = modelCtxColor(ctxResult, m, targetCtx);
    opt.textContent = m.ollama_tag;
    opt.style.color = color;
  });
  markComboboxItems(vramGB, targetCtx, flashOk);
}

// ── Capability filter ─────────────────────────────────────────────────────────

const _activeCaps = new Set();

function applyCap(cap) {
  if (cap === '') {
    _activeCaps.clear();
  } else {
    _activeCaps.has(cap) ? _activeCaps.delete(cap) : _activeCaps.add(cap);
  }
  document.querySelectorAll('.cap-pill').forEach(pill => {
    pill.classList.toggle('active', pill.dataset.cap === '' ? _activeCaps.size === 0 : _activeCaps.has(pill.dataset.cap));
  });
  filterModelList(document.getElementById('modelSearch')?.value || '', true);
}

// ── Model combobox ────────────────────────────────────────────────────────────

let _comboOpen = false;
let _comboHighlight = null;

function openCombobox() {
  const panel = document.getElementById('modelPanel');
  const face  = document.getElementById('modelFace');
  const input = document.getElementById('modelSearch');
  _comboOpen = true;
  face.classList.add('open');
  panel.hidden = false;
  input.value = '';
  filterModelList('');
  const selectedItem = document.querySelector('#modelList .combobox-item.selected');
  if (selectedItem) selectedItem.scrollIntoView({ block: 'nearest' });
  if (!window.matchMedia('(pointer: coarse)').matches) input.focus();
}

function closeCombobox() {
  _comboOpen = false;
  _comboHighlight = null;
  document.getElementById('modelFace').classList.remove('open');
  document.getElementById('modelPanel').hidden = true;
}

function comboHighlight(el) {
  if (_comboHighlight) _comboHighlight.classList.remove('highlighted');
  _comboHighlight = el || null;
  if (_comboHighlight) {
    _comboHighlight.classList.add('highlighted');
    _comboHighlight.scrollIntoView({ block: 'nearest' });
  }
}

function filterModelList(query, autoSelect = false) {
  const list = document.getElementById('modelList');
  const q    = query.toLowerCase().trim();
  let firstVisible = null;
  list.querySelectorAll('.combobox-item').forEach(item => {
    const textMatch = !q || item.dataset.tag.includes(q);
    const itemCaps  = new Set((item.dataset.caps || '').split(','));
    const capMatch  = _activeCaps.size === 0 || [..._activeCaps].every(c => itemCaps.has(c));
    const show = textMatch && capMatch;
    item.hidden = !show;
    if (show && !firstVisible) firstVisible = item;
  });
  comboHighlight(firstVisible);
  if (autoSelect) {
    const sel        = document.getElementById('modelSelect');
    const currentIdx = sel?.value;
    const currentItem = currentIdx !== '' && currentIdx !== undefined
      ? list.querySelector(`.combobox-item[data-idx="${currentIdx}"]`)
      : null;
    const currentFits = currentItem && !currentItem.hidden && currentItem.dataset.fit !== '3';
    if (!currentFits) {
      const items   = Array.from(list.querySelectorAll('.combobox-item'));
      const firstFit = items.find(el => !el.hidden && el.dataset.fit !== '3');
      sel.value = firstFit ? firstFit.dataset.idx : '';
      sel.dispatchEvent(new Event('change'));
    }
  }
}

function selectComboboxModel(idx) {
  const sel = document.getElementById('modelSelect');
  sel.value = idx;
  closeCombobox();
  syncComboboxFace();
  sel.dispatchEvent(new Event('change'));
}

function syncComboboxFace() {
  const sel      = document.getElementById('modelSelect');
  const faceText = document.getElementById('modelFaceText');
  const list     = document.getElementById('modelList');
  const modelIdx = parseInt(sel.value);
  const model    = MODELS[modelIdx];
  list && list.querySelectorAll('.combobox-item.selected').forEach(el => el.classList.remove('selected'));
  if (!model) {
    faceText.textContent = 'Select a model...';
    faceText.style.color = '';
    return;
  }
  const item = list && list.querySelector(`.combobox-item[data-idx="${modelIdx}"]`);
  if (item) {
    item.classList.add('selected');
    faceText.style.color = item.style.color || '';
    faceText.textContent = item.dataset.label;
  } else {
    faceText.textContent = model.ollama_tag;
  }
}

function markComboboxItems(vramGB, targetCtx, flashOk) {
  const list = document.getElementById('modelList');
  if (!list) return;
  const colorPriority = { '#56d88a': 0, '#f5a623': 1, '#f07418': 2, '#f06464': 3 };
  list.querySelectorAll('.combobox-item').forEach(item => {
    const m = MODELS[parseInt(item.dataset.idx)];
    if (!m) return;
    const weightsGB = m.variants && m.variants.length ? m.variants[0].weights_gb : 0;
    if (weightsGB >= vramGB - OVERHEAD_GB) {
      item.textContent   = '✗  ' + m.ollama_tag;
      item.style.color   = '#f06464';
      item.dataset.fit   = '3';
      return;
    }
    const bpe       = autoKvBpe(m, vramGB, weightsGB, targetCtx, flashOk);
    const ctxResult = calcMaxContext(m, vramGB, bpe, weightsGB);
    const color     = modelCtxColor(ctxResult, m, targetCtx);
    item.style.color   = color;
    item.dataset.fit   = colorPriority[color] ?? 4;
    item.textContent   = item.dataset.label;
  });
  // Sort by fit tier (data-fit), then alphabetically within tier
  const items = Array.from(list.querySelectorAll('.combobox-item'));
  items.sort((a, b) => {
    const pa = parseInt(a.dataset.fit ?? 4);
    const pb = parseInt(b.dataset.fit ?? 4);
    return pa !== pb ? pa - pb : a.dataset.tag.localeCompare(b.dataset.tag);
  });
  items.forEach(item => list.appendChild(item));
  const sel        = document.getElementById('modelSelect');
  const currentIdx = sel?.value;
  const currentItem = currentIdx !== '' && currentIdx !== undefined
    ? list.querySelector(`.combobox-item[data-idx="${currentIdx}"]`)
    : null;
  const currentFits = currentItem && currentItem.dataset.fit !== '3' && !currentItem.hidden;
  if (!currentFits) {
    // Auto-select the first visible green (fit=0) model; fall back to amber/orange
    const firstFit = items.find(el => !el.hidden && el.dataset.fit !== '3');
    if (firstFit) {
      sel.value = firstFit.dataset.idx;
      sel.dispatchEvent(new Event('change'));
      return;
    } else {
      sel.value = '';
      sel.dispatchEvent(new Event('change'));
      return;
    }
  }
  syncComboboxFace();
}

function buildModelCombobox() {
  const list = document.getElementById('modelList');
  list.innerHTML = '';
  MODELS.forEach((m, i) => {
    const [library] = m.ollama_tag.split(':');
    const info  = LIB_META[library];
    const caps  = [...(info?.capabilities || [])];
    if (caps.includes('embedding')) return;   // embedding models are not for chat — hide entirely
    if (info?.coding_role) caps.push('coding'); // synthetic: any coding_role (agent/code/fim)
    const flag  = flagFor(info?.origin);
    const label = `${flag} ${m.ollama_tag}`;
    const item  = document.createElement('div');
    item.className     = 'combobox-item';
    item.dataset.idx   = i;
    item.dataset.tag   = m.ollama_tag.toLowerCase();
    item.dataset.label = label;
    item.dataset.caps  = caps.join(',');
    item.textContent   = label;
    item.addEventListener('mousedown', e => { e.preventDefault(); selectComboboxModel(i); });
    list.appendChild(item);
  });

  const face  = document.getElementById('modelFace');
  const input = document.getElementById('modelSearch');
  const wrap  = document.getElementById('modelComboWrap');

  face.addEventListener('click', () => { _comboOpen ? closeCombobox() : openCombobox(); });
  face.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ' || e.key === 'ArrowDown') { e.preventDefault(); openCombobox(); }
  });
  input.addEventListener('input', () => filterModelList(input.value));
  input.addEventListener('keydown', e => {
    if (e.key === 'Escape') { e.preventDefault(); closeCombobox(); face.focus(); return; }
    const items = Array.from(list.querySelectorAll('.combobox-item:not([hidden])'));
    if (!items.length) return;
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      const cur = _comboHighlight ? items.indexOf(_comboHighlight) : -1;
      comboHighlight(e.key === 'ArrowDown' ? items[Math.min(cur + 1, items.length - 1)] : items[Math.max(cur - 1, 0)]);
    }
    if (e.key === 'Enter' && _comboHighlight) selectComboboxModel(parseInt(_comboHighlight.dataset.idx));
  });
  document.addEventListener('click', e => {
    if (_comboOpen && !wrap.contains(e.target)) closeCombobox();
  });
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
