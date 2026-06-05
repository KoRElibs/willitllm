// ─────────────────────────────────────────────────────────────────────────────
// APP.MAIN — shared state, render orchestrator, initialisation
//
// Entry point. Loaded last — all data files and app files must precede it.
//
// Globals:     GPUS, LIBRARIES, MODELS, QUANT_INFO, KV_CACHE (data files)
// ─────────────────────────────────────────────────────────────────────────────

// Lookup helpers derived from data files
function getKvCache(bytesPerElement) {
  return KV_CACHE.find(k => k.bytesPerElement === bytesPerElement) || KV_CACHE[0];
}

let activeOsTab = null;                   // 'linux' | 'windows' | null
const setupContent = { linux: '', windows: '' };

function getTargetCtx() {
  const v = document.getElementById('targetCtx').value;
  return v === 'max' ? null : parseInt(v);
}

function getFlashOk() {
  const opt = document.getElementById('vramInput').selectedOptions[0];
  return opt?.dataset.flash === 'yes' || opt?.dataset.flash === 'mixed';
}

const LIB_META = Object.fromEntries(LIBRARIES.map(l => [l.library, l]));

function getLibMeta(m) {
  return LIB_META[m.ollama_tag.split(':')[0]] || {};
}

// ─────────────────────────────────────────────────────────────────────────────
// URL STATE — encode/restore selections in the URL hash for shareability
// ─────────────────────────────────────────────────────────────────────────────

function pushHashState() {
  const gpuSel   = document.getElementById('vramInput');
  const modelSel = document.getElementById('modelSelect');
  const gpuOpt   = gpuSel.selectedOptions[0];
  const modelIdx = parseInt(modelSel.value);
  const model    = MODELS[modelIdx];
  const variant  = model ? getSelectedVariant(model) : null;
  if (!gpuOpt || gpuOpt.disabled || !model) return;
  const p = new URLSearchParams();
  p.set('g', gpuOpt.textContent.trim());
  p.set('m', model.ollama_tag);
  if (variant) p.set('v', variant.tag);
  p.set('t', document.getElementById('targetCtx').value);
  history.replaceState(null, '', '#' + p.toString());
}

function applyHashState() {
  const hash = window.location.hash.slice(1);
  if (!hash) return;
  const p = new URLSearchParams(hash);
  const gpuName  = p.get('g');
  const modelTag = p.get('m');
  const varTag   = p.get('v');
  const target   = p.get('t');

  if (gpuName) {
    const gpuSel = document.getElementById('vramInput');
    const opt = Array.from(gpuSel.options).find(o => o.textContent.trim() === gpuName);
    if (opt) opt.selected = true;
  }
  if (target) {
    const sel = document.getElementById('targetCtx');
    if (Array.from(sel.options).some(o => o.value === target)) sel.value = target;
  }
  if (modelTag) {
    const modelSel = document.getElementById('modelSelect');
    const modelIdx = MODELS.findIndex(m => m.ollama_tag === modelTag);
    if (modelIdx !== -1) {
      modelSel.value = modelIdx;
      const model = MODELS[modelIdx];
      populateVariants(model);
      if (varTag) {
        const varIdx = model.variants ? model.variants.findIndex(v => v.tag === varTag) : -1;
        if (varIdx !== -1) document.getElementById('variantSelect').value = varIdx;
      }
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// RENDER — orchestrator
// ─────────────────────────────────────────────────────────────────────────────
function render() {
  const modelIdx  = parseInt(document.getElementById('modelSelect').value);
  const vramGB    = parseFloat(document.getElementById('vramInput').value);
  const targetCtx = getTargetCtx();
  const flashOk   = getFlashOk();
  const model     = MODELS[modelIdx];
  updateSelectionSummary(model);

  if (!isNaN(vramGB) && vramGB > 0) markModelOptions(vramGB, targetCtx, flashOk);

  const noModel = document.getElementById('noModel');
  const results = document.getElementById('results');
  if (!model || isNaN(vramGB) || vramGB <= 0) {
    noModel.hidden = false;
    results.hidden = true;
    return;
  }
  noModel.hidden = true;
  results.hidden = false;

  const variant         = getSelectedVariant(model);
  const weightsGB       = variant ? variant.weights_gb : 0;
  const quantization    = variant ? variant.quantization : '—';
  const quantInfo       = variant ? QUANT_INFO[variant.quantization] : null;
  const libInfo         = getLibMeta(model);
  const bytesPerElement = autoKvBpe(model, vramGB, weightsGB, targetCtx, flashOk);
  const kvEntry         = getKvCache(bytesPerElement);
  const kvLabel         = kvEntry.label;
  const kvInfo          = kvEntry;

  const ctxResult = calcMaxContext(model, vramGB, bytesPerElement, weightsGB);
  const noFit     = weightsGB >= vramGB - OVERHEAD_GB;
  const scores    = computeScores(quantInfo, bytesPerElement, ctxResult, noFit, model, getTargetCtx());
  const { contextFitPct, scoreClass } = scores;

  document.getElementById('resultHeadline').className = `result-headline ${scoreClass}`;

  renderMembar(vramGB, weightsGB, ctxResult, noFit);
  renderBudget(vramGB, weightsGB, ctxResult, noFit);
  renderScorecard(scores, quantInfo, variant, kvLabel, kvInfo, libInfo, ctxResult, noFit);
  renderVerdict(noFit);
  renderDetails(model, libInfo, variant, weightsGB, quantization, bytesPerElement, kvLabel);

  let speedEsts = null;
  if (noFit) {
    renderOom(vramGB, weightsGB);
  } else {
    speedEsts = calcSpeedEstimates(model, variant, vramGB, quantInfo, ctxResult.maxCtx);
    renderAside(speedEsts, ctxResult, contextFitPct);
    renderCmd(model, ctxResult, kvLabel, bytesPerElement);
  }

  populateGpuTab(vramGB, speedEsts);
  renderFormula(model, variant, ctxResult, speedEsts, vramGB, weightsGB, bytesPerElement, kvLabel, quantInfo, noFit, contextFitPct);
  updateNudgeButtons(vramGB);
  pushHashState();
}

// ─────────────────────────────────────────────────────────────────────────────
// INIT
// ─────────────────────────────────────────────────────────────────────────────
function init() {
  document.getElementById('overheadReserved').textContent = fmtGB(OVERHEAD_GB);

  // GPU dropdown
  const vramSel = document.getElementById('vramInput');

  const vramPlaceholder = document.createElement('option');
  vramPlaceholder.value       = '';
  vramPlaceholder.disabled    = true;
  vramPlaceholder.selected    = true;
  vramPlaceholder.textContent = 'Select your GPU...';
  vramSel.appendChild(vramPlaceholder);

  // Generic entries — one per unique VRAM size
  const sizes = [...new Set(GPUS.map(g => g.vram))];
  sizes.forEach(vram => {
    const entries     = GPUS.filter(g => g.vram === vram);
    const flashValues = [...new Set(entries.map(g => g.flash))];
    const flash       = flashValues.length === 1 ? flashValues[0] : 'mixed';
    const opt         = document.createElement('option');
    opt.value         = vram;
    opt.dataset.flash = flash;
    opt.textContent   = `Generic ${vram} GB`;
    vramSel.appendChild(opt);
  });

  const sep       = document.createElement('option');
  sep.disabled    = true;
  sep.textContent = '— pick your card —';
  vramSel.appendChild(sep);

  // Individual cards sorted alphabetically
  const cards = GPUS.flatMap((gpu, gpuIdx) => gpu.names.map(name => ({ name, vram: gpu.vram, flash: gpu.flash, gpuIdx })));
  cards.sort((a, b) => a.name.localeCompare(b.name));
  cards.forEach(({ name, vram, flash, gpuIdx }) => {
    const opt          = document.createElement('option');
    opt.value          = vram;
    opt.dataset.flash  = flash;
    opt.dataset.gpuIdx = gpuIdx;
    opt.textContent    = name;
    vramSel.appendChild(opt);
  });

  // Model dropdown (hidden — combobox is the visible control)
  MODELS.sort((a, b) => a.ollama_tag.localeCompare(b.ollama_tag));
  const sel = document.getElementById('modelSelect');
  const modelPlaceholder = document.createElement('option');
  modelPlaceholder.value = ''; modelPlaceholder.disabled = true; modelPlaceholder.selected = true;
  modelPlaceholder.textContent = 'Select a model...';
  sel.appendChild(modelPlaceholder);
  MODELS.forEach((m, i) => {
    const opt = document.createElement('option');
    opt.value = i; opt.textContent = m.ollama_tag;
    sel.appendChild(opt);
  });

  buildModelCombobox();
  populateVariants(null);

  // Event listeners
  sel.addEventListener('change', () => {
    populateVariants(MODELS[parseInt(sel.value)]);
    render();
  });
  document.getElementById('vramInput').addEventListener('change', render);
  document.getElementById('targetCtx').addEventListener('change', render);
  document.getElementById('variantSelect').addEventListener('change', render);

  // Rebuild variant options on resize (mobile vs desktop format differs)
  let lastMobile = window.innerWidth <= 600;
  window.addEventListener('resize', () => {
    const isMobile = window.innerWidth <= 600;
    if (isMobile !== lastMobile) {
      lastMobile = isMobile;
      const modelIdx = parseInt(document.getElementById('modelSelect').value);
      if (MODELS[modelIdx]) populateVariants(MODELS[modelIdx]);
    }
  });

  document.getElementById('tabLinux').addEventListener('click',   () => setOsTab('linux'));
  document.getElementById('tabWindows').addEventListener('click', () => setOsTab('windows'));

  document.getElementById('nudge-speed').addEventListener('click',   () => nudgeVariant('speed'));
  document.getElementById('nudge-quality').addEventListener('click', () => nudgeVariant('quality'));

  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(btn.dataset.tab).classList.add('active');
    });
  });

  // Tooltip
  const tip = document.getElementById('tooltip');
  document.addEventListener('mouseover', e => {
    const el = e.target.closest('[data-tip]');
    if (!el) { tip.hidden = true; return; }
    tip.textContent = el.dataset.tip;
    tip.hidden = false;
    const rect = el.getBoundingClientRect();
    tip.style.top  = (rect.bottom + 8) + 'px';
    tip.style.left = Math.min(rect.left, window.innerWidth - 276) + 'px';
  });

  // Geek mode — persistent via localStorage, off by default
  const geekToggle  = document.getElementById('geekToggle');
  const geekSection = document.getElementById('geekSection');
  const applyGeek = on => {
    geekSection.hidden    = !on;
    geekToggle.textContent = on ? '▾ details' : '▸ details';
  };
  applyGeek(localStorage.getItem('geekMode') === 'true');
  geekToggle.addEventListener('click', () => {
    const on = geekSection.hidden;
    applyGeek(on);
    localStorage.setItem('geekMode', on);
  });

  applyHashState();
  window.addEventListener('hashchange', () => { applyHashState(); render(); });

  render();
}

init();
