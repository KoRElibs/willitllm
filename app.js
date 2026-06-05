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
  const vramGB    = parseFloat(document.getElementById('vramInput').value);
  const targetCtx = getTargetCtx();
  const flashOk   = getFlashOk();

  if (!isNaN(vramGB) && vramGB > 0) markModelOptions(vramGB, targetCtx, flashOk);

  // Re-read after markModelOptions — auto-selection may have changed sel.value
  const modelIdx = parseInt(document.getElementById('modelSelect').value);
  const model    = MODELS[modelIdx];
  updateSelectionSummary(model);

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
  document.getElementById('overheadReservedSheet').textContent = fmtGB(OVERHEAD_GB);

  // Info sheet
  const openBtn   = document.getElementById('infoSheetOpen');
  const closeBtn  = document.getElementById('infoSheetClose');
  const backdrop  = document.getElementById('infoSheetBackdrop');
  const sheet     = document.getElementById('infoSheet');
  function openSheet()  { sheet.hidden = false; backdrop.hidden = false; }
  function closeSheet() { sheet.hidden = true;  backdrop.hidden = true; }
  openBtn.addEventListener('click', openSheet);
  closeBtn.addEventListener('click', closeSheet);
  backdrop.addEventListener('click', closeSheet);
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeSheet(); });

  // GPU dropdown
  const vramSel = document.getElementById('vramInput');

  const vramPlaceholder = document.createElement('option');
  vramPlaceholder.value       = '';
  vramPlaceholder.disabled    = true;
  vramPlaceholder.selected    = true;
  vramPlaceholder.textContent = 'Select your GPU...';
  vramSel.appendChild(vramPlaceholder);

  // Helper: append an option to a group
  function addOpt(group, name, vram, flash, gpuIdx) {
    const opt          = document.createElement('option');
    opt.value          = vram;
    opt.dataset.flash  = flash;
    if (gpuIdx !== undefined) opt.dataset.gpuIdx = gpuIdx;
    opt.textContent    = name;
    group.appendChild(opt);
  }

  // GeForce = consumer GTX/RTX (4-digit model), excluding professional Ada/A-series
  const isGeForce = name => /^(GTX |RTX \d{4})/.test(name) && !name.includes('Ada');

  const groups = [
    { label: 'NVIDIA GeForce',     match: (gpu, name) => !gpu.vendor && isGeForce(name) },
    { label: 'NVIDIA Professional',match: (gpu, name) => !gpu.vendor && !isGeForce(name) },
    { label: 'AMD Radeon',         match: (gpu)       => gpu.vendor === 'AMD' },
    { label: 'Apple',              match: (gpu)       => gpu.vendor === 'Apple' },
  ];

  groups.forEach(({ label, match }) => {
    const cards = [];
    GPUS.forEach((gpu, gpuIdx) => {
      gpu.names.forEach(name => {
        if (match(gpu, name)) cards.push({ name, vram: gpu.vram, flash: gpu.flash, gpuIdx });
      });
    });
    if (!cards.length) return;

    cards.sort((a, b) => a.name.localeCompare(b.name));
    const group = document.createElement('optgroup');
    group.label = label;
    cards.forEach(({ name, vram, flash, gpuIdx }) => addOpt(group, name, vram, flash, gpuIdx));
    vramSel.appendChild(group);
  });

  // Generic entries — one per unique VRAM size, sorted descending
  const sizes = [...new Set(GPUS.map(g => g.vram))].sort((a, b) => b - a);
  const genericGroup       = document.createElement('optgroup');
  genericGroup.label       = 'Generic';
  sizes.forEach(vram => {
    const entries     = GPUS.filter(g => g.vram === vram);
    const flashValues = [...new Set(entries.map(g => g.flash))];
    const flash       = flashValues.length === 1 ? flashValues[0] : 'mixed';
    addOpt(genericGroup, `${vram} GB`, vram, flash);
  });
  vramSel.appendChild(genericGroup);

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

  // Capability filter pills
  document.querySelectorAll('.cap-pill').forEach(pill => {
    pill.addEventListener('click', () => applyCap(pill.dataset.cap));
  });

  // Swap target context option labels for narrow viewports (native selects truncate long text)
  const TARGET_LABELS = [
    { value: '8000',   wide: 'a chat · ~25 pages',        narrow: 'chat' },
    { value: '32000',  wide: 'a document · ~100 pages',   narrow: 'document' },
    { value: '64000',  wide: 'The Hobbit · ~200 pages',   narrow: 'The Hobbit' },
    { value: '100000', wide: 'Harry Potter · ~300 pages', narrow: 'Harry Potter' },
    { value: '200000', wide: 'several books · ~600 pages',narrow: 'several books' },
    { value: 'max',    wide: 'full model context',        narrow: 'full context' },
  ];
  function updateTargetCtxLabels() {
    const narrow = window.innerWidth <= 400;
    const sel    = document.getElementById('targetCtx');
    TARGET_LABELS.forEach(({ value, wide, narrow: short }) => {
      const opt = Array.from(sel.options).find(o => o.value === value);
      if (opt) opt.textContent = narrow ? short : wide;
    });
  }
  updateTargetCtxLabels();

  // Rebuild variant options on resize (mobile vs desktop format differs)
  let lastMobile = window.innerWidth <= 600;
  window.addEventListener('resize', () => {
    const isMobile = window.innerWidth <= 600;
    updateTargetCtxLabels();
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
