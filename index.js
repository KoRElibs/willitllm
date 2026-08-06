// ─── INDEX.JS — fit checker orchestrator and initialisation
//
// Entry point for index.html. Loaded last — all data and app files must precede it.
//
// Depends on:  MODELS, QUANT_INFO, KV_CACHE (data files),
//              app.calc.js (OVERHEAD_GB, autoKvBpe, calcMaxContext,
//                           calcSpeedEstimates, computeScores),
//              app.util.js (getLibMeta, getFlashOk),
//              app.shared.js (buildGpuSelector, initTooltip, initInfoSheet,
//                             muted),
//              index.combobox.js (buildModelCombobox, markComboboxItems,
//                                 syncComboboxFace, filterModelList),
//              index.variants.js (populateVariants, getSelectedVariant,
//                                 updateNudgeButtons, nudgeVariant),
//              index.ui.js (markModelOptions, applyCap, detectOs, setOs, syncOsSelect),
//              index.render.js (renderMembar, renderBudget, renderScorecard,
//                               renderVerdict, renderOom, renderAside, renderCmd),
//              index.details.js (renderDetails, populateGpuTab, renderFormula)
// Provides:    activeOsTab, runOpen, setupContent, getTargetCtx, getKvCache,
//              coderPageUrl, render (called by all ui files)

function getKvCache(bytesPerElement) {
  return KV_CACHE.find(k => k.bytesPerElement === bytesPerElement) || KV_CACHE[0];
}

// A stored choice always wins; otherwise guess from the browser so most visitors get
// their own OS's instructions without touching the dropdown. storedOs (index.render.js)
// also migrates retired keys; detectOs lives in index.ui.js.
let activeOsTab = storedOs(detectOs());
const setupContent = Object.fromEntries(OS_KEYS.map(k => [k, '']));

// Whether the "how to run it" panel is open. Persisted like geekMode, but defaults to
// closed — the verdict and scorecard answer the question the page asks; setup is the
// follow-up step.
let runOpen = localStorage.getItem('runOpen') === 'true';

function getTargetCtx() {
  const v = document.getElementById('targetCtx').value;
  // 'none' (neutral default) and 'max' both mean "no numeric target" — push context to the
  // model's arch limit for KV/scoring. They differ only in how fit is COLOURED (see
  // isNeutralTarget / contextFitColor): 'none' stays neutral, 'max' is judged vs capability.
  return (v === 'max' || v === 'none') ? null : parseInt(v);
}

// True when the user has not chosen a target — the default "as much as fits" view. In this
// mode the context-fit signal is neutral (no green/amber/orange): we just show how much
// context the model can use, without judging it against a target the user never set.
function isNeutralTarget() {
  return document.getElementById('targetCtx').value === 'none';
}

// The context length to evaluate a model against for KV/scoring. The default "as much as
// fits" view uses a SENSIBLE baseline — a normal 32k working context, capped at the model's
// own max — at default (f16) quality. That is deliberately NOT the "full model context"
// behaviour (autoKvBpe drives to q4_0 and grades against the whole trained window), which
// unfairly drags big-context models down. Explicit sizes and 'max' pass straight through.
function effectiveTargetCtx(model) {
  if (isNeutralTarget()) return Math.min(32000, model?.context_length ?? 32000);
  return getTargetCtx();
}

// ── URL state ─────────────────────────────────────────────────────────────────

function coderPageUrl() {
  const gpuOpt = document.getElementById('vramInput').selectedOptions[0];
  if (!gpuOpt || gpuOpt.disabled) return 'coder.html';
  const p = new URLSearchParams();
  p.set('g', gpuOpt.textContent.trim());
  return 'coder.html#' + p.toString();
}

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

// ── Render orchestrator ───────────────────────────────────────────────────────

function render() {
  const vramGB    = parseFloat(document.getElementById('vramInput').value);
  const targetCtx = getTargetCtx();
  const flashOk   = getFlashOk();

  // Keep cross-page nav link in sync with current GPU selection
  const coderUrl    = coderPageUrl();
  const vibeNavLink = document.getElementById('vibeNavLink');
  if (vibeNavLink) vibeNavLink.href = coderUrl;

  if (!isNaN(vramGB) && vramGB > 0) markModelOptions(vramGB, targetCtx, flashOk);

  // Re-read after markModelOptions — auto-selection may have changed sel.value
  const modelIdx = parseInt(document.getElementById('modelSelect').value);
  const model    = MODELS[modelIdx];

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
  const quantization    = variant ? (variant.quantization || variant.format || '—') : '—';
  const quantInfo       = variantRatings(model, variant);
  const libInfo         = getLibMeta(model);
  const bytesPerElement = autoKvBpe(model, vramGB, weightsGB, effectiveTargetCtx(model), flashOk);
  const kvEntry         = getKvCache(bytesPerElement);
  const kvLabel         = kvEntry.label;
  const kvInfo          = kvEntry;

  const ctxResult = calcMaxContext(model, vramGB, bytesPerElement, weightsGB);
  const noFit     = weightsGB >= vramGB - OVERHEAD_GB;
  const scores    = computeScores(quantInfo, bytesPerElement, ctxResult, noFit, model, effectiveTargetCtx(model));
  const { contextFitPct, scoreClass } = scores;

  // Coding capability verdict — degrees of "it will code!"
  const codeVerdictEl = document.getElementById('codeVerdict');
  if (codeVerdictEl) {
    if (noFit) {
      codeVerdictEl.hidden = true;
    } else if (libInfo.coding_role === 'agent') {
      codeVerdictEl.innerHTML = `<a href="${coderUrl}" class="code-verdict--agent">it will code!</a>`;
      codeVerdictEl.className = 'code-verdict';
      codeVerdictEl.hidden = false;
    } else if (libInfo.coding_role === 'code') {
      codeVerdictEl.innerHTML = `<a href="${coderUrl}" class="code-verdict--code">it will code</a>`;
      codeVerdictEl.className = 'code-verdict';
      codeVerdictEl.hidden = false;
    } else if (libInfo.coding_role === 'fim') {
      codeVerdictEl.innerHTML = `<a href="${coderUrl}" class="code-verdict--fim">it will autocomplete</a>`;
      codeVerdictEl.className = 'code-verdict';
      codeVerdictEl.hidden = false;
    } else {
      codeVerdictEl.textContent = 'it will not code';
      codeVerdictEl.className = 'code-verdict code-verdict--none';
      codeVerdictEl.hidden = false;
    }
  }

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
    speedEsts = calcSpeedEstimates(model, variant, vramGB, quantInfo, ctxResult.maxCtx, ctxResult.kvCacheGB, bytesPerElement);
    renderAside(speedEsts, ctxResult, contextFitPct);
    renderCmd(model, libInfo, ctxResult, kvLabel);
  }

  populateGpuTab(vramGB, speedEsts);
  renderFormula(model, variant, ctxResult, speedEsts, vramGB, weightsGB, bytesPerElement, kvLabel, quantInfo, noFit, contextFitPct);
  updateNudgeButtons(vramGB);
  pushHashState();
}

// ── Init ──────────────────────────────────────────────────────────────────────

function init() {
  initInfoSheet();
  buildGpuSelector();

  // Fixed model order: most capable (largest params) first, by tag for ties. Set once
  // here and never re-sorted — the combobox keeps this order and only recolours rows by
  // fit, so the list is stable and never reshuffles when the card or context changes.
  MODELS.sort((a, b) => modelParamSize(b) - modelParamSize(a) || a.ollama_tag.localeCompare(b.ollama_tag));
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
    { value: 'none',   wide: 'as much as fits',           narrow: 'as much as fits' },
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

  document.getElementById('osSelect').addEventListener('change', e => setOs(e.target.value));

  // "how to run it" toggle — lives in the result card, opens the setup panel below it.
  const runToggle  = document.getElementById('runToggle');
  const runSection = document.getElementById('runSection');
  const applyRun = on => {
    runOpen = on;
    runSection.hidden     = !on;
    runToggle.textContent = on ? '▾ how to run it' : '▸ how to run it';
    runToggle.setAttribute('aria-expanded', String(on));
  };
  runToggle.setAttribute('aria-controls', 'runSection');
  applyRun(runOpen);
  runToggle.addEventListener('click', () => {
    applyRun(runSection.hidden);
    localStorage.setItem('runOpen', runOpen);
  });

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

  initTooltip();

  // Breakdown toggle — persistent via localStorage, collapsed by default. Storage key stays
  // `geekMode` so an existing visitor's preference survives the label change.
  const geekToggle  = document.getElementById('geekToggle');
  const geekSection = document.getElementById('geekSection');
  const applyGeek = on => {
    geekSection.hidden     = !on;
    geekToggle.textContent = on ? '▾ the numbers' : '▸ the numbers';
    geekToggle.setAttribute('aria-expanded', String(on));
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
