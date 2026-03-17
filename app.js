// ─────────────────────────────────────────────────────────────────────────────
// CONSTANTS
// ─────────────────────────────────────────────────────────────────────────────
const KV_CACHE_LABELS = { '2': 'f16', '1': 'q8_0', '0.5': 'q4_0' };
const KV_CACHE_INFO = {
  'f16':  { summary: 'Full precision — works on every GPU, no setup needed.' },
  'q8_0': { summary: 'Half the memory per token — slight precision loss, fits more context in VRAM.' },
  'q4_0': { summary: 'Quarter the memory per token — more precision loss, maximum context for the VRAM.' },
};

function getBytesPerElement() {
  return parseFloat(document.getElementById('kvCacheType').value);
}

const OVERHEAD_FACTOR = 0.92;    // reserve ~8% for CUDA context, activations, OS
const CTX_LABELS = [
  [131072,'128k'],[65536,'64k'],[32768,'32k'],[16384,'16k'],
  [8192,'8k'],[4096,'4k'],[2048,'2k'],[1024,'1k'],
];
const POWERS_OF_2 = [131072, 65536, 32768, 16384, 8192, 4096, 2048, 1024];

const FLAGS = {
  'USA': '🇺🇸', 'France': '🇫🇷', 'EU': '🇪🇺', 'Canada': '🇨🇦',
  'UK': '🇬🇧', 'UAE': '🇦🇪', 'Switzerland': '🇨🇭', 'South Korea': '🇰🇷',
  'Singapore': '🇸🇬', 'Portugal': '🇵🇹', 'International': '🌍',
};
const LIB_META = Object.fromEntries(LIBRARIES.map(l => [l.library, l]));

function getLibMeta(m) {
  return LIB_META[m.ollama_tag.split(':')[0]] || {};
}

function formatModelOption(m) {
  const [library] = m.ollama_tag.split(':');
  const libInfo = LIB_META[library];
  const flag = libInfo?.origin ? (FLAGS[libInfo.origin] || '🌍') : '👥';
  return `${m.ollama_tag}  ${flag}`;
}

// ─────────────────────────────────────────────────────────────────────────────
// CORE CALC
// ─────────────────────────────────────────────────────────────────────────────
function calcMaxContext(model, vramGB, bytesPerElement, weightsGB) {
  const availableBytes = (vramGB * OVERHEAD_FACTOR - weightsGB) * 1024 ** 3;
  if (availableBytes <= 0) return { maxCtx: 0, kvCacheGB: 0, freeGB: 0, availableBytes };

  const perToken = model.block_count * model.head_count_kv * model.key_length * 2 * bytesPerElement;
  const rawTokens = availableBytes / perToken;
  const archLimit = model.context_length || Infinity;
  const archMaxRaw = Math.min(rawTokens, archLimit);
  const maxCtx = POWERS_OF_2.find(p => p <= archMaxRaw) || 0;
  const limitedByArch = isFinite(archLimit) && rawTokens > archLimit;
  const kvCacheGB = (maxCtx * perToken) / 1024 ** 3;
  const usedGB = weightsGB + kvCacheGB;
  const freeGB = vramGB - usedGB;

  return { maxCtx, kvCacheGB, freeGB, perToken, rawTokens, availableBytes, usedGB, archLimit, limitedByArch };
}

function fmtGB(n) { return n.toFixed(2) + ' GB'; }
function fmtTokensHuman(n) {
  // ~0.75 words per token; ~250 words per page
  const words = Math.round(n * 0.75 / 500) * 500;
  const pages = Math.round(n / 333 / 5) * 5;
  const w = words >= 1000 ? `${(words / 1000).toFixed(0)}k` : words;
  return `≈${w} words · ~${pages} pages`;
}
function fmtCtx(n) {
  return (CTX_LABELS.find(([t]) => n >= t) || [0, '0'])[1];
}

// ─────────────────────────────────────────────────────────────────────────────
// VARIANT DROPDOWN
// ─────────────────────────────────────────────────────────────────────────────
function buildRatingBar(val, filled, empty, max = 5) {
  const n = Math.round((val / 10) * max);
  return filled.repeat(n) + empty.repeat(max - n);
}

function updateVariantInfo(model) {
  const el = document.getElementById('variantInfo');
  if (!el) return;
  const selectedVariant = getSelectedVariant(model);
  const quantInfo = selectedVariant && QUANT_INFO[selectedVariant.quantization];
  el.textContent = quantInfo ? quantInfo.summary : '';
}

function populateVariants(model) {
  const sel = document.getElementById('variantSelect');
  sel.innerHTML = '';
  if (!model || !model.variants || model.variants.length === 0) {
    const opt = document.createElement('option');
    opt.textContent = 'no variants';
    sel.appendChild(opt);
    updateVariantInfo(model);
    return;
  }
  model.variants.forEach((variant, i) => {
    const opt = document.createElement('option');
    opt.value = i;
    const quantInfo     = QUANT_INFO[variant.quantization];
    const speedRating   = quantInfo ? buildRatingBar(quantInfo.speed,   '⚡︎', '·') : '·····';
    const qualityRating = quantInfo ? buildRatingBar(quantInfo.quality, '★',  '☆') : '☆☆☆☆☆';
    opt.textContent = `${speedRating} ${qualityRating}  ${variant.tag} — ${variant.weights_gb.toFixed(1)} GB`;
    sel.appendChild(opt);
  });
  updateVariantInfo(model);
}

function getSelectedVariantIdx(model) {
  if (!model || !model.variants || model.variants.length === 0) return 0;
  return Math.min(parseInt(document.getElementById('variantSelect').value) || 0, model.variants.length - 1);
}

function getSelectedVariant(model) {
  if (!model || !model.variants || model.variants.length === 0) return null;
  return model.variants[getSelectedVariantIdx(model)];
}

// Build the full ollama tag for a specific variant, e.g. "llama3.2:3b-q4_K_M".
// Uses the stored variant.tag (the original sub-tag from ollama.com).
function variantOllamaTag(model, variantIdx) {
  const variant = model.variants[variantIdx];
  const library = model.ollama_tag.split(':')[0];
  return `${library}:${variant.tag}`;
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK OOM OPTIONS IN DROPDOWN
// ─────────────────────────────────────────────────────────────────────────────
function markModelOptions(vramGB, bytesPerElement) {
  const sel = document.getElementById('modelSelect');
  Array.from(sel.options).forEach((opt, i) => {
    const m = MODELS[i];
    const weightsGB = m.variants && m.variants.length ? m.variants[0].weights_gb : 0;
    const fits = weightsGB < vramGB * OVERHEAD_FACTOR;
    if (!fits) {
      opt.textContent = `✗  ${formatModelOption(m)}`;
      opt.style.color = '#f06464';
      return;
    }
    const ctxResult     = calcMaxContext(m, vramGB, bytesPerElement, weightsGB);
    const contextFitPct = m.context_length ? Math.round((ctxResult.maxCtx / m.context_length) * 100) : 100;
    opt.textContent = formatModelOption(m);
    opt.style.color = contextFitPct >= 66 ? '#56d88a' : contextFitPct >= 33 ? '#f5a623' : '#f07418';
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// RENDER
// ─────────────────────────────────────────────────────────────────────────────
function render() {
  const modelIdx        = parseInt(document.getElementById('modelSelect').value);
  const vramGB          = parseFloat(document.getElementById('vramInput').value);
  const bytesPerElement = getBytesPerElement();
  const kvLabel         = KV_CACHE_LABELS[String(bytesPerElement)] || 'f16';
  const kvInfo          = KV_CACHE_INFO[kvLabel];
  if (kvInfo) { document.getElementById('kvCacheInfoText').textContent = `${kvLabel} — ${kvInfo.summary}`; }
  const model = MODELS[modelIdx];

  const noModel = document.getElementById('noModel');
  const results = document.getElementById('results');

  if (!model || isNaN(vramGB) || vramGB <= 0) {
    noModel.hidden = false;
    results.hidden = true;
    return;
  }

  noModel.hidden = true;
  results.hidden = false;

  const variant      = getSelectedVariant(model);
  const weightsGB    = variant ? variant.weights_gb : 0;
  const quantization = variant ? variant.quantization : '—';

  const quantInfo = variant ? QUANT_INFO[variant.quantization] : null;
  const infoEl    = document.getElementById('variantInfo');
  infoEl.textContent = quantInfo ? `${variant.quantization} — ${quantInfo.summary}` : '';

  markModelOptions(vramGB, bytesPerElement);

  const ctxResult = calcMaxContext(model, vramGB, bytesPerElement, weightsGB);
  const noFit     = weightsGB >= vramGB * OVERHEAD_FACTOR;

  // ── memory bar
  const modelPct   = Math.min(100, (weightsGB / vramGB) * 100);
  const contextPct = noFit ? 0 : Math.min(100 - modelPct, (ctxResult.kvCacheGB / vramGB) * 100);
  const freePct    = Math.max(0, 100 - modelPct - contextPct);

  document.getElementById('barTotal').textContent = fmtGB(vramGB) + ' total';

  const segModel = document.getElementById('segModel');
  segModel.className   = 'membar-seg ' + (noFit ? 'seg-overflow' : 'seg-model');
  segModel.style.width = modelPct.toFixed(1) + '%';
  segModel.textContent = modelPct > 12 ? fmtGB(weightsGB) : '';

  const segContext = document.getElementById('segContext');
  segContext.className   = 'membar-seg ' + (noFit ? 'seg-overflow' : 'seg-context');
  segContext.style.width = contextPct.toFixed(1) + '%';
  segContext.textContent = contextPct > 8 ? fmtCtx(ctxResult.maxCtx) : '';

  const segFree = document.getElementById('segFree');
  segFree.style.width  = freePct.toFixed(1) + '%';
  segFree.style.flex   = freePct < 1 ? `0 0 ${freePct.toFixed(1)}%` : '1';
  segFree.textContent  = freePct > 8 ? fmtGB(ctxResult.freeGB) : '';

  document.getElementById('legendModel').textContent   = `Model weights (${fmtGB(weightsGB)})`;
  document.getElementById('legendContext').textContent = `KV cache @ ${fmtCtx(ctxResult.maxCtx)} ctx (${fmtGB(ctxResult.kvCacheGB)})`;
  document.getElementById('legendFree').textContent    = `Free (${fmtGB(Math.max(0, ctxResult.freeGB))})`;

  // ── result headline
  const headline  = document.getElementById('resultHeadline');
  const labelOom  = document.getElementById('resultLabelOom');
  const verdictEl = document.getElementById('verdict');
  const scorecard = document.getElementById('scorecard');
  const ollamaCmd = document.getElementById('ollamaCmd');

  const libInfo = getLibMeta(model);

  // ── Scorecard
  function stars(n) { return '★'.repeat(n) + '☆'.repeat(5 - n); }
  function colorForScore(n) {
    return n >= 4 ? 'var(--green)' : n === 3 ? 'var(--amber)' : n === 2 ? 'var(--orange)' : 'var(--red)';
  }

  const scoreSpeed     = quantInfo ? Math.max(1, Math.round((quantInfo.speed   / 10) * 5)) : 0;
  const scoreQuality   = quantInfo ? Math.max(1, Math.round((quantInfo.quality / 10) * 5)) : 0;
  const contextFitPct  = (!noFit && model.context_length) ? Math.round((ctxResult.maxCtx / model.context_length) * 100) : null;
  const scoreContext   = contextFitPct === null ? 0 : contextFitPct >= 90 ? 5 : contextFitPct >= 66 ? 4 : contextFitPct >= 40 ? 3 : contextFitPct >= 15 ? 2 : 1;
  const scorePrecision = bytesPerElement === 2 ? 5 : bytesPerElement === 1 ? 3 : 2;
  const scoreAvg       = (scoreSpeed + scoreQuality + scoreContext + scorePrecision) / 4;
  const scoreClass     = noFit ? 'error' : scoreAvg >= 4 ? 'score-high' : scoreAvg >= 3 ? 'score-mid' : scoreAvg >= 2 ? 'score-low' : 'score-poor';

  headline.className = `result-headline ${scoreClass}`;

  if (!noFit) {
    [
      ['scoreSpeed',     scoreSpeed],
      ['scoreQuality',   scoreQuality],
      ['scorePrecision', scorePrecision],
      ['scoreContext',   scoreContext],
    ].forEach(([id, score]) => {
      const el = document.getElementById(id);
      el.textContent = stars(score);
      el.style.color = colorForScore(score);
    });

    const ctxSub = `${fmtCtx(ctxResult.maxCtx)} tokens${contextFitPct !== null && contextFitPct < 100 ? ` · ${contextFitPct}% of ${fmtCtx(model.context_length)} max` : ''}`;
    document.getElementById('scoreContextSub').textContent = libInfo.multimodal
      ? ctxSub + ' · images consume tokens'
      : ctxSub;
    scorecard.hidden = false;
  } else {
    scorecard.hidden = true;
  }

  // Restart animation on every render
  verdictEl.classList.remove('verdict-anim');
  void verdictEl.offsetWidth;
  verdictEl.textContent = noFit ? "IT WON'T LLM!" : "IT WILL LLM!";
  verdictEl.classList.add('verdict-anim');

  // ── model info
  document.getElementById('detailOrganization').textContent = libInfo.organization || '—';

  const originEl = document.getElementById('detailOrigin');
  if (libInfo.origin) {
    originEl.textContent = `${FLAGS[libInfo.origin] || ''} ${libInfo.origin}`.trim();
    originEl.className = 'detail-val';
  } else {
    originEl.textContent = 'community project';
    originEl.className = 'detail-val community-origin';
  }
  document.getElementById('detailMoeRow').hidden        = !model.moe;
  document.getElementById('detailMultimodalRow').hidden = !libInfo.multimodal;
  document.getElementById('detailMaxCtx').textContent   = model.context_length
    ? model.context_length.toLocaleString() + ' tokens' : '—';

  if (noFit) {
    labelOom.textContent = `Model weights (${fmtGB(weightsGB)}) exceed available VRAM (${fmtGB(vramGB * OVERHEAD_FACTOR)} usable). This model will not load.`;
    labelOom.hidden = false;
    ollamaCmd.hidden = true;
    document.getElementById('ollamaHint').hidden = true;
  } else {
    labelOom.hidden = true;

    const ctxHint = document.getElementById('ctxHint');
    if (scoreContext <= 3) {
      ctxHint.textContent = `Runs fine at ${fmtCtx(ctxResult.maxCtx)} — ${fmtTokensHuman(ctxResult.maxCtx)} of typical English text. Keep num_ctx at or below this limit (the command below sets it).`;
      ctxHint.hidden = false;
    } else {
      ctxHint.hidden = true;
    }

    const flashSel     = document.getElementById('vramInput').selectedOptions[0];
    const flashSupport = flashSel ? flashSel.dataset.flash : 'yes';
    const flashTip     = document.getElementById('kvFlashTip');
    if (bytesPerElement < 2) {
      const tip = flashSupport === 'mixed'
        ? `AMD support varies by ollama build and driver. Verify your setup before setting OLLAMA_KV_CACHE_TYPE.`
        : flashSupport !== 'yes'
        ? `Not supported on Turing (RTX 20xx) or older NVIDIA GPUs. This setting will have no effect or may cause errors.`
        : `Gains are modest below 8k context — most effective at the longer context windows this VRAM allows.`;
      flashTip.dataset.tip = tip;
      flashTip.hidden = false;
    } else {
      flashTip.hidden = true;
    }

    const hintEl         = document.getElementById('ollamaHint');
    const winToggle      = document.getElementById('winToggle');
    const winToggleLabel = document.getElementById('winToggleLabel');
    const useWindows     = winToggle.checked;
    winToggleLabel.hidden = false;

    const muted      = s => `<span class="cmd-muted">${s}</span>`;
    const variantIdx = getSelectedVariantIdx(model);
    const runTag     = variantOllamaTag(model, variantIdx);
    const runLines   = `ollama run ${runTag}\n>>> /set parameter num_ctx ${ctxResult.maxCtx}`;

    if (useWindows) {
      ollamaCmd.innerHTML = [
        muted(`# 1. Open: System Properties → Environment Variables → New user variable`),
        muted(`#    Name:  OLLAMA_KV_CACHE_TYPE`),
        muted(`#    Value: ${kvLabel}`),
        muted(`# 2. Right-click Ollama in the system tray → Quit, then relaunch Ollama`),
        muted(`# 3. Run:`),
        runLines,
      ].join('\n');
      hintEl.textContent = 'Set the environment variable via Windows System Properties, restart Ollama from the system tray, then run the command.';
    } else {
      ollamaCmd.innerHTML = [
        muted(`# Stop ollama if running, then restart with the KV cache setting:`),
        `OLLAMA_KV_CACHE_TYPE=${kvLabel} ollama serve`,
        muted(`# In a new terminal:`),
        runLines,
      ].join('\n');
      hintEl.textContent = 'Restart ollama with the env var set, then run the command in a new terminal.';
    }
    hintEl.hidden = false;
    ollamaCmd.hidden = false;
  }

  // ── details table
  const detailValues = {
    detailLayers:       model.block_count,
    detailAttnHeads:    model.head_count,
    detailKvHeads:      model.head_count_kv,
    detailHeadDim:      model.key_length,
    detailHiddenSize:   model.embedding_length,
    detailAttnHeadsDiv: model.head_count,
    detailBpe:          bytesPerElement,
    detailBpeLabel:     kvLabel,
    detailWeights:      fmtGB(weightsGB),
    detailQuantization: quantization,
  };
  Object.entries(detailValues).forEach(([id, val]) => {
    document.getElementById(id).textContent = val;
  });

  // Architecture source cells → link to ollama library page
  const [library] = model.ollama_tag.split(':');
  const ollamaLibUrl  = `https://ollama.com/library/${library}`;
  const ollamaSrcLink = `<a href="${ollamaLibUrl}" target="_blank" rel="noopener noreferrer">ollama.com ↗</a>`;
  document.querySelectorAll('.src-config-json').forEach(el => { el.innerHTML = ollamaSrcLink; });
  document.getElementById('srcAttnHeads').innerHTML = ollamaSrcLink + ' · not used in calc';
  const derivedHeadDim = model.key_length === Math.floor(model.embedding_length / model.head_count);
  document.getElementById('srcHeadDim').innerHTML = derivedHeadDim
    ? ollamaSrcLink + ' · derived'
    : ollamaSrcLink + ' · explicit';

  // Model weights source
  const modelName     = model.ollama_tag.split(':')[0];
  const ollamaPageUrl = modelName.includes('/')
    ? `https://ollama.com/${modelName}`
    : `https://ollama.com/library/${modelName}`;
  document.getElementById('detailSource').innerHTML =
    `<a href="${ollamaPageUrl}" target="_blank" rel="noopener noreferrer">ollama.com ↗</a>`;

  document.getElementById('provenanceAlert').hidden = true;

  // ── formula breakdown
  const formulaBox  = document.getElementById('formulaBox');
  formulaBox.hidden = noFit;
  if (!noFit) {
    document.getElementById('formulaBlockCount').textContent = model.block_count;
    document.getElementById('formulaKvHeads').textContent    = model.head_count_kv;
    document.getElementById('formulaBpeLabel').textContent   = `${bytesPerElement}(${kvLabel})`;
    document.getElementById('formulaKeyLength').textContent  = model.key_length;
    document.getElementById('formulaPerToken').textContent   = `${ctxResult.perToken.toLocaleString()} bytes`;
    document.getElementById('formulaPerTokenKB').textContent = `(${(ctxResult.perToken / 1024).toFixed(1)} KB)`;
    document.getElementById('formulaAvailLabel').textContent = `available_vram = ${fmtGB(vramGB)} × ${OVERHEAD_FACTOR} overhead factor − ${fmtGB(weightsGB)} weights`;
    document.getElementById('formulaAvailGB').textContent    = fmtGB(vramGB * OVERHEAD_FACTOR - weightsGB);
    document.getElementById('formulaAvailBytes').textContent = `${Math.round((vramGB * OVERHEAD_FACTOR - weightsGB) * 1024 ** 3).toLocaleString()} bytes`;
    document.getElementById('formulaRawTokens').textContent  = Math.round(ctxResult.rawTokens).toLocaleString();
    document.getElementById('formulaMaxCtx').textContent     = `${ctxResult.maxCtx.toLocaleString()} tokens (${fmtCtx(ctxResult.maxCtx)})`;
    const archCapNote = document.getElementById('formulaArchCapNote');
    if (ctxResult.limitedByArch) {
      document.getElementById('formulaArchLimit').textContent = ctxResult.archLimit.toLocaleString();
      archCapNote.hidden = false;
    } else {
      archCapNote.hidden = true;
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// INIT
// ─────────────────────────────────────────────────────────────────────────────
function init() {
  document.getElementById('overheadPct').textContent = Math.round((1 - OVERHEAD_FACTOR) * 100);

  // Build GPU dropdown from gpus.js data
  const vramSel = document.getElementById('vramInput');

  // Generic entries — one per unique VRAM size
  const sizes = [...new Set(GPUS.map(g => g.vram))];
  sizes.forEach(vram => {
    const entries     = GPUS.filter(g => g.vram === vram);
    const flashValues = [...new Set(entries.map(g => g.flash))];
    const flash       = flashValues.length === 1 ? flashValues[0] : 'mixed';
    const opt         = document.createElement('option');
    opt.value         = vram;
    opt.dataset.flash = flash;
    opt.textContent   = `${vram} GB`;
    if (entries.some(g => g.default)) opt.selected = true;
    vramSel.appendChild(opt);
  });

  // Separator
  const sep = document.createElement('option');
  sep.disabled    = true;
  sep.textContent = '— pick your card —';
  vramSel.appendChild(sep);

  // Individual card entries sorted alphabetically
  const cards = GPUS.flatMap(gpu => gpu.names.map(name => ({ name, vram: gpu.vram, flash: gpu.flash })));
  cards.sort((a, b) => a.name.localeCompare(b.name));
  cards.forEach(({ name, vram, flash }) => {
    const opt         = document.createElement('option');
    opt.value         = vram;
    opt.dataset.flash = flash;
    opt.textContent   = `${name} — ${vram} GB`;
    vramSel.appendChild(opt);
  });

  MODELS.sort((a, b) => a.ollama_tag.localeCompare(b.ollama_tag));

  const sel = document.getElementById('modelSelect');
  MODELS.forEach((m, i) => {
    const opt       = document.createElement('option');
    opt.value       = i;
    opt.textContent = formatModelOption(m);
    sel.appendChild(opt);
  });

  sel.addEventListener('change', () => {
    populateVariants(MODELS[parseInt(sel.value)]);
    render();
  });

  const initialIdx = parseInt(sel.value) || 0;
  if (MODELS[initialIdx]) populateVariants(MODELS[initialIdx]);

  document.getElementById('vramInput').addEventListener('change', render);
  document.getElementById('kvCacheType').addEventListener('change', render);
  document.getElementById('variantSelect').addEventListener('change', render);
  document.getElementById('winToggle').addEventListener('change', render);

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

  render();
}

init();
