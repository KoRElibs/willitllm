// ─────────────────────────────────────────────────────────────────────────────
// CONSTANTS
// ─────────────────────────────────────────────────────────────────────────────
const KV_CACHE_LABELS = { '2': 'f16', '1': 'q8_0', '0.5': 'q4_0' };
const KV_CACHE_INFO = {
  'f16':  { summary: 'full precision — works on every GPU, no setup needed.' },
  'q8_0': { summary: 'half the memory per token — slight precision loss, fits more context in VRAM.' },
  'q4_0': { summary: 'quarter the memory per token — more precision loss, maximum context for the VRAM.' },
};

function getBytesPerElement() {
  return parseFloat(document.getElementById('kvCacheType').value);
}

const OVERHEAD_GB = 0.5;    // fixed reservation for CUDA context, driver, ollama overhead

let activeOsTab = null;                  // 'linux' | 'windows' | null
const setupContent = { linux: '', windows: '' };
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
  return m.ollama_tag;
}

// ─────────────────────────────────────────────────────────────────────────────
// CORE CALC
// ─────────────────────────────────────────────────────────────────────────────
function calcMaxContext(model, vramGB, bytesPerElement, weightsGB) {
  const availableBytes = (vramGB - OVERHEAD_GB - weightsGB) * 1024 ** 3;
  if (availableBytes <= 0) return { maxCtx: 0, kvCacheGB: 0, freeGB: 0, availableBytes };

  const valueDim = model.value_length ?? model.key_length;
  const perToken = model.block_count * model.head_count_kv * (model.key_length + valueDim) * bytesPerElement;
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

function fmtGB(n) { return parseFloat(n.toFixed(1)) + ' GB'; }
function fmtSpeed(lo, hi) {
  const fmt = n => n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(Math.round(n));
  return lo === hi ? `~${fmt(lo)} t/s` : `~${fmt(lo)}–${fmt(hi)} t/s`;
}
function fmtSpeedHuman(lo, hi) {
  // Convert tokens/s → words/s (~0.75 words per token)
  const fmt = n => {
    const w = Math.round(n * 0.75);
    return w >= 1000 ? `~${(w / 1000).toFixed(1)}k` : `~${w}`;
  };
  const loS = fmt(lo), hiS = fmt(hi);
  return loS === hiS ? `${loS} words/s` : `${loS}–${hiS} words/s`;
}

// Returns a human-relatable speech pace label, e.g. "3× speech pace"
// Reference: average speech ~2.5 words/s (150 wpm)
function fmtSpeechPace(lo, hi) {
  const SPEECH_WPS = 2.5;
  const avgWps = ((lo + hi) / 2) * 0.75;
  const mult = Math.round(avgWps / SPEECH_WPS);
  if (mult <= 1) return 'speech pace';
  return `${mult}× speech pace`;
}

// Returns { bwLo, bwHi, tflopsLo, tflopsHi, isExact } or null if no data available.
// For a named card option (dataset.gpuIdx set), returns exact values (lo === hi).
// For a generic VRAM-tier option, returns [min, max] across all named entries at that tier.
function getGpuSpecs(vramGB) {
  const sel = document.getElementById('vramInput');
  const opt = sel.selectedOptions[0];
  const gpuIdx = opt ? parseInt(opt.dataset.gpuIdx) : NaN;

  if (!isNaN(gpuIdx) && GPUS[gpuIdx] && GPUS[gpuIdx].bandwidth) {
    const gpu = GPUS[gpuIdx];
    return { bwLo: gpu.bandwidth, bwHi: gpu.bandwidth,
             tflopsLo: gpu.tflops_fp16, tflopsHi: gpu.tflops_fp16, isExact: true };
  }

  // Generic entry — range from all named entries at this VRAM tier
  const entries = GPUS.filter(g => g.vram === vramGB && g.bandwidth);
  if (entries.length === 0) return null;
  return {
    bwLo:     Math.min(...entries.map(g => g.bandwidth)),
    bwHi:     Math.max(...entries.map(g => g.bandwidth)),
    tflopsLo: Math.min(...entries.map(g => g.tflops_fp16)),
    tflopsHi: Math.max(...entries.map(g => g.tflops_fp16)),
    isExact:  false,
  };
}

function updateKvOptions() {
  const gpuOpt     = document.getElementById('vramInput').selectedOptions[0];
  const flash      = gpuOpt ? gpuOpt.dataset.flash : 'yes';
  const flashOk    = flash === 'yes' || flash === 'mixed';
  const kvSel      = document.getElementById('kvCacheType');
  Array.from(kvSel.options).forEach(opt => {
    const needsFlash = parseFloat(opt.value) < 2;
    opt.hidden   = needsFlash && !flashOk;
    opt.disabled = needsFlash && !flashOk;
  });
  // Reset to f16 if current selection is no longer available
  if (parseFloat(kvSel.value) < 2 && !flashOk) kvSel.value = '2';
}

function populateGpuTab(vramGB, speedEsts) {
  const sel    = document.getElementById('vramInput');
  const opt    = sel.selectedOptions[0];
  const gpuIdx = opt ? parseInt(opt.dataset.gpuIdx) : NaN;

  let name, bandwidth, tflops, flash;
  if (!isNaN(gpuIdx) && GPUS[gpuIdx]) {
    const gpu = GPUS[gpuIdx];
    name      = gpu.names.join(' / ');
    bandwidth = gpu.bandwidth + ' GB/s';
    tflops    = gpu.tflops_fp16 + ' TFLOPS';
    flash     = gpu.flash;
  } else {
    const entries = GPUS.filter(g => g.vram === vramGB && g.bandwidth);
    name = `${vramGB} GB (generic)`;
    if (entries.length) {
      const bwLo = Math.min(...entries.map(g => g.bandwidth));
      const bwHi = Math.max(...entries.map(g => g.bandwidth));
      const tLo  = Math.min(...entries.map(g => g.tflops_fp16));
      const tHi  = Math.max(...entries.map(g => g.tflops_fp16));
      bandwidth  = bwLo === bwHi ? `${bwLo} GB/s` : `${bwLo}–${bwHi} GB/s`;
      tflops     = tLo  === tHi  ? `${tLo} TFLOPS` : `${tLo}–${tHi} TFLOPS`;
      const flashVals = [...new Set(entries.map(g => g.flash))];
      flash = flashVals.length === 1 ? flashVals[0] : 'varies';
    } else {
      bandwidth = '—'; tflops = '—'; flash = '—';
    }
  }

  document.getElementById('gpuName').textContent         = name;
  document.getElementById('gpuVramDisplay').textContent  = vramGB + ' GB';
  document.getElementById('gpuBandwidth').textContent    = bandwidth;
  document.getElementById('gpuTflops').textContent       = tflops;
  document.getElementById('gpuFlash').textContent        = flash;
  document.getElementById('gpuGenSpeedDetail').textContent     = speedEsts ? fmtSpeed(speedEsts.genLo,     speedEsts.genHi)     : '—';
  document.getElementById('gpuPrefillSpeedDetail').textContent = speedEsts ? fmtSpeed(speedEsts.prefillLo, speedEsts.prefillHi) : '—';
}

function calcSpeedEstimates(model, variant, vramGB, quantInfo) {
  if (!quantInfo || !quantInfo.gen_eff || !quantInfo.prefill_eff) return null;
  const gpuSpecs = getGpuSpecs(vramGB);
  if (!gpuSpecs) return null;

  // MoE: only active experts load per token — scale weights proportionally
  const activeFraction = (model.params_b_active && model.params_b)
    ? model.params_b_active / model.params_b : 1.0;
  const activeWeightsGB = variant.weights_gb * activeFraction;

  const [genEffLo,     genEffHi]     = quantInfo.gen_eff;
  const [prefillEffLo, prefillEffHi] = quantInfo.prefill_eff;

  // Generation speed: bandwidth-bound. tokens/sec = bandwidth × efficiency / active_weights
  const genLo = Math.round((gpuSpecs.bwLo * genEffLo) / activeWeightsGB);
  const genHi = Math.round((gpuSpecs.bwHi * genEffHi) / activeWeightsGB);

  // Processing speed: compute-bound. tokens/sec = tflops × 1e12 × efficiency / (2 × params × 1e9)
  const paramsActive = (model.params_b_active || model.params_b) * 1e9;
  const prefillLo = Math.round((gpuSpecs.tflopsLo * 1e12 * prefillEffLo) / (2 * paramsActive));
  const prefillHi = Math.round((gpuSpecs.tflopsHi * 1e12 * prefillEffHi) / (2 * paramsActive));

  return { genLo, genHi, prefillLo, prefillHi, isExact: gpuSpecs.isExact };
}
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

function updateSelectionSummary(model) {
  const el = document.getElementById('selectionSummary');
  if (!el) return;

  const modelSel = document.getElementById('modelSelect');
  const modelOpt = modelSel.selectedOptions[0];
  if (!modelOpt || modelOpt.value === '') { el.textContent = 'VRAM allocation'; return; }

  const library = model ? model.ollama_tag.split(':')[0] : modelOpt.textContent.trim().split(/[:\s]/)[0];
  const variant = model ? getSelectedVariant(model) : null;
  const fullTag = variant ? `${library}:${variant.tag}` : library;

  const kvOpt = document.getElementById('kvCacheType').selectedOptions[0];
  const kvLabel = kvOpt ? kvOpt.textContent.trim().replace(/^[■□\s]+/, '') : '';

  const gpuOpt = document.getElementById('vramInput').selectedOptions[0];
  const gpuName = gpuOpt ? gpuOpt.textContent.trim() : '';

  const modelParts = [fullTag, kvLabel ? `KV ${kvLabel}` : ''].filter(Boolean);
  el.textContent = gpuName
    ? `${gpuName}: ${modelParts.join(' · ')}`
    : modelParts.join(' · ');
}

function getVariantGroup(tag, quantization) {
  if (/^\d+(\.\d+)?b$/i.test(tag)) return '(default)';
  const rest = tag.replace(/^\d+(\.\d+)?b-/i, '');
  const qSuffix = '-' + quantization.toLowerCase();
  return rest.toLowerCase().endsWith(qSuffix)
    ? rest.slice(0, -qSuffix.length) || '(default)'
    : rest;
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

  // Group variants by custominfo (everything between size prefix and quantization suffix)
  const groups = new Map();
  model.variants.forEach((variant, i) => {
    const group = getVariantGroup(variant.tag, variant.quantization);
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
      const opt        = document.createElement('option');
      opt.value        = i;
      const quantInfo  = QUANT_INFO[variant.quantization];
      const quant      = variant.quantization || '?';
      const gb         = variant.weights_gb.toFixed(1);
      const defMark    = i === 0 ? ' ← default' : '';
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

// ─────────────────────────────────────────────────────────────────────────────
// RENDER
// ─────────────────────────────────────────────────────────────────────────────
function render() {
  const modelIdx        = parseInt(document.getElementById('modelSelect').value);
  const vramGB          = parseFloat(document.getElementById('vramInput').value);
  const bytesPerElement = getBytesPerElement();
  const kvLabel         = KV_CACHE_LABELS[String(bytesPerElement)] || 'f16';
  const kvInfo          = KV_CACHE_INFO[kvLabel];
  const model = MODELS[modelIdx];
  let speedEsts = null;
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

  const variant      = getSelectedVariant(model);
  const weightsGB    = variant ? variant.weights_gb : 0;
  const quantization = variant ? variant.quantization : '—';

  const quantInfo = variant ? QUANT_INFO[variant.quantization] : null;

  markModelOptions(vramGB, bytesPerElement);

  const ctxResult = calcMaxContext(model, vramGB, bytesPerElement, weightsGB);
  const noFit     = weightsGB >= vramGB - OVERHEAD_GB;

  // ── memory bar
  const modelPct   = Math.min(100, (weightsGB / vramGB) * 100);
  const contextPct = noFit ? 0 : Math.min(100 - modelPct, (ctxResult.kvCacheGB / vramGB) * 100);
  const freePct    = Math.max(0, 100 - modelPct - contextPct);

  document.getElementById('barTotal').textContent = Math.round(vramGB) + ' GB';

  const segModel = document.getElementById('segModel');
  segModel.className   = 'membar-seg ' + (noFit ? 'seg-overflow' : 'seg-model');
  segModel.style.width = modelPct.toFixed(1) + '%';
  segModel.textContent = modelPct > 12 ? fmtGB(weightsGB) : '';

  const segContext = document.getElementById('segContext');
  segContext.className   = 'membar-seg ' + (noFit ? 'seg-overflow' : 'seg-context');
  segContext.style.width = contextPct.toFixed(1) + '%';
  segContext.textContent = contextPct > 8 ? fmtGB(ctxResult.kvCacheGB) : '';

  const segFree = document.getElementById('segFree');
  segFree.style.width  = freePct.toFixed(1) + '%';
  segFree.style.flex   = freePct < 1 ? `0 0 ${freePct.toFixed(1)}%` : '1';
  segFree.textContent  = freePct > 8 ? fmtGB(ctxResult.freeGB) : '';

  document.getElementById('legendModel').textContent   = `Model weights (${fmtGB(weightsGB)})`;
  document.getElementById('legendContext').textContent = `KV cache (${fmtGB(ctxResult.kvCacheGB)}) · ${fmtCtx(ctxResult.maxCtx)} ctx`;
  document.getElementById('legendFree').textContent    = `Free (${fmtGB(Math.max(0, ctxResult.freeGB))})`;

  // ── result headline
  const headline  = document.getElementById('resultHeadline');
  const labelOom  = document.getElementById('resultLabelOom');
  const verdictEl = document.getElementById('verdict');
  const scorecard = document.getElementById('scorecard');
  const ollamaCmd = document.getElementById('ollamaCmd');

  const libInfo = getLibMeta(model);

  // ── Scorecard
  function bar10(n10) { return '■'.repeat(n10) + '□'.repeat(10 - n10); }
  function colorForScore(n5) {
    return n5 >= 4 ? 'var(--green)' : n5 === 3 ? 'var(--amber)' : n5 === 2 ? 'var(--orange)' : 'var(--red)';
  }

  const ctxTradeoff = 'Memory clarity vs. attention span: crisper recall (f16) costs more VRAM per token, leaving less room for a long conversation.';
  const scoreSpeed     = quantInfo ? Math.max(1, Math.round((quantInfo.speed   / 10) * 5)) : 0;
  const scoreQuality   = quantInfo ? Math.max(1, Math.round((quantInfo.quality / 10) * 5)) : 0;
  const contextFitPct  = (!noFit && model.context_length) ? Math.round((ctxResult.maxCtx / model.context_length) * 100) : null;
  const scoreContext   = contextFitPct === null ? 0 : contextFitPct >= 90 ? 5 : contextFitPct >= 66 ? 4 : contextFitPct >= 40 ? 3 : contextFitPct >= 15 ? 2 : 1;
  const scoreContext10 = contextFitPct === null ? 0 : Math.min(10, Math.max(1, Math.ceil(contextFitPct / 10)));
  const scorePrecision = bytesPerElement === 2 ? 5 : bytesPerElement === 1 ? 3 : 2;
  const scoreAvg       = (scoreSpeed + scoreQuality + scoreContext + scorePrecision) / 4;
  const scoreClass     = noFit ? 'error' : scoreAvg >= 4 ? 'score-high' : scoreAvg >= 3 ? 'score-mid' : scoreAvg >= 2 ? 'score-low' : 'score-poor';

  headline.className = `result-headline ${scoreClass}`;

  if (!noFit) {
    // Speed & Quality: use raw 1-10 from quantInfo for full precision
    // Precision & Context: scale 1-5 → 1-10
    const bars = [
      ['scoreSpeed',     quantInfo ? quantInfo.speed   : 0, scoreSpeed],
      ['scoreQuality',   quantInfo ? quantInfo.quality : 0, scoreQuality],
      ['scorePrecision', scorePrecision * 2,                scorePrecision],
      ['scoreContext',   scoreContext10,                    scoreContext],
    ];
    bars.forEach(([id, n10, n5]) => {
      const el = document.getElementById(id);
      el.textContent = bar10(Math.round(n10));
      el.style.color = colorForScore(n5);
    });

    scorecard.hidden = false;

    const quantInfoFull = variant ? QUANT_INFO[variant.quantization] : null;
    if (quantInfoFull) {
      const tradeoff = 'Thinking speed and sharpness trade off — a lighter quantization means faster responses but a duller mind. You cannot have both at maximum. (Technical: quantization level)';
      document.getElementById('scoreQuality').dataset.tip =
        `${variant.quantization} · ${quantInfoFull.summary} · ${tradeoff}`;
      document.getElementById('scoreSpeed').dataset.tip =
        `${variant.quantization} · ${quantInfoFull.summary} · ${tradeoff}`;
    }
    if (kvInfo) {
      document.getElementById('scorePrecision').dataset.tip =
        `${kvLabel} · ${ctxTradeoff}`;
    }
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
    labelOom.textContent = `Model weights (${fmtGB(weightsGB)}) exceed available VRAM (${fmtGB(vramGB - OVERHEAD_GB)} usable). This model will not load.`;
    labelOom.hidden = false;
    ollamaCmd.hidden = true;
    document.getElementById('osTabs').hidden = true;
    document.getElementById('ollamaSetup').hidden = true;
    document.getElementById('resultAside').hidden = true;
  } else {
    labelOom.hidden = true;

    const pages = Math.round(ctxResult.maxCtx / 333 / 5) * 5;
    const pctPart = contextFitPct !== null && contextFitPct < 100
      ? `${contextFitPct}% of max context`
      : 'full context';
    const mmPart = libInfo.multimodal ? ' · images use tokens' : '';
    document.getElementById('scoreContext').dataset.tip =
      `${fmtCtx(ctxResult.maxCtx)} · ${pctPart}${mmPart} · ${ctxTradeoff}`;
    // ── speed estimates
    speedEsts = calcSpeedEstimates(model, variant, vramGB, quantInfo);

    // ── result aside (right panel)
    const asideEl = document.getElementById('resultAside');
    const genEl    = document.getElementById('asideGenSpeed');
    const prefEl   = document.getElementById('asidePrefillSpeed');
    if (speedEsts) {
      genEl.textContent    = fmtSpeedHuman(speedEsts.genLo, speedEsts.genHi);
      genEl.dataset.tip    = `Writing its response · ${fmtSpeechPace(speedEsts.genLo, speedEsts.genHi)} · ${fmtSpeed(speedEsts.genLo, speedEsts.genHi)} (generation — output tokens/s, bandwidth-bound)`;
      prefEl.textContent   = fmtSpeedHuman(speedEsts.prefillLo, speedEsts.prefillHi);
      prefEl.dataset.tip   = `Reading your prompt · ${fmtSpeechPace(speedEsts.prefillLo, speedEsts.prefillHi)} · ${fmtSpeed(speedEsts.prefillLo, speedEsts.prefillHi)} (prefill — input tokens/s, compute-bound)`;
      document.getElementById('asideGenStat').dataset.tip    = '';
      document.getElementById('asidePrefillStat').dataset.tip = '';
    } else {
      genEl.textContent  = '—';
      prefEl.textContent = '—';
    }
    const humanCtx = fmtTokensHuman(ctxResult.maxCtx);
    const [pagePart] = humanCtx.split(' · ').reverse();
    const ctxPagesEl = document.getElementById('asideCtxPages');
    ctxPagesEl.textContent  = pagePart;
    ctxPagesEl.dataset.tip  = `${fmtCtx(ctxResult.maxCtx)} tokens · ${humanCtx} (attention span — how much text fits in VRAM at once)`;
    document.getElementById('asideCtxLabel').textContent = 'in one go';
    // Show caveat when using >50% of the model's trained context — degradation becomes
    // practically significant at that point (lost-in-the-middle effect).
    const ctxCaveat = document.getElementById('ctxCaveat');
    if (ctxCaveat) ctxCaveat.hidden = !contextFitPct || contextFitPct <= 50;
    asideEl.hidden = false;


    const muted      = s => `<span class="cmd-muted">${s}</span>`;
    const variantIdx = getSelectedVariantIdx(model);
    const runTag     = variantOllamaTag(model, variantIdx);

    ollamaCmd.textContent = `ollama run ${runTag}\n>>> /set parameter num_ctx ${ctxResult.maxCtx}`;
    ollamaCmd.hidden = false;

    const osTabs = document.getElementById('osTabs');
    if (bytesPerElement < 2) {
      setupContent.linux = [
        muted(`# Stop ollama if running, then restart with the KV cache setting:`),
        `OLLAMA_KV_CACHE_TYPE=${kvLabel} ollama serve`,
        muted(`# In a new terminal, run the command above`),
      ].join('\n');
      setupContent.windows = [
        muted(`# 1. Open: System Properties → Environment Variables → New user variable`),
        muted(`#    Name:  OLLAMA_KV_CACHE_TYPE`),
        muted(`#    Value: ${kvLabel}`),
        muted(`# 2. Right-click Ollama in system tray → Quit, then relaunch Ollama`),
        muted(`# 3. Run the command above`),
      ].join('\n');
      if (activeOsTab) {
        document.getElementById('ollamaSetup').innerHTML = setupContent[activeOsTab];
      }
      osTabs.hidden = false;
    } else {
      osTabs.hidden = true;
      document.getElementById('ollamaSetup').hidden = true;
      activeOsTab = null;
      document.getElementById('tabLinux').textContent  = '▶ Linux / Mac';
      document.getElementById('tabWindows').textContent = '▶ Windows';
      document.getElementById('tabLinux').classList.remove('active');
      document.getElementById('tabWindows').classList.remove('active');
    }
  }

  // ── details table
  const detailValues = {
    detailLayers:       model.block_count,
    detailKvHeads:      model.head_count_kv,
    detailHeadDim:      model.key_length,
    detailValueLength:  model.value_length ?? model.key_length,
    detailBpe:          bytesPerElement,
    detailBpeLabel:     kvLabel,
    detailWeights:      fmtGB(weightsGB),
    detailQuantization: quantization,
  };
  Object.entries(detailValues).forEach(([id, val]) => {
    document.getElementById(id).textContent = val;
  });

  // ── Quantization section (model tab)
  document.getElementById('quantType').textContent    = quantization;
  document.getElementById('quantSummary').textContent = quantInfo?.summary || '—';

  // Single variant-specific link at the bottom of the details table
  const [library] = model.ollama_tag.split(':');
  const variantTag    = variant ? variant.tag : model.ollama_tag.split(':')[1];
  const ollamaVariantUrl = library.includes('/')
    ? `https://ollama.com/${library}:${variantTag}`
    : `https://ollama.com/library/${library}:${variantTag}`;
  const linkEl = document.getElementById('detailOllamaLink');
  linkEl.href        = ollamaVariantUrl;
  linkEl.textContent = `ollama.com/library/${library}:${variantTag} ↗`;

  document.getElementById('provenanceAlert').hidden = true;

  // ── GPU tab
  populateGpuTab(vramGB, speedEsts);

  // ── formula breakdown
  const formulaBox  = document.getElementById('formulaBox');
  formulaBox.hidden = noFit;
  document.getElementById('formulaNoFit').hidden = !noFit;
  if (!noFit) {
    const valueDim = model.value_length ?? model.key_length;
    document.getElementById('formulaHeader').textContent      = `Context window: ${fmtCtx(ctxResult.maxCtx)}`;
    document.getElementById('formulaBlockCount').textContent  = model.block_count;
    document.getElementById('formulaKvHeads').textContent     = model.head_count_kv;
    document.getElementById('formulaBpeLabel').textContent    = `${bytesPerElement}(${kvLabel})`;
    document.getElementById('formulaValueLength').textContent = valueDim;
    document.getElementById('formulaKeyLength').textContent   = model.key_length;
    document.getElementById('formulaPerToken').textContent    = `${ctxResult.perToken.toLocaleString()} bytes`;
    document.getElementById('formulaPerTokenKB').textContent  = `(${(ctxResult.perToken / 1024).toFixed(1)} KB)`;
    document.getElementById('formulaAvailLabel').textContent  = `available_vram = ${fmtGB(vramGB)} − ${fmtGB(OVERHEAD_GB)} overhead − ${fmtGB(weightsGB)} weights`;
    document.getElementById('formulaAvailGB').textContent     = fmtGB(vramGB - OVERHEAD_GB - weightsGB);
    document.getElementById('formulaAvailBytes').textContent  = `${Math.round((vramGB - OVERHEAD_GB - weightsGB) * 1024 ** 3).toLocaleString()} bytes`;
    document.getElementById('formulaRawTokens').textContent   = Math.round(ctxResult.rawTokens).toLocaleString();
    document.getElementById('formulaMaxCtx').textContent      = `${ctxResult.maxCtx.toLocaleString()} tokens (${fmtCtx(ctxResult.maxCtx)})`;
    const archCapNote = document.getElementById('formulaArchCapNote');
    if (ctxResult.limitedByArch) {
      document.getElementById('formulaArchLimit').textContent = ctxResult.archLimit.toLocaleString();
      archCapNote.hidden = false;
    } else {
      archCapNote.hidden = true;
    }
    const fCtxCaveat = document.getElementById('formulaCtxCaveatNote');
    if (fCtxCaveat) fCtxCaveat.hidden = !contextFitPct || contextFitPct <= 50;
  }

  // ── speed formula
  const speedSection = document.getElementById('formulaSpeedSection');
  const speedBody    = document.getElementById('formulaSpeedBody');
  if (speedEsts) {
    const gpuSpecs   = getGpuSpecs(vramGB);
    const fmtRange   = (lo, hi) => lo === hi ? String(lo) : `${lo}–${hi}`;
    const activeFrac = (model.params_b_active && model.params_b) ? model.params_b_active / model.params_b : 1.0;
    const activeWeightsGB = (variant ? variant.weights_gb : 0) * activeFrac;
    const [genLo, genHi]       = quantInfo ? quantInfo.gen_eff     : [0, 0];
    const [preLo, preHi]       = quantInfo ? quantInfo.prefill_eff : [0, 0];
    const paramsB = model.params_b_active || model.params_b;

    document.getElementById('formulaGenBw').textContent         = gpuSpecs ? fmtRange(gpuSpecs.bwLo, gpuSpecs.bwHi) : '?';
    document.getElementById('formulaGenEff').textContent        = `[${genLo}–${genHi}]`;
    document.getElementById('formulaGenWeights').textContent    = activeWeightsGB.toFixed(2);
    document.getElementById('formulaGenResult').textContent     = fmtSpeed(speedEsts.genLo, speedEsts.genHi);
    document.getElementById('formulaPrefillTflops').textContent = gpuSpecs ? fmtRange(gpuSpecs.tflopsLo, gpuSpecs.tflopsHi) : '?';
    document.getElementById('formulaPrefillEff').textContent    = `[${preLo}–${preHi}]`;
    document.getElementById('formulaPrefillParams').textContent = paramsB;
    document.getElementById('formulaPrefillResult').textContent = fmtSpeed(speedEsts.prefillLo, speedEsts.prefillHi);
    speedSection.hidden = false;
    speedBody.hidden    = false;
  } else {
    speedSection.hidden = true;
    speedBody.hidden    = true;
  }

  updateNudgeButtons(ctxResult ? ctxResult.limitedByArch : false, vramGB);
}

// Returns variants in the same group as variantIdx, sorted by quality ascending (fastest first).
function groupVariantsSorted(model, variantIdx) {
  const v = model.variants[variantIdx];
  const group = getVariantGroup(v.tag, v.quantization);
  const all = model.variants.map((v, i) => ({ v, i, qi: QUANT_INFO[v.quantization] }));
  const inGroup = all.filter(({ v }) => getVariantGroup(v.tag, v.quantization) === group);
  // Fall back to all variants when the current group has only one member (e.g. the bare default tag)
  const candidates = inGroup.length > 1 ? inGroup : all;
  return candidates.sort((a, b) => {
    const qa = a.qi ? a.qi.quality : 5, qb = b.qi ? b.qi.quality : 5;
    return qa !== qb ? qa - qb : a.v.weights_gb - b.v.weights_gb;
  });
}

function nudgeVariant(direction) {
  const modelIdx = parseInt(document.getElementById('modelSelect').value);
  const model = MODELS[modelIdx];
  if (!model || !model.variants) return;
  const idx = getSelectedVariantIdx(model);
  const sorted = groupVariantsSorted(model, idx);
  const pos = sorted.findIndex(({ i }) => i === idx);
  const target = direction === 'quality' ? pos + 1 : pos - 1;
  if (target < 0 || target >= sorted.length) return;
  document.getElementById('variantSelect').value = sorted[target].i;
  render();
}

function nudgeKv(direction) {
  const sel = document.getElementById('kvCacheType');
  const visible = Array.from(sel.options).filter(o => !o.hidden && !o.disabled);
  const curIdx = visible.findIndex(o => o.value === sel.value);
  const target = direction === 'quality' ? curIdx - 1 : curIdx + 1;
  if (target < 0 || target >= visible.length) return;
  sel.value = visible[target].value;
  render();
}

function updateNudgeButtons(ctxAtMax, vramGB) {
  const modelIdx = parseInt(document.getElementById('modelSelect').value);
  const model = MODELS[modelIdx];
  const kvSel = document.getElementById('kvCacheType');
  const show = (id, visible) => { const el = document.getElementById(id); if (el) el.hidden = !visible; };

  if (!model || !model.variants) {
    ['nudge-speed','nudge-quality','nudge-ctx-quality','nudge-ctx-size'].forEach(id => show(id, false));
    return;
  }
  const idx = getSelectedVariantIdx(model);
  const sorted = groupVariantsSorted(model, idx);
  const pos = sorted.findIndex(({ i }) => i === idx);

  const fits = v => !vramGB || v.weights_gb < vramGB - OVERHEAD_GB;
  show('nudge-speed',   pos > 0                    && fits(sorted[pos - 1].v));
  show('nudge-quality', pos < sorted.length - 1    && fits(sorted[pos + 1].v));
  const kvVisible = Array.from(kvSel.options).filter(o => !o.hidden && !o.disabled);
  const kvCurIdx  = kvVisible.findIndex(o => o.value === kvSel.value);
  show('nudge-ctx-quality', kvCurIdx > 0);
  show('nudge-ctx-size',    !ctxAtMax && kvCurIdx < kvVisible.length - 1);
}

function setOsTab(os) {
  const setupEl  = document.getElementById('ollamaSetup');
  const tabLinux  = document.getElementById('tabLinux');
  const tabWindows = document.getElementById('tabWindows');

  if (activeOsTab === os) {
    // collapse
    activeOsTab = null;
    setupEl.hidden = true;
    tabLinux.textContent  = '▶ Linux / Mac';
    tabWindows.textContent = '▶ Windows';
    tabLinux.classList.remove('active');
    tabWindows.classList.remove('active');
  } else {
    activeOsTab = os;
    setupEl.innerHTML = setupContent[os];
    setupEl.hidden = false;
    tabLinux.textContent  = os === 'linux'   ? '▼ Linux / Mac' : '▶ Linux / Mac';
    tabWindows.textContent = os === 'windows' ? '▼ Windows'     : '▶ Windows';
    tabLinux.classList.toggle('active',   os === 'linux');
    tabWindows.classList.toggle('active', os === 'windows');
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// INIT
// ─────────────────────────────────────────────────────────────────────────────
function init() {
  document.getElementById('overheadReserved').textContent = fmtGB(OVERHEAD_GB);

  // Build GPU dropdown from gpus.js data
  const vramSel = document.getElementById('vramInput');

  // Placeholder
  const vramPlaceholder = document.createElement('option');
  vramPlaceholder.value    = '';
  vramPlaceholder.disabled = true;
  vramPlaceholder.selected = true;
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

  // Separator
  const sep = document.createElement('option');
  sep.disabled    = true;
  sep.textContent = '— pick your card —';
  vramSel.appendChild(sep);

  // Individual card entries sorted alphabetically
  const cards = GPUS.flatMap((gpu, gpuIdx) => gpu.names.map(name => ({ name, vram: gpu.vram, flash: gpu.flash, gpuIdx })));
  cards.sort((a, b) => a.name.localeCompare(b.name));
  cards.forEach(({ name, vram, flash, gpuIdx }) => {
    const opt            = document.createElement('option');
    opt.value            = vram;
    opt.dataset.flash    = flash;
    opt.dataset.gpuIdx   = gpuIdx;
    opt.textContent      = name;
    vramSel.appendChild(opt);
  });

  MODELS.sort((a, b) => a.ollama_tag.localeCompare(b.ollama_tag));

  const sel = document.getElementById('modelSelect');

  // Placeholder
  const modelPlaceholder = document.createElement('option');
  modelPlaceholder.value    = '';
  modelPlaceholder.disabled = true;
  modelPlaceholder.selected = true;
  modelPlaceholder.textContent = 'Select a model...';
  sel.appendChild(modelPlaceholder);

  // Group by organization, with flag in the optgroup label
  const groups = new Map(); // org label → [{ m, i }]
  MODELS.forEach((m, i) => {
    const [library] = m.ollama_tag.split(':');
    const info = LIB_META[library];
    const org  = info?.organization || 'Other';
    const flag = info?.origin ? (FLAGS[info.origin] || '🌍') : '👥';
    const key  = `${org}||${flag}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push({ m, i });
  });

  // Sort groups alphabetically by org name
  const sortedGroups = [...groups.entries()].sort(([a], [b]) => a.localeCompare(b));
  sortedGroups.forEach(([key, items]) => {
    const [org, flag] = key.split('||');
    const grp = document.createElement('optgroup');
    grp.label = `${org}  ${flag}`;
    items.forEach(({ m, i }) => {
      const opt       = document.createElement('option');
      opt.value       = i;
      opt.textContent = m.ollama_tag;
      grp.appendChild(opt);
    });
    sel.appendChild(grp);
  });

  sel.addEventListener('change', () => {
    populateVariants(MODELS[parseInt(sel.value)]);
    render();
  });

  document.getElementById('vramInput').addEventListener('change', () => { updateKvOptions(); render(); });
  document.getElementById('kvCacheType').addEventListener('change', render);
  document.getElementById('variantSelect').addEventListener('change', render);

  // Re-build variant options on resize (mobile vs desktop format differs)
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

  document.getElementById('nudge-speed').addEventListener('click',       () => nudgeVariant('speed'));
  document.getElementById('nudge-quality').addEventListener('click',     () => nudgeVariant('quality'));
  document.getElementById('nudge-ctx-quality').addEventListener('click', () => nudgeKv('quality'));
  document.getElementById('nudge-ctx-size').addEventListener('click',    () => nudgeKv('size'));

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

  render();
}

init();
