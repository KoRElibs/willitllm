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
  const [library] = m.ollama_tag.split(':');
  const libInfo = LIB_META[library];
  const flag = libInfo?.origin ? (FLAGS[libInfo.origin] || '🌍') : '👥';
  return `${m.ollama_tag}  ${flag}`;
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

function fmtGB(n) { return n.toFixed(2) + ' GB'; }
function fmtSpeed(lo, hi) {
  const fmt = n => n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(Math.round(n));
  return lo === hi ? `~${fmt(lo)} t/s` : `~${fmt(lo)}–${fmt(hi)} t/s`;
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

function setLabel(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value ? ': ' + value : '';
}

function updateVariantLabel(model) {
  const selectedVariant = getSelectedVariant(model);
  if (!selectedVariant) { setLabel('labelVariant', ''); return; }
  const group = getVariantGroup(selectedVariant.tag, selectedVariant.quantization);
  const label = group === '(default)'
    ? selectedVariant.quantization
    : `${group} · ${selectedVariant.quantization}`;
  setLabel('labelVariant', label);
}

function updateGpuLabel() {
  const sel = document.getElementById('vramInput');
  if (sel.value) setLabel('labelGpu', sel.value + ' GB');
}

function updateKvLabel() {
  const sel = document.getElementById('kvCacheType');
  const opt = sel.selectedOptions[0];
  if (!opt) return;
  // Strip symbols, keep just the type name (f16 / q8_0 / q4_0)
  setLabel('labelKv', opt.textContent.trim().replace(/^[■□\s]+/, ''));
}

function updateModelLabel() {
  const sel = document.getElementById('modelSelect');
  const opt = sel.selectedOptions[0];
  if (!opt) return;
  // Format: "library:size  🇺🇸" → "library size" (strip flag, replace colon)
  const base = opt.textContent.trim().split(/\s{2,}/)[0].replace(':', ' ');
  setLabel('labelModel', base);
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
    updateVariantLabel(model);
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
      if (window.innerWidth <= 600) {
        const s = quantInfo ? quantInfo.speed   : '?';
        const q = quantInfo ? quantInfo.quality : '?';
        opt.textContent = `${s}S ${q}Q  ${quant}`;
      } else {
        const speedRating   = quantInfo ? buildRatingBar(quantInfo.speed,   '▶', '▷') : '▷▷▷▷▷';
        const qualityRating = quantInfo ? buildRatingBar(quantInfo.quality, '★', '☆') : '☆☆☆☆☆';
        opt.textContent = `${speedRating} ${qualityRating}  ${gb} GB  ${quant}`;
      }
      container.appendChild(opt);
    });
  });

  updateVariantLabel(model);
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
    const fits = weightsGB < vramGB - OVERHEAD_GB;
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
  const model = MODELS[modelIdx];
  updateVariantLabel(model);

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

  document.getElementById('barTotal').textContent = fmtGB(vramGB) + ' total';

  const segModel = document.getElementById('segModel');
  segModel.className   = 'membar-seg ' + (noFit ? 'seg-overflow' : 'seg-model');
  segModel.style.width = modelPct.toFixed(1) + '%';
  segModel.textContent = modelPct > 12 ? fmtGB(weightsGB) : '';

  const segContext = document.getElementById('segContext');
  segContext.className   = 'membar-seg ' + (noFit ? 'seg-overflow' : 'seg-context');
  segContext.style.width = contextPct.toFixed(1) + '%';
  segContext.textContent = contextPct > 20
    ? `${fmtCtx(ctxResult.maxCtx)} · ${fmtGB(ctxResult.kvCacheGB)}`
    : contextPct > 8 ? fmtCtx(ctxResult.maxCtx) : '';

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

    scorecard.hidden = false;

    const quantInfoFull = variant ? QUANT_INFO[variant.quantization] : null;
    if (quantInfoFull) {
      document.getElementById('scoreQuality').dataset.tip =
        `${variant.quantization} · ${quantInfoFull.summary}`;
    }
    if (kvInfo) {
      document.getElementById('scorePrecision').dataset.tip =
        `${kvLabel} · ${kvInfo.summary}`;
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
    document.getElementById('speedGen').hidden = true;
    document.getElementById('speedPrefill').hidden = true;
  } else {
    labelOom.hidden = true;

    const pages = Math.round(ctxResult.maxCtx / 333 / 5) * 5;
    const pctPart = contextFitPct !== null && contextFitPct < 100
      ? `${contextFitPct}% of max context`
      : 'full context';
    const mmPart = libInfo.multimodal ? ' · images use tokens' : '';
    document.getElementById('scoreContext').dataset.tip =
      `${fmtCtx(ctxResult.maxCtx)} · ${pctPart} · ~${pages} pages of typical English text${mmPart}`;

    // ── speed estimates (inline in scorecard rows)
    const speedEsts  = calcSpeedEstimates(model, variant, vramGB, quantInfo);
    const genEl      = document.getElementById('speedGen');
    const prefillEl  = document.getElementById('speedPrefill');
    const genericNote = speedEsts && !speedEsts.isExact ? ' Select your exact GPU for a tighter estimate.' : '';
    if (speedEsts) {
      genEl.textContent    = fmtSpeed(speedEsts.genLo, speedEsts.genHi);
      genEl.dataset.tip    = `Generation speed — output tokens per second.${genericNote}`;
      genEl.hidden         = false;
      prefillEl.textContent = fmtSpeed(speedEsts.prefillLo, speedEsts.prefillHi);
      prefillEl.dataset.tip = `Processing speed — prompt tokens ingested per second.${genericNote}`;
      prefillEl.hidden      = false;
    } else {
      genEl.hidden     = true;
      prefillEl.hidden = true;
    }

    const flashSel     = document.getElementById('vramInput').selectedOptions[0];
    const flashSupport = flashSel ? flashSel.dataset.flash : 'yes';
    if (bytesPerElement < 2 && kvInfo) {
      const flashNote = flashSupport === 'mixed'
        ? ` AMD support varies by ollama build and driver — verify before setting OLLAMA_KV_CACHE_TYPE.`
        : flashSupport !== 'yes'
        ? ` Not supported on Turing (RTX 20xx) or older NVIDIA GPUs — this setting may have no effect or cause errors.`
        : ` Gains are modest below 8k context.`;
      document.getElementById('scorePrecision').dataset.tip =
        `${kvLabel} · ${kvInfo.summary}${flashNote}`;
    }

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

  // ── formula breakdown
  const formulaBox  = document.getElementById('formulaBox');
  formulaBox.hidden = noFit;
  if (!noFit) {
    const valueDim = model.value_length ?? model.key_length;
    document.getElementById('formulaHeader').textContent     = `Max context: ${fmtCtx(ctxResult.maxCtx)} tokens`;
    document.getElementById('formulaBlockCount').textContent = model.block_count;
    document.getElementById('formulaKvHeads').textContent    = model.head_count_kv;
    document.getElementById('formulaBpeLabel').textContent   = `${bytesPerElement}(${kvLabel})`;
    document.getElementById('formulaValueLength').textContent = valueDim;
    document.getElementById('formulaKeyLength').textContent  = model.key_length;
    document.getElementById('formulaPerToken').textContent   = `${ctxResult.perToken.toLocaleString()} bytes`;
    document.getElementById('formulaPerTokenKB').textContent = `(${(ctxResult.perToken / 1024).toFixed(1)} KB)`;
    document.getElementById('formulaAvailLabel').textContent = `available_vram = ${fmtGB(vramGB)} − ${fmtGB(OVERHEAD_GB)} overhead − ${fmtGB(weightsGB)} weights`;
    document.getElementById('formulaAvailGB').textContent    = fmtGB(vramGB - OVERHEAD_GB - weightsGB);
    document.getElementById('formulaAvailBytes').textContent = `${Math.round((vramGB - OVERHEAD_GB - weightsGB) * 1024 ** 3).toLocaleString()} bytes`;
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
    if (entries.some(g => g.default)) opt.selected = true;
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
  MODELS.forEach((m, i) => {
    const opt       = document.createElement('option');
    opt.value       = i;
    opt.textContent = formatModelOption(m);
    sel.appendChild(opt);
  });

  sel.addEventListener('change', () => {
    populateVariants(MODELS[parseInt(sel.value)]);
    updateModelLabel();
    render();
  });

  const initialIdx = parseInt(sel.value) || 0;
  if (MODELS[initialIdx]) populateVariants(MODELS[initialIdx]);

  updateGpuLabel();
  updateModelLabel();
  updateKvLabel();

  document.getElementById('vramInput').addEventListener('change', () => { updateGpuLabel(); render(); });
  document.getElementById('kvCacheType').addEventListener('change', () => { updateKvLabel(); render(); });
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
