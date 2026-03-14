// ─────────────────────────────────────────────────────────────────────────────
// CONSTANTS
// ─────────────────────────────────────────────────────────────────────────────
const BYTES_PER_ELEMENT = 2;       // fp16 KV cache (ollama default)
const OVERHEAD_FACTOR   = 0.92;    // reserve ~8% for CUDA context, activations, OS
const POWERS_OF_2       = [131072, 65536, 32768, 16384, 8192, 4096, 2048, 1024];

const ORIGIN_FLAGS = {
  'USA':         '🇺🇸',
  'France':      '🇫🇷',
  'EU':          '🇪🇺',
  'Canada':      '🇨🇦',
  'UK':          '🇬🇧',
  'UAE':         '🇦🇪',
  'Switzerland': '🇨🇭',
  'South Korea': '🇰🇷',
  'Singapore':   '🇸🇬',
  'International': '🌍',
};

const SPECIALTY_ICONS = {
  'general':       '💬',
  'code':          '💻',
  'multilingual':  '🌐',
  'translation':   '🔄',
  'medical':       '🏥',
  'legal':         '⚖️',
  'vision':        '👁️',
  'reasoning':     '🧠',
};

function modelLabel(m) {
  const flag = ORIGIN_FLAGS[m.origin] || '🌍';
  const icon = SPECIALTY_ICONS[m.specialty] || '💬';
  return `${m.name}  ${flag} ${icon}`;
}

// ─────────────────────────────────────────────────────────────────────────────
// CORE CALC
// ─────────────────────────────────────────────────────────────────────────────
function calcMaxContext(model, vramGB) {
  const availableBytes = (vramGB - model.weights_gb) * OVERHEAD_FACTOR * 1024 ** 3;
  if (availableBytes <= 0) return { maxCtx: 0, kvCacheGB: 0, freeGB: 0, availableBytes };

  const perToken = model.layers * model.num_key_value_heads * model.head_dim * 2 * BYTES_PER_ELEMENT;
  const rawTokens = availableBytes / perToken;
  const archLimit = model.max_context || Infinity;
  const archMaxRaw = Math.min(rawTokens, archLimit);
  const maxCtx = POWERS_OF_2.find(p => p <= archMaxRaw) || 0;
  const limitedByArch = isFinite(archLimit) && rawTokens > archLimit;
  const kvCacheGB = (maxCtx * perToken) / 1024 ** 3;
  const usedGB = model.weights_gb + kvCacheGB;
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
  if (n >= 131072) return '128k';
  if (n >= 65536)  return '64k';
  if (n >= 32768)  return '32k';
  if (n >= 16384)  return '16k';
  if (n >= 8192)   return '8k';
  if (n >= 4096)   return '4k';
  if (n >= 2048)   return '2k';
  if (n >= 1024)   return '1k';
  return '0';
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK OOM OPTIONS IN DROPDOWN
// ─────────────────────────────────────────────────────────────────────────────
function markModelOptions(vramGB) {
  const sel = document.getElementById('modelSelect');
  Array.from(sel.options).forEach((opt, i) => {
    const fits = MODELS[i].weights_gb < vramGB * OVERHEAD_FACTOR;
    opt.textContent = fits ? modelLabel(MODELS[i]) : `✗  ${modelLabel(MODELS[i])}`;
    opt.style.color = fits ? '' : '#f06464';
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// RENDER
// ─────────────────────────────────────────────────────────────────────────────
function render() {
  const modelIdx = parseInt(document.getElementById('modelSelect').value);
  const vramGB   = parseFloat(document.getElementById('vramInput').value);
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

  markModelOptions(vramGB);

  const r      = calcMaxContext(model, vramGB);
  const noFit  = model.weights_gb >= vramGB * OVERHEAD_FACTOR;

  // ── memory bar
  const modelPct   = Math.min(100, (model.weights_gb / vramGB) * 100);
  const contextPct = noFit ? 0 : Math.min(100 - modelPct, (r.kvCacheGB / vramGB) * 100);
  const freePct    = Math.max(0, 100 - modelPct - contextPct);

  document.getElementById('barTotal').textContent = fmtGB(vramGB) + ' total';

  const segModel = document.getElementById('segModel');
  segModel.className     = 'membar-seg ' + (noFit ? 'seg-overflow' : 'seg-model');
  segModel.style.width   = modelPct.toFixed(1) + '%';
  segModel.textContent   = modelPct > 12 ? fmtGB(model.weights_gb) : '';

  const segContext = document.getElementById('segContext');
  segContext.className   = 'membar-seg ' + (noFit ? 'seg-overflow' : 'seg-context');
  segContext.style.width = contextPct.toFixed(1) + '%';
  segContext.textContent = contextPct > 8 ? fmtCtx(r.maxCtx) : '';

  const segFree = document.getElementById('segFree');
  segFree.style.width = freePct.toFixed(1) + '%';
  segFree.style.flex  = freePct < 1 ? `0 0 ${freePct.toFixed(1)}%` : '1';
  segFree.textContent = freePct > 8 ? fmtGB(r.freeGB) : '';

  document.getElementById('legendModel').textContent   = `Model weights (${fmtGB(model.weights_gb)})`;
  document.getElementById('legendContext').textContent = `KV cache @ ${fmtCtx(r.maxCtx)} ctx (${fmtGB(r.kvCacheGB)})`;
  document.getElementById('legendFree').textContent    = `Free (${fmtGB(Math.max(0, r.freeGB))})`;

  // ── result headline
  const headline  = document.getElementById('resultHeadline');
  const labelOom  = document.getElementById('resultLabelOom');
  const verdictEl = document.getElementById('verdict');
  const ctxRow    = document.getElementById('resultCtxRow');
  const ollamaCmd = document.getElementById('ollamaCmd');

  // Color grade: how much of the architectural max context fits?
  let pctClass = 'pct-high';
  let pct = null;
  if (!noFit && model.max_context) {
    pct = Math.round((r.maxCtx / model.max_context) * 100);
    pctClass = pct >= 66 ? 'pct-high' : pct >= 33 ? 'pct-mid' : 'pct-low';
  }
  headline.className = 'result-headline' + (noFit ? ' error' : ` ${pctClass}`);

  // Restart animation on every render (restart trick: remove class, force reflow, re-add)
  verdictEl.classList.remove('verdict-anim');
  void verdictEl.offsetWidth;
  verdictEl.textContent = noFit ? "IT WON'T LLM!" : "IT WILL LLM!";
  verdictEl.classList.add('verdict-anim');

  // ── model info
  // Organization: link to HuggingFace org page, extracted from config_url
  const hfOrgMatch = model.config_url && model.config_url.match(/huggingface\.co\/([^/]+)/);
  const orgEl = document.getElementById('detailOrganization');
  if (hfOrgMatch) {
    const hfOrgUrl = `https://huggingface.co/${hfOrgMatch[1]}`;
    orgEl.innerHTML = `<a href="${hfOrgUrl}" target="_blank" rel="noopener noreferrer">${model.organization} ↗</a>`;
    document.getElementById('detailOrgSrc').textContent = 'models.js';
  } else {
    orgEl.textContent = model.organization;
  }
  document.getElementById('detailOrigin').textContent    = `${ORIGIN_FLAGS[model.origin] || ''} ${model.origin}`;
  document.getElementById('detailSpecialty').textContent = `${SPECIALTY_ICONS[model.specialty] || ''} ${model.specialty}`;
  document.getElementById('detailMoeRow').hidden            = !model.moe;
  document.getElementById('detailMaxCtx').textContent       = model.max_context ? model.max_context.toLocaleString() + ' tokens' : '—';

  if (noFit) {
    labelOom.textContent = `Model weights (${fmtGB(model.weights_gb)}) exceed available VRAM (${fmtGB(vramGB * OVERHEAD_FACTOR)} usable). This model will not load.`;
    labelOom.hidden = false;
    ctxRow.hidden   = true;
    ollamaCmd.hidden = true;
    document.getElementById('ollamaHint').hidden = true;
  } else {
    document.getElementById('resultValue').textContent = `${fmtCtx(r.maxCtx)} tokens`;

    const ctxPctEl = document.getElementById('ctxPct');
    ctxPctEl.textContent = pct !== null
      ? `· ${pct}% of ${fmtCtx(model.max_context)} architectural max`
      : '';

    labelOom.hidden = true;
    ctxRow.hidden   = false;

    const ctxHint = document.getElementById('ctxHint');
    if (pctClass === 'pct-mid' || pctClass === 'pct-low') {
      ctxHint.textContent = `Runs fine at ${fmtCtx(r.maxCtx)} — ${fmtTokensHuman(r.maxCtx)}. Keep num_ctx at or below this limit (the command below sets it).`;
      ctxHint.hidden = false;
    } else {
      ctxHint.hidden = true;
    }

    if (model.ollama_tag) {
      ollamaCmd.textContent = `$ ollama run ${model.ollama_tag}\n>>> /set parameter num_ctx ${r.maxCtx}`;
      document.getElementById('ollamaHint').hidden = false;
    } else {
      ollamaCmd.textContent = `not available in the ollama library\nsee source link in details below`;
      document.getElementById('ollamaHint').hidden = true;
    }
    ollamaCmd.hidden = false;
  }

  // ── details table
  document.getElementById('detailLayers').textContent    = model.layers;
  document.getElementById('detailAttnHeads').textContent = model.num_attention_heads;
  document.getElementById('detailKvHeads').textContent   = model.num_key_value_heads;
  document.getElementById('detailHeadDim').textContent   = model.head_dim;
  document.getElementById('detailHiddenSize').textContent   = model.hidden_size;
  document.getElementById('detailAttnHeadsDiv').textContent = model.num_attention_heads;
  document.getElementById('detailBpe').textContent       = BYTES_PER_ELEMENT;
  document.getElementById('detailWeights').textContent      = fmtGB(model.weights_gb);
  document.getElementById('detailQuantization').textContent = model.quantization || '—';

  // Architecture source cells → link to config.json when available
  const configLink = model.config_url && model.config_url.startsWith('https://')
    ? `<a href="${model.config_url}" target="_blank" rel="noopener noreferrer">config.json ↗</a>`
    : 'config.json';
  document.querySelectorAll('.src-config-json').forEach(el => { el.innerHTML = configLink; });
  // num_attention_heads note appended after the link
  document.getElementById('srcAttnHeads').innerHTML = configLink + ' · not used in calc';
  // head_dim: note whether it was derived or read from config
  const derivedHeadDim = model.head_dim === Math.floor(model.hidden_size / model.num_attention_heads);
  document.getElementById('srcHeadDim').innerHTML = derivedHeadDim
    ? configLink + ' · derived'
    : configLink + ' · explicit';

  // Model weights source → ollama library (official) or community upload (unverified)
  // Official: plain "name:tag" — hosted in Ollama's curated library namespace, sourced from official repos.
  // Community: "namespace/name:tag" — uploaded by a third party, provenance unverified.
  const modelName      = model.ollama_tag ? model.ollama_tag.split(':')[0] : null;
  const isCommunity    = modelName ? modelName.includes('/') : false;
  const ollamaPageUrl  = modelName
    ? (isCommunity ? `https://ollama.com/${modelName}` : `https://ollama.com/library/${modelName}`)
    : null;
  const srcEl = document.getElementById('detailSource');
  if (ollamaPageUrl) {
    const linkText = isCommunity ? 'ollama.com ↗' : 'ollama.com/library ↗';
    const warning  = isCommunity
      ? ' <span class="community-warning">⚠ community upload — not verified by Ollama</span>'
      : '';
    srcEl.innerHTML = `<a href="${ollamaPageUrl}" target="_blank" rel="noopener noreferrer">${linkText}</a>${warning}`;
  } else {
    srcEl.textContent = 'ollama registry';
  }

  // Provenance alert — shown for all models; escalated for community uploads
  const provenanceAlert = document.getElementById('provenanceAlert');
  const ollamaLink = ollamaPageUrl
    ? `<a href="${ollamaPageUrl}" target="_blank" rel="noopener noreferrer">${isCommunity ? ollamaPageUrl.replace('https://', '') : `ollama.com/library/${modelName}`} ↗</a>`
    : null;
  const hfOrgLink = hfOrgMatch
    ? `<a href="https://huggingface.co/${hfOrgMatch[1]}" target="_blank" rel="noopener noreferrer">huggingface.co/${hfOrgMatch[1]} ↗</a>`
    : null;
  const crossCheck = [ollamaLink, hfOrgLink].filter(Boolean).join(' · ');
  const orgName = model.organization || 'the originating organization';
  if (isCommunity) {
    provenanceAlert.className = 'provenance-alert provenance-alert--community';
    provenanceAlert.innerHTML = `<strong>⚠ Unverified upload.</strong> This model was uploaded by a community member — not by Ollama or ${orgName}. There is no guarantee these are the genuine ${orgName} weights.${crossCheck ? ' Check the source yourself: ' + crossCheck : ''}`;
  } else {
    provenanceAlert.className = 'provenance-alert';
    provenanceAlert.innerHTML = `<strong>Provenance:</strong> Listed in Ollama's official library. Not independently verified by will-it-llm — check it yourself:${crossCheck ? ' ' + crossCheck : ' (no source links available)'}`;
  }
  provenanceAlert.hidden = false;

  // ── formula breakdown
  const formulaBox    = document.getElementById('formulaBox');
  formulaBox.hidden   = noFit;
  if (!noFit) {
    document.getElementById('fLayers').textContent    = model.layers;
    document.getElementById('fKvHeads').textContent   = model.num_key_value_heads;
    document.getElementById('fHeadDim').textContent   = model.head_dim;
    document.getElementById('fPerToken').textContent  = `${r.perToken.toLocaleString()} bytes`;
    document.getElementById('fPerTokenKB').textContent = `(${(r.perToken / 1024).toFixed(1)} KB)`;
    document.getElementById('fAvailLabel').textContent = `available_vram = (${fmtGB(vramGB)} − ${fmtGB(model.weights_gb)}) × ${OVERHEAD_FACTOR} overhead factor`;
    document.getElementById('fAvailGB').textContent   = fmtGB((vramGB - model.weights_gb) * OVERHEAD_FACTOR);
    document.getElementById('fAvailBytes').textContent = `${Math.round((vramGB - model.weights_gb) * OVERHEAD_FACTOR * 1024 ** 3).toLocaleString()} bytes`;
    document.getElementById('fRawTokens').textContent = Math.round(r.rawTokens).toLocaleString();
    document.getElementById('fMaxCtx').textContent    = `${r.maxCtx.toLocaleString()} tokens (${fmtCtx(r.maxCtx)})`;
    const archNote = document.getElementById('archLimitNote');
    if (r.limitedByArch) {
      document.getElementById('fArchLimit').textContent = r.archLimit.toLocaleString();
      archNote.hidden = false;
    } else {
      archNote.hidden = true;
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// INIT
// ─────────────────────────────────────────────────────────────────────────────
function init() {
  document.getElementById('overheadPct').textContent = Math.round((1 - OVERHEAD_FACTOR) * 100);

  MODELS.sort((a, b) => a.name.localeCompare(b.name));

  // Remove models not available in the ollama library
  for (let i = MODELS.length - 1; i >= 0; i--) {
    if (!MODELS[i].ollama_tag) MODELS.splice(i, 1);
  }

  const sel = document.getElementById('modelSelect');
  MODELS.forEach((m, i) => {
    const opt = document.createElement('option');
    opt.value = i;
    opt.textContent = modelLabel(m);
    sel.appendChild(opt);
  });
  sel.addEventListener('change', render);
  document.getElementById('vramInput').addEventListener('change', render);
  render();
}

init();
