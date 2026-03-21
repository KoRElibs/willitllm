// ─────────────────────────────────────────────────────────────────────────────
// APP.CALC — pure calculations and formatting helpers
//
// Minimal DOM access (getGpuSpecs reads the GPU select element).
// Provides constants and functions consumed by all other app files.
//
// Globals:     GPUS, QUANT_INFO                             (data files)
// ─────────────────────────────────────────────────────────────────────────────

const OVERHEAD_GB  = 0.5;    // fixed reservation: CUDA context, driver, ollama
const POWERS_OF_2  = [131072, 65536, 32768, 16384, 8192, 4096, 2048, 1024];
const CTX_LABELS   = [
  [131072,'128k'],[65536,'64k'],[32768,'32k'],[16384,'16k'],
  [8192,'8k'],[4096,'4k'],[2048,'2k'],[1024,'1k'],
];

// ── Formatters ────────────────────────────────────────────────────────────────

function fmtGB(n) { return parseFloat(n.toFixed(1)) + ' GB'; }

function fmtSpeed(lo, hi) {
  const fmt = n => n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(Math.round(n));
  return lo === hi ? `~${fmt(lo)} t/s` : `~${fmt(lo)}–${fmt(hi)} t/s`;
}

// Converts tokens/s → words/s (~0.75 words per token)
function fmtSpeedHuman(lo, hi) {
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

// ~0.75 words per token; ~250 words per page (333 tokens/page)
function fmtTokensHuman(n) {
  const words = Math.round(n * 0.75 / 500) * 500;
  const pages = Math.round(n / 333 / 5) * 5;
  const w = words >= 1000 ? `${(words / 1000).toFixed(0)}k` : words;
  return `≈${w} words · ~${pages} pages`;
}

function fmtCtx(n) {
  return (CTX_LABELS.find(([t]) => n >= t) || [0, '0'])[1];
}

function bar10(n10) { return '■'.repeat(n10) + '□'.repeat(10 - n10); }

function colorForScore(n5) {
  return n5 >= 4 ? 'var(--green)' : n5 === 3 ? 'var(--amber)' : n5 === 2 ? 'var(--orange)' : 'var(--red)';
}

// ── GPU specs ─────────────────────────────────────────────────────────────────

// Returns { bwLo, bwHi, tflopsLo, tflopsHi, isExact } or null if no data.
// For a named card (dataset.gpuIdx set), returns exact values (lo === hi).
// For a generic VRAM-tier entry, returns [min, max] across all named cards at that tier.
function getGpuSpecs(vramGB) {
  const sel    = document.getElementById('vramInput');
  const opt    = sel.selectedOptions[0];
  const gpuIdx = opt ? parseInt(opt.dataset.gpuIdx) : NaN;

  if (!isNaN(gpuIdx) && GPUS[gpuIdx] && GPUS[gpuIdx].bandwidth) {
    const gpu = GPUS[gpuIdx];
    return { bwLo: gpu.bandwidth, bwHi: gpu.bandwidth,
             tflopsLo: gpu.tflops_fp16, tflopsHi: gpu.tflops_fp16, isExact: true };
  }

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

// ── Calculations ──────────────────────────────────────────────────────────────

function calcMaxContext(model, vramGB, bytesPerElement, weightsGB) {
  const availableBytes = (vramGB - OVERHEAD_GB - weightsGB) * 1024 ** 3;
  if (availableBytes <= 0) return { maxCtx: 0, kvCacheGB: 0, freeGB: 0, availableBytes };

  const valueDim  = model.value_length ?? model.key_length;
  const perToken  = model.block_count * model.head_count_kv * (model.key_length + valueDim) * bytesPerElement;
  const rawTokens = availableBytes / perToken;
  const archLimit = model.context_length || Infinity;
  const archMaxRaw    = Math.min(rawTokens, archLimit);
  const maxCtx        = POWERS_OF_2.find(p => p <= archMaxRaw) || 0;
  const limitedByArch = isFinite(archLimit) && rawTokens > archLimit;
  const kvCacheGB = (maxCtx * perToken) / 1024 ** 3;
  const usedGB    = weightsGB + kvCacheGB;
  const freeGB    = vramGB - usedGB;

  return { maxCtx, kvCacheGB, freeGB, perToken, rawTokens, availableBytes, usedGB, archLimit, limitedByArch };
}

function calcSpeedEstimates(model, variant, vramGB, quantInfo) {
  if (!quantInfo || !quantInfo.gen_eff || !quantInfo.prefill_eff) return null;
  const gpuSpecs = getGpuSpecs(vramGB);
  if (!gpuSpecs) return null;

  // MoE: only active experts load per token — scale weights proportionally
  const activeFraction  = (model.params_b_active && model.params_b)
    ? model.params_b_active / model.params_b : 1.0;
  const activeWeightsGB = variant.weights_gb * activeFraction;

  const [genEffLo,     genEffHi]     = quantInfo.gen_eff;
  const [prefillEffLo, prefillEffHi] = quantInfo.prefill_eff;

  // Generation: bandwidth-bound. tokens/s = bandwidth × efficiency / active_weights
  const genLo = Math.round((gpuSpecs.bwLo * genEffLo) / activeWeightsGB);
  const genHi = Math.round((gpuSpecs.bwHi * genEffHi) / activeWeightsGB);

  // Prefill: compute-bound. tokens/s = tflops × 1e12 × efficiency / (2 × params × 1e9)
  const paramsActive = (model.params_b_active || model.params_b) * 1e9;
  const prefillLo = Math.round((gpuSpecs.tflopsLo * 1e12 * prefillEffLo) / (2 * paramsActive));
  const prefillHi = Math.round((gpuSpecs.tflopsHi * 1e12 * prefillEffHi) / (2 * paramsActive));

  return { genLo, genHi, prefillLo, prefillHi, isExact: gpuSpecs.isExact };
}

// Returns all scores + scoreClass for the current selection.
function computeScores(quantInfo, bytesPerElement, ctxResult, noFit, model) {
  const contextFitPct = (!noFit && model.context_length)
    ? Math.round((ctxResult.maxCtx / model.context_length) * 100) : null;
  const scoreSpeed     = quantInfo ? Math.max(1, Math.round((quantInfo.speed   / 10) * 5)) : 0;
  const scoreQuality   = quantInfo ? Math.max(1, Math.round((quantInfo.quality / 10) * 5)) : 0;
  const scoreContext   = contextFitPct === null ? 0
    : contextFitPct >= 90 ? 5 : contextFitPct >= 66 ? 4 : contextFitPct >= 40 ? 3 : contextFitPct >= 15 ? 2 : 1;
  const scoreContext10 = contextFitPct === null ? 0 : Math.min(10, Math.max(1, Math.ceil(contextFitPct / 10)));
  const scorePrecision = bytesPerElement === 2 ? 5 : bytesPerElement === 1 ? 3 : 2;
  const scoreAvg       = (scoreSpeed + scoreQuality + scoreContext + scorePrecision) / 4;
  const scoreClass     = noFit ? 'error'
    : scoreAvg >= 4 ? 'score-high' : scoreAvg >= 3 ? 'score-mid' : scoreAvg >= 2 ? 'score-low' : 'score-poor';
  return { scoreSpeed, scoreQuality, scoreContext, scoreContext10, scorePrecision, scoreClass, contextFitPct };
}
