// ─────────────────────────────────────────────────────────────────────────────
// APP.CALC — pure calculations and formatting helpers
//
// Minimal DOM access (getGpuSpecs reads the GPU select element).
// Provides constants and functions consumed by all other app files.
//
// Globals:     GPUS, QUANT_INFO                             (data files)
// ─────────────────────────────────────────────────────────────────────────────

const OVERHEAD_GB    = 0.8;  // fixed reservation: CUDA context, driver, ollama runtime + driver-reserved
                            // VRAM. Raised 0.5→0.8 after a GTX 1660S spill at a context the old value
                            // predicted would fit (real usable VRAM < rated; see meta/knowledge/).
const SAFETY_FACTOR  = 0.9;  // 10% margin — overhead estimate is imprecise (0.5–1.0 GB in practice)
const CTX_ROUND      = 128;  // round down to nearest 128 (natural head-dimension granularity)
const DECODE_ATTN_EFF = 0.015; // batch-1 decode attention: fraction of fp16 TFLOPS reached by the
                               // per-token attention GEMV. Calibrated against RTX 3090 + GTX 1660S
                               // sweeps (meta/benchmarks/); see meta/knowledge/benchmark-rtx3090-devstral.md

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
  if (!isFinite(mult) || mult <= 1) return 'speech pace';
  return `${mult}× speech pace`;
}

// ~0.75 words per token; ~250 words per page (333 tokens/page)
function fmtCtxWords(n) {
  const words = Math.round(n * 0.75 / 500) * 500;
  const w = words >= 1000 ? `${(words / 1000).toFixed(0)}k` : String(words);
  return `≈${w} words`;
}

function fmtCtxPages(n) {
  const pages = Math.round(n / 333 / 5) * 5;
  return `~${pages} pages`;
}

function fmtTokensHuman(n) {
  return `${fmtCtxWords(n)} · ${fmtCtxPages(n)}`;
}

function fmtCtx(n) {
  if (n >= 1000) return Math.round(n / 1000) + 'k';
  return String(n);
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
  if (availableBytes <= 0) return { maxCtx: 0, kvCacheGB: 0, safetyGB: 0, genuinelyFreeGB: 0, freeGB: 0, availableBytes };

  const valueDim  = model.value_length ?? model.key_length;
  const perToken  = model.block_count * model.head_count_kv * (model.key_length + valueDim) * bytesPerElement;
  const rawTokens = availableBytes / perToken;
  const archLimit = model.context_length || Infinity;
  const archMaxRaw    = Math.min(rawTokens, archLimit);
  const maxCtx        = Math.floor(archMaxRaw * SAFETY_FACTOR / CTX_ROUND) * CTX_ROUND;
  const limitedByArch = isFinite(archLimit) && rawTokens > archLimit;
  // IMPORTANT: kvCacheGB is ≈ constant when VRAM-limited — halving bytesPerElement
  // doubles maxCtx, so their product stays flat. What changes for the user is maxCtx
  // (the context window size), not memory. kvCacheGB only shrinks when maxCtx hits
  // archLimit. Always surface maxCtx as the primary metric, not kvCacheGB.
  const kvCacheGB       = (maxCtx * perToken) / 1024 ** 3;
  // Split the non-weights, non-KV remainder into three honest components:
  //   overheadGB      — fixed driver/runtime reservation (OVERHEAD_GB)
  //   safetyGB        — tokens we could fit after safety factor but held back by SAFETY_FACTOR + CTX_ROUND rounding
  //   genuinelyFreeGB — only >0 when arch-limited (more VRAM than the model can ever use)
  const safetyGB        = (archMaxRaw - maxCtx) * perToken / 1024 ** 3;
  const genuinelyFreeGB = Math.max(0, rawTokens - archMaxRaw) * perToken / 1024 ** 3;
  const usedGB          = weightsGB + kvCacheGB;
  const freeGB          = vramGB - usedGB;   // = OVERHEAD_GB + safetyGB + genuinelyFreeGB

  return { maxCtx, kvCacheGB, safetyGB, genuinelyFreeGB, freeGB, perToken, rawTokens, availableBytes, usedGB, archLimit, limitedByArch };
}

function calcSpeedEstimates(model, variant, vramGB, quantInfo, maxCtx, kvCacheGB = 0) {
  if (!quantInfo || !quantInfo.gen_eff || !quantInfo.prefill_eff) return null;
  const gpuSpecs = getGpuSpecs(vramGB);
  if (!gpuSpecs) return null;

  // MoE: only active experts load per token — scale weights proportionally
  const activeFraction  = (model.params_b_active && model.params_b)
    ? model.params_b_active / model.params_b : 1.0;
  const activeWeightsGB = variant.weights_gb * activeFraction;

  const [genEffLo,     genEffHi]     = quantInfo.gen_eff;
  const [prefillEffLo, prefillEffHi] = quantInfo.prefill_eff;

  const paramsActive   = (model.params_b_active || model.params_b) * 1e9;
  const valueDim       = model.value_length ?? model.key_length;
  const kvDimsPerToken = model.block_count * model.head_count_kv * (model.key_length + valueDim);

  // Effective attended context. Sliding-window models (Gemma 2/3/4) only attend to a
  // fixed window of recent tokens in most layers, so their attention cost — and hence
  // generation speed — stays nearly flat as context grows. Full-attention models attend
  // to the whole context. Capping at the window is what makes the attention term general
  // across architectures rather than a per-model fudge.
  const attnCtx = Math.min(maxCtx || 0, model.sliding_window ?? Infinity);

  // Generation (decode): two serial costs per token.
  //   1. Memory streaming — read all active weights + the full KV cache once.
  //      t_mem = (active_weights + kv_cache) / (bandwidth × gen_eff)
  //   2. Attention compute — the per-token QKᵀ/AV GEMV over the attended context.
  //      t_attn = 2 × attnCtx × kvDimsPerToken / (tflops × DECODE_ATTN_EFF)
  // gen t/s = 1 / (t_mem + t_attn). The compute term is negligible at small context
  // (weights dominate) and grows with context for full-attention models — matching the
  // measured efficiency collapse of large dense models at 100k+ tokens.
  const decodeAttnFlops = 2 * attnCtx * kvDimsPerToken;
  const tMemFast  = (activeWeightsGB + kvCacheGB) / (gpuSpecs.bwHi * genEffHi);
  const tMemSlow  = (activeWeightsGB + kvCacheGB) / (gpuSpecs.bwLo * genEffLo);
  const tAttnFast = decodeAttnFlops / (gpuSpecs.tflopsHi * 1e12 * DECODE_ATTN_EFF);
  const tAttnSlow = decodeAttnFlops / (gpuSpecs.tflopsLo * 1e12 * DECODE_ATTN_EFF);
  const genHi = Math.round(1 / (tMemFast + tAttnFast));
  const genLo = Math.round(1 / (tMemSlow + tAttnSlow));

  // Prefill: compute-bound.
  // FLOPs per token = linear term (MLP + projections) + quadratic attention term.
  // Quadratic term: each token attends to attnCtx tokens across all layers —
  // 2 × attnCtx × block_count × head_count_kv × (key_length + value_length).
  // attnCtx (sliding-window aware) keeps this honest for Gemma-family models too.
  // Uses KV head dims (same data as the KV cache formula); slightly conservative
  // for MHA models but correct for GQA and internally consistent.
  const flopsPerToken  = 2 * paramsActive + 2 * attnCtx * kvDimsPerToken;
  const prefillLo = Math.round((gpuSpecs.tflopsLo * 1e12 * prefillEffLo) / flopsPerToken);
  const prefillHi = Math.round((gpuSpecs.tflopsHi * 1e12 * prefillEffHi) / flopsPerToken);

  return { genLo, genHi, prefillLo, prefillHi, isExact: gpuSpecs.isExact };
}

// Returns the best KV bytes-per-element for a model given a target context and GPU flash support.
// Prefers highest quality (f16) that still meets the target; falls back to most efficient available.
function autoKvBpe(model, vramGB, weightsGB, targetCtx, flashOk) {
  const candidates  = flashOk ? [2, 1, 0.5] : [2];
  const effective   = targetCtx ?? model.context_length ?? Infinity;
  for (const bpe of candidates) {
    if (calcMaxContext(model, vramGB, bpe, weightsGB).maxCtx >= effective) return bpe;
  }
  return candidates[candidates.length - 1];
}

// Returns all scores + scoreClass for the current selection.
function computeScores(quantInfo, bytesPerElement, ctxResult, noFit, model, targetCtx) {
  // contextFitPct stays as % of architectural max — used for ⓘ caveats, not for scoring
  const contextFitPct = (!noFit && model.context_length)
    ? Math.round((ctxResult.maxCtx / model.context_length) * 100) : null;
  const scoreSpeed     = quantInfo ? Math.max(1, Math.round((quantInfo.speed   / 10) * 5)) : 0;
  const scoreQuality   = quantInfo ? Math.max(1, Math.round((quantInfo.quality / 10) * 5)) : 0;

  let scoreContext, scoreContext10;
  if (contextFitPct === null || noFit) {
    scoreContext = 0; scoreContext10 = 0;
  } else if (targetCtx === null) {
    // "full model context" — score against architectural max.
    // If VRAM is not the constraint (arch-limited), the model is giving everything it has → 10/10.
    scoreContext   = contextFitPct >= 90 ? 5 : contextFitPct >= 66 ? 4 : contextFitPct >= 40 ? 3 : contextFitPct >= 15 ? 2 : 1;
    scoreContext10 = ctxResult?.limitedByArch ? 10 : Math.min(10, Math.max(1, Math.ceil(contextFitPct / 10)));
  } else {
    // actual / desired — calcMaxContext already caps maxCtx at model's arch limit,
    // so a model that tops out at 131k when you need 200k correctly scores 131/200 = 65%
    const ratio    = Math.min(1, ctxResult.maxCtx / targetCtx);
    scoreContext   = ratio >= 0.9 ? 5 : ratio >= 0.66 ? 4 : ratio >= 0.40 ? 3 : ratio >= 0.15 ? 2 : 1;
    scoreContext10 = Math.max(1, Math.round(ratio * 10));
  }

  const scorePrecision = bytesPerElement === 2 ? 5 : bytesPerElement === 1 ? 3 : 2;
  const scoreAvg       = (scoreSpeed + scoreQuality + scoreContext + scorePrecision) / 4;
  const scoreClass     = noFit ? 'error'
    : scoreAvg >= 4 ? 'score-high' : scoreAvg >= 3 ? 'score-mid' : scoreAvg >= 2 ? 'score-low' : 'score-poor';
  return { scoreSpeed, scoreQuality, scoreContext, scoreContext10, scorePrecision, scoreClass, contextFitPct };
}
