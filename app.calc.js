// ─── APP.CALC — pure calculations: KV context, speed estimates, scoring
//
// Minimal DOM access (getGpuSpecs reads the GPU select element).
// Formatters live in app.fmt.js.
//
// Depends on:  GPUS, QUANT_INFO (data files), app.fmt.js (formatters)
// Provides:    OVERHEAD_GB, SAFETY_FACTOR, CTX_ROUND, DECODE_ATTN_EFF,
//              getGpuSpecs, calcMaxContext, calcSpeedEstimates,
//              autoKvBpe, computeScores

const OVERHEAD_GB     = 0.8;
const SAFETY_FACTOR   = 0.9;
const CTX_ROUND       = 128;
const DECODE_ATTN_EFF = 0.015;

// ── GPU specs ─────────────────────────────────────────────────────────────────

// Returns { bwLo, bwHi, tflopsLo, tflopsHi, flash, isExact } or null if no data.
// For a named card (dataset.gpuIdx set), returns exact values (lo === hi).
// For a generic VRAM-tier entry, returns [min, max] across all named cards at that tier.
// flash ('yes'|'no'|'mixed') is read from the selected option's dataset.
function getGpuSpecs(vramGB) {
  const sel    = document.getElementById('vramInput');
  const opt    = sel.selectedOptions[0];
  const gpuIdx = opt ? parseInt(opt.dataset.gpuIdx) : NaN;
  const flash  = opt?.dataset.flash || 'no';

  if (!isNaN(gpuIdx) && GPUS[gpuIdx] && GPUS[gpuIdx].bandwidth) {
    const gpu = GPUS[gpuIdx];
    return { bwLo: gpu.bandwidth, bwHi: gpu.bandwidth,
             tflopsLo: gpu.tflops_fp16, tflopsHi: gpu.tflops_fp16, flash, isExact: true };
  }

  const entries = GPUS.filter(g => g.vram === vramGB && g.bandwidth);
  if (entries.length === 0) return null;
  return {
    flash,
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

function calcSpeedEstimates(model, variant, vramGB, quantInfo, maxCtx, kvCacheGB = 0, bytesPerElement = 2) {
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
  //   2. KV-access slowdown — grows with the attended context (∝ attnCtx × kvDims).
  //      t_attn = 2 × attnCtx × kvDimsPerToken / (tflops × DECODE_ATTN_EFF)
  // gen t/s = 1 / (t_mem + t_attn).
  //
  // The slowdown only applies when KV access is NOT free-flowing:
  //   • quantized KV (bpe < 2) — per-element dequantization on every read, OR
  //   • no flash attention (GPU flash ≠ 'yes') — unfused attention.
  // With f16 KV on a flash GPU, decode stays flat with context (measured ~0.80
  // efficiency to ~48k on both llama-arch devstral and mistral3 mistral-small) —
  // so the term is gated off there to avoid under-predicting speed.
  const kvSlowdown      = bytesPerElement < 2 || gpuSpecs.flash !== 'yes';
  const decodeAttnFlops = kvSlowdown ? 2 * attnCtx * kvDimsPerToken : 0;
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
