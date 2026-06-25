// ─── INDEX.DETAILS — detail panel tabs: Model Info, GPU Specs, Formula
//
// Depends on:  GPUS, QUANT_INFO (data files),
//              app.calc.js (OVERHEAD_GB, SAFETY_FACTOR, CTX_ROUND, getGpuSpecs),
//              app.fmt.js (fmtGB, fmtCtx, fmtSpeed),
//              data.flags.js (flagFor)
// Provides:    renderDetails, populateGpuTab, renderFormula

function renderDetails(model, libInfo, variant, weightsGB, quantization, bytesPerElement, kvLabel) {
  document.getElementById('detailOrganization').textContent = libInfo.organization || '—';

  const originEl = document.getElementById('detailOrigin');
  if (libInfo.origin) {
    originEl.textContent = `${flagFor(libInfo.origin)} ${libInfo.origin}`.trim();
    originEl.className   = 'detail-val';
  } else {
    originEl.textContent = 'community project';
    originEl.className   = 'detail-val community-origin';
  }

  document.getElementById('detailMoeRow').hidden        = !model.moe;
  document.getElementById('detailMultimodalRow').hidden = !(libInfo.capabilities || []).includes('vision');
  document.getElementById('detailMaxCtx').textContent   = model.context_length
    ? model.context_length.toLocaleString() + ' tokens' : '—';

  const quantInfo = variant ? QUANT_INFO[variant.quantization] : null;
  Object.entries({
    detailLayers:       model.block_count,
    detailKvHeads:      model.head_count_kv,
    detailHeadDim:      model.key_length,
    detailValueLength:  model.value_length ?? model.key_length,
    detailBpe:          bytesPerElement,
    detailBpeLabel:     kvLabel,
    detailWeights:      fmtGB(weightsGB),
    detailQuantization: quantization,
  }).forEach(([id, val]) => document.getElementById(id).textContent = val);

  document.getElementById('quantType').textContent    = quantization;
  document.getElementById('quantSummary').textContent = quantInfo?.summary || '—';

  const [library]  = model.ollama_tag.split(':');
  const variantTag = variant ? variant.tag : model.ollama_tag.split(':')[1];
  const ollamaUrl  = library.includes('/')
    ? `https://ollama.com/${library}:${variantTag}`
    : `https://ollama.com/library/${library}:${variantTag}`;
  const linkEl       = document.getElementById('detailOllamaLink');
  linkEl.href        = ollamaUrl;
  linkEl.textContent = `ollama.com/library/${library}:${variantTag} ↗`;

  document.getElementById('provenanceAlert').hidden = true;
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

function renderFormula(model, variant, ctxResult, speedEsts, vramGB, weightsGB, bytesPerElement, kvLabel, quantInfo, noFit, contextFitPct) {
  document.getElementById('formulaBox').hidden   = noFit;
  document.getElementById('formulaNoFit').hidden = !noFit;
  if (noFit) return;

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
  document.getElementById('formulaSafetyStep').textContent  = `×${SAFETY_FACTOR} safety, ÷${CTX_ROUND}`;
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

  const speedSection = document.getElementById('formulaSpeedSection');
  const speedBody    = document.getElementById('formulaSpeedBody');
  if (speedEsts) {
    const gpuSpecs        = getGpuSpecs(vramGB);
    const fmtRange        = (lo, hi) => lo === hi ? String(lo) : `${lo}–${hi}`;
    const activeFrac      = (model.params_b_active && model.params_b) ? model.params_b_active / model.params_b : 1.0;
    const activeWeightsGB = (variant ? variant.weights_gb : 0) * activeFrac;
    const [genLo, genHi]  = quantInfo ? quantInfo.gen_eff     : [0, 0];
    const [preLo, preHi]  = quantInfo ? quantInfo.prefill_eff : [0, 0];
    document.getElementById('formulaGenBw').textContent         = gpuSpecs ? fmtRange(gpuSpecs.bwLo, gpuSpecs.bwHi) : '?';
    document.getElementById('formulaGenEff').textContent        = `[${genLo}–${genHi}]`;
    document.getElementById('formulaGenWeights').textContent    = activeWeightsGB.toFixed(2);
    document.getElementById('formulaGenResult').textContent     = fmtSpeed(speedEsts.genLo, speedEsts.genHi);
    const valueDimF      = model.value_length ?? model.key_length;
    const linearFlops    = 2 * (model.params_b_active || model.params_b) * 1e9;
    const attnFlops      = 2 * ctxResult.maxCtx * model.block_count * model.head_count_kv * (model.key_length + valueDimF);
    const totalFlops     = linearFlops + attnFlops;
    const fmtGF          = n => `${(n / 1e9).toFixed(0)} GF/tok`;
    document.getElementById('formulaPrefillTflops').textContent = gpuSpecs ? fmtRange(gpuSpecs.tflopsLo, gpuSpecs.tflopsHi) : '?';
    document.getElementById('formulaPrefillEff').textContent    = `[${preLo}–${preHi}]`;
    document.getElementById('formulaPrefillFlops').textContent  = `${fmtGF(totalFlops)} (${fmtGF(linearFlops)} linear + ${fmtGF(attnFlops)} attn)`;
    document.getElementById('formulaPrefillResult').textContent = fmtSpeed(speedEsts.prefillLo, speedEsts.prefillHi);
    speedSection.hidden = false;
    speedBody.hidden    = false;
  } else {
    speedSection.hidden = true;
    speedBody.hidden    = true;
  }
}
