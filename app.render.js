// ─────────────────────────────────────────────────────────────────────────────
// APP.RENDER — DOM render sub-functions
//
// Each function takes all its inputs as arguments; no module-level state.
// Calls helper functions from app.calc.js and app.ui.js.
//
// Globals:     GPUS, QUANT_INFO                             (data files)
//              OVERHEAD_GB                                  (app.calc.js)
//              activeOsTab, setupContent                    (app.js, shared mutable state)
// ─────────────────────────────────────────────────────────────────────────────

function renderMembar(vramGB, weightsGB, ctxResult, noFit) {
  const pct = gb => Math.min(100, (gb / vramGB) * 100);

  const modelPct    = pct(weightsGB);
  const contextPct  = noFit ? 0 : pct(ctxResult.kvCacheGB);
  const overheadPct = noFit ? 0 : pct(OVERHEAD_GB);
  const safetyPct   = noFit ? 0 : pct(ctxResult.safetyGB);
  const freePct     = noFit ? 0 : pct(ctxResult.genuinelyFreeGB);

  document.getElementById('barTotal').textContent = Math.round(vramGB) + ' GB';

  const segModel = document.getElementById('segModel');
  segModel.className   = 'membar-seg ' + (noFit ? 'seg-overflow' : 'seg-model');
  segModel.style.width = modelPct.toFixed(1) + '%';
  segModel.textContent = modelPct > 12 ? fmtGB(weightsGB) : '';

  const segContext = document.getElementById('segContext');
  segContext.className   = 'membar-seg ' + (noFit ? 'seg-overflow' : 'seg-context');
  segContext.style.width = contextPct.toFixed(1) + '%';
  segContext.textContent = contextPct > 8 ? fmtGB(ctxResult.kvCacheGB) : '';

  const segOverhead = document.getElementById('segOverhead');
  segOverhead.style.width  = overheadPct.toFixed(1) + '%';
  segOverhead.textContent  = overheadPct > 6 ? '~' + fmtGB(OVERHEAD_GB) : '';

  const segSafety = document.getElementById('segSafety');
  segSafety.style.width  = safetyPct.toFixed(1) + '%';
  segSafety.textContent  = safetyPct > 6 ? fmtGB(ctxResult.safetyGB) : '';

  const segFree = document.getElementById('segFree');
  if (freePct > 0.5) {
    // flex: 1 fills any remaining space after the other fixed-width segments
    segFree.style.flex  = '1';
    segFree.style.width = '';
    segFree.textContent = freePct > 6 ? fmtGB(ctxResult.genuinelyFreeGB) : '';
  } else {
    segFree.style.flex  = '';
    segFree.style.width = '0%';
    segFree.textContent = '';
  }

  document.getElementById('legendModel').textContent    = `Model weights · ${fmtGB(weightsGB)}`;
  document.getElementById('legendContext').textContent  = noFit ? '' : `${fmtCtx(ctxResult.maxCtx)} context · KV cache ${fmtGB(ctxResult.kvCacheGB)}`;
  document.getElementById('legendOverhead').textContent = noFit ? '' : `Overhead ~${fmtGB(OVERHEAD_GB)}`;
  const legendSafetyItem = document.getElementById('legendSafetyItem');
  legendSafetyItem.hidden = noFit || ctxResult.safetyGB < 0.05;
  if (!noFit) document.getElementById('legendSafety').textContent = `Safety ${fmtGB(ctxResult.safetyGB)}`;
  const legendFreeItem = document.getElementById('legendFreeItem');
  legendFreeItem.hidden = noFit || ctxResult.genuinelyFreeGB < 0.05;
  if (!noFit) document.getElementById('legendFree').textContent = `Free ${fmtGB(ctxResult.genuinelyFreeGB)}`;
}

function renderBudget(vramGB, weightsGB, ctxResult, noFit) {
  const show = id => { document.getElementById(id).hidden = false; };
  const hide = id => { document.getElementById(id).hidden = true; };

  if (noFit) {
    ['budgetHeader','budgetSection','budgetKvRow','budgetOverheadRow','budgetSafetyRow','budgetFreeRow','budgetTotalRow'].forEach(hide);
    return;
  }

  ['budgetHeader','budgetSection','budgetKvRow','budgetOverheadRow','budgetSafetyRow','budgetTotalRow'].forEach(show);

  document.getElementById('budgetWeights').textContent  = fmtGB(weightsGB);
  document.getElementById('budgetKv').textContent       = `${fmtGB(ctxResult.kvCacheGB)} (${fmtCtx(ctxResult.maxCtx)} tokens)`;
  document.getElementById('budgetOverhead').textContent = `~${fmtGB(OVERHEAD_GB)}`;
  document.getElementById('budgetSafety').textContent   = fmtGB(ctxResult.safetyGB);

  if (ctxResult.genuinelyFreeGB > 0.05) {
    show('budgetFreeRow');
    document.getElementById('budgetFree').textContent = fmtGB(ctxResult.genuinelyFreeGB);
  } else {
    hide('budgetFreeRow');
  }

  const totalUsed = weightsGB + ctxResult.kvCacheGB + OVERHEAD_GB + ctxResult.safetyGB;
  document.getElementById('budgetTotal').textContent = `${fmtGB(totalUsed)} of ${fmtGB(vramGB)}`;
}

function renderScorecard(scores, quantInfo, variant, kvLabel, kvInfo, libInfo, ctxResult, noFit) {
  const scorecard = document.getElementById('scorecard');
  if (noFit) { scorecard.hidden = true; return; }

  const { scoreSpeed, scoreQuality, scoreContext, scoreContext10, scorePrecision, contextFitPct } = scores;

  // Speed & Quality: raw 1–10 from quantInfo. Precision & Context: scale 1–5 → 1–10.
  [
    ['scoreSpeed',     quantInfo ? quantInfo.speed   : 0, scoreSpeed],
    ['scoreQuality',   quantInfo ? quantInfo.quality : 0, scoreQuality],
    ['scorePrecision', scorePrecision * 2,                scorePrecision],
    ['scoreContext',   scoreContext10,                    scoreContext],
  ].forEach(([id, n10, n5]) => {
    const el = document.getElementById(id);
    el.textContent = bar10(Math.round(n10));
    el.style.color = colorForScore(n5);
  });
  scorecard.hidden = false;

  const ctxTradeoff = 'Memory clarity vs. context fit: crisper recall (f16) costs more VRAM per token, leaving less room for a long conversation.';
  if (quantInfo) {
    const tradeoff = 'Thinking speed and sharpness trade off — a lighter quantization means faster responses but a duller mind. You cannot have both at maximum. (Technical: quantization level)';
    document.getElementById('scoreSpeed').dataset.tip   = `${variant.quantization} · ${quantInfo.summary} · ${tradeoff}`;
    document.getElementById('scoreQuality').dataset.tip = `${variant.quantization} · ${quantInfo.summary} · ${tradeoff}`;
  }
  if (kvInfo) {
    document.getElementById('scorePrecision').dataset.tip = `${kvLabel} · ${ctxTradeoff}`;
  }
  const scoreTgtCtx    = getTargetCtx();
  const scoreTgtFitPct = scoreTgtCtx ? Math.round(Math.min(1, ctxResult.maxCtx / scoreTgtCtx) * 100) : null;
  const pctPart = scoreTgtFitPct !== null && scoreTgtFitPct < 100
    ? `${fmtCtx(ctxResult.maxCtx)} of ${fmtCtx(scoreTgtCtx)} token target (${scoreTgtFitPct}%)`
    : (contextFitPct !== null && contextFitPct < 100
        ? `${contextFitPct}% of model limit`
        : 'full target');
  const mmPart = (libInfo.capabilities || []).includes('vision') ? ' · images use tokens' : '';
  document.getElementById('scoreContext').dataset.tip = `${pctPart}${mmPart} · ${ctxTradeoff}`;
}

function renderVerdict(noFit) {
  const verdictEl = document.getElementById('verdict');
  verdictEl.classList.remove('verdict-anim');
  verdictEl.textContent = noFit ? "IT WON'T LLM!" : "IT WILL LLM!";
  requestAnimationFrame(() => requestAnimationFrame(() => verdictEl.classList.add('verdict-anim')));
}

function renderOom(vramGB, weightsGB) {
  const labelOom = document.getElementById('resultLabelOom');
  labelOom.textContent = `Model weights (${fmtGB(weightsGB)}) exceed available VRAM (${fmtGB(vramGB - OVERHEAD_GB)} usable). This model will not load.`;
  labelOom.hidden = false;
  document.getElementById('ollamaCmd').hidden   = true;
  document.getElementById('osTabs').hidden      = true;
  document.getElementById('ollamaSetup').hidden = true;
  document.getElementById('resultAside').hidden = true;
}

function renderAside(speedEsts, ctxResult, contextFitPct) {
  const genEl  = document.getElementById('asideGenSpeed');
  const prefEl = document.getElementById('asidePrefillSpeed');
  const speedCaveat = document.getElementById('speedCaveat');
  if (speedEsts) {
    genEl.textContent  = fmtSpeechPace(speedEsts.genLo, speedEsts.genHi);
    genEl.dataset.tip  = `Writing its response · ${fmtSpeedHuman(speedEsts.genLo, speedEsts.genHi)} · ${fmtSpeed(speedEsts.genLo, speedEsts.genHi)} (generation — output tokens/s, bandwidth-bound) · rough estimate, ±2×`;
    prefEl.textContent = fmtSpeechPace(speedEsts.prefillLo, speedEsts.prefillHi);
    prefEl.dataset.tip = `Reading your prompt · ${fmtSpeedHuman(speedEsts.prefillLo, speedEsts.prefillHi)} · ${fmtSpeed(speedEsts.prefillLo, speedEsts.prefillHi)} (prefill — input tokens/s, compute-bound) · rough estimate, ±2×`;
    const genLabelEl  = document.getElementById('asideGenLabel');
    const prefLabelEl = document.getElementById('asidePrefillLabel');
    genLabelEl.textContent  = `writing · ${fmtSpeedHuman(speedEsts.genLo, speedEsts.genHi)}`;
    prefLabelEl.textContent = `reading · ${fmtSpeedHuman(speedEsts.prefillLo, speedEsts.prefillHi)}`;
    genLabelEl.dataset.tip  = `${fmtSpeed(speedEsts.genLo, speedEsts.genHi)} · output tokens/s · bandwidth-bound (model weights stream from VRAM every token)`;
    prefLabelEl.dataset.tip = `${fmtSpeed(speedEsts.prefillLo, speedEsts.prefillHi)} · input tokens/s · compute-bound (all prompt tokens processed in parallel)`;
    document.getElementById('asideGenStat').dataset.tip     = '';
    document.getElementById('asidePrefillStat').dataset.tip = '';
    if (speedCaveat) speedCaveat.hidden = false;
  } else {
    genEl.textContent  = '—';
    prefEl.textContent = '—';
    document.getElementById('asideGenLabel').textContent  = 'writing';
    document.getElementById('asidePrefillLabel').textContent = 'reading';
    document.getElementById('asideGenLabel').dataset.tip  = '';
    document.getElementById('asidePrefillLabel').dataset.tip = '';
    if (speedCaveat) speedCaveat.hidden = true;
  }

  const ctxPagesEl  = document.getElementById('asideCtxPages');
  const ctxLabelEl  = document.getElementById('asideCtxLabel');
  const caveatMark = contextFitPct && contextFitPct > 50
    ? ' <span data-tip="Like human memory — most models recall the start and end of a long text better than the middle." style="font-size:0.75em;opacity:0.5;cursor:help;">ⓘ</span>'
    : '';

  const targetCtx    = getTargetCtx();
  // targetFitPct: how much of the user's chosen target this model can deliver
  // (contextFitPct from scores is % of model's arch limit — separate concept)
  const targetFitPct = targetCtx ? Math.round(Math.min(1, ctxResult.maxCtx / targetCtx) * 100) : null;
  const showGap      = targetFitPct !== null && targetFitPct < 95;

  // Color the context stat to reflect how well the target is met
  ctxPagesEl.style.color = !showGap               ? 'var(--text)'
                         : targetFitPct >= 90      ? 'var(--green)'
                         : targetFitPct >= 50      ? 'var(--amber)'
                         :                           'var(--orange)';

  // Show achieved / target when there's a meaningful gap
  if (showGap) {
    ctxPagesEl.innerHTML = `${fmtCtxPages(ctxResult.maxCtx)}<span class="ctx-target-gap"> / ${fmtCtxPages(targetCtx)}</span>` + caveatMark;
  } else {
    ctxPagesEl.innerHTML = fmtCtxPages(ctxResult.maxCtx) + caveatMark;
  }
  ctxPagesEl.dataset.tip = showGap
    ? `${fmtCtx(ctxResult.maxCtx)} tokens achieved · target: ${fmtCtx(targetCtx)} tokens · ${targetFitPct}% of target`
    : `${fmtCtx(ctxResult.maxCtx)} tokens · ${fmtTokensHuman(ctxResult.maxCtx)} (context fit — how much text fits in VRAM at once)`;

  ctxLabelEl.textContent = showGap
    ? `context · ${fmtCtxWords(ctxResult.maxCtx)} · ${targetFitPct}% of target`
    : `context · ${fmtCtxWords(ctxResult.maxCtx)}`;
  ctxLabelEl.dataset.tip = `${fmtCtx(ctxResult.maxCtx)} tokens · ~0.75 words per token`;

  document.getElementById('resultAside').hidden = false;
}

function renderCmd(model, libInfo, ctxResult, kvLabel, bytesPerElement) {
  document.getElementById('resultLabelOom').hidden = true;

  const muted      = s => `<span class="cmd-muted">${s}</span>`;
  const variantIdx = getSelectedVariantIdx(model);
  const runTag     = variantOllamaTag(model, variantIdx);
  const isCoding   = !!libInfo.coding_role;
  const ollamaCmd  = document.getElementById('ollamaCmd');

  ollamaCmd.innerHTML = isCoding
    ? `ollama pull ${runTag}\n${muted('# then set contextLength in your editor config (see vibe coder →)')}`
    : `ollama pull ${runTag}\nOLLAMA_NUM_CTX=${ctxResult.maxCtx} ollama run ${runTag}`;
  ollamaCmd.hidden = false;

  const osTabs = document.getElementById('osTabs');
  if (bytesPerElement < 2) {
    const afterSetup = muted('# then run the command above');
    setupContent.linux = [
      muted('# First — stop ollama, restart with the KV cache setting:'),
      `OLLAMA_KV_CACHE_TYPE=${kvLabel} ollama serve`,
      muted('# In a new terminal:'),
      afterSetup,
    ].join('\n');
    setupContent.windows = [
      muted('# 1. Open: System Properties → Environment Variables → New user variable'),
      muted('#    Name:  OLLAMA_KV_CACHE_TYPE'),
      muted(`#    Value: ${kvLabel}`),
      muted('# 2. Right-click Ollama in system tray → Quit, then relaunch Ollama'),
      muted('# 3. In a new terminal:'),
      afterSetup,
    ].join('\n');
    if (activeOsTab) document.getElementById('ollamaSetup').innerHTML = setupContent[activeOsTab];
    osTabs.hidden = false;
  } else {
    osTabs.hidden = true;
    document.getElementById('ollamaSetup').hidden = true;
    activeOsTab = null;
    document.getElementById('tabLinux').textContent   = '▶ Linux / Mac';
    document.getElementById('tabWindows').textContent = '▶ Windows';
    document.getElementById('tabLinux').classList.remove('active');
    document.getElementById('tabWindows').classList.remove('active');
  }
}

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
