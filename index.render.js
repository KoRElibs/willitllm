// ─── INDEX.RENDER — main result card rendering
//
// Renders: VRAM bar, budget breakdown, scorecard, verdict, OOM message,
//          aside (speed+context stats), and Ollama setup commands.
//
// Depends on:  OVERHEAD_GB (app.calc.js),
//              app.fmt.js (fmtGB, fmtCtx, fmtCtxPages, fmtCtxWords,
//                          fmtTokensHuman, fmtSpeechPace, fmtSpeedHuman,
//                          fmtSpeed, bar10, colorForScore),
//              app.shared.js (osKvContent, muted),
//              app.util.js (metricLabel),
//              index.variants.js (getSelectedVariantIdx, variantOllamaTag),
//              index.ui.js (syncOsTabs),
//              index.js (activeOsTab, setupContent, getTargetCtx — at runtime)
// Provides:    renderMembar, renderBudget, renderScorecard, renderVerdict,
//              renderOom, renderAside, renderCmd

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

  // Benchmark row — cited model-capability score (distinct from quant "Sharpness").
  const benchRow = document.getElementById('scoreBenchRow');
  if (libInfo && libInfo.capability != null) {
    const label = metricLabel(libInfo.capability_metric);
    const proto = libInfo.capability_protocol ? ` ${libInfo.capability_protocol}` : '';
    document.getElementById('scoreBench').textContent = `${label} ${libInfo.capability}%`;
    document.getElementById('scoreBenchLabel').dataset.tip =
      `${label}${proto} ${libInfo.capability}% — model-capability benchmark (not quantization). `
      + (libInfo.capability_ref ? `Measured on ${libInfo.capability_ref}; family-level, so smaller sizes score lower. ` : '')
      + (libInfo.capability_source ? `Source: ${libInfo.capability_source}` : '');
    benchRow.hidden = false;
  } else if (benchRow) {
    benchRow.hidden = true;
  }

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
  document.getElementById('codeVerdict').hidden = true;
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

function renderCmd(model, libInfo, ctxResult, kvLabel) {
  document.getElementById('resultLabelOom').hidden = true;

  const idx      = getSelectedVariantIdx(model);
  const runTag   = variantOllamaTag(model, idx);
  const isCoding = !!libInfo.coding_role;
  const pull     = `ollama pull ${runTag}`;

  const runLinux = isCoding
    ? muted('# then set contextLength in your editor config (see vibe coder →)')
    : `OLLAMA_NUM_CTX=${ctxResult.maxCtx} ollama run ${runTag}`;
  const runWin = isCoding
    ? muted('# then set contextLength in your editor config (see vibe coder →)')
    : `$env:OLLAMA_NUM_CTX=${ctxResult.maxCtx}; ollama run ${runTag}`;

  const transition = { generic: '# in a new terminal:', linux: '# in a new terminal:',
    'linux-service': '# in a new terminal:', macos: '# in a new terminal:', windows: '# in PowerShell:' };

  ['generic', 'linux', 'linux-service', 'macos', 'windows'].forEach(tab => {
    setupContent[tab] = [
      osKvContent(tab, kvLabel),
      muted(transition[tab]),
      pull,
      tab === 'windows' ? runWin : runLinux,
    ].join('\n');
  });

  document.getElementById('ollamaSetup').innerHTML = setupContent[activeOsTab];
  document.getElementById('ollamaSetup').hidden = false;
  document.getElementById('osTabs').hidden = false;
  syncOsTabs();
}
