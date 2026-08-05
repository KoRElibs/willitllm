// ─── INDEX.RENDER — main result card rendering
//
// Renders: VRAM bar, budget breakdown, scorecard, verdict, OOM message,
//          aside (speed+context stats), and Ollama setup commands.
//
// Depends on:  OVERHEAD_GB (app.calc.js),
//              app.fmt.js (fmtGB, fmtCtx, fmtCtxPages, fmtCtxWords,
//                          fmtTokensHuman, fmtSpeechPace, fmtSpeedHuman,
//                          fmtSpeed, bar10, colorForScore),
//              app.shared.js (muted),
//              app.util.js (metricLabel),
//              index.variants.js (getSelectedVariantIdx, variantOllamaTag),
//              index.ui.js (syncOsSelect),
//              index.js (activeOsTab, runOpen, setupContent, getTargetCtx — at runtime)
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
    // Only Context fit carries the status scale (green/amber/orange) — it's the "will it
    // meet my target" signal. Speed/Sharpness/Memory-clarity are neutral tradeoffs, not
    // good/bad verdicts, so they use a single calm fill instead of the traffic-light hues.
    // Context uses contextFitColor (the same scale as the model list and the ~pages number),
    // so it is green only when the target is actually met — never at a shortfall. With no
    // target chosen (neutral default) it drops to the same calm fill as the other meters.
    el.style.color = (id === 'scoreContext' && !isNeutralTarget())
      ? contextFitColor(ctxResult.maxCtx, getTargetCtx(), contextFitPct)
      : 'var(--meter)';
  });

  // Benchmark row — cited model-capability score (distinct from quant "Sharpness").
  // This row also hosts the coding verdict (set in render()), so it's shown for every
  // fitting model; the `has-bench` class drives whether the benchmark parts appear and
  // whether the verdict is pushed to the right (CSS). noFit already returned above.
  const benchRow = document.getElementById('scoreBenchRow');
  if (libInfo && libInfo.capability != null) {
    const label = metricLabel(libInfo.capability_metric);
    const proto = libInfo.capability_protocol ? ` ${libInfo.capability_protocol}` : '';
    document.getElementById('scoreBench').textContent = `${label} ${libInfo.capability}%`;
    document.getElementById('scoreBenchLabel').dataset.tip =
      `${label}${proto} ${libInfo.capability}% — model-capability benchmark (not quantization). `
      + (libInfo.capability_ref ? `Measured on ${libInfo.capability_ref}; family-level, so smaller sizes score lower. ` : '')
      + (libInfo.capability_source ? `Source: ${libInfo.capability_source}` : '');
    benchRow.classList.add('has-bench');
  } else {
    benchRow.classList.remove('has-bench');
  }
  benchRow.hidden = false;

  scorecard.hidden = false;

  const ctxTradeoff = 'Memory clarity vs. context fit: crisper recall (f16) costs more VRAM per token, leaving less room for a long conversation.';
  if (quantInfo) {
    const tradeoff = 'Thinking speed and sharpness trade off — a lighter quantization means faster responses but a duller mind. You cannot have both at maximum. (Technical: quantization level)';
    const quantLabel = `${quantInfo.approx ? '~' : ''}${variant.quantization || variant.format || '?'}`;
    document.getElementById('scoreSpeed').dataset.tip   = `${quantLabel} · ${quantInfo.summary} · ${tradeoff}`;
    document.getElementById('scoreQuality').dataset.tip = `${quantLabel} · ${quantInfo.summary} · ${tradeoff}`;
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
  document.getElementById('resultAside').hidden = true;
  // Nothing to run — hide both the toggle and the panel, whatever the stored open state.
  document.getElementById('runToggle').hidden  = true;
  document.getElementById('runSection').hidden = true;
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

  // Color the context stat with the SAME scale as the scorecard Context meter and the model
  // list (contextFitColor), so the two halves of the context signal can never disagree.
  // Neutral default ('none'): pass no archFitPct so contextFitColor returns the neutral --text
  // (just show the number). 'max' passes archFitPct → judged vs capability; sizes judged vs target.
  ctxPagesEl.style.color = contextFitColor(ctxResult.maxCtx, targetCtx, isNeutralTarget() ? null : contextFitPct);

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

// The OS keys the setup panel understands, in dropdown order. Index-only — coder.html
// no longer renders setup commands, it links here instead.
const OS_KEYS = ['linux-quick', 'linux-service', 'macos', 'windows'];

// Older builds stored other keys under `osTab`; map them forward so a returning visitor
// never lands on a key that no longer exists (which would render an empty panel).
const OS_LEGACY = { generic: 'linux-quick', linux: 'linux-service' };

function storedOs(fallback) {
  const v = localStorage.getItem('osTab');
  if (OS_LEGACY[v]) return OS_LEGACY[v];
  return OS_KEYS.includes(v) ? v : fallback;
}

// Does this recommendation actually require touching the server?
//
// Only a QUANTIZED KV cache does. `OLLAMA_KV_CACHE_TYPE` is read by the server at startup,
// and it needs `OLLAMA_FLASH_ATTENTION=1` alongside it or Ollama silently keeps f16.
//
// f16 is Ollama's default, so when that is what we recommend there is nothing to configure:
// emitting `OLLAMA_KV_CACHE_TYPE=f16` would set the value it already has, and making the user
// stop a system service and run a foreground server to do it is pure ceremony. Context length
// alone never justifies it either — `/set parameter num_ctx` sets it per run from inside the
// REPL, at the highest precedence (API option > env var > Modelfile > default).
function needsServerSetup(kvLabel) {
  return kvLabel !== 'f16';
}

// Server-side vars, only ever called when needsServerSetup() is true.
// OLLAMA_CONTEXT_LENGTH is the current name — NOT `OLLAMA_NUM_CTX`, which was dropped in
// Ollama 0.6 and is silently ignored today (BUG-26). Since we are restarting the server
// anyway here, setting it there is set-and-forget and saves a `/set parameter` every run.
function serverEnv(kvLabel, maxCtx) {
  return [
    'OLLAMA_FLASH_ATTENTION=1',
    `OLLAMA_KV_CACHE_TYPE=${kvLabel}`,
    `OLLAMA_CONTEXT_LENGTH=${maxCtx}`,
  ];
}

function osKvContent(tab, kvLabel, maxCtx) {
  const env = serverEnv(kvLabel, maxCtx);

  // Temporary: run the server in this terminal with the settings applied. The service has
  // to be stopped first — it holds port 11434, so `ollama serve` would fail while it runs.
  if (tab === 'linux-quick') return [
    muted(`# ${kvLabel} KV cache is a server setting — restart Ollama with it, this session only:`),
    muted("# ('sudo systemctl start ollama' hands it back to the service afterwards)"),
    'sudo systemctl stop ollama',
    `${env.join(' ')} ollama serve`,
  ].join('\n');

  // Permanent: a drop-in for the packaged service. `systemctl edit ollama.service` is the
  // documented route but opens an editor, so it can't be copy-pasted — this writes the same
  // override.conf directly. Idempotent: tee overwrites, so a later change replaces cleanly.
  if (tab === 'linux-service') return [
    muted('# configure the Ollama service — permanent, survives reboot:'),
    `sudo mkdir -p /etc/systemd/system/ollama.service.d && printf '[Service]\\n${env.map(v => `Environment="${v}"\\n`).join('')}' | sudo tee /etc/systemd/system/ollama.service.d/override.conf`,
    muted('# reload and restart Ollama service:'),
    'sudo systemctl daemon-reload && sudo systemctl restart ollama',
  ].join('\n');

  // The menubar app owns the server, so it has to be quit before a terminal one can bind
  // the port. Reopening the app restores the defaults — nothing to undo.
  if (tab === 'macos') return [
    muted(`# ${kvLabel} KV cache is a server setting — restart Ollama with it, this session only:`),
    muted('# quit Ollama from the menu bar (⌘Q) — reopening it restores the defaults:'),
    `${env.join(' ')} ollama serve`,
  ].join('\n');

  // Padded to the longest name so the values line up as a readable column.
  if (tab === 'windows') {
    const pad = Math.max(...env.map(v => v.split('=')[0].length));
    return [
      muted('# System Properties → Environment Variables → New user variable, for each:'),
      ...env.map(v => {
        const [name, val] = v.split('=');
        return muted(`#    ${name.padEnd(pad)}  =  ${val}`);
      }),
      muted('# then right-click Ollama in the tray → Quit, and reopen Ollama'),
    ].join('\n');
  }

  return '';
}

function renderCmd(model, libInfo, ctxResult, kvLabel) {
  document.getElementById('resultLabelOom').hidden = true;

  const idx    = getSelectedVariantIdx(model);
  const runTag = variantOllamaTag(model, idx);
  const needsServer = needsServerSetup(kvLabel);

  // f16 needs no server change, so the recipe is just pull, run, set the context from inside
  // the REPL — identical on every platform. The OS selector is hidden in this case: offering
  // a choice that changes nothing is noise, and it made the panel look like it was demanding
  // sudo and a foreground server for a default setting.
  if (!needsServer) {
    const universal = [
      `ollama pull ${runTag}`,
      `ollama run ${runTag}`,
      muted(`>>> /set parameter num_ctx ${ctxResult.maxCtx}`),
    ].join('\n');
    OS_KEYS.forEach(tab => { setupContent[tab] = universal; });
  } else {
    // Windows is the only one whose follow-up commands are PowerShell rather than a shell.
    const transition = tab => tab === 'windows' ? '# in PowerShell:' : '# in a new terminal:';

    OS_KEYS.forEach(tab => {
      // The systemd block reconfigures a background service that keeps running, so it ends at
      // the restart — starting a model afterwards is ordinary use, not part of the setup. The
      // other three occupy the terminal you just typed into, so the recipe continues in a new
      // one and finishes with the model actually running. Context comes from the env var set
      // above, so no `/set parameter` is needed here.
      const lines = [
        osKvContent(tab, kvLabel, ctxResult.maxCtx),
        muted(transition(tab)),
        `ollama pull ${runTag}`,
      ];
      if (tab !== 'linux-service') lines.push(`ollama run ${runTag}`);
      setupContent[tab] = lines.join('\n');
    });
  }

  document.querySelector('.run-os-row').hidden = !needsServer;

  document.getElementById('ollamaSetup').innerHTML = setupContent[activeOsTab];
  syncOsSelect();
  // The toggle becomes available; whether the panel is open is the user's stored choice,
  // applied once at init and preserved across re-renders.
  document.getElementById('runToggle').hidden = false;
  document.getElementById('runSection').hidden = !runOpen;
}
