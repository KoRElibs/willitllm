// ─── INDEX.UI — model scoring colors, cap filter, OS tabs, selection summary
//
// Depends on:  MODELS, QUANT_INFO (data files),
//              app.calc.js (OVERHEAD_GB, autoKvBpe, calcMaxContext, computeScores),
//              index.combobox.js (markComboboxItems),
//              index.variants.js (getSelectedVariant),
//              index.js (activeOsTab, setupContent, render — at runtime)
// Provides:    _activeCaps, updateSelectionSummary, modelScoreColor,
//              markModelOptions, applyCap, detectOs, setOs, syncOsSelect

function updateSelectionSummary(model) {
  const el = document.getElementById('selectionSummary');
  if (!el) return;

  const modelSel = document.getElementById('modelSelect');
  const modelOpt = modelSel.selectedOptions[0];
  if (!modelOpt || modelOpt.value === '') { el.textContent = 'VRAM allocation'; return; }

  const library = model ? model.ollama_tag.split(':')[0] : modelOpt.textContent.trim().split(/[:\s]/)[0];
  const variant = model ? getSelectedVariant(model) : null;
  const fullTag = variant ? `${library}:${variant.tag}` : library;

  const gpuOpt = document.getElementById('vramInput').selectedOptions[0];
  const gpuName = gpuOpt ? gpuOpt.textContent.trim() : '';

  el.textContent = gpuName ? `${gpuName}: ${fullTag}` : fullTag;
}

// Unified colour + sort priority using the same formula as the result headline.
// fit: 0=green 1=amber 2=orange 3=red(poor) 4=OOM(excluded from auto-select)
function modelScoreColor(m, vramGB, targetCtx, flashOk) {
  const weightsGB = m.variants?.length ? m.variants[0].weights_gb : 0;
  if (weightsGB >= vramGB - OVERHEAD_GB) return { color: '#f06464', fit: 4 };
  // Per-model effective target: in the default "as much as fits" view this is min(32k, the
  // model's own max) at default KV, so a model is judged on a normal working context — not
  // penalised for using compressed KV or for not filling a huge trained window (see
  // effectiveTargetCtx). Supersedes the passed targetCtx, which can't be per-model.
  const effTarget  = effectiveTargetCtx(m);
  const bpe        = autoKvBpe(m, vramGB, weightsGB, effTarget, flashOk);
  const ctxResult  = calcMaxContext(m, vramGB, bpe, weightsGB);
  const quantInfo  = variantRatings(m, m.variants?.[0]);
  const { scoreClass } = computeScores(quantInfo, bpe, ctxResult, false, m, effTarget);
  return { 'score-high': { color: '#56d88a', fit: 0 },
           'score-mid':  { color: '#f5a623', fit: 1 },
           'score-low':  { color: '#f07418', fit: 2 },
           'score-poor': { color: '#f06464', fit: 3 } }[scoreClass] ?? { color: '#f07418', fit: 2 };
}

function markModelOptions(vramGB, targetCtx, flashOk) {
  const sel = document.getElementById('modelSelect');
  Array.from(sel.options).forEach(opt => {
    const m = MODELS[parseInt(opt.value)];
    if (!m) return;
    const { color, fit } = modelScoreColor(m, vramGB, targetCtx, flashOk);
    opt.textContent = fit === 4 ? `✗  ${m.ollama_tag}` : m.ollama_tag;
    opt.style.color = color;
  });
  markComboboxItems(vramGB, targetCtx, flashOk);
}

// ── Capability filter ─────────────────────────────────────────────────────────

const _activeCaps = new Set();

function applyCap(cap) {
  if (cap === '') {
    _activeCaps.clear();
  } else {
    _activeCaps.has(cap) ? _activeCaps.delete(cap) : _activeCaps.add(cap);
  }
  document.querySelectorAll('.cap-pill').forEach(pill => {
    pill.classList.toggle('active', pill.dataset.cap === '' ? _activeCaps.size === 0 : _activeCaps.has(pill.dataset.cap));
  });
  // "clear" pill (data-cap="") only appears when there's a filter to clear — shown last (CSS order).
  const clearPill = document.querySelector('.cap-pill[data-cap=""]');
  if (clearPill) clearPill.hidden = _activeCaps.size === 0;
  filterModelList(document.getElementById('modelSearch')?.value || '', true);
}

// ── OS selection ──────────────────────────────────────────────────────────────

// Best-effort guess so most visitors land on the right instructions without touching
// the dropdown. userAgentData.platform is the modern signal; navigator.platform is the
// fallback (deprecated but still the only thing Firefox/Safari expose). Anything
// unrecognised falls through to the Linux quick start — the least destructive option,
// since it changes nothing permanently and its commands read the same everywhere.
function detectOs() {
  const p = (navigator.userAgentData?.platform || navigator.platform || '').toLowerCase();
  const ua = navigator.userAgent.toLowerCase();
  if (p.includes('win') || ua.includes('windows'))    return 'windows';
  if (p.includes('mac') || ua.includes('mac os'))     return 'macos';
  return 'linux-quick';
}

function syncOsSelect() {
  const sel = document.getElementById('osSelect');
  if (sel) sel.value = activeOsTab;
}

function setOs(os) {
  activeOsTab = os;
  // Same localStorage key as coder.html's OS tabs — one OS preference across both pages.
  localStorage.setItem('osTab', os);
  document.getElementById('ollamaSetup').innerHTML = setupContent[os];
  syncOsSelect();
}
