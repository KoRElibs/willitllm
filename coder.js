// ─── CODER.JS — vibe coder orchestrator and initialisation
//
// Entry point for coder.html. Loaded last — all data and app files must precede it.
//
// Depends on:  MODELS (data files),
//              app.calc.js (OVERHEAD_GB),
//              app.shared.js (buildGpuSelector, initTooltip, initInfoSheet, osKvContent),
//              app.util.js (getLibMeta, getFlashOk),
//              coder.rank.js (buildEntries),
//              coder.rows.js (sectionDivider, makeRow)
// Provides:    getBaseUrl, render (called by coder.rows.js at runtime)

function getBaseUrl() {
  return document.getElementById('baseUrlInput')?.value.trim() || 'http://localhost:11434';
}

// ── URL state ─────────────────────────────────────────────────────────────────

function pushHashState() {
  const gpuOpt = document.getElementById('vramInput').selectedOptions[0];
  if (!gpuOpt || gpuOpt.disabled) return;
  const p = new URLSearchParams();
  p.set('g', gpuOpt.textContent.trim());
  const url = document.getElementById('baseUrlInput').value.trim();
  if (url && url !== 'http://localhost:11434') p.set('u', url);
  history.replaceState(null, '', '#' + p.toString());
}

function applyHashState() {
  const hash = window.location.hash.slice(1);
  if (!hash) return;
  const p = new URLSearchParams(hash);
  const gpuName = p.get('g');
  if (gpuName) {
    const gpuSel = document.getElementById('vramInput');
    const opt = Array.from(gpuSel.options).find(o => o.textContent.trim() === gpuName);
    if (opt) opt.selected = true;
  }
  const url = p.get('u');
  if (url) document.getElementById('baseUrlInput').value = url;
}

function fitCheckerUrl() {
  const gpuOpt = document.getElementById('vramInput').selectedOptions[0];
  if (!gpuOpt || gpuOpt.disabled) return 'index.html';
  const p = new URLSearchParams();
  p.set('g', gpuOpt.textContent.trim());
  return 'index.html#' + p.toString();
}

// ── List render ───────────────────────────────────────────────────────────────

function renderList(vramGB) {
  const { agent, code, fim } = buildEntries(vramGB, getFlashOk());
  const list = document.getElementById('coderList');
  list.innerHTML = '';

  if (!agent.length && !code.length && !fim.length) {
    list.innerHTML = '<div class="no-model">No coding models fit this GPU. Try a larger card.</div>';
    return;
  }

  // Attach the plain-language "why" clause to every entry, relative to its bucket.
  [agent, code, fim].forEach(bucket => bucket.forEach(e => { e.why = whyLine(e, bucket); }));

  if (agent.length) {
    list.appendChild(sectionDivider('Coding agents — autonomous planning, file edits, shell commands', 'divider-agent'));
    agent[0].recommended = true;
    agent.forEach(e => list.appendChild(makeRow(e)));
  }

  if (code.length) {
    list.appendChild(sectionDivider('Code chat &amp; assistance — explanation, generation, review'));
    code.forEach(e => list.appendChild(makeRow(e)));
  }

  if (fim.length) {
    list.appendChild(sectionDivider('Autocomplete — fill-in-the-middle IDE completion'));
    fim.forEach(e => list.appendChild(makeRow(e)));
  }
}

// ── Render + Init ─────────────────────────────────────────────────────────────

function render() {
  const vramGB   = parseFloat(document.getElementById('vramInput').value);
  const noGpu    = document.getElementById('noGpu');
  const list     = document.getElementById('coderList');
  const backLink = document.getElementById('backLink');
  if (backLink) backLink.href = fitCheckerUrl();
  if (isNaN(vramGB) || vramGB <= 0) {
    noGpu.hidden = false; list.hidden = true; return;
  }
  pushHashState();
  noGpu.hidden = true; list.hidden = false;
  renderList(vramGB);
}

function init() {
  buildGpuSelector();
  applyHashState();
  document.getElementById('vramInput').addEventListener('change', render);
  document.getElementById('baseUrlInput').addEventListener('change', render);
  window.addEventListener('hashchange', () => { applyHashState(); render(); });

  initTooltip();
  initInfoSheet();

  render();
}

init();
