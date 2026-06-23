// ─────────────────────────────────────────────────────────────────────────────
// CODER.JS — vibe coder page: ranking, config output, init
//
// Single-file entry point. Loaded after data files and app.calc.js.
// Does NOT load app.render.js / app.ui.js / app.js — those are index.html only.
//
// Globals: GPUS, LIBRARIES, MODELS, QUANT_INFO            (data files)
//          OVERHEAD_GB, calcMaxContext, calcSpeedEstimates,
//          autoKvBpe, fmtSpeed, fmtCtx                    (app.calc.js)
// ─────────────────────────────────────────────────────────────────────────────

const LIB_META_C = Object.fromEntries(LIBRARIES.map(l => [l.library, l]));

function getLibMeta(m) {
  return LIB_META_C[m.ollama_tag.split(':')[0]] || {};
}

// A model belongs on the coder page only if it has a curated coding_role.
// The generic `tools` capability is NOT used — it lets in general chat models
// (llama, mistral, mixtral, …) that merely support function calling while
// excluding real code models (codellama, codegemma, starcoder2) that lack the badge.
const CODING_ROLES = ['agent', 'code', 'fim'];
function isCodingModel(lib) {
  return CODING_ROLES.includes(lib.coding_role);
}

// Escape HTML special chars for safe insertion into innerHTML <pre> content
function esc(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// ── Formatting ────────────────────────────────────────────────────────────────

// Context in developer units: files when ≥5, lines below that.
// ~1000 tokens per average source file; ~3 tokens per line of code.
function fmtCtxCoding(maxCtx) {
  const files = Math.round(maxCtx / 1000);
  if (files >= 5) return `~${Math.round(files / 5) * 5} files`;
  const lines = Math.round(maxCtx / 3 / 100) * 100;
  return `~${lines.toLocaleString()} lines`;
}

// ── Ranking ───────────────────────────────────────────────────────────────────

// Weighted coding score 0–1: speed 50%, context 30%, quality 20%.
// Speed weighted highest because agentic coding sessions chain 30–100 sequential tool calls.
function codingRank(genLo, maxCtx, quality) {
  const speedNorm = Math.min(1, (genLo || 0) / 30);
  const ctxNorm   = Math.min(1, maxCtx / 65536);
  const qualNorm  = (quality || 5) / 10;
  return speedNorm * 0.5 + ctxNorm * 0.3 + qualNorm * 0.2;
}

// ── GPU selector ──────────────────────────────────────────────────────────────
// Mirrors the GPU selector in app.js — same optgroup structure.

function getFlashOk() {
  const opt = document.getElementById('vramInput').selectedOptions[0];
  return opt?.dataset.flash === 'yes' || opt?.dataset.flash === 'mixed';
}

function getBaseUrl() {
  return document.getElementById('baseUrlInput')?.value.trim() || 'http://localhost:11434';
}

function buildGpuSelector() {
  const vramSel = document.getElementById('vramInput');

  const ph = document.createElement('option');
  ph.value = ''; ph.disabled = true; ph.selected = true;
  ph.textContent = 'Select your GPU...';
  vramSel.appendChild(ph);

  const isGeForce = name => /^(GTX |RTX \d{4})/.test(name) && !name.includes('Ada');
  const groups = [
    { label: 'NVIDIA GeForce',      match: (g, n) => !g.vendor && isGeForce(n) },
    { label: 'NVIDIA Professional', match: (g, n) => !g.vendor && !isGeForce(n) },
    { label: 'AMD Radeon',          match: (g)    => g.vendor === 'AMD' },
    { label: 'Apple',               match: (g)    => g.vendor === 'Apple' },
  ];

  groups.forEach(({ label, match }) => {
    const cards = [];
    GPUS.forEach((gpu, idx) => {
      gpu.names.forEach(name => {
        if (match(gpu, name)) cards.push({ name, vram: gpu.vram, flash: gpu.flash, idx });
      });
    });
    if (!cards.length) return;
    cards.sort((a, b) => a.name.localeCompare(b.name));
    const group = document.createElement('optgroup');
    group.label = label;
    cards.forEach(({ name, vram, flash, idx }) => {
      const opt = document.createElement('option');
      opt.value = vram; opt.dataset.flash = flash; opt.dataset.gpuIdx = idx;
      opt.textContent = name;
      group.appendChild(opt);
    });
    vramSel.appendChild(group);
  });

  const sizes = [...new Set(GPUS.map(g => g.vram))].sort((a, b) => b - a);
  const genericGroup = document.createElement('optgroup');
  genericGroup.label = 'Generic';
  sizes.forEach(vram => {
    const entries   = GPUS.filter(g => g.vram === vram);
    const flashVals = [...new Set(entries.map(g => g.flash))];
    const opt = document.createElement('option');
    opt.value = vram;
    opt.dataset.flash = flashVals.length === 1 ? flashVals[0] : 'mixed';
    opt.textContent = `${vram} GB`;
    genericGroup.appendChild(opt);
  });
  vramSel.appendChild(genericGroup);
}

// ── Data / ranking ────────────────────────────────────────────────────────────

function buildEntries(vramGB, flashOk) {
  const agent = [];
  const code  = [];
  const fim   = [];

  MODELS.forEach(model => {
    const lib     = getLibMeta(model);
    if (!isCodingModel(lib)) return;

    const variant   = model.variants?.[0];
    if (!variant) return;

    const weightsGB = variant.weights_gb;
    if (weightsGB >= vramGB - OVERHEAD_GB) return;   // hide sizes that don't fit

    const bpe       = autoKvBpe(model, vramGB, weightsGB, null, flashOk);
    const ctx       = calcMaxContext(model, vramGB, bpe, weightsGB);
    const quantInfo = QUANT_INFO[variant.quantization];
    const speedEsts = calcSpeedEstimates(model, variant, vramGB, quantInfo, ctx.maxCtx, ctx.kvCacheGB, bpe);
    const score     = speedEsts ? codingRank(speedEsts.genLo, ctx.maxCtx, quantInfo?.quality) : 0;
    const ctxF16    = calcMaxContext(model, vramGB, 2, weightsGB);

    const entry = { model, lib, variant, weightsGB, bpe, ctx, ctxF16, fits: true, speedEsts, quantInfo, score };
    const bucket = lib.coding_role === 'fim'  ? fim
                 : lib.coding_role === 'code' ? code
                 : agent;
    bucket.push(entry);
  });

  const sortFn = (a, b) => b.score - a.score;
  agent.sort(sortFn);
  code.sort(sortFn);
  fim.sort(sortFn);
  return { agent, code, fim };
}

// ── Config HTML ───────────────────────────────────────────────────────────────

// OS-specific instructions for setting the KV cache env var permanently.
// Linux is split into write (idempotent) + apply (restarts ollama) so users can
// re-run the write step without unnecessarily bouncing the service.
function kvSetupHtml(kvType) {
  const linuxWrite = `sudo mkdir -p /etc/systemd/system/ollama.service.d && printf '[Service]\\nEnvironment="OLLAMA_KV_CACHE_TYPE=${kvType}"\\n' | sudo tee /etc/systemd/system/ollama.service.d/override.conf`;
  const linuxApply = `sudo systemctl daemon-reload && sudo systemctl restart ollama`;
  const macos      = `# Quit the Ollama menu bar app first (⌘Q)\necho 'export OLLAMA_KV_CACHE_TYPE=${kvType}' >> ~/.zshrc && source ~/.zshrc\nollama serve`;
  const win        = `setx OLLAMA_KV_CACHE_TYPE ${kvType}\n# Restart Ollama: right-click tray icon → Quit, then reopen`;

  const cmd = (label, text) =>
    `<div class="kv-cmd"><div class="config-label">${label} <button class="copy-btn">copy</button></div><pre class="ollama-cmd">${esc(text)}</pre></div>`;

  return `
    <div class="config-section">
      <div class="config-label">1. Set KV cache type — permanent</div>
      <div class="kv-os-tabs-wrap">
        <button class="kv-os-tab active" data-kvos="linux">Linux</button>
        <button class="kv-os-tab" data-kvos="macos">macOS</button>
        <button class="kv-os-tab" data-kvos="windows">Windows</button>
      </div>
      <div class="kv-os-block" data-kvos="linux">
        ${cmd('Write config (safe to re-run)', linuxWrite)}
        ${cmd('Apply — restarts ollama', linuxApply)}
      </div>
      <div class="kv-os-block" data-kvos="macos" hidden>${cmd('Add to shell + restart', macos)}</div>
      <div class="kv-os-block" data-kvos="windows" hidden>${cmd('Set env var + restart', win)}</div>
    </div>`;
}

// Renders one set of ollama + editor instructions for a given context size.
// showCline: false for code-role models (Continue only — not designed for agentic loops).
function modeHtml(runTag, maxCtx, kvType, baseUrl, showCline) {
  const ollamaRun    = `ollama pull ${runTag}`;
  const clineFields  = [
    ['API Provider',   'Ollama'],
    ['Base URL',       baseUrl],
    ['Model',          runTag],
    ['Context Window', String(maxCtx)],
  ];
  const clineText    = clineFields.map(([k, v]) => `${k.padEnd(18)}${v}`).join('\n');
  const continueJson = JSON.stringify({
    title: runTag, provider: 'ollama', model: runTag,
    apiBase: baseUrl, contextLength: maxCtx,
  }, null, 2);

  const kvSection = kvType ? kvSetupHtml(kvType) : '';
  const runLabel  = kvType ? '2. Pull model' : 'Ollama — pull model';

  const clientTabs = showCline
    ? `<button class="client-tab active" data-client="cline">Cline</button>
       <button class="client-tab" data-client="continue">Continue</button>`
    : `<button class="client-tab active" data-client="continue">Continue</button>`;

  const clineBlock = showCline
    ? `<div class="client-block" data-client="cline">
        <div class="config-label"><span>Cline sidebar → ⚙️ → enter these values <a class="doc-link" href="https://docs.cline.bot/running-models-locally/ollama" target="_blank" rel="noopener">docs ↗</a></span><button class="copy-btn">copy</button></div>
        <pre class="client-config">${esc(clineText)}</pre>
        <div class="config-note">Model is a dropdown — it lists models already pulled to ollama. Also recommended: ⚙️ → Features → enable <strong>Use Compact Prompt</strong>.</div>
      </div>`
    : '';

  return `
    ${kvSection}
    <div class="config-section">
      <div class="config-label"><span>${runLabel} <a class="doc-link" href="https://ollama.com" target="_blank" rel="noopener">ollama.com ↗</a></span><button class="copy-btn">copy</button></div>
      <pre class="ollama-cmd">${esc(ollamaRun)}</pre>
    </div>
    <div class="config-section">
      <div class="client-tabs-wrap">
        ${clientTabs}
      </div>
      ${clineBlock}
      <div class="client-block" data-client="continue"${showCline ? ' hidden' : ''}>
        <div class="config-label"><span>Continue config — add to the <code>models</code> array <a class="doc-link" href="https://docs.continue.dev" target="_blank" rel="noopener">docs ↗</a></span><button class="copy-btn">copy</button></div>
        <pre class="client-config">${esc(continueJson)}</pre>
        <div class="config-note">Open Continue config: click the Continue icon in the sidebar → ⚙️ → <em>Open Config</em>. Paste this inside the <code>models: [ ]</code> array. Also works in Cursor, VSCodium, Windsurf, and JetBrains.</div>
      </div>
    </div>
  `;
}

// Config for FIM / autocomplete models.
// Ollama hosting (num_ctx, KV cache) works the same as any other model — the two-mode
// Quick start / More context layout is preserved. Only the editor plugin section differs:
// tabAutocompleteModel instead of Cline/Continue chat configs.
function fimConfigHtml(entry) {
  const { model, variant, ctx, ctxF16, bpe } = entry;
  const [library] = model.ollama_tag.split(':');
  const runTag    = `${library}:${variant.tag}`;
  const baseUrl   = getBaseUrl();
  const kvType    = bpe === 1 ? 'q8_0' : bpe === 0.5 ? 'q4_0' : null;
  const f16Ctx    = ctxF16?.maxCtx ?? ctx.maxCtx;
  const hasOptimized = kvType !== null && ctx.maxCtx > f16Ctx;

  const note = `<div class="config-section"><div class="config-note">FIM models use fill-in-the-middle tokens for cursor autocomplete — not for chat or agent loops. Add the block below to your <code>.continue/config.json</code> as <code>tabAutocompleteModel</code>.</div></div>`;

  function fimMode(maxCtx, kv) {
    const ollamaRun = `ollama pull ${runTag}`;
    const tabJson   = JSON.stringify({
      tabAutocompleteModel: {
        title: runTag, provider: 'ollama', model: runTag,
        apiBase: baseUrl, contextLength: maxCtx,
      },
    }, null, 2);
    const kvSection = kv ? kvSetupHtml(kv) : '';
    const runLabel  = kv ? '2. Pull model' : 'Ollama — pull model';
    return `
      ${kvSection}
      <div class="config-section">
        <div class="config-label">${runLabel} <button class="copy-btn">copy</button></div>
        <pre class="ollama-cmd">${esc(ollamaRun)}</pre>
      </div>
      <div class="config-section">
        <div class="config-label"><span>Continue config — set as <code>tabAutocompleteModel</code> <a class="doc-link" href="https://docs.continue.dev" target="_blank" rel="noopener">docs ↗</a></span><button class="copy-btn">copy</button></div>
        <pre class="client-config">${esc(tabJson)}</pre>
        <div class="config-note">Open Continue config: Continue sidebar → ⚙️ → <em>Open Config</em>. Paste this as a top-level key (not inside the <code>models</code> array).</div>
      </div>
    `;
  }

  if (!hasOptimized) return note + fimMode(ctx.maxCtx, null);

  const gain        = Math.round((ctx.maxCtx / f16Ctx - 1) * 100);
  const qualityNote = kvType === 'q8_0'
    ? 'Quality: nearly lossless (~0.5% perplexity hit)'
    : 'Quality: modest hit (~2–5% perplexity, degrades further at long contexts)';
  return `
    ${note}
    <div class="mode-tabs-wrap">
      <button class="mode-tab active" data-mode="quick"
        data-tip="Full f16 KV quality. Works immediately — no server restart needed.">
        Quick start · ${fmtCtxCoding(f16Ctx)}
      </button>
      <button class="mode-tab" data-mode="optimized"
        data-tip="${qualityNote} · ~${gain}% more context than Quick start · needs OLLAMA_KV_CACHE_TYPE=${kvType} restart">
        More context · ${fmtCtxCoding(ctx.maxCtx)}
      </button>
    </div>
    <div class="mode-block" data-mode="quick">
      ${fimMode(f16Ctx, null)}
    </div>
    <div class="mode-block" data-mode="optimized" hidden>
      ${fimMode(ctx.maxCtx, kvType)}
    </div>
  `;
}

// Two-mode config for agent/code models: "Quick start" (f16 KV) vs "Optimized" (q8_0/q4_0).
// Shows a single mode when both would give the same context (arch-limited or no flash support).
// FIM models use fimConfigHtml() instead — this function is not called for them.
function makeConfigHtml(entry) {
  const { model, variant, ctx, ctxF16, bpe, lib } = entry;
  const [library] = model.ollama_tag.split(':');
  const runTag    = `${library}:${variant.tag}`;
  const kvType    = bpe === 1 ? 'q8_0' : bpe === 0.5 ? 'q4_0' : null;
  const f16Ctx    = ctxF16?.maxCtx ?? ctx.maxCtx;
  const hasOptimized = kvType !== null && ctx.maxCtx > f16Ctx;

  // Code-role models: hide Cline — these are not designed for autonomous agent loops.
  const showCline = lib.coding_role === 'agent';
  const baseUrl   = getBaseUrl();

  if (!hasOptimized) return modeHtml(runTag, ctx.maxCtx, null, baseUrl, showCline);

  const gain        = Math.round((ctx.maxCtx / f16Ctx - 1) * 100);
  const qualityNote = kvType === 'q8_0'
    ? 'Quality: nearly lossless (~0.5% perplexity hit)'
    : 'Quality: modest hit (~2–5% perplexity, degrades further at long contexts)';
  return `
    <div class="mode-tabs-wrap">
      <button class="mode-tab active" data-mode="quick"
        data-tip="Full f16 KV quality. Works immediately — no server restart needed.">
        Quick start · ${fmtCtxCoding(f16Ctx)}
      </button>
      <button class="mode-tab" data-mode="optimized"
        data-tip="${qualityNote} · ~${gain}% more context than Quick start · needs OLLAMA_KV_CACHE_TYPE=${kvType} restart">
        More context · ${fmtCtxCoding(ctx.maxCtx)}
      </button>
    </div>
    <div class="mode-block" data-mode="quick">
      ${modeHtml(runTag, f16Ctx, null, baseUrl, showCline)}
    </div>
    <div class="mode-block" data-mode="optimized" hidden>
      ${modeHtml(runTag, ctx.maxCtx, kvType, baseUrl, showCline)}
    </div>
  `;
}

// ── Row building ──────────────────────────────────────────────────────────────

function makeRow(entry) {
  const { model, lib, variant, speedEsts, ctx, score, recommended } = entry;
  const [library] = model.ollama_tag.split(':');
  const runTag    = `${library}:${variant.tag}`;

  const role    = lib.coding_role === 'agent' ? 'AGENT' : lib.coding_role === 'code' ? 'CODE' : 'FIM';
  const roleCls = lib.coding_role === 'agent' ? 'badge-agent' : lib.coding_role === 'code' ? 'badge-code' : 'badge-fim';
  const roleTip = lib.coding_role === 'agent'
    ? 'Purpose-built for tool-calling agent loops (Cline, Continue, Aider). Multi-step planning, file edits, shell commands.'
    : lib.coding_role === 'code'
    ? 'Code chat & generation — excellent for code explanation, review, and generation. Use with Continue for IDE coding assistance; not designed for autonomous agent loops.'
    : 'Fill-in-the-middle autocomplete — uses FIM tokens for single-cursor completion in IDEs. Configure as tabAutocompleteModel in Continue; not for chat or agent use.';

  const flag      = flagFor(lib.origin);
  const speedText = speedEsts ? fmtSpeed(speedEsts.genLo, speedEsts.genHi) : '—';
  const ctxText   = fmtCtxCoding(ctx.maxCtx);
  const ctxTip    = `${ctx.maxCtx.toLocaleString()} tokens · ~${Math.round(ctx.maxCtx / 3).toLocaleString()} lines · ~${Math.round(ctx.maxCtx / 1000)} avg files`;
  const barPct    = Math.round(Math.max(0, score) * 100);

  const row = document.createElement('div');
  row.className = 'coder-row' + (recommended ? ' coder-row-recommended' : '');

  const recTag = recommended
    ? '<span class="rec-tag" data-tip="Top-ranked agent model for this GPU — the recommended starting point.">★ recommended</span>'
    : '';

  // FIM models get a different config panel (tabAutocomplete format).
  const configHtml = lib.coding_role === 'fim'
    ? fimConfigHtml(entry)
    : makeConfigHtml(entry);

  const editorLink  = `<a class="prereq-link" href="https://code.visualstudio.com" target="_blank" rel="noopener">VS Code</a> or <a class="prereq-link" href="https://cursor.com/download" target="_blank" rel="noopener">Cursor</a>`;
  const ollamaLink  = `<a class="prereq-link" href="https://ollama.com" target="_blank" rel="noopener">Ollama</a>`;
  const extLinks = lib.coding_role === 'agent'
    ? `<a class="prereq-link" href="https://marketplace.visualstudio.com/items?itemName=saoudrizwan.claude-dev" target="_blank" rel="noopener">Cline</a> or <a class="prereq-link" href="https://marketplace.visualstudio.com/items?itemName=Continue.continue" target="_blank" rel="noopener">Continue</a> extension`
    : `<a class="prereq-link" href="https://marketplace.visualstudio.com/items?itemName=Continue.continue" target="_blank" rel="noopener">Continue</a> extension`;
  const prereqStrip = `<div class="prereq-strip"><span class="prereq-step">① Get an editor — ${editorLink}</span><span class="prereq-sep">·</span><span class="prereq-step">② Install ${ollamaLink}</span><span class="prereq-sep">·</span><span class="prereq-step">③ Add the ${extLinks} — open Extensions in VS Code (<kbd>Ctrl+Shift+X</kbd>), search the name, click Install</span></div>`;

  row.innerHTML = `
    <div class="coder-row-header">
      <span class="coder-badge ${roleCls}" data-tip="${roleTip}">${role}</span>
      ${recTag}
      <span class="coder-flag" data-tip="${esc(lib.organization || '')}${lib.origin ? ' · ' + esc(lib.origin) : ''}">${flag}</span>
      <span class="coder-name">${esc(runTag)}</span>
      <span class="coder-speed" data-tip="Est. generation speed (lower bound). Speed matters most in agentic loops — tool calls chain sequentially.">${speedText}</span>
      <span class="coder-ctx" data-tip="${ctxTip}">${ctxText}</span>
      <div class="coder-score-bar" data-tip="Coding rank: 50% speed + 30% context + 20% quality"><div class="coder-score-fill" style="width:${barPct}%"></div></div>
    </div>
    <div class="coder-config" hidden>
      ${prereqStrip}
      ${configHtml}
    </div>
  `;

  // Toggle config panel on header click
  row.querySelector('.coder-row-header').addEventListener('click', () => {
    const config  = row.querySelector('.coder-config');
    const wasOpen = !config.hidden;
    document.querySelectorAll('.coder-config').forEach(c => c.hidden = true);
    document.querySelectorAll('.coder-row').forEach(r => r.classList.remove('open'));
    if (!wasOpen) { config.hidden = false; row.classList.add('open'); }
  });

  // KV OS tab switching (Linux / macOS / Windows) — scoped to the containing config-section
  row.querySelectorAll('.kv-os-tab').forEach(btn => {
    btn.addEventListener('click', e => {
      e.stopPropagation();
      const kvos    = btn.dataset.kvos;
      const section = btn.closest('.config-section');
      section.querySelectorAll('.kv-os-tab').forEach(b => b.classList.toggle('active', b === btn));
      section.querySelectorAll('.kv-os-block').forEach(b => b.hidden = b.dataset.kvos !== kvos);
    });
  });

  // Mode tab switching (Quick start / Optimized)
  row.querySelectorAll('.mode-tab').forEach(btn => {
    btn.addEventListener('click', e => {
      e.stopPropagation();
      const mode = btn.dataset.mode;
      row.querySelectorAll('.mode-tab').forEach(b => b.classList.toggle('active', b === btn));
      row.querySelectorAll('.mode-block').forEach(b => b.hidden = b.dataset.mode !== mode);
    });
  });

  // Client tab switching (Cline / Continue) — scoped to active mode block
  row.querySelectorAll('.client-tab').forEach(btn => {
    btn.addEventListener('click', e => {
      e.stopPropagation();
      const client  = btn.dataset.client;
      const section = btn.closest('.config-section');
      section.querySelectorAll('.client-tab').forEach(b => b.classList.toggle('active', b === btn));
      section.querySelectorAll('.client-block').forEach(b => b.hidden = b.dataset.client !== client);
    });
  });

  // Copy buttons — scope to .kv-cmd if present, else .config-section
  row.querySelectorAll('.copy-btn').forEach(btn => {
    btn.addEventListener('click', e => {
      e.stopPropagation();
      const scope = btn.closest('.kv-cmd') ?? btn.closest('.config-section');
      const pre   = scope?.querySelector('pre');
      if (!pre) return;
      navigator.clipboard.writeText(pre.textContent.trim()).then(() => {
        const orig = btn.textContent;
        btn.textContent = 'copied!';
        setTimeout(() => { btn.textContent = orig; }, 1500);
      });
    });
  });

  return row;
}

// ── List render ───────────────────────────────────────────────────────────────

function sectionDivider(text) {
  const d = document.createElement('div');
  d.className = 'fim-divider';
  d.innerHTML = `<span>${text}</span>`;
  return d;
}

function renderList(vramGB) {
  const { agent, code, fim } = buildEntries(vramGB, getFlashOk());
  const list = document.getElementById('coderList');
  list.innerHTML = '';

  if (!agent.length && !code.length && !fim.length) {
    list.innerHTML = '<div class="no-model">No coding models fit this GPU. Try a larger card.</div>';
    return;
  }

  if (agent.length) {
    agent[0].recommended = true;   // top-ranked agent = recommended starting point
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
  const vramGB = parseFloat(document.getElementById('vramInput').value);
  const noGpu  = document.getElementById('noGpu');
  const list   = document.getElementById('coderList');
  if (isNaN(vramGB) || vramGB <= 0) {
    noGpu.hidden = false; list.hidden = true; return;
  }
  noGpu.hidden = true; list.hidden = false;
  renderList(vramGB);
}

function init() {
  buildGpuSelector();
  document.getElementById('vramInput').addEventListener('change', render);
  document.getElementById('baseUrlInput').addEventListener('change', render);

  // Tooltip — same pattern as index.html
  const tip = document.getElementById('tooltip');
  document.addEventListener('mouseover', e => {
    const el = e.target.closest('[data-tip]');
    if (!el || !el.dataset.tip) { tip.hidden = true; return; }
    tip.textContent = el.dataset.tip;
    tip.hidden = false;
    const rect = el.getBoundingClientRect();
    tip.style.top  = (rect.bottom + 8) + 'px';
    tip.style.left = Math.min(rect.left, window.innerWidth - 276) + 'px';
  });
  document.addEventListener('mouseout', e => {
    if (!e.target.closest('[data-tip]')) tip.hidden = true;
  });

  render();
}

init();
