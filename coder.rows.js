// ─── CODER.ROWS — HTML generation for coder rows and config panels
//
// Depends on:  app.shared.js (osKvContent),
//              app.fmt.js (fmtSpeed),
//              data.flags.js (flagFor),
//              coder.rank.js (fmtCtxCoding),
//              coder.js (getBaseUrl — at runtime)
// Provides:    esc, sectionDivider, kvSetupHtml, modeHtml,
//              fimConfigHtml, makeConfigHtml, makeRow

function esc(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function sectionDivider(text, extraClass) {
  const d = document.createElement('div');
  d.className = 'section-divider' + (extraClass ? ' ' + extraClass : '');
  d.innerHTML = `<span>${text}</span>`;
  return d;
}

function kvSetupHtml(kvLabel) {
  const initOs = localStorage.getItem('osTab') || 'generic';
  const tabs = ['generic', 'linux', 'linux-service', 'macos', 'windows'];
  const tabLabels = { generic: 'Generic', linux: 'Linux', 'linux-service': 'Linux service', macos: 'macOS', windows: 'Windows' };
  const tabHtml = tabs.map(os =>
    `<button class="os-tab${os === initOs ? ' active' : ''}" data-os="${os}">${tabLabels[os]}</button>`
  ).join('\n        ');
  return `
    <div class="config-section">
      <div class="config-label">1. Start Ollama with KV cache type (${kvLabel}) <button class="copy-btn">copy</button></div>
      <div class="os-tabs">
        ${tabHtml}
      </div>
      <pre class="ollama-cmd ollama-setup" data-kv="${kvLabel}">${osKvContent(initOs, kvLabel)}</pre>
    </div>`;
}

function modeHtml(runTag, maxCtx, kvLabel, baseUrl, showCline) {
  const ollamaRun   = `ollama pull ${runTag}`;
  const clineFields = [
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

  const kvSection = kvSetupHtml(kvLabel);
  const runLabel  = '2. Pull model';

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
    return `
      ${kvSetupHtml(kv)}
      <div class="config-section">
        <div class="config-label">2. Pull model <button class="copy-btn">copy</button></div>
        <pre class="ollama-cmd">${esc(ollamaRun)}</pre>
      </div>
      <div class="config-section">
        <div class="config-label"><span>Continue config — set as <code>tabAutocompleteModel</code> <a class="doc-link" href="https://docs.continue.dev" target="_blank" rel="noopener">docs ↗</a></span><button class="copy-btn">copy</button></div>
        <pre class="client-config">${esc(tabJson)}</pre>
        <div class="config-note">Open Continue config: Continue sidebar → ⚙️ → <em>Open Config</em>. Paste this as a top-level key (not inside the <code>models</code> array).</div>
      </div>
    `;
  }

  if (!hasOptimized) return note + fimMode(ctx.maxCtx, kvType || 'f16');

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
      ${fimMode(f16Ctx, 'f16')}
    </div>
    <div class="mode-block" data-mode="optimized" hidden>
      ${fimMode(ctx.maxCtx, kvType)}
    </div>
  `;
}

function makeConfigHtml(entry) {
  const { model, variant, ctx, ctxF16, bpe, lib } = entry;
  const [library] = model.ollama_tag.split(':');
  const runTag    = `${library}:${variant.tag}`;
  const kvType    = bpe === 1 ? 'q8_0' : bpe === 0.5 ? 'q4_0' : null;
  const f16Ctx    = ctxF16?.maxCtx ?? ctx.maxCtx;
  const hasOptimized = kvType !== null && ctx.maxCtx > f16Ctx;

  const showCline = lib.coding_role === 'agent';
  const baseUrl   = getBaseUrl();

  if (!hasOptimized) return modeHtml(runTag, ctx.maxCtx, kvType || 'f16', baseUrl, showCline);

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
      ${modeHtml(runTag, f16Ctx, 'f16', baseUrl, showCline)}
    </div>
    <div class="mode-block" data-mode="optimized" hidden>
      ${modeHtml(runTag, ctx.maxCtx, kvType, baseUrl, showCline)}
    </div>
  `;
}

function makeRow(entry) {
  const { model, lib, variant, speedEsts, ctx, score, recommended } = entry;
  const [library] = model.ollama_tag.split(':');
  const runTag    = `${library}:${variant.tag}`;

  const role    = lib.coding_role === 'agent' ? 'AGENT' : lib.coding_role === 'code' ? 'CODE' : 'FIM';
  const roleCls = lib.coding_role === 'agent' ? 'badge-agent' : lib.coding_role === 'code' ? 'badge-code' : 'badge-fim';
  const roleTip = lib.coding_role === 'agent'
    ? 'Purpose-trained for agentic coding loops — multi-step planning, file edits, shell commands, tool calling. The right choice for vibe coding with Cline or Continue.'
    : lib.coding_role === 'code'
    ? 'Code chat & generation — explanation, review, and generation. Great with Continue. Some support tool-calling but are not optimized for autonomous coding loops — use an AGENT model for that.'
    : 'Fill-in-the-middle autocomplete — uses FIM tokens for single-cursor completion in IDEs. Configure as tabAutocompleteModel in Continue; not for chat or autonomous agent use.';

  const flag      = flagFor(lib.origin);
  const speedText = speedEsts ? fmtSpeed(speedEsts.genLo, speedEsts.genHi) : '—';
  const ctxText   = fmtCtxCoding(ctx.maxCtx);
  const ctxTip    = `${ctx.maxCtx.toLocaleString()} tokens · ~${Math.round(ctx.maxCtx / 3).toLocaleString()} lines · ~${Math.round(ctx.maxCtx / 1000)} avg files`;
  const barPct    = Math.round(Math.max(0, score) * 100);

  const row = document.createElement('div');
  row.className = 'coder-row' + (recommended ? ' coder-row-recommended' : '');

  const recTag = recommended
    ? '<span class="rec-tag" data-tip="Top-ranked coding agent for this GPU — best starting point for vibe coding.">★ recommended</span>'
    : '';

  const configHtml = lib.coding_role === 'fim'
    ? fimConfigHtml(entry)
    : makeConfigHtml(entry);

  const editorLink = `<a class="prereq-link" href="https://code.visualstudio.com" target="_blank" rel="noopener">VS Code</a> or <a class="prereq-link" href="https://cursor.com/download" target="_blank" rel="noopener">Cursor</a>`;
  const ollamaLink = `<a class="prereq-link" href="https://ollama.com" target="_blank" rel="noopener">Ollama</a>`;
  const extLinks   = lib.coding_role === 'agent'
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

  row.querySelector('.coder-row-header').addEventListener('click', () => {
    const config  = row.querySelector('.coder-config');
    const wasOpen = !config.hidden;
    document.querySelectorAll('.coder-config').forEach(c => c.hidden = true);
    document.querySelectorAll('.coder-row').forEach(r => r.classList.remove('open'));
    if (!wasOpen) { config.hidden = false; row.classList.add('open'); }
  });

  row.querySelectorAll('.os-tab[data-os]').forEach(btn => {
    btn.addEventListener('click', e => {
      e.stopPropagation();
      const os      = btn.dataset.os;
      const section = btn.closest('.config-section');
      const pre     = section.querySelector('.ollama-setup');
      if (!pre) return;
      const kvLabel = pre.dataset.kv;
      localStorage.setItem('osTab', os);
      section.querySelectorAll('.os-tab[data-os]').forEach(b => b.classList.toggle('active', b === btn));
      pre.innerHTML = osKvContent(os, kvLabel);
    });
  });

  row.querySelectorAll('.mode-tab').forEach(btn => {
    btn.addEventListener('click', e => {
      e.stopPropagation();
      const mode = btn.dataset.mode;
      row.querySelectorAll('.mode-tab').forEach(b => b.classList.toggle('active', b === btn));
      row.querySelectorAll('.mode-block').forEach(b => b.hidden = b.dataset.mode !== mode);
    });
  });

  row.querySelectorAll('.client-tab').forEach(btn => {
    btn.addEventListener('click', e => {
      e.stopPropagation();
      const client  = btn.dataset.client;
      const section = btn.closest('.config-section');
      section.querySelectorAll('.client-tab').forEach(b => b.classList.toggle('active', b === btn));
      section.querySelectorAll('.client-block').forEach(b => b.hidden = b.dataset.client !== client);
    });
  });

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
