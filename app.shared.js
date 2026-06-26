// ─────────────────────────────────────────────────────────────────────────────
// APP.SHARED — utilities loaded by both index.html and coder.html
//
// Globals: GPUS (data.gpus.js)
// ─────────────────────────────────────────────────────────────────────────────

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
    GPUS.forEach((gpu, gpuIdx) => {
      gpu.names.forEach(name => {
        if (match(gpu, name)) cards.push({ name, vram: gpu.vram, flash: gpu.flash, gpuIdx });
      });
    });
    if (!cards.length) return;
    cards.sort((a, b) => a.name.localeCompare(b.name));
    const group = document.createElement('optgroup');
    group.label = label;
    cards.forEach(({ name, vram, flash, gpuIdx }) => {
      const opt = document.createElement('option');
      opt.value = vram; opt.dataset.flash = flash; opt.dataset.gpuIdx = gpuIdx;
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

function initTooltip() {
  const tip  = document.getElementById('tooltip');
  const hide = () => { tip.hidden = true; };
  const show = el => {
    tip.textContent = el.dataset.tip;
    tip.hidden = false;
    const rect = el.getBoundingClientRect();
    tip.style.top  = (rect.bottom + 8) + 'px';
    tip.style.left = Math.min(rect.left, window.innerWidth - 276) + 'px';
  };

  // Desktop: follow the pointer on hover.
  document.addEventListener('mouseover', e => {
    const el = e.target.closest('[data-tip]');
    el && el.dataset.tip ? show(el) : hide();
  });
  document.addEventListener('mouseout', e => {
    if (!e.target.closest('[data-tip]')) hide();
  });

  // Touch (no hover): tap an info element to reveal its tip — and swallow that
  // tap so it doesn't also toggle the row it sits in. Interactive controls
  // (buttons/links) keep their normal action; tapping elsewhere dismisses.
  // Capture phase so stopPropagation beats the row's own click handler.
  document.addEventListener('click', e => {
    if (!matchMedia('(hover: none)').matches) return;
    const el = e.target.closest('[data-tip]');
    if (el && el.dataset.tip && !el.closest('button, a')) { e.stopPropagation(); show(el); }
    else hide();
  }, true);
}

function initInfoSheet() {
  const openBtn  = document.getElementById('infoSheetOpen');
  const closeBtn = document.getElementById('infoSheetClose');
  const backdrop = document.getElementById('infoSheetBackdrop');
  const sheet    = document.getElementById('infoSheet');
  if (!openBtn || !sheet) return;
  const open  = () => { sheet.hidden = false; backdrop.hidden = false; };
  const close = () => { sheet.hidden = true;  backdrop.hidden = true; };
  openBtn.addEventListener('click', open);
  closeBtn.addEventListener('click', close);
  backdrop.addEventListener('click', close);
  document.addEventListener('keydown', e => { if (e.key === 'Escape') close(); });
}

const muted = s => `<span class="cmd-muted">${s}</span>`;

function osKvContent(tab, kvLabel) {
  if (tab === 'generic') return [
    muted(`# KV cache type: ${kvLabel} (session only — resets on restart):`),
    `OLLAMA_KV_CACHE_TYPE=${kvLabel} ollama serve`,
  ].join('\n');
  if (tab === 'linux') return [
    muted(`# KV cache type: ${kvLabel} (permanent — adds to ~/.bashrc):`),
    `echo 'export OLLAMA_KV_CACHE_TYPE=${kvLabel}' >> ~/.bashrc && source ~/.bashrc`,
    muted('# then start Ollama:'),
    'ollama serve',
  ].join('\n');
  if (tab === 'linux-service') return [
    muted(`# KV cache type: ${kvLabel} (permanent — systemd override):`),
    `sudo mkdir -p /etc/systemd/system/ollama.service.d && printf '[Service]\\nEnvironment="OLLAMA_KV_CACHE_TYPE=${kvLabel}"\\n' | sudo tee /etc/systemd/system/ollama.service.d/override.conf`,
    muted('# reload and restart Ollama service:'),
    'sudo systemctl daemon-reload && sudo systemctl restart ollama',
  ].join('\n');
  if (tab === 'macos') return [
    muted(`# KV cache type: ${kvLabel} — quit Ollama from menu bar (⌘Q), then:`),
    `echo 'export OLLAMA_KV_CACHE_TYPE=${kvLabel}' >> ~/.zshrc && source ~/.zshrc`,
    'ollama serve',
  ].join('\n');
  if (tab === 'windows') return [
    muted(`# KV cache type: ${kvLabel} (permanent — System Properties):`),
    muted('# 1. System Properties → Environment Variables → New user variable:'),
    muted(`#    Name:  OLLAMA_KV_CACHE_TYPE`),
    muted(`#    Value: ${kvLabel}`),
    muted('# 2. right-click Ollama in tray → Quit, then reopen Ollama'),
  ].join('\n');
  return '';
}
