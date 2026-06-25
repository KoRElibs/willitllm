// ─── INDEX.COMBOBOX — searchable model combobox state machine
//
// Depends on:  MODELS (data files), app.util.js (LIB_META),
//              data.flags.js (flagFor), index.ui.js (_activeCaps),
//              index.js (render — called at runtime only)
// Provides:    buildModelCombobox, openCombobox, closeCombobox,
//              filterModelList, syncComboboxFace, markComboboxItems

let _comboOpen = false;
let _comboHighlight = null;

function openCombobox() {
  const panel = document.getElementById('modelPanel');
  const face  = document.getElementById('modelFace');
  const input = document.getElementById('modelSearch');
  _comboOpen = true;
  face.classList.add('open');
  panel.hidden = false;
  input.value = '';
  filterModelList('');
  const selectedItem = document.querySelector('#modelList .combobox-item.selected');
  if (selectedItem) selectedItem.scrollIntoView({ block: 'nearest' });
  if (!window.matchMedia('(pointer: coarse)').matches) input.focus();
}

function closeCombobox() {
  _comboOpen = false;
  _comboHighlight = null;
  document.getElementById('modelFace').classList.remove('open');
  document.getElementById('modelPanel').hidden = true;
}

function comboHighlight(el) {
  if (_comboHighlight) _comboHighlight.classList.remove('highlighted');
  _comboHighlight = el || null;
  if (_comboHighlight) {
    _comboHighlight.classList.add('highlighted');
    _comboHighlight.scrollIntoView({ block: 'nearest' });
  }
}

function filterModelList(query, autoSelect = false) {
  const list = document.getElementById('modelList');
  const q    = query.toLowerCase().trim();
  let firstVisible = null;
  list.querySelectorAll('.combobox-item').forEach(item => {
    const textMatch = !q || item.dataset.tag.includes(q);
    const itemCaps  = new Set((item.dataset.caps || '').split(','));
    const capMatch  = _activeCaps.size === 0 || [..._activeCaps].every(c => itemCaps.has(c));
    const show = textMatch && capMatch;
    item.hidden = !show;
    if (show && !firstVisible) firstVisible = item;
  });
  comboHighlight(firstVisible);
  if (autoSelect) {
    const sel        = document.getElementById('modelSelect');
    const currentIdx = sel?.value;
    const currentItem = currentIdx !== '' && currentIdx !== undefined
      ? list.querySelector(`.combobox-item[data-idx="${currentIdx}"]`)
      : null;
    const currentFits = currentItem && !currentItem.hidden && parseInt(currentItem.dataset.fit) < 4;
    if (!currentFits) {
      const items    = Array.from(list.querySelectorAll('.combobox-item'));
      const firstFit = items.find(el => !el.hidden && parseInt(el.dataset.fit) < 4);
      const newVal   = firstFit ? firstFit.dataset.idx : '';
      if (sel.value !== newVal) { sel.value = newVal; sel.dispatchEvent(new Event('change')); }
    }
  }
}

function selectComboboxModel(idx) {
  const sel = document.getElementById('modelSelect');
  sel.value = idx;
  closeCombobox();
  syncComboboxFace();
  sel.dispatchEvent(new Event('change'));
}

function syncComboboxFace() {
  const sel      = document.getElementById('modelSelect');
  const faceText = document.getElementById('modelFaceText');
  const list     = document.getElementById('modelList');
  const modelIdx = parseInt(sel.value);
  const model    = MODELS[modelIdx];
  list && list.querySelectorAll('.combobox-item.selected').forEach(el => el.classList.remove('selected'));
  if (!model) {
    faceText.textContent = 'Select a model...';
    faceText.style.color = '';
    return;
  }
  const item = list && list.querySelector(`.combobox-item[data-idx="${modelIdx}"]`);
  if (item) {
    item.classList.add('selected');
    faceText.style.color = item.style.color || '';
    faceText.textContent = item.dataset.label;
  } else {
    faceText.textContent = model.ollama_tag;
  }
}

function markComboboxItems(vramGB, targetCtx, flashOk) {
  const list = document.getElementById('modelList');
  if (!list) return;
  list.querySelectorAll('.combobox-item').forEach(item => {
    const m = MODELS[parseInt(item.dataset.idx)];
    if (!m) return;
    const { color, fit } = modelScoreColor(m, vramGB, targetCtx, flashOk);
    item.style.color = color;
    item.dataset.fit = fit;
    item.textContent = fit === 4 ? '✗  ' + m.ollama_tag : item.dataset.label;
  });
  const items = Array.from(list.querySelectorAll('.combobox-item'));
  items.sort((a, b) => {
    const pa = parseInt(a.dataset.fit ?? 4);
    const pb = parseInt(b.dataset.fit ?? 4);
    return pa !== pb ? pa - pb : a.dataset.tag.localeCompare(b.dataset.tag);
  });
  items.forEach(item => list.appendChild(item));
  const sel         = document.getElementById('modelSelect');
  const currentIdx  = sel?.value;
  const currentItem = currentIdx !== '' && currentIdx !== undefined
    ? list.querySelector(`.combobox-item[data-idx="${currentIdx}"]`)
    : null;
  const currentFits = currentItem && parseInt(currentItem.dataset.fit) < 4 && !currentItem.hidden;
  if (!currentFits) {
    const firstFit = items.find(el => !el.hidden && parseInt(el.dataset.fit) < 4);
    const newVal   = firstFit ? firstFit.dataset.idx : '';
    if (sel.value !== newVal) { sel.value = newVal; sel.dispatchEvent(new Event('change')); return; }
  }
  syncComboboxFace();
}

function buildModelCombobox() {
  const list = document.getElementById('modelList');
  list.innerHTML = '';
  MODELS.forEach((m, i) => {
    const [library] = m.ollama_tag.split(':');
    const info  = LIB_META[library];
    const caps  = [...(info?.capabilities || [])];
    if (caps.includes('embedding')) return;   // embedding models are not for chat — hide entirely
    if (info?.coding_role) caps.push('coding'); // synthetic: any coding_role (agent/code/fim)
    const flag  = flagFor(info?.origin);
    const label = `${flag} ${m.ollama_tag}`;
    const item  = document.createElement('div');
    item.className     = 'combobox-item';
    item.dataset.idx   = i;
    item.dataset.tag   = m.ollama_tag.toLowerCase();
    item.dataset.label = label;
    item.dataset.caps  = caps.join(',');
    item.textContent   = label;
    item.addEventListener('mousedown', e => { e.preventDefault(); selectComboboxModel(i); });
    list.appendChild(item);
  });

  const face  = document.getElementById('modelFace');
  const input = document.getElementById('modelSearch');
  const wrap  = document.getElementById('modelComboWrap');

  face.addEventListener('click', () => { _comboOpen ? closeCombobox() : openCombobox(); });
  face.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ' || e.key === 'ArrowDown') { e.preventDefault(); openCombobox(); }
  });
  input.addEventListener('input', () => filterModelList(input.value));
  input.addEventListener('keydown', e => {
    if (e.key === 'Escape') { e.preventDefault(); closeCombobox(); face.focus(); return; }
    const items = Array.from(list.querySelectorAll('.combobox-item:not([hidden])'));
    if (!items.length) return;
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      const cur = _comboHighlight ? items.indexOf(_comboHighlight) : -1;
      comboHighlight(e.key === 'ArrowDown' ? items[Math.min(cur + 1, items.length - 1)] : items[Math.max(cur - 1, 0)]);
    }
    if (e.key === 'Enter' && _comboHighlight) selectComboboxModel(parseInt(_comboHighlight.dataset.idx));
  });
  document.addEventListener('click', e => {
    if (_comboOpen && !wrap.contains(e.target)) closeCombobox();
  });
}
