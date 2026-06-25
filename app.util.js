// ─── APP.UTIL — shared domain helpers
//
// Depends on:  LIBRARIES (data.libraries.js), DOM (#vramInput)
// Provides:    LIB_META, getLibMeta, getFlashOk

const LIB_META = Object.fromEntries(LIBRARIES.map(l => [l.library, l]));

function getLibMeta(m) {
  return LIB_META[m.ollama_tag.split(':')[0]] || {};
}

function getFlashOk() {
  const opt = document.getElementById('vramInput').selectedOptions[0];
  return opt?.dataset.flash === 'yes' || opt?.dataset.flash === 'mixed';
}
