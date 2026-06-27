// ─── APP.UTIL — shared domain helpers
//
// Depends on:  LIBRARIES (data.libraries.js), DOM (#vramInput)
// Provides:    LIB_META, getLibMeta, getFlashOk, modelParamSize,
//              METRIC_LABELS, metricLabel, metricLabelShort

const LIB_META = Object.fromEntries(LIBRARIES.map(l => [l.library, l]));

function getLibMeta(m) {
  return LIB_META[m.ollama_tag.split(':')[0]] || {};
}

// Parameter count parsed from the ollama tag size suffix — "gemma3:270m" → 2.7e8,
// "codellama:13b" → 1.3e10, "qwen:0.5b" → 5e8. The capability proxy used to order
// models "most capable first" (bigger ≈ more capable). Unsized tags (e.g. ":latest")
// return 0 so they sort last. Single source of truth for size ordering.
function modelParamSize(model) {
  const m = (model.ollama_tag.split(':')[1] || '').match(/(\d+(?:\.\d+)?)\s*([bm])/i);
  return m ? parseFloat(m[1]) * (m[2].toLowerCase() === 'm' ? 1e6 : 1e9) : 0;
}

// Capability-benchmark display names, keyed by the metric ids in data.libraries.js.
// Single source of truth — coder ranking/rows and the index scorecard all read it.
const METRIC_LABELS = {
  'swe-bench-verified': { full: 'SWE-bench Verified', short: 'SWE-bench' },
  'humaneval':          { full: 'HumanEval',          short: 'HumanEval' },
  'humaneval-plus':     { full: 'HumanEval+',         short: 'HumanEval+' },
  'mmlu':               { full: 'MMLU',               short: 'MMLU' },
};
function metricLabel(metric)      { return METRIC_LABELS[metric]?.full || metric || 'benchmark'; }
function metricLabelShort(metric) { return METRIC_LABELS[metric]?.short || 'bench'; }

function getFlashOk() {
  const opt = document.getElementById('vramInput').selectedOptions[0];
  return opt?.dataset.flash === 'yes' || opt?.dataset.flash === 'mixed';
}
