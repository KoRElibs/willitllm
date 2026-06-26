// ─── APP.UTIL — shared domain helpers
//
// Depends on:  LIBRARIES (data.libraries.js), DOM (#vramInput)
// Provides:    LIB_META, getLibMeta, getFlashOk,
//              METRIC_LABELS, metricLabel, metricLabelShort

const LIB_META = Object.fromEntries(LIBRARIES.map(l => [l.library, l]));

function getLibMeta(m) {
  return LIB_META[m.ollama_tag.split(':')[0]] || {};
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
