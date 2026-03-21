// ─────────────────────────────────────────────────────────────────────────────
// KV CACHE OPTIONS
//
// The three quantization levels ollama supports for the KV cache.
// Lower precision uses less VRAM per token, allowing a longer context window
// at the cost of some recall accuracy.
//
// ── FIELDS ───────────────────────────────────────────────────────────────────
//
// bytesPerElement  Bytes per KV element. Matches the <select> option values in
//                  index.html. Used directly in the KV cache VRAM formula.
//
// label            OLLAMA_KV_CACHE_TYPE environment variable value.
//
// summary          One-line description shown in the scorecard tooltip.
//
// ─────────────────────────────────────────────────────────────────────────────

const KV_CACHE = [
  { bytesPerElement: 2,   label: 'f16',  summary: 'full precision — works on every GPU, no setup needed.' },
  { bytesPerElement: 1,   label: 'q8_0', summary: 'half the memory per token — slight precision loss, fits more context in VRAM.' },
  { bytesPerElement: 0.5, label: 'q4_0', summary: 'quarter the memory per token — more precision loss, maximum context for the VRAM.' },
];
