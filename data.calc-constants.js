// ─────────────────────────────────────────────────────────────────────────────
// CALCULATION CONSTANTS
//
// Single source of truth for formula constants shared between the browser and
// Python tooling (benchmark.py). Change a value here; it takes effect everywhere.
//
// Dual-use file:
//   1. Browser     — loaded as <script src="data.calc-constants.js">, exposes
//                    CALC_CONSTANTS global. Must be loaded before app.calc.js.
//   2. Python      — strips "const CALC_CONSTANTS =" wrapper, parses as JSON.
//
// Keys are quoted so the object is valid JSON after stripping the JS wrapper.
//
// ── FIELDS ───────────────────────────────────────────────────────────────────
//
// overhead_gb      Fixed VRAM reservation: CUDA context, driver overhead, ollama
//                  runtime, and driver-reserved VRAM (rated capacity overstates
//                  addressable by ~4–6%). Raised 0.5 → 0.8 after a GTX 1660 Super
//                  spilled at a context the 0.5 GB value predicted would fit.
//                  See meta/benchmarks/README.md.
//
// safety_factor    Margin applied before rounding. Keeps recommended num_ctx safely
//                  inside the VRAM budget given that overhead_gb is a rough estimate.
//
// ctx_round        Round max context down to nearest N tokens. 128 is the natural
//                  head-dimension granularity for most transformer architectures;
//                  wastes <1% of available context.
//
// decode_attn_eff  Fraction of fp16 TFLOPS reached by the per-token attention GEMV
//                  during decode. Calibrated from RTX 3090 + GTX 1660 Super sweeps
//                  across f16/q8_0/q4_0 KV — see meta/benchmarks/README.md.
//                  Applied only on quantized KV or non-flash GPUs.
//
// ─────────────────────────────────────────────────────────────────────────────

const CALC_CONSTANTS = {
  "overhead_gb":      0.8,
  "safety_factor":    0.9,
  "ctx_round":        128,
  "decode_attn_eff":  0.015
};
