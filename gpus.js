// ─────────────────────────────────────────────────────────────────────────────
// GPU DATABASE
//
// Each entry represents one selectable GPU tier in the VRAM dropdown.
//
// Fields:
//   vram    {number}  — VRAM in GB
//   flash   {string}  — Flash Attention support: 'yes' | 'no' | 'mixed'
//                       'yes'   = NVIDIA Ampere (sm_80+) or newer — fully supported
//                       'no'    = NVIDIA Turing (sm_75) or older   — not supported
//                       'mixed' = AMD / other — support uncertain, varies by build
//   names   {string[]}— GPU model names shown in the dropdown label
//   vendor  {string?} — Optional vendor tag appended in parentheses, e.g. 'AMD'
//   default {bool?}   — Pre-selected entry on page load
//
// Flash Attention in ollama requires NVIDIA Ampere (compute capability ≥ 8.0).
// Turing (RTX 20xx, GTX 16xx) is sm_75 and does NOT support it.
// AMD support is reported as working in some builds but is not officially
// guaranteed — mark as 'mixed' and surface a warning to the user.
// ─────────────────────────────────────────────────────────────────────────────

const GPUS = [
  { vram: 4,   flash: 'no',    names: ['GTX 1650', 'GTX 1050 Ti'] },
  { vram: 4,   flash: 'yes',   names: ['RTX 3050'] },
  { vram: 6,   flash: 'no',    names: ['GTX 1060', 'RTX 2060'] },
  { vram: 8,   flash: 'no',    names: ['RTX 2070', 'RTX 2080'] },
  { vram: 8,   flash: 'yes',   names: ['RTX 3070', 'RTX 3070 Ti', 'RTX 4060 Ti'] },
  { vram: 10,  flash: 'yes',   names: ['RTX 3080'] },
  { vram: 11,  flash: 'no',    names: ['RTX 2080 Ti'] },
  { vram: 12,  flash: 'yes',   names: ['RTX 3060', 'RTX 4070', 'RTX 4070 Ti'] },
  { vram: 16,  flash: 'yes',   names: ['RTX 4070 Ti Super', 'RTX A4000'] },
  { vram: 16,  flash: 'mixed', names: ['RX 7900 GRE'],  vendor: 'AMD' },
  { vram: 20,  flash: 'mixed', names: ['RX 7900 XT'],   vendor: 'AMD' },
  { vram: 24,  flash: 'yes',   names: ['RTX 3090', 'RTX 4090'], default: true },
  { vram: 24,  flash: 'mixed', names: ['RX 7900 XTX'],  vendor: 'AMD' },
  { vram: 32,  flash: 'yes',   names: ['RTX 5090', 'RTX 5000 Ada'] },
  { vram: 40,  flash: 'yes',   names: ['A100 40G'] },
  { vram: 48,  flash: 'yes',   names: ['RTX 6000 Ada', 'L40S', 'A40'] },
  { vram: 80,  flash: 'yes',   names: ['A100 80G', 'H100'] },
  { vram: 96,  flash: 'yes',   names: ['H100 NVL'] },
  { vram: 192, flash: 'yes',   names: ['2× H100'] },
];
