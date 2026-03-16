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
// "series" notation is used when multiple SKUs share the same VRAM and Flash
// Attention support and differ only by suffix (Ti, Super, XT, etc.).
//
// Flash Attention in ollama requires NVIDIA Ampere (compute capability ≥ 8.0).
// Turing (RTX 20xx, GTX 16xx) is sm_75 and does NOT support it.
// AMD support is reported as working in some builds but is not officially
// guaranteed — mark as 'mixed' and surface a warning to the user.
// ─────────────────────────────────────────────────────────────────────────────

const GPUS = [
  // ── 4 GB ──────────────────────────────────────────────────────────────────
  { vram: 4,   flash: 'no',    names: ['GTX 1050 Ti', 'GTX 1650'] },

  // ── 6 GB ──────────────────────────────────────────────────────────────────
  { vram: 6,   flash: 'no',    names: ['GTX 1060', 'GTX 1660 series', 'RTX 2060'] },

  // ── 8 GB ──────────────────────────────────────────────────────────────────
  { vram: 8,   flash: 'no',    names: ['RTX 2070', 'RTX 2080'] },
  { vram: 8,   flash: 'yes',   names: ['RTX 3050', 'RTX 3060 Ti', 'RTX 3070 series', 'RTX 4060 series'] },

  // ── 10 GB ─────────────────────────────────────────────────────────────────
  { vram: 10,  flash: 'yes',   names: ['RTX 3080'] },

  // ── 11 GB ─────────────────────────────────────────────────────────────────
  { vram: 11,  flash: 'no',    names: ['RTX 2080 Ti'] },

  // ── 12 GB ─────────────────────────────────────────────────────────────────
  { vram: 12,  flash: 'yes',   names: ['RTX 3060', 'RTX 3080 Ti', 'RTX 4070 series', 'RTX 5070'] },
  { vram: 12,  flash: 'mixed', names: ['RX 6700 XT', 'RX 7700 XT'], vendor: 'AMD' },

  // ── 16 GB ─────────────────────────────────────────────────────────────────
  { vram: 16,  flash: 'yes',   names: ['RTX 4060 Ti 16G', 'RTX 4070 Ti Super', 'RTX 4080 series', 'RTX 5070 Ti', 'RTX 5080', 'RTX A4000'] },
  { vram: 16,  flash: 'mixed', names: ['RX 6800 series', 'RX 6900 / 6950 XT', 'RX 7800 XT', 'RX 7900 GRE'], vendor: 'AMD' },

  // ── 20 GB ─────────────────────────────────────────────────────────────────
  { vram: 20,  flash: 'mixed', names: ['RX 7900 XT'], vendor: 'AMD' },

  // ── 24 GB ─────────────────────────────────────────────────────────────────
  { vram: 24,  flash: 'yes',   names: ['RTX 3090 series', 'RTX 4090', 'RTX A5000'], default: true },
  { vram: 24,  flash: 'mixed', names: ['RX 7900 XTX'], vendor: 'AMD' },

  // ── 32 GB ─────────────────────────────────────────────────────────────────
  { vram: 32,  flash: 'yes',   names: ['RTX 5090', 'RTX 5000 Ada'] },

  // ── Data centre / workstation ─────────────────────────────────────────────
  { vram: 40,  flash: 'yes',   names: ['A100 40G'] },
  { vram: 48,  flash: 'yes',   names: ['RTX 6000 Ada', 'L40S', 'A40'] },
  { vram: 80,  flash: 'yes',   names: ['A100 80G', 'H100'] },
  { vram: 96,  flash: 'yes',   names: ['H100 NVL'] },
  { vram: 192, flash: 'yes',   names: ['2× H100'] },
];
