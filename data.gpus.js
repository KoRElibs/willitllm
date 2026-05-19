// ─────────────────────────────────────────────────────────────────────────────
// GPU DATABASE
//
// One entry per distinct GPU model. Generic tier entries (one per VRAM size)
// are generated automatically in app.js from these entries.
//
// ── DATA SOURCES ─────────────────────────────────────────────────────────────
//
//   NVIDIA GeForce:          sources/nvidia-tflops-derived.md  (screenshots, 2026-05-19)
//   Primary (all vendors):   https://www.techpowerup.com/gpu-specs/
//   Laptop GPUs:             https://www.notebookcheck.net/Best-Laptops/GPU-Benchmark-List.html
//   Data centre / pro:       vendor spec sheets (NVIDIA, AMD product pages)
//
//   bandwidth   — "Memory Bandwidth" on TechPowerUp or NVIDIA compare page (GB/s)
//   tflops_fp16 — FP16 TFLOPS, dense (no sparsity). Derivation by generation:
//                 NVIDIA Ada (RTX 40xx):       AI TOPS ÷ 8   (Gen 4 tensor, INT8 sparse)
//                 NVIDIA Blackwell (RTX 50xx): AI TOPS ÷ 16  (Gen 5 tensor, FP4 sparse)
//                 NVIDIA Ampere/Turing RTX:    TechPowerUp "FP16 (half)"
//                 NVIDIA GTX 16xx:             2 × FP32 (no tensor cores, CUDA FP16)
//                 NVIDIA GTX 10xx:             = FP32 (no native FP16)
//                 Laptop GPUs: max TGP variant; real-world speed varies by laptop power limit.
//
// ── FIELDS ───────────────────────────────────────────────────────────────────
//
// vram         VRAM in GB.
//
// flash        Flash Attention support: 'yes' | 'no' | 'mixed'
//              'yes'   — NVIDIA Ampere (sm_80+) or newer, fully supported.
//              'no'    — NVIDIA Turing (sm_75) or older, not supported.
//              'mixed' — AMD / other, support varies by ollama build and ROCm version.
//
// bandwidth    Memory bandwidth in GB/s. Used for generation speed estimates.
//
// tflops_fp16  FP16 TFLOPS, dense (no sparsity). Used for prefill speed estimates.
//              Efficiency ranges in data.quantizations.js bridge peak → real-world.
//
// names        GPU model name(s) shown in the dropdown.
//
// vendor       Optional vendor tag, e.g. 'AMD'. Omit for NVIDIA.
//
// ─────────────────────────────────────────────────────────────────────────────

const GPUS = [
  // ── 4 GB ──────────────────────────────────────────────────────────────────
  { vram: 4,   flash: 'no',    bandwidth:  112, tflops_fp16:   1.9, names: ['GTX 1050 Ti'] },
  { vram: 4,   flash: 'no',    bandwidth:  128, tflops_fp16:   5.9, names: ['GTX 1650'] },
  { vram: 4,   flash: 'yes',   bandwidth:  192, tflops_fp16:   8.0, names: ['RTX 3050 Laptop'] },   // desktop is 8 GB

  // ── 6 GB ──────────────────────────────────────────────────────────────────
  { vram: 6,   flash: 'no',    bandwidth:  192, tflops_fp16:   3.9, names: ['GTX 1060'] },
  { vram: 6,   flash: 'no',    bandwidth:  192, tflops_fp16:  10.1, names: ['GTX 1660'] },
  { vram: 6,   flash: 'no',    bandwidth:  336, tflops_fp16:  10.1, names: ['GTX 1660 Super'] },
  { vram: 6,   flash: 'no',    bandwidth:  288, tflops_fp16:  10.9, names: ['GTX 1660 Ti'] },
  { vram: 6,   flash: 'no',    bandwidth:  336, tflops_fp16:  51.6, names: ['RTX 2060'] },          // Turing — cores×boost×16; sources/nvidia-tflops-derived.md
  { vram: 6,   flash: 'yes',   bandwidth:  336, tflops_fp16:  11.0, names: ['RTX 3060 Laptop'] },   // desktop is 12 GB
  { vram: 6,   flash: 'yes',   bandwidth:  192, tflops_fp16:  18.0, names: ['RTX 4050 Laptop'] },

  // ── 8 GB ──────────────────────────────────────────────────────────────────
  { vram: 8,   flash: 'no',    bandwidth:  448, tflops_fp16:  57.4, names: ['RTX 2060 Super'] },     // Turing — cores×boost×16; sources/nvidia-tflops-derived.md
  { vram: 8,   flash: 'no',    bandwidth:  448, tflops_fp16:  59.7, names: ['RTX 2070'] },          // Turing — cores×boost×16; sources/nvidia-tflops-derived.md
  { vram: 8,   flash: 'no',    bandwidth:  448, tflops_fp16:  72.4, names: ['RTX 2070 Super'] },    // Turing — cores×boost×16; sources/nvidia-tflops-derived.md
  { vram: 8,   flash: 'no',    bandwidth:  448, tflops_fp16:  81.8, names: ['RTX 2080'] },          // Turing — cores×boost×16; sources/nvidia-tflops-derived.md
  { vram: 8,   flash: 'no',    bandwidth:  496, tflops_fp16:  89.2, names: ['RTX 2080 Super'] },    // Turing — cores×boost×16; sources/nvidia-tflops-derived.md
  { vram: 8,   flash: 'yes',   bandwidth:  224, tflops_fp16:  18.0, names: ['RTX 3050 Desktop'] },  // laptop is 4 GB
  { vram: 8,   flash: 'yes',   bandwidth:  448, tflops_fp16:  32.7, names: ['RTX 3060 Ti'] },
  { vram: 8,   flash: 'yes',   bandwidth:  448, tflops_fp16:  40.0, names: ['RTX 3070'] },
  { vram: 8,   flash: 'yes',   bandwidth:  608, tflops_fp16:  43.1, names: ['RTX 3070 Ti'] },
  { vram: 8,   flash: 'yes',   bandwidth:  256, tflops_fp16:  30.0, names: ['RTX 3080 Laptop 8G'] }, // desktop is 10 GB
  { vram: 8,   flash: 'yes',   bandwidth:  272, tflops_fp16:  30.0, names: ['RTX 4060'] },
  { vram: 8,   flash: 'yes',   bandwidth:  288, tflops_fp16:  44.2, names: ['RTX 4060 Ti'] },
  { vram: 8,   flash: 'yes',   bandwidth:  256, tflops_fp16:  29.0, names: ['RTX 4070 Laptop'] },   // desktop is 12 GB
  { vram: 8,   flash: 'yes',   bandwidth:  384, tflops_fp16:  46.0, names: ['RTX 5070 Laptop'] },   // desktop is 12 GB — laptop TFLOPS approximate (not on desktop compare page)
  { vram: 8,   flash: 'yes',   bandwidth:  448, tflops_fp16:  47.4, names: ['RTX 5060 Ti'] },       // Blackwell — AI TOPS 759 ÷ 16; sources/nvidia-tflops-derived.md
  { vram: 8,   flash: 'yes',   bandwidth:  448, tflops_fp16:  38.4, names: ['RTX 5060'] },          // Blackwell — AI TOPS 614 ÷ 16; sources/nvidia-tflops-derived.md
  { vram: 8,   flash: 'yes',   bandwidth:  384, tflops_fp16:  12.5, names: ['RTX 5060 Laptop'] },   // Blackwell — 128-bit GDDR7; laptop TFLOPS approximate (not on desktop compare page)
  { vram: 8,   flash: 'yes',   bandwidth:  320, tflops_fp16:  26.3, names: ['RTX 5050'] },          // Blackwell — AI TOPS 421 ÷ 16; sources/nvidia-tflops-derived.md

  // ── 10 GB ─────────────────────────────────────────────────────────────────
  { vram: 10,  flash: 'yes',   bandwidth:  760, tflops_fp16:  59.6, names: ['RTX 3080 Desktop'] },  // laptop is 8 or 16 GB

  // ── 11 GB ─────────────────────────────────────────────────────────────────
  { vram: 11,  flash: 'no',    bandwidth:  616, tflops_fp16: 107.6, names: ['RTX 2080 Ti'] },

  // ── 12 GB ─────────────────────────────────────────────────────────────────
  { vram: 12,  flash: 'yes',   bandwidth:  360, tflops_fp16:  25.3, names: ['RTX 3060 Desktop'] },  // laptop is 6 GB
  { vram: 12,  flash: 'yes',   bandwidth:  912, tflops_fp16:  68.3, names: ['RTX 3080 Ti Desktop'] }, // laptop is 16 GB
  { vram: 12,  flash: 'yes',   bandwidth:  432, tflops_fp16:  49.0, names: ['RTX 4080 Laptop'] },   // desktop is 16 GB
  { vram: 12,  flash: 'yes',   bandwidth:  504, tflops_fp16:  58.5, names: ['RTX 4070 Desktop'] },  // laptop is 8 GB
  { vram: 12,  flash: 'yes',   bandwidth:  504, tflops_fp16:  70.0, names: ['RTX 4070 Super'] },
  { vram: 12,  flash: 'yes',   bandwidth:  504, tflops_fp16:  80.0, names: ['RTX 4070 Ti'] },
  { vram: 12,  flash: 'yes',   bandwidth:  672, tflops_fp16:  61.8, names: ['RTX 5070 Desktop'] },  // laptop is 8 GB — AI TOPS 988 ÷ 16; sources/nvidia-tflops-derived.md
  { vram: 12,  flash: 'mixed', bandwidth:  384, tflops_fp16:  26.4, names: ['RX 6700 XT'],  vendor: 'AMD' },
  { vram: 12,  flash: 'mixed', bandwidth:  432, tflops_fp16:  35.2, names: ['RX 7700 XT'],  vendor: 'AMD' },

  // ── 16 GB ─────────────────────────────────────────────────────────────────
  { vram: 16,  flash: 'yes',   bandwidth:  288, tflops_fp16:  44.2, names: ['RTX 4060 Ti 16G'] },
  { vram: 16,  flash: 'yes',   bandwidth:  448, tflops_fp16:  60.0, names: ['RTX 3080 Ti Laptop'] }, // desktop is 12 GB
  { vram: 16,  flash: 'yes',   bandwidth:  448, tflops_fp16:  38.0, names: ['RTX 3080 Laptop 16G'] }, // desktop is 10 GB
  { vram: 16,  flash: 'yes',   bandwidth:  576, tflops_fp16:  82.0, names: ['RTX 4090 Laptop'] },   // desktop is 24 GB
  { vram: 16,  flash: 'yes',   bandwidth:  672, tflops_fp16:  88.3, names: ['RTX 4070 Ti Super'] }, // AI TOPS 706 ÷ 8; sources/nvidia-tflops-derived.md
  { vram: 16,  flash: 'yes',   bandwidth:  717, tflops_fp16:  97.5, names: ['RTX 4080 Desktop'] },  // laptop is 12 GB
  { vram: 16,  flash: 'yes',   bandwidth:  736, tflops_fp16: 104.5, names: ['RTX 4080 Super'] },    // AI TOPS 836 ÷ 8; sources/nvidia-tflops-derived.md
  { vram: 16,  flash: 'yes',   bandwidth:  896, tflops_fp16:  87.9, names: ['RTX 5070 Ti'] },       // Blackwell — AI TOPS 1406 ÷ 16; sources/nvidia-tflops-derived.md
  { vram: 16,  flash: 'yes',   bandwidth:  960, tflops_fp16: 112.6, names: ['RTX 5080'] },          // Blackwell — AI TOPS 1801 ÷ 16; sources/nvidia-tflops-derived.md
  { vram: 16,  flash: 'yes',   bandwidth:  448, tflops_fp16:  47.4, names: ['RTX 5060 Ti 16G'] },   // Blackwell — AI TOPS 759 ÷ 16; sources/nvidia-tflops-derived.md
  { vram: 16,  flash: 'yes',   bandwidth:  448, tflops_fp16:  38.8, names: ['RTX A4000'] },
  { vram: 16,  flash: 'mixed', bandwidth:  512, tflops_fp16:  26.8, names: ['RX 6800'],             vendor: 'AMD' },
  { vram: 16,  flash: 'mixed', bandwidth:  512, tflops_fp16:  32.3, names: ['RX 6800 XT'],          vendor: 'AMD' },
  { vram: 16,  flash: 'mixed', bandwidth:  544, tflops_fp16:  46.2, names: ['RX 6900 / 6950 XT'],   vendor: 'AMD' },
  { vram: 16,  flash: 'mixed', bandwidth:  624, tflops_fp16:  37.3, names: ['RX 7800 XT'],          vendor: 'AMD' },
  { vram: 16,  flash: 'mixed', bandwidth:  576, tflops_fp16:  45.9, names: ['RX 7900 GRE'],         vendor: 'AMD' },
  { vram: 16,  flash: 'mixed', bandwidth:  640, tflops_fp16:  97.0, names: ['RX 9070 XT'],          vendor: 'AMD' }, // RDNA 4
  { vram: 16,  flash: 'mixed', bandwidth:  576, tflops_fp16:  72.0, names: ['RX 9070'],             vendor: 'AMD' }, // RDNA 4

  // ── 20 GB ─────────────────────────────────────────────────────────────────
  { vram: 20,  flash: 'mixed', bandwidth:  800, tflops_fp16:  52.4, names: ['RX 7900 XT'],  vendor: 'AMD' },

  // ── 24 GB ─────────────────────────────────────────────────────────────────
  { vram: 24,  flash: 'yes',   bandwidth: 1024, tflops_fp16: 150.0, names: ['RTX 5090 Laptop'] },   // desktop is 32 GB — approximate
  { vram: 24,  flash: 'yes',   bandwidth:  936, tflops_fp16:  71.0, names: ['RTX 3090'] },
  { vram: 24,  flash: 'yes',   bandwidth: 1008, tflops_fp16:  79.9, names: ['RTX 3090 Ti'] },
  { vram: 24,  flash: 'yes',   bandwidth: 1008, tflops_fp16: 165.2, names: ['RTX 4090 Desktop'], default: true }, // laptop is 16 GB
  { vram: 24,  flash: 'yes',   bandwidth:  768, tflops_fp16:  55.4, names: ['RTX A5000'] },
  { vram: 24,  flash: 'mixed', bandwidth:  960, tflops_fp16:  61.4, names: ['RX 7900 XTX'], vendor: 'AMD' },

  // ── 32 GB ─────────────────────────────────────────────────────────────────
  { vram: 32,  flash: 'yes',   bandwidth: 1792, tflops_fp16: 209.5, names: ['RTX 5090 Desktop'] },  // laptop is 24 GB — AI TOPS 3352 ÷ 16; sources/nvidia-tflops-derived.md
  { vram: 32,  flash: 'yes',   bandwidth:  576, tflops_fp16:  57.7, names: ['RTX 5000 Ada'] },

  // ── Data centre / workstation ─────────────────────────────────────────────
  { vram: 40,  flash: 'yes',   bandwidth: 1555, tflops_fp16: 312.0, names: ['A100 40G'] },
  { vram: 48,  flash: 'yes',   bandwidth:  960, tflops_fp16:  91.1, names: ['RTX 6000 Ada'] },
  { vram: 48,  flash: 'yes',   bandwidth:  864, tflops_fp16: 183.0, names: ['L40S'] },
  { vram: 48,  flash: 'yes',   bandwidth:  696, tflops_fp16:  74.8, names: ['A40'] },
  { vram: 80,  flash: 'yes',   bandwidth: 2000, tflops_fp16: 312.0, names: ['A100 80G'] },
  { vram: 80,  flash: 'yes',   bandwidth: 3350, tflops_fp16: 989.0, names: ['H100'] },
  { vram: 96,  flash: 'yes',   bandwidth: 3350, tflops_fp16: 989.0, names: ['H100 NVL'] },
  { vram: 192, flash: 'yes',   bandwidth: 6700, tflops_fp16: 1978.0, names: ['2× H100'] },
];
