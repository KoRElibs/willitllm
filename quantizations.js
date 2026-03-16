// speed  1 (slowest) → 10 (fastest)
// quality 1 (worst)  → 10 (best / lossless)
const QUANT_INFO = {
  // IQ — importance-matrix quantization
  'IQ1_S':   { speed: 10, quality:  1, summary: 'Last resort — model barely fits; expect obvious quality loss.' },
  'IQ1_M':   { speed: 10, quality:  2, summary: 'Fractionally better than IQ1_S; still only when nothing else fits.' },
  'IQ2_XXS': { speed:  9, quality:  2, summary: 'Very tight VRAM; smarter compression than Q2_K at a similar size.' },
  'IQ2_XS':  { speed:  9, quality:  3, summary: 'A step up from IQ2_XXS when you have a little more room.' },
  'IQ2_S':   { speed:  9, quality:  4, summary: 'Approaches 3-bit results in a 2-bit file; good when VRAM is scarce.' },
  'IQ2_M':   { speed:  9, quality:  4, summary: 'Best 2-bit option; close to Q3_K_S quality in a smaller file.' },
  'IQ3_XXS': { speed:  8, quality:  4, summary: '3-bit size with better efficiency than plain Q3; use when size is the limit.' },
  'IQ3_XS':  { speed:  8, quality:  5, summary: 'Middle of the IQ3 family; more efficient than Q3_K_M at the same size.' },
  'IQ3_S':   { speed:  8, quality:  5, summary: 'Beats plain Q3 variants consistently; prefer over Q3_K_S or Q3_K_M.' },
  'IQ3_M':   { speed:  8, quality:  6, summary: 'Best 3-bit overall; approaches Q4_K_S results while staying small.' },
  'IQ4_XS':  { speed:  7, quality:  6, summary: 'Near Q4_K_M quality in a smaller file — often the smart 4-bit pick.' },
  'IQ4_NL':  { speed:  7, quality:  7, summary: 'Non-linear IQ4; marginal edge over IQ4_XS, use when available.' },
  // Q — standard block quantization
  'Q2_K':    { speed: 10, quality:  2, summary: 'Use only when the model would not otherwise fit at all.' },
  'Q3_K_S':  { speed:  9, quality:  3, summary: 'Fits on very limited VRAM; noticeable quality loss.' },
  'Q3_K_M':  { speed:  9, quality:  4, summary: 'Small step up from Q3_K_S; reasonable if VRAM is tight.' },
  'Q3_K_L':  { speed:  8, quality:  4, summary: 'Best standard 3-bit option; worth it over S/M when you have room.' },
  'Q4_0':    { speed:  8, quality:  5, summary: 'Older 4-bit method; Q4_K_M gives better results for the same VRAM.' },
  'Q4_1':    { speed:  7, quality:  5, summary: 'Marginal step up from Q4_0; Q4_K_M is nearly always the better pick.' },
  'Q4_K_S':  { speed:  7, quality:  6, summary: 'Slightly smaller than Q4_K_M with similar quality; good when VRAM is close.' },
  'Q4_K_M':  { speed:  7, quality:  6, summary: 'The most popular choice — best size-to-quality ratio at 4-bit.' },
  'Q5_0':    { speed:  6, quality:  7, summary: 'Older 5-bit method; Q5_K_M is a better use of the same VRAM.' },
  'Q5_1':    { speed:  6, quality:  7, summary: 'Marginal step up from Q5_0; Q5_K_M is nearly always the better pick.' },
  'Q5_K_S':  { speed:  6, quality:  7, summary: 'Good 5-bit option when you need to save a little VRAM over Q5_K_M.' },
  'Q5_K_M':  { speed:  5, quality:  8, summary: 'Near-original quality; use this when Q4 feels insufficient.' },
  'Q6_K':    { speed:  4, quality:  9, summary: 'Barely distinguishable from full precision; for when quality cannot be compromised.' },
  'Q8_0':    { speed:  3, quality:  9, summary: 'Effectively lossless; great for benchmarking or ruling out compression effects.' },
  // Full precision
  'F16':     { speed:  1, quality: 10, summary: 'Full 16-bit precision; use when VRAM is not a constraint.' },
  'FP16':    { speed:  1, quality: 10, summary: 'Same as F16; full precision baseline.' },
  'BF16':    { speed:  1, quality: 10, summary: 'Brain float 16-bit; same quality as F16, better suited to modern hardware.' },
  'F32':     { speed:  1, quality: 10, summary: 'Full 32-bit float; maximum precision, rarely needed for inference.' },
};
