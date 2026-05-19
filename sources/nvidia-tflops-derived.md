# NVIDIA GeForce — FP16 Dense TFLOPS

Derived from `nvidia-geforce-compare.md`. Not published by NVIDIA directly.  
The compare page shows AI TOPS (sparse tensor), not FP16 dense. Conversion formulas below.

---

## Conversion formulas

| Generation              | Formula              | Tensor gen | Sparse type |
|-------------------------|----------------------|------------|-------------|
| Blackwell (RTX 50xx)    | `AI TOPS ÷ 16`       | Gen 5      | FP4 sparse  |
| Ada Lovelace (RTX 40xx) | `AI TOPS ÷ 8`        | Gen 4      | INT8 sparse |
| Ampere (RTX 30xx)       | `cores × boost GHz × 4`  | Gen 3      | —           |
| Turing RTX (RTX 20xx)   | `cores × boost GHz × 16` | Gen 2      | —           |
| Turing GTX (GTX 16xx)   | `2 × FP32`           | —          | No tensor cores |
| Pascal (GTX 10xx)       | `= FP32`             | —          | No FP16 support |

Cross-check: `FP16 dense ≈ CUDA cores × boost GHz × 4` for Ampere/Ada/Blackwell (SM is 2× FP32, tensor FP16 ≈ 2× FP32).  
Turing is different: SM is 1× FP32 and Gen 2 tensor cores give FP16 ≈ 8× FP32, so `× 16` total.  
Source for clocks: `nvidia-geforce-compare.md` (RTX 40/50) and Wikipedia GeForce RTX 20 series (RTX 20xx).

---

## RTX 50 Series (Blackwell) — AI TOPS ÷ 16

| GPU         | AI TOPS | FP16 dense (TF) | Cross-check            |
|-------------|---------|-----------------|------------------------|
| RTX 5090    | 3352    | 209.5           | 21760 × 2.41 × 4 = 209.9 |
| RTX 5080    | 1801    | 112.6           | 10752 × 2.62 × 4 = 112.7 |
| RTX 5070 Ti | 1406    |  87.9           |  8960 × 2.45 × 4 =  87.9 |
| RTX 5070    |  988    |  61.8           |  6144 × 2.51 × 4 =  61.7 |
| RTX 5060 Ti |  759    |  47.4           |  4608 × 2.57 × 4 =  47.4 |
| RTX 5060    |  614    |  38.4           |  3840 × 2.50 × 4 =  38.4 |
| RTX 5050    |  421    |  26.3           |  2560 × 2.57 × 4 =  26.3 |

---

## RTX 40 Series (Ada Lovelace) — AI TOPS ÷ 8

| GPU               | AI TOPS | FP16 dense (TF) | Cross-check              |
|-------------------|---------|-----------------|--------------------------|
| RTX 4090          | 1321    | 165.1           | 16384 × 2.52 × 4 = 165.1 |
| RTX 4080 Super    |  836    | 104.5           | 10240 × 2.55 × 4 = 104.4 |
| RTX 4080          |  780    |  97.5           |  9728 × 2.51 × 4 =  97.7 |
| RTX 4070 Ti Super |  706    |  88.3           |  8448 × 2.61 × 4 =  88.2 |
| RTX 4070 Ti       |  641    |  80.1           |  7680 × 2.61 × 4 =  80.2 |
| RTX 4070 Super    |  568    |  71.0           |  7168 × 2.48 × 4 =  71.1 |
| RTX 4070          |  466    |  58.3           |  5888 × 2.48 × 4 =  58.4 |
| RTX 4060 Ti       |  353    |  44.1           |  4352 × 2.54 × 4 =  44.2 |
| RTX 4060          |  242    |  30.3           |  3072 × 2.46 × 4 =  30.2 |

---

## RTX 20 Series (Turing) — cores × boost GHz × 16

Turing SMs are 1× FP32 (vs 2× for Ampere+). Gen 2 tensor cores give FP16 ≈ 8× FP32, hence ×16 total.  
Wikipedia "FP16" column = CUDA shader FP16 (2× FP32 only) — do NOT use that column.  
Clocks from: https://en.wikipedia.org/wiki/GeForce_RTX_20_series (fetched 2026-05-20).

| GPU            | Cores | Boost (GHz) | FP16 dense (TF) | Cross-check               |
|----------------|-------|-------------|-----------------|---------------------------|
| RTX 2080 Ti    | 4352  | 1.545       | 107.5           | 4352 × 1.545 × 16 = 107.5 |
| RTX 2080 Super | 3072  | 1.815       |  89.2           | 3072 × 1.815 × 16 =  89.2 |
| RTX 2080       | 2944  | 1.710       |  80.5           | 2944 × 1.710 × 16 =  80.5 |
| RTX 2070 Super | 2560  | 1.770       |  72.4           | 2560 × 1.770 × 16 =  72.4 |
| RTX 2070       | 2304  | 1.620       |  59.7           | 2304 × 1.620 × 16 =  59.7 |
| RTX 2060 Super | 2176  | 1.650       |  57.4           | 2176 × 1.650 × 16 =  57.4 |
| RTX 2060       | 1920  | 1.680       |  51.6           | 1920 × 1.680 × 16 =  51.6 |
