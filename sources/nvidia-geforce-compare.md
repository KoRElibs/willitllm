# NVIDIA GeForce GPU Specifications — Compare Page

**Source:** https://www.nvidia.com/en-gb/geforce/graphics-cards/compare/  
**Date:** 2026-05-19  

> **DO NOT EDIT** — this file is a verbatim replica of the NVIDIA compare page as captured on
> the date above. It must only be updated when the user explicitly provides new screenshots from
> the source page. Never modify it as a side effect of other work (corrections, derivations,
> data.gpus.js updates, etc.). Derived values belong in `nvidia-tflops-derived.md`.

---

## Technologies

| Feature                  | RTX 50 Series  | RTX 40 Series  | RTX 30 Series  | RTX 20 Series  | GTX 16 Series  | GTX 10 Series  |
|--------------------------|----------------|----------------|----------------|----------------|----------------|----------------|
| Architecture             | Blackwell      | Ada Lovelace   | Ampere         | Turing         | Turing         | Pascal         |
| Streaming Multiprocessors| 2× FP32        | 2× FP32        | 2× FP32        | 1× FP32        | 1× FP32        | 1× FP32        |
| Tensor Cores (AI)        | Gen 5          | Gen 4          | Gen 3          | Gen 2          | —              | —              |
| Ray Tracing Cores        | Gen 4          | Gen 3          | Gen 2          | Gen 1          | —              | —              |
| CUDA Capability          | 12.0           | 8.9            | 8.6            | 7.5            | 7.5            | 6.1            |
| PCIe                     | Gen 5          | Gen 4          | Gen 4          | Gen 3          | Gen 3          | Gen 3          |
| NVENC                    | Gen 9          | Gen 8          | Gen 7          | Gen 7          | Gen 6          | Gen 6          |
| NVDEC                    | Gen 6          | Gen 5          | Gen 5          | Gen 4          | Gen 4          | Gen 3          |
| AV1 Encode               | Yes            | Yes            | —              | —              | —              | —              |
| AV1 Decode               | Yes            | Yes            | Yes            | —              | —              | —              |
| DLSS                     | 4.5            | 3.5            | 2              | 2              | —              | —              |

---

## Compare 50 Series Specs

|                          | RTX 5090       | RTX 5080       | RTX 5070 Ti    | RTX 5070       | RTX 5060 Ti     | RTX 5060       | RTX 5050       |
|--------------------------|----------------|----------------|----------------|----------------|-----------------|----------------|----------------|
| **GPU Engine Specs**     |                |                |                |                |                 |                |                |
| NVIDIA CUDA® Cores       | 21 760         | 10 752         | 8 960          | 6 144          | 4 608           | 3 840          | 2 560          |
| Shader Cores             | Blackwell      | Blackwell      | Blackwell      | Blackwell      | Blackwell       | Blackwell      | Blackwell      |
| Tensor Cores (AI)        | 5th Gen / 3352 AI TOPS | 5th Gen / 1801 AI TOPS | 5th Gen / 1406 AI TOPS | 5th Gen / 988 AI TOPS | 5th Gen / 759 AI TOPS | 5th Gen / 614 AI TOPS | 5th Gen / 421 AI TOPS |
| Ray Tracing Cores        | 4th Gen / 318 TFLOPS | 4th Gen / 171 TFLOPS | 4th Gen / 133 TFLOPS | 4th Gen / 94 TFLOPS | 4th Gen / 72 TFLOPS | 4th Gen / 58 TFLOPS | 4th Gen / 40 TFLOPS |
| Boost Clock (GHz)        | 2.41           | 2.62           | 2.45           | 2.51           | 2.57            | 2.50           | 2.57           |
| Base Clock (GHz)         | 2.01           | 2.30           | 2.30           | 2.33           | 2.41            | 2.28           | 2.31           |
| **Memory Specs**         |                |                |                |                |                 |                |                |
| Standard Memory Config   | 32 GB GDDR7    | 16 GB GDDR7    | 16 GB GDDR7    | 12 GB GDDR7    | 16 / 8 GB GDDR7 | 8 GB GDDR7     | 8 GB GDDR6     |
| Memory Interface Width   | 512-bit        | 256-bit        | 256-bit        | 192-bit        | 128-bit         | 128-bit        | 128-bit        |

---

## Compare 40 Series Specs

|                          | RTX 4090       | RTX 4080 Super | RTX 4080       | RTX 4070 Ti Super | RTX 4070 Ti    | RTX 4070 Super | RTX 4070       | RTX 4060 Ti      | RTX 4060       |
|--------------------------|----------------|----------------|----------------|-------------------|----------------|----------------|----------------|------------------|----------------|
| **GPU Engine Specs**     |                |                |                |                   |                |                |                |                  |                |
| NVIDIA CUDA® Cores       | 16 384         | 10 240         | 9 728          | 8 448             | 7 680          | 7 168          | 5 888          | 4 352            | 3 072          |
| Shader Cores             | Ada Lovelace / 83 TFLOPS | Ada Lovelace / 52 TFLOPS | Ada Lovelace / 49 TFLOPS | Ada Lovelace / 44 TFLOPS | Ada Lovelace / 40 TFLOPS | Ada Lovelace / 36 TFLOPS | Ada Lovelace / 29 TFLOPS | Ada Lovelace / 22 TFLOPS | Ada Lovelace / 15 TFLOPS |
| Ray Tracing Cores        | 3rd Gen / 191 TFLOPS | 3rd Gen / 121 TFLOPS | 3rd Gen / 113 TFLOPS | 3rd Gen / 102 TFLOPS | 3rd Gen / 93 TFLOPS | 3rd Gen / 82 TFLOPS | 3rd Gen / 67 TFLOPS | 3rd Gen / 51 TFLOPS | 3rd Gen / 35 TFLOPS |
| Tensor Cores (AI)        | 4th Gen / 1321 AI TOPS | 4th Gen / 836 AI TOPS | 4th Gen / 780 AI TOPS | 4th Gen / 706 AI TOPS | 4th Gen / 641 AI TOPS | 4th Gen / 568 AI TOPS | 4th Gen / 466 AI TOPS | 4th Gen / 353 AI TOPS | 4th Gen / 242 AI TOPS |
| Boost Clock (GHz)        | 2.52           | 2.55           | 2.51           | 2.61              | 2.61           | 2.48           | 2.48           | 2.54             | 2.46           |
| Base Clock (GHz)         | 2.23           | 2.29           | 2.21           | 2.34              | 2.31           | 1.98           | 1.92           | 2.31             | 1.83           |
| **Memory Specs**         |                |                |                |                   |                |                |                |                  |                |
| Standard Memory Config   | 24 GB GDDR6X   | 16 GB GDDR6X   | 16 GB GDDR6X   | 16 GB GDDR6X      | 12 GB GDDR6X   | 12 GB GDDR6X   | 12 GB GDDR6X   | 16 / 8 GB GDDR6  | 8 GB GDDR6     |
| Memory Interface Width   | 384-bit        | 256-bit        | 256-bit        | 256-bit           | 192-bit        | 192-bit        | 192-bit        | 128-bit          | 128-bit        |

---

## Compare 30 Series Specs

|                          | RTX 3090 Ti    | RTX 3090       | RTX 3080 Ti    | RTX 3080          | RTX 3070 Ti    | RTX 3070       | RTX 3060 Ti      | RTX 3060          | RTX 3050 (8G)  | RTX 3050 (6G)  |
|--------------------------|----------------|----------------|----------------|-------------------|----------------|----------------|------------------|-------------------|----------------|----------------|
| **GPU Engine Specs**     |                |                |                |                   |                |                |                  |                   |                |                |
| NVIDIA CUDA® Cores       | 10 752         | 10 496         | 10 240         | 8960 / 8704       | 6 144          | 5 888          | 4 864            | 3 584             | 2 560          | 2 304          |
| Boost Clock (GHz)        | 1.86           | 1.70           | 1.67           | 1.71              | 1.77           | 1.73           | 1.67             | 1.78              | 1.78           | 1.47           |
| Base Clock (GHz)         | 1.56           | 1.40           | 1.37           | 1.26 / 1.44       | 1.58           | 1.50           | 1.41             | 1.32              | 1.55           | 1.04           |
| **Memory Specs**         |                |                |                |                   |                |                |                  |                   |                |                |
| Standard Memory Config   | 24 GB GDDR6X   | 24 GB GDDR6X   | 12 GB GDDR6X   | 12 / 10 GB GDDR6X | 8 GB GDDR6X    | 8 GB GDDR6     | 8 GB GDDR6/6X    | 12 / 8 GB GDDR6   | 8 GB GDDR6     | 6 GB GDDR6     |
| Memory Interface Width   | 384-bit        | 384-bit        | 384-bit        | 384 / 320-bit     | 256-bit        | 256-bit        | 256-bit          | 192 / 128-bit     | 128-bit        | 96-bit         |

---

## Compare 20 Series Specs

|                          | RTX 2080 Ti    | RTX 2080 Super | RTX 2080       | RTX 2070 Super  | RTX 2070       | RTX 2060 Super | RTX 2060         |
|--------------------------|----------------|----------------|----------------|-----------------|----------------|----------------|------------------|
| **GPU Engine Specs**     |                |                |                |                 |                |                |                  |
| NVIDIA CUDA® Cores       | 4 352          | 3 072          | 2 944          | 2 560           | 2 304          | 2 176          | 2176 / 1920      |
| Boost Clock (GHz)        | 1.64           | 1.82           | 1.80           | 1.77            | 1.71           | 1.65           | 1.65 / 1.68      |
| Base Clock (GHz)         | 1.35           | 1.65           | 1.52           | 1.61            | 1.41           | 1.47           | 1.47 / 1.37      |
| **Memory Specs**         |                |                |                |                 |                |                |                  |
| Standard Memory Config   | 11 GB GDDR6    | 8 GB GDDR6     | 8 GB GDDR6     | 8 GB GDDR6      | 8 GB GDDR6     | 8 GB GDDR6     | 12 / 6 GB GDDR6  |
| Memory Interface Width   | 352-bit        | 256-bit        | 256-bit        | 256-bit         | 256-bit        | 256-bit        | 192-bit          |

---

## Compare 16 Series Specs

|                          | GTX 1660 Ti    | GTX 1660 Super | GTX 1660       | GTX 1650 Super  | GTX 1650 (G5)  | GTX 1650 (G6)  | GTX 1630        |
|--------------------------|----------------|----------------|----------------|-----------------|----------------|----------------|-----------------|
| **GPU Engine Specs**     |                |                |                |                 |                |                |                 |
| NVIDIA CUDA® Cores       | 1 536          | 1 408          | 1 408          | 1 280           | 896            | 896            | 512             |
| Boost Clock (MHz)        | 1770           | 1785           | 1785           | 1725            | 1665           | 1590           | 1785            |
| Base Clock (MHz)         | 1500           | 1530           | 1530           | 1530            | 1485           | 1410           | 1740            |
| **Memory Specs**         |                |                |                |                 |                |                |                 |
| Standard Memory Config   | 6 GB GDDR6     | 6 GB GDDR6     | 6 GB GDDR5     | 4 GB GDDR6      | 4 GB GDDR5     | 4 GB GDDR6     | 4 GB GDDR6      |
| Memory Interface Width   | 192-bit        | 192-bit        | 192-bit        | 128-bit         | 128-bit        | 128-bit        | 64-bit          |
