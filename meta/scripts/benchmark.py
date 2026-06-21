#!/usr/bin/env python3
"""
Benchmark ollama generation speed at varying context sizes.
Auto-detects model architecture, computes the max context that fits in VRAM,
checks for VRAM spill, and compares measured results against willitllm estimates.

Does NOT modify any data files — results are printed only.

Usage:
  python3 meta/scripts/benchmark.py --vram 24 --bandwidth 936 --host http://rtx:11434
  python3 meta/scripts/benchmark.py --vram 24 --bandwidth 936 --fill --runs 2
  python3 meta/scripts/benchmark.py --vram 24 --bandwidth 936 --model llama3.1:8b --fill

Without --fill: num_ctx sets VRAM allocation only; actual context is ~50 tokens.
  Measures gen speed with minimal KV cache pressure. Fast, not realistic.

With --fill: prompt is padded to ~num_ctx tokens before generating.
  Measures gen speed with KV cache fully populated. Slow but matches real Cline usage.
"""

import argparse
import datetime
import json
import os
import re
import statistics
import time
import urllib.request

REPO_ROOT    = os.path.join(os.path.dirname(__file__), '..', '..')
BENCH_DIR    = os.path.join(REPO_ROOT, 'meta', 'benchmarks')

# gen_eff [lo, hi] per quantization — mirrors data.quantizations.js
GEN_EFF = {
    'Q2_K':   (0.38, 0.75),
    'Q3_K_S': (0.40, 0.78), 'Q3_K_M': (0.40, 0.78), 'Q3_K_L': (0.40, 0.78),
    'Q4_0':   (0.43, 0.82), 'Q4_1':   (0.43, 0.82),
    'Q4_K_S': (0.43, 0.82), 'Q4_K_M': (0.43, 0.82),
    'IQ4_XS': (0.43, 0.82), 'IQ4_NL': (0.43, 0.82),
    'Q5_0':   (0.45, 0.85), 'Q5_1':   (0.45, 0.85),
    'Q5_K_S': (0.45, 0.85), 'Q5_K_M': (0.45, 0.85),
    'Q6_K':   (0.48, 0.88),
    'Q8_0':   (0.50, 0.90),
    'F16':    (0.55, 0.92), 'BF16':   (0.55, 0.92),
}

KV_BPE = {'f16': 2.0, 'q8_0': 1.0, 'q4_0': 0.5}

OVERHEAD_GB = 0.5   # CUDA context + driver + ollama overhead
SAFETY      = 0.9   # 10% margin, same as willitllm
CTX_ROUND   = 128

# Short coding prompt (~50 tokens). With --fill this becomes a suffix.
BASE_PROMPT = (
    "Write a Python class implementing a min-heap with push, pop, and peek methods. "
    "Include clear docstrings and inline comments explaining the heap property."
)
FILLER_CHUNK = "The quick brown fox jumps over the lazy dog. " * 20


# ── Ollama API helpers ────────────────────────────────────────────────────────

def api_post(host, path, body, timeout=600):
    req = urllib.request.Request(
        f"{host}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def api_get(host, path, timeout=15):
    req = urllib.request.Request(f"{host}{path}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get_model_meta(host, model):
    """
    Query /api/show and extract architecture params + weights size.
    Returns (arch_params dict, weights_gb, quantization_str) or raises.
    """
    data = api_post(host, "/api/show", {"model": model}, timeout=30)
    info = data.get("model_info", {})

    # Architecture prefix varies per family: llama, qwen2, mistral, gemma, phi3…
    # Find it by locating the block_count key.
    arch = None
    for key in info:
        if key.endswith(".block_count"):
            arch = key.split(".")[0]
            break

    if not arch:
        raise RuntimeError(
            f"Cannot parse model_info for {model}. "
            f"Keys: {list(info.keys())[:10]}"
        )

    def get(suffix, default=None):
        for candidate in [f"{arch}.{suffix}", f"general.{suffix}"]:
            if candidate in info:
                return info[candidate]
        return default

    params = {
        "block_count":    int(get("block_count",            32)),
        "head_count_kv":  int(get("attention.head_count_kv",
                                  get("head_count_kv",      8))),
        "key_length":     int(get("attention.key_length",
                                  get("key_length",         128))),
        "value_length":   int(get("attention.value_length",
                                  get("value_length",       128))),
        "context_length": int(get("context_length",         131072)),
    }

    # /api/show doesn't reliably return size; fall back to /api/tags lookup
    size_bytes = data.get("size", 0)
    if not size_bytes:
        try:
            tags_data = api_get(host, "/api/tags", timeout=10)
            for m in tags_data.get("models", []):
                if m.get("model", "").startswith(model.split(":")[0]):
                    size_bytes = m.get("size", 0)
                    break
        except Exception:
            pass
    weights_gb = size_bytes / 1024**3 if size_bytes else None

    quant = (data.get("details", {}).get("quantization_level") or "Q4_K_M").upper()

    return params, weights_gb, quant


def check_vram(host, model):
    """
    Query /api/ps and return (size_gb, size_vram_gb, spilling).
    spilling=True means some layers are offloaded to system RAM.
    """
    try:
        data = api_get(host, "/api/ps")
        for m in data.get("models", []):
            if model.split(":")[0] in m.get("name", ""):
                size      = m.get("size", 0)
                size_vram = m.get("size_vram", 0)
                return size / 1024**3, size_vram / 1024**3, size_vram < size
    except Exception:
        pass
    return None, None, False


# ── Context calculations ──────────────────────────────────────────────────────

def calc_max_ctx(arch, weights_gb, vram_gb, bpe):
    """Replicate willitllm's calcMaxContext formula."""
    available = (vram_gb - OVERHEAD_GB - weights_gb) * 1024**3
    per_token = (arch["block_count"] * arch["head_count_kv"]
                 * (arch["key_length"] + arch["value_length"]) * bpe)
    raw       = available / per_token
    arch_lim  = arch["context_length"]
    arch_raw  = min(raw, arch_lim)
    return int(arch_raw * SAFETY / CTX_ROUND) * CTX_ROUND, per_token


def kv_gb_at_ctx(ctx, per_token_bytes):
    return ctx * per_token_bytes / 1024**3


# ── Prompt helpers ────────────────────────────────────────────────────────────

def make_prompt(num_ctx, fill, num_predict, tpc=0.27):
    if not fill or not num_ctx:
        return BASE_PROMPT
    target_chars = int((num_ctx - num_predict - 10) / tpc)
    filler = (FILLER_CHUNK * (target_chars // len(FILLER_CHUNK) + 1))[:target_chars]
    return filler + "\n\n" + BASE_PROMPT


# ── Save helpers ─────────────────────────────────────────────────────────────

def slugify(s):
    return re.sub(r'[^a-z0-9]+', '-', s.lower()).strip('-')

def save_results(gpu_name, args, arch, weights_gb, quant, gen_eff, max_ctx,
                 per_token_bytes, rows):
    os.makedirs(BENCH_DIR, exist_ok=True)

    gpu_slug   = slugify(gpu_name)
    model_slug = slugify(args.model)
    filename   = f"{gpu_slug}-{model_slug}-{args.kv_type}.md"
    path       = os.path.join(BENCH_DIR, filename)
    date       = datetime.date.today().isoformat()

    lines = [
        f"# {gpu_name} · {args.model} · {args.kv_type}",
        "",
        f"Date: {date}  |  Fill: {'yes' if args.fill else 'no'}  |  Runs: {args.runs}",
        "",
        "## Setup",
        "",
        f"| | |",
        f"|---|---|",
        f"| GPU | {gpu_name} ({args.vram:.0f} GB · {args.bandwidth:.0f} GB/s) |",
        f"| Model | {args.model} ({quant}, {weights_gb:.1f} GB) |",
        f"| KV cache | {args.kv_type} (bpe={KV_BPE[args.kv_type]}) |",
        f"| Arch | {arch['block_count']} layers · {arch['head_count_kv']} KV heads · key={arch['key_length']} · ctx_limit={arch['context_length']//1024}k |",
        f"| Max ctx | {max_ctx//1024}k tokens (willitllm formula) |",
        "",
        "## Measured",
        "",
        "| Ctx | Prompt tok | KV GB | Gen avg | Prefill | VRAM |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        vram = f"{r['vram_gb']:.1f} GB" if r['vram_gb'] else "—"
        if r['spilling']:
            vram += " ⚠ SPILL"
        lines.append(
            f"| {r['label']} | {r['prompt_tokens']:,} | {r['kv_gb']:.2f} "
            f"| {r['gen_avg']:.1f} | {r['pfill_avg']:.0f} | {vram} |"
        )

    lines += ["", "## vs willitllm", "",
              f"gen_eff [{gen_eff[0]}, {gen_eff[1]}] for {quant}",
              "",
              "| Ctx | Measured | Old est | New est | Old err | New err |",
              "|---|---|---|---|---|---|"]
    for r in rows:
        if r['num_ctx'] == 0:
            continue
        kv_gb  = kv_gb_at_ctx(r['num_ctx'], per_token_bytes)
        old_lo = (args.bandwidth * gen_eff[0]) / weights_gb
        old_hi = (args.bandwidth * gen_eff[1]) / weights_gb
        new_lo = (args.bandwidth * gen_eff[0]) / (weights_gb + kv_gb)
        new_hi = (args.bandwidth * gen_eff[1]) / (weights_gb + kv_gb)
        old_err = (old_lo / r['gen_avg'] - 1) * 100
        new_err = (new_lo / r['gen_avg'] - 1) * 100
        lines.append(
            f"| {r['label']} | {r['gen_avg']:.1f} | {old_lo:.0f}–{old_hi:.0f} "
            f"| {new_lo:.0f}–{new_hi:.0f} | {old_err:+.0f}% | {new_err:+.0f}% |"
        )

    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"\nSaved → {os.path.relpath(path)}")
    return path


# ── Benchmark ────────────────────────────────────────────────────────────────

def bench_ctx(host, model, num_ctx, runs, fill, num_predict):
    label  = f"{num_ctx // 1024}k" if num_ctx else "default"
    prompt = make_prompt(num_ctx, fill, num_predict)
    est_prompt_tok = int(len(prompt) * 0.27)

    suffix = f" (~{est_prompt_tok:,} prompt tok)" if fill and num_ctx else ""
    print(f"\n  ctx={label:<10}{suffix}", end="", flush=True)

    samples = []
    for i in range(runs):
        t0 = time.time()
        print(f"  [run {i+1}/{runs}]", end="", flush=True)
        try:
            payload = {
                "model":   model,
                "prompt":  prompt,
                "stream":  False,
                "options": {"num_predict": num_predict},
            }
            if num_ctx:
                payload["options"]["num_ctx"] = num_ctx
            r = api_post(host, "/api/generate", payload)

            gen_tps   = r["eval_count"]        / (r["eval_duration"]        / 1e9)
            pfill_tps = r["prompt_eval_count"] / (r["prompt_eval_duration"] / 1e9)
            samples.append((gen_tps, pfill_tps, r["eval_count"], r["prompt_eval_count"]))
            elapsed = time.time() - t0
            print(f"  {gen_tps:.1f} t/s ({elapsed:.0f}s)", end="", flush=True)
        except Exception as e:
            print(f"  ERR({e})", end="", flush=True)

    return samples


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",       default="devstral-small-2:24b")
    ap.add_argument("--host",        default="http://localhost:11434")
    ap.add_argument("--vram",        type=float, required=True,
                    help="GPU VRAM in GB (e.g. 24)")
    ap.add_argument("--bandwidth",   type=float, required=True,
                    help="GPU memory bandwidth in GB/s (e.g. 936 for RTX 3090)")
    ap.add_argument("--kv-type",     default="q4_0", choices=["q4_0", "q8_0", "f16"],
                    dest="kv_type",
                    help="KV cache type configured on the server (OLLAMA_KV_CACHE_TYPE)")
    ap.add_argument("--weights-gb",  type=float, default=None, dest="weights_gb",
                    help="Override weights size in GB (auto-detected from /api/show if omitted)")
    ap.add_argument("--ctx",         type=int, nargs="+",
                    help="Override context sizes to test. By default: [default, 4k, 8k, 32k, max]")
    ap.add_argument("--fill",        action="store_true",
                    help="Pad prompt to fill ctx (slow but realistic — matches Cline workload)")
    ap.add_argument("--runs",        type=int, default=2)
    ap.add_argument("--num-predict", type=int, default=150, dest="num_predict")
    ap.add_argument("--gpu",         default=None,
                    help="GPU name for saved filename, e.g. 'RTX 3090' or 'GTX 1660 Super'")
    args = ap.parse_args()

    bpe = KV_BPE[args.kv_type]

    # ── Detect model architecture ──────────────────────────────────────────
    print(f"Querying model info from {args.host} …", flush=True)
    try:
        arch, detected_weights_gb, quant = get_model_meta(args.host, args.model)
    except Exception as e:
        print(f"ERROR: {e}")
        return

    weights_gb = args.weights_gb or detected_weights_gb
    if not weights_gb:
        print("ERROR: Could not detect weights size. Pass --weights-gb manually.")
        return

    gen_eff = GEN_EFF.get(quant, GEN_EFF["Q4_K_M"])
    max_ctx, per_token_bytes = calc_max_ctx(arch, weights_gb, args.vram, bpe)

    print(f"\nModel     : {args.model}  ({quant}, {weights_gb:.1f} GB)")
    print(f"GPU       : {args.vram:.0f} GB VRAM  ·  {args.bandwidth:.0f} GB/s bandwidth")
    print(f"KV type   : {args.kv_type}  (bpe={bpe})")
    print(f"Arch      : {arch['block_count']} layers  ·  {arch['head_count_kv']} KV heads  ·"
          f"  key={arch['key_length']}  ·  ctx_limit={arch['context_length']//1024}k")
    print(f"Max ctx   : {max_ctx//1024}k tokens  (willitllm formula, {args.vram}GB VRAM)")
    print(f"Runs      : {args.runs} per context size")
    print(f"Fill      : {'yes — prompt padded to ctx size' if args.fill else 'no — short prompt only'}")

    # 0 in --ctx means "default" (no num_ctx set, uses model built-in default)
    if args.ctx:
        ctx_sizes = [None if c == 0 else c for c in args.ctx]
    else:
        ctx_sizes = [None, 4096, 8192, 32768, max_ctx]

    # ── Run benchmarks ────────────────────────────────────────────────────
    print("\nRunning…")
    rows = []
    for num_ctx in ctx_sizes:
        samples = bench_ctx(args.host, args.model, num_ctx, args.runs, args.fill, args.num_predict)
        if not samples:
            continue

        # Check for VRAM spill after first run at this ctx
        _sz, sz_vram, spilling = check_vram(args.host, args.model)

        gen_vals   = [s[0] for s in samples]
        pfill_vals = [s[1] for s in samples]
        actual_ctx = num_ctx or 0
        kv_gb      = kv_gb_at_ctx(samples[0][3], per_token_bytes)  # prompt_eval_count × per_token

        rows.append({
            "label":        f"{num_ctx // 1024}k" if num_ctx else "default",
            "num_ctx":      num_ctx or 0,
            "gen_avg":      statistics.mean(gen_vals),
            "gen_lo":       min(gen_vals),
            "gen_hi":       max(gen_vals),
            "pfill_avg":    statistics.mean(pfill_vals),
            "prompt_tokens": samples[0][3],
            "kv_gb":        kv_gb,
            "spilling":     spilling,
            "vram_gb":      sz_vram,
        })

    # ── Results table ─────────────────────────────────────────────────────
    print("\n\n── Measured ─────────────────────────────────────────────────────────")
    print(f"{'Ctx':<9} {'Prompt tok':>10} {'KV GB':>6} {'Gen avg':>8} {'Prefill':>9}  VRAM")
    print("─" * 72)
    for r in rows:
        vram_note = ""
        if r["vram_gb"]:
            vram_note = f"  {r['vram_gb']:.1f} GB"
            if r["spilling"]:
                vram_note += "  ⚠ SPILLING TO RAM"
        print(f"{r['label']:<9} {r['prompt_tokens']:>10,} {r['kv_gb']:>6.2f} "
              f"{r['gen_avg']:>7.1f}  {r['pfill_avg']:>8.1f}{vram_note}")

    # ── willitllm comparison ──────────────────────────────────────────────
    if not rows:
        return

    print(f"\n── vs willitllm ({quant}  ·  gen_eff [{gen_eff[0]}, {gen_eff[1]}]) ─────────────────────────────")
    print(f"{'Ctx':<9} {'Measured':>9} {'Old est':>9} {'New est':>9}  {'Old err':>8}  {'New err':>8}")
    print("─" * 72)

    for r in rows:
        if r["num_ctx"] == 0:
            continue
        kv_gb = kv_gb_at_ctx(r["num_ctx"], per_token_bytes)

        # Old willitllm: weights only in denominator
        old_lo = (args.bandwidth * gen_eff[0]) / weights_gb
        old_hi = (args.bandwidth * gen_eff[1]) / weights_gb

        # New willitllm (proposed): weights + KV cache in denominator
        new_lo = (args.bandwidth * gen_eff[0]) / (weights_gb + kv_gb)
        new_hi = (args.bandwidth * gen_eff[1]) / (weights_gb + kv_gb)

        measured = r["gen_avg"]
        old_err  = (old_lo / measured - 1) * 100  # % by which old estimate exceeds measured (at lo)
        new_err  = (new_lo / measured - 1) * 100

        print(f"{r['label']:<9} {measured:>8.1f}  "
              f"{old_lo:.0f}–{old_hi:.0f}  "
              f"{new_lo:.0f}–{new_hi:.0f}  "
              f"{old_err:>+7.0f}%  {new_err:>+7.0f}%")

    print(f"\n── willitllm hint (do not paste — verify first) ────────────────────")
    measured_rows = [r for r in rows if r.get("num_ctx", 0) > 0 and r.get("gen_avg")]
    if measured_rows:
        lo = min(r["gen_lo"] for r in measured_rows)
        hi = max(r["gen_hi"] for r in measured_rows)
        print(f'  "gen_lo": {lo:.0f},  // t/s · measured at max ctx  · {args.model}')
        print(f'  "gen_hi": {hi:.0f},  // t/s · measured at min ctx  · {args.model}')
    else:
        print("  (no successful non-default runs to compute hint)")

    if args.gpu:
        save_results(args.gpu, args, arch, weights_gb, quant, gen_eff,
                     max_ctx, per_token_bytes, rows)
    else:
        print("\n(Pass --gpu 'GPU Name' to auto-save results to meta/benchmarks/)")


if __name__ == "__main__":
    main()
