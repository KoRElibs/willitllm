# will-it-llm — bugs

Keep this file updated on every change — see `SPEC.md §12`.

Open

- [BUG-20 — Stale model result when cap pill filter matches no models](#bug-20)
- [BUG-18 — Generation-speed range too tight for small Q8_0 at low context](#bug-18)
- [BUG-17 — Super-linear decode collapse on very large dense models at extreme context](#bug-17)

Fixed

- [BUG-19 — Decode slowdown term wrongly penalized f16 + flash-attention setups](#bug-19)
- [BUG-16 — Generation-speed formula ignored attention compute; over-predicted at long context](#bug-16)
- [BUG-15 — render() used stale model variable after auto-selection](#bug-15)
- [BUG-14 — Selected model stayed sticky when GPU or capability filter changed](#bug-14)
- [BUG-13 — write_libraries_js emitted entries without commas between fields](#bug-13)
- [BUG-12 — Scraper path constants pointed to dev/ instead of project root](#bug-12)
- [BUG-11b — Recommended context could spill to system RAM (overhead under-estimated)](#bug-11b)
- [BUG-11 — Model dropdown colours wrong when target context exceeds model's arch limit](#bug-11)
- [BUG-10 — Model list sort broken by browser hex-to-RGB colour normalisation](#bug-10)
- [BUG-09 — "Full model context" target scored 9/10 when model is arch-limited](#bug-09)
- [BUG-08 — Verdict pop animation may not trigger on mobile](#bug-08)
- [BUG-07 — Ollama command block wraps awkwardly on narrow mobile](#bug-07)
- [BUG-06 — Model face briefly shows plain tag without flag on URL hash restore](#bug-06)
- [BUG-05 — TARGET CONTEXT option text truncates on narrow mobile (375px)](#bug-05)
- [BUG-04 — Attention span score unresponsive and incorrectly scored against target context](#bug-04)
- [BUG-03 — KV cache options selectable before GPU is chosen](#bug-03)
- [BUG-02 — Target context pills appeared to have no effect](#bug-02)
- [BUG-01 — Nudge button missing after quality nudge](#bug-01)

---

## Open

### BUG-20

When a capability pill (e.g. "coding") is active and the current model does not satisfy it,
`filterModelList` correctly sets `sel.value = ''` and dispatches a `change` event. However,
`render()` then calls `markModelOptions()` → `markComboboxItems()`, which auto-selects the first
VRAM-fitting model without checking whether that item is currently hidden by the cap filter. The
stale model is re-selected and results render for it even though it does not match the active filter.

Candidate fix: in `markComboboxItems` (and/or `filterModelList`'s `autoSelect` path), skip items
where `item.hidden` is true when looking for the auto-select candidate. When no visible+fitting item
exists, leave `sel.value = ''` so `render()` shows no results.

Related: [BUG-14](#bug-14) (prior auto-select fix that introduced this edge case).

---

### BUG-18

llama3.2:1b (Q8_0) on a GTX 1660 Super measured ~78.5 t/s at 27k where the formula's upper bound
is 78 — and runs near peak bandwidth at low context (effective gen_eff ~0.98 vs the Q8_0 `gen_hi`
of 0.90). Very small models reach near-100% bandwidth utilisation, exceeding the quant's `gen_hi`.
Low impact (it under-promises speed, the safe direction). Candidate fix: raise Q8_0 `gen_hi`
0.90 → ~0.95. Not changed pending more small-model data points.

---

### BUG-17

For large *full-attention* dense models at very long context the measured generation speed falls
faster than the (now context-aware) formula predicts: devstral-small-2:24b at 200k measured ~7 t/s
where the formula still predicts ~18–23. The added decode attention-compute term ([BUG-16](#bug-16)
fix) captures most of the decline for ≤115k but a residual super-linear effect remains beyond that
on 24B-class dense models. Documented limitation; modelling it would need a quadratic/empirical
term and more data. Sliding-window models (Gemma) are unaffected.

---

## Fixed

### BUG-19

The BUG-16 slowdown term fired unconditionally, but a three-way KV-type sweep (f16/q8_0/q4_0)
on the RTX 3090 showed the context-decline is caused by KV **dequantization** (quantized KV) and
**unfused attention** (no-flash GPUs) — not attention compute per se. With f16 KV on a flash GPU,
decode is flat (~0.80 effective gen_eff to 48k, confirmed on two architectures: llama-arch
devstral:24b and mistral3 mistral-small3.2:24b). The unconditional term under-predicted those
setups (devstral f16 @32k: measured 41.4, formula said [20–35]). Fixed by gating the term on
`bytes_per_element < 2 OR gpu.flash ≠ 'yes'`; f16+flash now predicts [22–42] (brackets 41.4).
`calcSpeedEstimates` gained a `bytesPerElement` arg and `getGpuSpecs` now returns `flash`; call
sites in app.js/coder.js updated. Quantized/no-flash paths unchanged (no regression). Also noted:
per-token speed is f16 ≥ quantized at any context — q4_0's benefit is capacity, not speed.

---

### BUG-16

The decode estimate was `bandwidth × gen_eff / (active_weights + kv_cache)` — purely memory-bound.
It over-predicted generation speed for full-attention models as context grew (e.g. devstral:24b on
RTX 3090: the predicted range missed the measured 19.6 t/s at 112k, sitting entirely above it). Root
cause: at batch-1 decode there is also a per-token attention-compute cost that grows with the
attended context. Fixed by adding a serial attention-compute term to `calcSpeedEstimates`
(`gen = 1/(t_mem + t_attn)`), made general across architectures by capping the attended context at
each model's `sliding_window` (Gemma 2/3/4) — so sliding-window models stay flat while full-attention
models decline. New `sliding_window` field added to `data.models.js` (16 Gemma entries) and the
scraper now captures `{arch}.attention.sliding_window`. Calibrated `DECODE_ATTN_EFF = 0.015` against
RTX 3090 + GTX 1660 Super full-context sweeps (`meta/benchmarks/`); high-context bracketing error
dropped from ~32% to ~15% RMS. The same `attn_ctx` cap was applied to the prefill quadratic term.

---

### BUG-15

`render()` captured `modelIdx` and `model` at the top before calling `markModelOptions()`. When
`markModelOptions` → `markComboboxItems` auto-selected a model and dispatched a synchronous
`change` event, a second `render()` ran correctly — but the first `render()` then resumed with its
stale `model = undefined`, hiding `#results` again. Fixed by moving the `modelIdx`/`model` reads
to after `markModelOptions()` so they reflect any auto-selection that occurred.

---

### BUG-14

Changing GPU VRAM or capability pills did not deselect the current model even when it no longer
fit in VRAM or was filtered out. The result was a stale model displayed in the face button that
didn't match the visible list state. Fixed: `markComboboxItems` now auto-selects the first fitting
(non-✗) visible model whenever the current selection doesn't fit or is hidden; `filterModelList`
does the same when called with `autoSelect = true` (from `applyCap`).

---

### BUG-13

The f-string format in `write_libraries_js` concatenated key-value pairs without `,` separators,
producing invalid JSON. The function was never exercised in practice (the file was hand-maintained)
so the bug was latent. Rewrote the function to build field lists properly and use `json.dumps()`
per value.

---

### BUG-12

`MODELS_JS` and `LIBRARIES_JS` were defined with `.parent.parent` — correct when the script was
at `scripts/update_models.py` but broken after it was moved to `dev/scripts/update_models.py`.
Updated to `.parent.parent.parent`.

---

### BUG-11b

willitllm recommended a context that did not fit: llama3.2:3b on a GTX 1660 Super (6 GB) was told
~29k fit, but 28k spilled to system RAM (generation collapsed to 1.6 t/s). Root cause: rated VRAM
overstates addressable VRAM (driver/system reserve ~4–6%) and runtime overhead exceeded the fixed
0.5 GB. Fixed by raising `OVERHEAD_GB` 0.5 → 0.8 (within SPEC's documented range). This reduces
but does not fully eliminate boundary-spill risk on the smallest cards, where the unusable fraction
is proportionally largest — see SPEC §11 note on modelling usable VRAM as a fraction.

---

### BUG-11

When `targetCtx` was set (user chose a target context size), `modelCtxColor` capped the target at
the model's `context_length` before comparing — so a model with a 32k arch limit would always show
green even when the user wanted 200k tokens, because `min(200448, 32768) = 32768` and its
VRAM-capped max equals 32768.

Root cause: the `Math.min(targetCtx, model.context_length)` guard in `modelCtxColor` was meant to
avoid penalising a model for not exceeding its trained limit, but it also masked the case where the
model can't serve the user's desired context at all.

Fix: removed the arch cap from `modelCtxColor`. `ctxResult.maxCtx` is already bounded by both VRAM
and arch limit (from `calcMaxContext`), so comparing it directly against the raw `targetCtx` is
correct — a model whose best is 32k will show amber/orange when the user wants 200k.

---

### BUG-10

`item.style.color = '#56d88a'` is read back as `rgb(86, 216, 138)` — the hex-keyed `fitPriority`
lookup always returned `undefined`, so all models sorted as priority 4 (unknown) and the sort was
a no-op. Fixed by storing fit priority in `item.dataset.fit` at mark time and sorting on that
integer attribute instead.

---

### BUG-09

When `targetCtx = null` and the model is architecture-limited (VRAM can provide more than the
model's trained max), `contextFitPct ≈ 90%` due to the safety factor, giving
`scoreContext10 = ceil(9) = 9` — never 10/10 even though the model is giving everything it has.
Fixed: when `ctxResult.limitedByArch` is true, force `scoreContext10 = 10`.

---

### BUG-08

The `verdict-pop` keyframe relied on `void el.offsetWidth` to force a reflow before re-adding the
animation class. This reflow trick is not guaranteed on all mobile browsers. Fixed by replacing the
reflow with `requestAnimationFrame(() => requestAnimationFrame(...))` — the double RAF ensures the
class removal has been committed to the render pipeline before the class is re-added.

---

### BUG-07

`>>> /set parameter num_ctx 33152` broke mid-line at 375px because `.ollama-cmd` used
`white-space: pre-wrap; word-break: break-all`. Fixed: `white-space: pre; overflow-x: auto` —
command now scrolls horizontally rather than wrapping.

---

### BUG-06

On page load from a URL hash, `syncComboboxFace` was called before `markComboboxItems` had set
item colours — so the face showed the model tag without colour. Fixed by removing the premature
standalone `syncComboboxFace()` calls from `init()` and the `hashchange` handler;
`render()` → `markComboboxItems` → `syncComboboxFace()` now handles both in one pass with colours
already set.

---

### BUG-05

On very narrow phones the select option text ("a document · ~100 pages") was clipped by the native
select control. The "fit count" badge that also competed for space was removed in a prior commit.
Fixed by swapping to shorter option labels (e.g. "document", "The Hobbit") at ≤400px viewport
width via JS; full labels are restored at wider viewports.

---

### BUG-04

Two issues: (1) pill handler only called `markModelOptions`, never `render()`, so scorecard never
updated. (2) scoring logic was wrong — capped desired span at `model.context_length`, making a
model that gives its full arch limit score 10/10 even when the target is larger. Correct formula:
`ratio = min(1, maxCtx / targetCtx)` — actual divided by desired, no arch cap.

---

### BUG-03

q8_0 and q4_0 started visible and selectable on fresh page load because `updateKvOptions()` was
never called at startup. Fixed by calling `updateKvOptions()` once in `init()` immediately after
the GPU dropdown is built. (KV cache is now auto-selected; this function has since been removed
entirely.)

---

### BUG-02

Clicking between presets correctly recoloured the hidden combobox list, but with the combobox
closed the only visible change was the face button text colour — unnoticeable when the selected
model stayed the same colour tier. Fixed by adding a live "X fit" count next to the target label
that updates on every `markModelOptions` call, giving clear feedback regardless of combobox state.

---

### BUG-01

After nudging to higher sharpness, the "faster" nudge button disappeared even when lower-quality
variants existed and fit in VRAM. Reproducible with `translategemma:12b` on RTX 3090: the "it"
group contained only q8_0 and bf16; bf16 didn't fit, so both nudge directions were hidden despite
the cheaper Q4_K_M "(default)" variant being available.

Root cause: `groupVariantsSorted()` filtered candidates to the current variant's group, which
excluded cross-group alternatives. The `group` field is a UI label for `<optgroup>` sections — it
was never meant to constrain nudge direction.

Fix: replaced `groupVariantsSorted()` with `variantsSortedByQuality()` — sorts all variants by
quality regardless of group. 12 lines → 4 lines.
