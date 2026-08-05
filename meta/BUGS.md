# will-it-llm — bugs

Keep this file updated on every change — see `SPEC.md §12`.

Open

- [BUG-21 — Coder rows cannot be expanded on touch devices](#bug-21)
- [BUG-23 — Coder model name truncates to zero width on mobile](#bug-23)
- [BUG-24 — iOS Safari zooms on focus for both text inputs (font-size < 16px)](#bug-24)
- [BUG-20 — Stale model result when cap pill filter matches no models](#bug-20)
- [BUG-18 — Generation-speed range too tight for small Q8_0 at low context](#bug-18)
- [BUG-17 — Super-linear decode collapse on very large dense models at extreme context](#bug-17)

Fixed

- [BUG-26 — Setup commands emitted a dead env var and omitted the one that makes KV cache work](#bug-26)
- [BUG-25 — Linux/macOS setup appended to shell rc files: non-idempotent, and wrong on Linux](#bug-25)
- [BUG-22 — Result headline forced the page to 534px on every phone width](#bug-22)
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

### BUG-21

On touch devices a coder row cannot be opened. `.coder-row-header` is composed almost entirely of
`[data-tip]` spans (badge, ★ recommended tag, flag, speed, context, benchmark, score bar), and the
touch branch of `initTooltip` (`app.shared.js:81`) calls `e.stopPropagation()` in the capture phase
so the tap shows a tooltip instead of reaching the row's toggle handler.

Measured at 393px by sampling every x across each header at mid-height: 86% of the top-ranked
(★ recommended) row's width swallows the tap, 66–69% on the other rows. The remaining ~50px on the
top row is not contiguous — 22px of left padding plus four 8px flex gaps. Aggravated by
[BUG-23](#bug-23), which collapses `.coder-name` (the largest non-tip child) to 0px, and by the
6px-wide caret being pushed off-screen.

Impact: the ready-to-paste Cline/Continue configs — the entire purpose of `coder.html` — are
unreachable on a phone.

Candidate fix: give the row a contiguous non-`[data-tip]` tap area on mobile (wrapping the header
so `.coder-name` gets its own full-width line does this as a side effect — see BUG-23), and/or make
the touch handler skip `stopPropagation()` when the tip element sits inside a toggleable row.

Reproduce: `python3 meta/scripts/mobile_audit.py`; see `meta/UX-MOBILE.md` finding 1.

---

### BUG-23

`.coder-name` — the string the user has to type after `ollama pull` — truncates to nothing on
mobile. `.coder-row-header` is `flex-wrap: nowrap` and the name is the only child that shrinks;
badge, ★ recommended tag, flag, speed range and benchmark chip all keep their full width.

Measured at 393px: `devstral-small-2:24b` renders at **0px** (needs 157px) on the ★ recommended row,
`devstral:24b` at 49px (needs 94), `granite-code:8b` and `granite-code:20b` at 57px (need 117/125).
The list reads "de…", "gra…", "phi…", and the top-ranked recommendation shows no name at all.

The same `nowrap` also pushes `.coder-bench` and `.coder-caret` past the right edge, giving
`coder.html` a 406px document at a 393px viewport.

Candidate fix: allow the header to wrap below 600px and give `.coder-name` its own full-width line
(`flex: 1 1 100%; order: -1`). Verified in-browser: document width 406 → 393px, name fully visible.
This also creates the contiguous tap area [BUG-21](#bug-21) needs.

---

### BUG-24

iOS Safari auto-zooms the page when an input with `font-size < 16px` receives focus, and does not
zoom back out on blur. Both text inputs on the site are 12px:

- `.combobox-search` (`styles.css:206`) — the model search box on `index.html`
- `.coder-controls input[type="text"]` (`styles.css:740`) — the Ollama URL field on `coder.html`

These are the two primary text-entry points, so an iPhone user is left zoomed in and horizontally
panned after using either. Native `<select>` controls are unaffected.

Candidate fix: `font-size: 16px` on both inputs inside the mobile media query. Do **not** fix it by
adding `maximum-scale=1` to the viewport meta — that disables pinch-zoom for everyone.

Not reproducible in Firefox/Playwright (WebKit-specific); identified from the computed styles.

---

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

### BUG-26

The "how to run it" setup block emitted commands that could not do what the site promised. Verified
against docs.ollama.com on 2026-08-05:

1. **`OLLAMA_NUM_CTX` does not exist.** The run line was
   `OLLAMA_NUM_CTX=<maxCtx> ollama run <tag>`. That variable name was dropped in Ollama 0.6 and is
   silently ignored by current releases; the current name is `OLLAMA_CONTEXT_LENGTH`. Every user
   who followed our instructions got Ollama's own default context, not the one we calculated.
2. **Context length is server-side.** Even under the correct name it only takes effect on
   `ollama serve` — `ollama run` is a client talking to an already-running server, so an env var
   prefixed to it is discarded. The setting had to move onto the serve command / service config.
3. **`OLLAMA_KV_CACHE_TYPE` needs `OLLAMA_FLASH_ATTENTION=1`.** Ollama only quantizes the KV cache
   when flash attention is on, and it is off by default. We emitted the cache type alone, so every
   `q8_0`/`q4_0` recommendation silently ran at f16 — meaning the *whole point* of the
   recommendation (fitting a longer context in the same VRAM) did not happen, and the context
   figure shown next to it was unreachable. This was the most damaging of the three: it broke the
   calculation the site exists to perform.

Fixed in `osKvContent()` / `renderCmd()` (`index.render.js`): all three variables are now emitted
together on whatever starts the server for that platform, `OLLAMA_FLASH_ATTENTION=1` is included
whenever the recommended cache type is not f16 (and omitted when it is, since f16 does not need it),
and the block ends with `ollama run <tag>` carrying no environment prefix. The systemd option ends
at the service restart instead — it reconfigures a background service that keeps running, so
starting a model afterwards is ordinary use rather than part of the setup.

A fourth problem surfaced once the above was in: the block demanded `sudo systemctl stop ollama`,
a foreground `ollama serve`, and `OLLAMA_KV_CACHE_TYPE=f16` **even when f16 was the recommendation**
— i.e. setting the value Ollama already defaults to, at the cost of taking down the user's service.
Since f16 is the common case (`autoKvBpe` reaches for it first), most visitors were shown that.
Fixed by gating the whole server section on `needsServerSetup(kvLabel)`: f16 now emits just
`ollama pull` / `ollama run` / `>>> /set parameter num_ctx <n>`, which needs no server change and is
identical on every platform, so the OS selector row is hidden too. `coder.html`'s assumption note
got the same treatment — an f16 row states there is nothing to configure, since `contextLength` in
the editor config is an API option and outranks any server setting.

`meta/knowledge/external-tools.md` § Ollama now records all these facts so this cannot recur.

---

### BUG-25

The `linux` and `macos` branches of `osKvContent()` told the user to append an export to a shell
rc file:

```sh
echo 'export OLLAMA_KV_CACHE_TYPE=q8_0' >> ~/.bashrc && source ~/.bashrc   # linux
echo 'export OLLAMA_KV_CACHE_TYPE=q8_0' >> ~/.zshrc  && source ~/.zshrc    # macos
```

Two problems:

1. **Non-idempotent.** Nothing guards the append. A user who changes model or context — and so
   gets a different `kvLabel` — and runs the snippet again silently accumulates a second,
   conflicting `export` line. Whichever is last wins, nothing ever cleans them up, and the site
   gave no way to undo it.
2. **Ineffective on Linux.** The Linux installer registers a systemd service running as the
   `ollama` user; it never reads an interactive shell's rc file. Worse, that service holds port
   11434, so the `ollama serve` the snippet then told the user to run would fail with
   `address already in use`. The instructions could not work as written on a default install.
   `meta/knowledge/external-tools.md` already recorded the systemd drop-in as *the* Linux
   mechanism — the branch contradicted our own notes and Ollama's docs.

Fixed by dropping rc-file mutation entirely, and by collapsing the OS list to **one entry per
platform** — four options instead of five:

- `linux` → the systemd drop-in (previously the separate `linux-service` entry), now the only
  Linux path. A first attempt kept a temporary `Linux · quick start` beside it (stop the service,
  run a rival `ollama serve` in the foreground); that was removed as strictly worse than the
  drop-in — more steps, it occupied a terminal, and closing that terminal left ollama dead with
  the service still stopped. The drop-in is two commands, idempotent (`tee` overwrites), survives
  reboot, and is what Ollama documents. Non-systemd Linux setups use `generic`.
- `macos` → quit the menubar app (⌘Q) then `OLLAMA_KV_CACHE_TYPE=<kv> ollama serve`; reopening the
  app restores the default. The docs do describe the `~/.zshrc` route, but you have to quit the app
  and run `ollama serve` in that terminal either way, so the persistent export bought nothing.
- `generic` and `windows` unchanged.

Retiring the `linux-service` key needed a migration: `storedOs()` in `app.shared.js` normalises it
to `linux` on read (and any other unrecognised value to a passed default), since the `osTab`
preference is shared with `coder.html` and persists across visits. Without it a returning user would
have hit a missing branch and an empty setup block.

SPEC §7.5b now forbids rc-file appends in `osKvContent()` and records why there is no
temporary/permanent pair. Verified on both pages: all four options render in order, the retired
`linux-service` and a garbage value both resolve to a populated block, no JS errors.

---

### BUG-22

The result headline forced the document to **534px wide at every phone viewport** (measured
identically at 344, 375 and 393px), so the whole site scrolls horizontally into dead space once a
model is selected.

Root cause: `.result-main` / `.result-cmd` / `.result-aside` were grid items without `min-width: 0`.
A grid item's automatic minimum size is its min-content size, so the track was sized by
`<pre class="ollama-cmd">` (`white-space: pre`, min-content 497px) rather than by the viewport.

This also silently defeated the [BUG-07](#bug-07) fix: `overflow-x: auto` on `.ollama-cmd` never
engaged because the element was never constrained.

Off-screen as a result: the card's right border, the `it will code!` link, the `ROUGH ESTIMATE`
caveat, the end of the reading-speed line, the **macOS** and **Windows** OS tabs (2 of 5, with no
scroll affordance), and the tail of every command line.

Fixed by removing the `result-cmd` grid area entirely — the OS tab strip and setup block moved
out of the card into the new collapsible `#runSection` below it (SPEC §7.5b) — and adding
`min-width: 0` to `.result-main` so no future wide child can re-create the problem. Verified at 344,
375, 393 and 1280px: `document.scrollWidth === clientWidth` in both the closed and open states, and
the setup block now scrolls internally as intended (343px visible / 540px scrollable at 393px),
which also restores the [BUG-07](#bug-07) fix.

---

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
