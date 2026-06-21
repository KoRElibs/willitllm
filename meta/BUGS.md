# will-it-llm — bugs

Status tags: `open` · `fixed`

Keep this file updated on every change — see `SPEC.md §12`.

---

## Open

**BUG-15 — render() used stale model variable after auto-selection** `fixed`
`render()` captured `modelIdx` and `model` at the top before calling `markModelOptions()`. When `markModelOptions` → `markComboboxItems` auto-selected a model and dispatched a synchronous `change` event, a second `render()` ran correctly — but the first `render()` then resumed with its stale `model = undefined`, hiding `#results` again. Fixed by moving the `modelIdx`/`model` reads to after `markModelOptions()` so they reflect any auto-selection that occurred.

**BUG-14 — Selected model stayed sticky when GPU or capability filter changed** `fixed`
Changing GPU VRAM or capability pills did not deselect the current model even when it no longer fit in VRAM or was filtered out. The result was a stale model displayed in the face button that didn't match the visible list state. Fixed: `markComboboxItems` now auto-selects the first fitting (non-✗) visible model whenever the current selection doesn't fit or is hidden; `filterModelList` does the same when called with `autoSelect = true` (from `applyCap`).

**BUG-12 — Scraper path constants pointed to dev/ instead of project root** `fixed`
`MODELS_JS` and `LIBRARIES_JS` were defined with `.parent.parent` — correct when the script was at `scripts/update_models.py` but broken after it was moved to `dev/scripts/update_models.py`. Updated to `.parent.parent.parent`.

**BUG-13 — write_libraries_js emitted entries without commas between fields** `fixed`
The f-string format in `write_libraries_js` concatenated key-value pairs without `,` separators, producing invalid JSON. The function was never exercised in practice (the file was hand-maintained) so the bug was latent. Rewrote the function to build field lists properly and use `json.dumps()` per value.

**BUG-11 — Model dropdown colours wrong when target context exceeds model's arch limit** `fixed`
When `targetCtx` was set (user chose a target context size), `modelCtxColor` capped the target at the model's `context_length` before comparing — so a model with a 32k arch limit would always show green even when the user wanted 200k tokens, because `min(200448, 32768) = 32768` and its VRAM-capped max equals 32768.

Root cause: the `Math.min(targetCtx, model.context_length)` guard in `modelCtxColor` was meant to avoid penalising a model for not exceeding its trained limit, but it also masked the case where the model can't serve the user's desired context at all.

Fix: removed the arch cap from `modelCtxColor`. `ctxResult.maxCtx` is already bounded by both VRAM and arch limit (from `calcMaxContext`), so comparing it directly against the raw `targetCtx` is correct — a model whose best is 32k will show amber/orange when the user wants 200k.

**BUG-05 — TARGET CONTEXT option text truncates on narrow mobile (375px)** `fixed`
On very narrow phones the select option text ("a document · ~100 pages") was clipped by the native select control. The "fit count" badge that also competed for space was removed in a prior commit. Fixed by swapping to shorter option labels (e.g. "document", "The Hobbit") at ≤400px viewport width via JS; full labels are restored at wider viewports.

**BUG-06 — Model face briefly shows plain tag without flag on URL hash restore** `fixed`
On page load from a URL hash, `syncComboboxFace` was called before `markComboboxItems` had set item colours — so the face showed the model tag without colour. Fixed by removing the premature standalone `syncComboboxFace()` calls from `init()` and the `hashchange` handler; `render()` → `markComboboxItems` → `syncComboboxFace()` now handles both in one pass with colours already set.

**BUG-07 — Ollama command block wraps awkwardly on narrow mobile** `fixed`
`>>> /set parameter num_ctx 33152` broke mid-line at 375px because `.ollama-cmd` used `white-space: pre-wrap; word-break: break-all`. Fixed: `white-space: pre; overflow-x: auto` — command now scrolls horizontally rather than wrapping.

**BUG-08 — Verdict pop animation may not trigger on mobile** `fixed`
The `verdict-pop` keyframe relied on `void el.offsetWidth` to force a reflow before re-adding the animation class. This reflow trick is not guaranteed on all mobile browsers. Fixed by replacing the reflow with `requestAnimationFrame(() => requestAnimationFrame(...))` — the double RAF ensures the class removal has been committed to the render pipeline before the class is re-added.

**BUG-09 — "Full model context" target scored 9/10 when model is arch-limited** `fixed`
When `targetCtx = null` and the model is architecture-limited (VRAM can provide more than the model's trained max), `contextFitPct ≈ 90%` due to the safety factor, giving `scoreContext10 = ceil(9) = 9` — never 10/10 even though the model is giving everything it has. Fixed: when `ctxResult.limitedByArch` is true, force `scoreContext10 = 10`.

---

## Fixed

**BUG-01 — Nudge button missing after quality nudge** `fixed`
After nudging to higher sharpness, the "faster" nudge button disappeared even when lower-quality variants existed and fit in VRAM. Reproducible with `translategemma:12b` on RTX 3090: the "it" group contained only q8_0 and bf16; bf16 didn't fit, so both nudge directions were hidden despite the cheaper Q4_K_M "(default)" variant being available.

Root cause: `groupVariantsSorted()` filtered candidates to the current variant's group, which excluded cross-group alternatives. The `group` field is a UI label for `<optgroup>` sections — it was never meant to constrain nudge direction.

Fix: replaced `groupVariantsSorted()` with `variantsSortedByQuality()` — sorts all variants by quality regardless of group. 12 lines → 4 lines.

**BUG-02 — Target context pills appeared to have no effect** `fixed`
Clicking between presets correctly recoloured the hidden combobox list, but with the combobox closed the only visible change was the face button text colour — unnoticeable when the selected model stayed the same colour tier. Fixed by adding a live "X fit" count next to the target label that updates on every `markModelOptions` call, giving clear feedback regardless of combobox state.

**BUG-03 — KV cache options selectable before GPU is chosen** `fixed`
q8_0 and q4_0 started visible and selectable on fresh page load because `updateKvOptions()` was never called at startup. Fixed by calling `updateKvOptions()` once in `init()` immediately after the GPU dropdown is built. (KV cache is now auto-selected; this function has since been removed entirely.)

**BUG-04 — Attention span score unresponsive and incorrectly scored against target context** `fixed`
Two issues: (1) pill handler only called `markModelOptions`, never `render()`, so scorecard never updated. (2) scoring logic was wrong — capped desired span at `model.context_length`, making a model that gives its full arch limit score 10/10 even when the target is larger. Correct formula: `ratio = min(1, maxCtx / targetCtx)` — actual divided by desired, no arch cap.

**BUG-10 — Model list sort broken by browser hex-to-RGB colour normalisation** `fixed`
`item.style.color = '#56d88a'` is read back as `rgb(86, 216, 138)` — the hex-keyed `fitPriority` lookup always returned `undefined`, so all models sorted as priority 4 (unknown) and the sort was a no-op. Fixed by storing fit priority in `item.dataset.fit` at mark time and sorting on that integer attribute instead.
