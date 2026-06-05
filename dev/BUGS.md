# will-it-llm — bugs

Status tags: `open` · `fixed`

Keep this file updated on every change — see `SPEC.md §12`.

---

## Open

**BUG-05 — TARGET CONTEXT label and fit count truncate on narrow mobile (375px)** `open`
On very narrow phones the "TARGET CONTEXT" label competes with the "24 FIT" badge for horizontal space, and the select option text ("a document · ~100 pages") gets clipped by the native select element. Native select truncation is hard to fix; the label fit-count clash can be addressed with a media query.

**BUG-06 — Model face briefly shows plain tag without flag on URL hash restore** `fixed`
On page load from a URL hash, `syncComboboxFace` was called before `markComboboxItems` had set item colours — so the face showed the model tag without colour. Fixed by removing the premature standalone `syncComboboxFace()` calls from `init()` and the `hashchange` handler; `render()` → `markComboboxItems` → `syncComboboxFace()` now handles both in one pass with colours already set.

**BUG-07 — Ollama command block wraps awkwardly on narrow mobile** `fixed`
`>>> /set parameter num_ctx 33152` broke mid-line at 375px because `.ollama-cmd` used `white-space: pre-wrap; word-break: break-all`. Fixed: `white-space: pre; overflow-x: auto` — command now scrolls horizontally rather than wrapping.

**BUG-08 — Verdict pop animation may not trigger on mobile** `open`
The `verdict-pop` keyframe relies on `void el.offsetWidth` to force a reflow before re-adding the animation class. This reflow trick is not guaranteed on all mobile browsers. Needs investigation on real hardware.

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
