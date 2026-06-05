# will-it-llm — bugs

Status tags: `open` · `fixed`

Keep this file updated on every change — see `SPEC.md §12`.

---

## Open

**BUG-02 — Target context pills appear to have no effect** `fixed`
Clicking between presets correctly recoloured the hidden combobox list, but with the combobox closed the only visible change was the face button text colour — unnoticeable when the selected model stayed the same colour tier. Fixed by adding a live "X fit" count next to the pills that updates on every `markModelOptions` call, giving clear feedback regardless of combobox state.

**BUG-03 — KV cache options selectable before GPU is chosen** `fixed`
q8_0 and q4_0 started visible and selectable on fresh page load because `updateKvOptions()` was never called at startup — only on GPU `change` events and inside `applyHashState`. Fixed by calling `updateKvOptions()` once in `init()` immediately after the GPU dropdown is built.

---

**BUG-04 — Attention span score unresponsive and incorrectly scored against target context** `fixed`
Two issues: (1) pill handler only called `markModelOptions`, never `render()`, so scorecard never updated — fixed by calling `render()` from pill handler directly. (2) scoring logic was wrong: `computeScores` had no knowledge of the target context and always scored against architectural max; an attempt to fix it incorrectly capped the desired span at `model.context_length`, making a model that gives its full arch limit score 10/10 even when the target is larger. Correct formula: `ratio = min(1, maxCtx / targetCtx)` — actual divided by desired, no arch cap (calcMaxContext already caps maxCtx). A model topping out at 131k when you need 200k correctly scores 66%.

---

## Fixed

**BUG-01 — Nudge button missing after quality nudge** `fixed`
After nudging to higher sharpness, the "faster" nudge button disappeared even when lower-quality variants existed and fit in VRAM. Reproducible with `translategemma:12b` on RTX 3090: the "it" group contained only q8_0 and bf16; bf16 didn't fit, so both nudge directions were hidden despite the cheaper Q4_K_M "(default)" variant being available.

Root cause: `groupVariantsSorted()` filtered candidates to the current variant's group, which excluded cross-group alternatives. The `group` field is a UI label for `<optgroup>` sections — it was never meant to constrain nudge direction.

Fix: replaced `groupVariantsSorted()` with `variantsSortedByQuality()` — sorts all variants by quality regardless of group. 12 lines → 4 lines.
