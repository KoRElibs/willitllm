# will-it-llm — todo

## UX

- **Searchable model dropdown** — text input that filters the model list in real-time. The
  current `<select>` with 100+ models is painful to navigate. Filter by name, then pick from
  the filtered set. Keyboard-friendly. ✓ done

## Bugs

- **Nudge button missing after quality nudge** — reproducible with e.g. `translategemma`: nudging
  to higher sharpness makes the "faster" nudge button disappear even though lower-quality variants
  exist and fit in VRAM. Investigate `updateNudgeButtons` / `groupVariantsSorted` — the "faster"
  direction should show whenever a lower-quality variant that fits in VRAM exists, regardless of
  what the current selection is.

## Data / scoring

- **Large-context models unfairly penalised red** — models designed for long contexts (e.g. 128k+
  trained context) show red in the dropdown because the user's GPU can only fit a fraction of that
  context. This is misleading — the model is perfectly usable, just VRAM-constrained. Consider:
  1. A user-adjustable "target context" slider (default e.g. 8k or 16k); colour coding based on
     whether the model fits *that* target, not its architectural max.
  2. When the user selects a model that achieves only a small fraction of its trained context
     (e.g. <20%), surface a contextual note: "This model is designed for Xk context but your GPU
     can only fit Yk — it will still work well within that window."

- **"What fits?" ranked list mode** — flip the interaction: given the selected GPU, show all
  models ranked by context fit %, with speed and quality scores visible. Lets users discover
  what they can run rather than checking one model at a time. Sortable by fit, speed, quality.
