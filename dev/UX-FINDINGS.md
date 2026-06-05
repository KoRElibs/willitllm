# will-it-llm — UX findings

**Tested:** 2026-06-05, Firefox headless (Playwright), desktop 1280×900 and mobile 390×844 / 375×667  

Bugs found during testing live in `BUGS.md` (BUG-05 through BUG-09). This file contains design and UX suggestions only.

---

## High priority

**H1 — Ollama command sets `num_ctx` to GPU maximum, not user's target**
When target is "a chat (~8k)" and the GPU can deliver 48k, the command says `num_ctx 48640`. The user stated they need 8k but gets a command for 48k — a disconnect between stated need and output.

Options:
1. Set `num_ctx` to `min(maxCtx, targetCtx)` — give exactly what the user asked for
2. Always use `maxCtx` but add a note: "your GPU can provide up to 48k — this uses the full capacity"
3. Show two commands: one for target, one for maximum

**H2 — Variant dropdown has up to 66 options (e.g. mistral:7b)**
A 66-entry dropdown is unusable. Users either get lost or blindly pick the default.

Suggestion: show only the top 8–10 variants by default — one per meaningful quantization tier (Q2_K, Q4_K_M, Q8_0, F16) — with a "show all X variants" toggle. The current default is always selected, which helps, but users exploring options are overwhelmed.

**H3 — KV cache auto-selection is invisible — Memory clarity score changes silently**
The Memory clarity row may show ■■■■■■□□□□ but nothing explains that q8_0 was automatically chosen to meet the target. The score looks like a quality penalty with no visible cause.

Suggestion: surface the auto-selected KV type somewhere near the result — e.g. a small muted label below the ollama command: `KV cache: q8_0 — auto-selected to reach your target`. Update the Memory clarity tooltip to mention auto-selection.

---

## Medium priority

**M1 — Variant field looks like a broken input before a model is chosen**
The bottom-left field shows "select a model first" (grey, disabled) which reads like a broken or locked control to first-time users. The visual weight matches the active controls above it, creating a false impression that something is wrong.

Suggestion: visually de-emphasise the Variant field until it becomes active — lighter border, more muted label, or hide it entirely until a model is selected and animate it in.

**M2 — Mobile nudge buttons ("faster" / "better") dominate the scorecard rows**
On 390px, the nudge buttons are wide text labels that push the score bars to the right, making rows uneven and the scorecard cramped.

Suggestion: on mobile (≤600px), replace text labels with compact icon-only buttons (e.g. `▶▶` / `★★`) or move them to a separate row below the scorecard.

**M3 — "▸ details" geek mode toggle is very easy to miss**
Right-aligned, 11px monospace, muted colour — it blends into the page footer. Most casual users will never discover it.

Suggestion: left-align the toggle or increase to 12px. Consider a one-time subtle prompt on first load ("need the full breakdown? ▸ details") that disappears after it's been clicked once.

---

## Low priority

**L1 — Empty space between controls and hint text after GPU selection**
After picking a GPU, there's a large blank area above "Pick your GPU, then choose a model…" while waiting for a model to be selected. The page feels unfinished at this state.

Suggestion: show a preview of the top 3–5 green models as soon as a GPU is selected, with just name and fit status. Gives instant reward for GPU selection and prompts the next step.

**L2 — Disclaimer and AI notice are too long on mobile**
On 390px the disclaimer, AI notice, and scores explanation push far below the fold and feel like a wall of fine print.

Suggestion: collapse the disclaimer behind a "legal & data notes ▸" toggle. Keep the one-line AI notice always visible.

---

## What works well (confirmed in testing)

- 2×2 grid layout flows logically: hardware → need → model (sorted by fit) → refine
- Model list sorted green-first on GPU selection — best candidates immediately visible
- Flag emoji per model is clean and informative without org group clutter
- Fit count in TARGET CONTEXT label updates in real time — clear feedback
- Attention span correctly reflects actual vs desired context
- KV auto-selection is functionally correct and invisible to users
- OOM models correctly shown red with ✗ prefix, sorted to bottom of list
- Keyboard navigation in model combobox (ArrowDown / Enter) works
- URL hash round-trip preserves GPU, model, variant, and target context
- Mobile combobox closes on selection, no JS errors on any platform
