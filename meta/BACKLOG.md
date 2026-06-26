# will-it-llm — engineering backlog

Items that are **neither bugs** (`BUGS.md`) **nor user-facing features** (`FEATURES.md`)
**nor UX findings** (`UX-FINDINGS.md`): design/modelling observations and follow-ups worth
deciding on, but not broken and not promised. Each is a judgement call for a human, not a
defect.

Open

- [BL-01 — Capability weight vs. speed in the `code` bucket](#bl-01)
- [BL-02 — Capability scores are family-level, not per-size](#bl-02)

---

### BL-01 — Capability weight vs. speed in the `code` bucket
**Raised:** 2026-06-26 · **Area:** `coder.rank.js` (`ROLE_WEIGHTS`)

With `code` weighted `cap 0.50 · speed 0.25 · ctx 0.25`, a smaller-but-faster model can
outrank a clearly stronger one on a roomy GPU. Observed at 24 GB: `granite-code:8b`
(HumanEval 57.9) ranks above `phi4:14b` (82.6), because the 8B's speed + context edge
(each saturating) outweighs the 0.50-weighted ~25-point capability gap.

This is a **weight-tuning question**, not a correctness bug, and it pre-dates the
canonical-scoring restructure (the restructure only changed *which* scores are eligible,
not the weights). Decide: should `code` weight capability higher (e.g. 0.6+) so a much
stronger model isn't beaten on latency alone, or is "fast enough + decent" the right call
for an interactive code-chat model? Note any change interacts with the speed/ctx
saturation targets (`SPEED_TARGET_TPS`, `CTX_TARGET_TOKENS`).

### BL-02 — Capability scores are family-level, not per-size
**Raised:** 2026-06-26 · **Area:** `data.libraries.js`, `coder.rank.js`

`capability` is one number per *library*, measured on a single reference size
(`capability_ref`, e.g. `CodeLlama-13B-Instruct`). Every variant of that library inherits
it, so `codellama:7b` is ranked on the 13B's 42.7, `granite-code:34b` on the 8B's 57.9,
etc. The ranking then compares a borrowed score against another library's real-size score.

`capability_ref` documents this honestly, but the math ignores size. The accurate fix is
**per-size (variant-level) scoring**: record a benchmark per size where a citable number
exists, and fall back to "unscored" (not a borrowed number) where it doesn't. This was
explicitly scoped *out* of the canonical-scoring change as a larger curation effort —
captured here so it isn't lost. Biggest remaining accuracy lever for the coder ranking.
