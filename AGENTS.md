# will-it-llm — AI agent guide

Entry point for any AI agent working on this codebase.

---

## Read order

Start with `OVERVIEW.md` — it has the file map, formula summary, and maintenance rule.
Then continue here for tools and agent-specific setup.

**Full context: also read:**

1. `SPEC.md` — complete specification: data structures, formulas, all UI behaviour
2. `meta/FEATURES.md` — what is built vs. planned
3. `meta/BUGS.md` — known open bugs

**Read `SPEC.md` before making any code change.** It is the authoritative specification.
The codebase must match it, and changes to the codebase must update it.

---

## Critical constraints

Non-obvious rules not immediately visible from reading the code. Check these before touching anything.

**`data.models.js` is scraper-maintained — never edit manually.** Changes will be overwritten the
next time `meta/scripts/update_models.py` runs. Use the scraper workflow in
`meta/scripts/update-models.md` instead.

**`capabilities` and `pulls` in `data.libraries.js` are scraper-owned — do not set manually.**
Run `update_models.py --capabilities --apply` to refresh them. `coding_role` is the opposite —
hand-curated, the scraper preserves it but never sets it.

**`?v=N` cache-bust strings in `<script>` tags must be bumped on every deploy.** Both `index.html`
and `coder.html` load JS files as `<script src="file.js?v=N">`. Increment N for every JS file you
changed, or the CDN will serve stale code to users.

**Formula constants in `app.calc.js` are empirically calibrated — do not adjust from intuition.**
`OVERHEAD_GB`, `SAFETY_FACTOR`, `DECODE_ATTN_EFF`, and `CTX_ROUND` were set from real hardware
benchmarks in `meta/benchmarks/`. Read the relevant benchmark files before changing any of them,
and update `SPEC.md §4–5` if you do.

---

## Tools and skills

### Browser verification — Playwright + Firefox

When to use: after any change to HTML, CSS, or JS.

The script pattern lives at `meta/skills/browser-verifier.md`. Write the test script to a temp
path, run with `python3`, read the screenshot after.

**Always save screenshots to `meta/cache/screenshots/`.** Name them descriptively:
`index_baseline.png`, `coder_after_refactor.png`, etc. This lets you compare before/after visually
across sessions — the directory is gitignored but persists locally.

Playwright and Firefox are already installed — no setup needed.

### Mobile UX audit — `meta/scripts/mobile_audit.py`

When to use: after any layout/CSS change, and whenever reviewing how the site behaves on phones.

Runs both pages at 375×667, 393×852 and 344×882 with touch emulation, captures full-page and
viewport screenshots to `meta/cache/screenshots/`, and reports horizontal overflow (with the
offending elements), sub-44px tap targets, and sub-12px text as JSON on stdout.

```bash
python3 meta/scripts/mobile_audit.py > /tmp/mobile.json
```

Latest results: `meta/UX-MOBILE.md`. The device-profile and in-page check patterns are documented
in `meta/skills/browser-verifier.md` under *Mobile verification*.

### Model scraper — `meta/scripts/update_models.py`

When to use: when adding or refreshing model data in `data.models.js` or `data.libraries.js`.

Full workflow: `meta/scripts/update-models.md`. Never edit `data.models.js` manually — it is
scraper-maintained and manual edits will be overwritten.

### Benchmark runner — `meta/scripts/benchmark.py`

When to use: when calibrating or verifying speed estimates against real hardware.

Requires a live ollama instance. Results belong in `meta/benchmarks/`.

### Deploy — `meta/deploy/deploy.sh`

When to use: when deploying to production.

Requires `meta/deploy/deploy.cfg` — not in the repo (machine-specific credentials). The file
`meta/deploy/deploy.cfg.example` shows the required format. Never commit `deploy.cfg`.

---

## External knowledge

Consult these before changing anything that touches their domain:

| Document | Consult when |
| --- | --- |
| `meta/knowledge/ollama-context-and-kv-cache.md` | **Required reading before touching any setup command.** How context window size and K/V cache quantization are actually set in Ollama, and their limitations — the cache type is a *global* server option with silent failure modes. Getting this wrong produces commands that look right and do nothing (see BUG-26). |
| `meta/knowledge/external-tools.md` | Changing any UI copy or config that references Cline, Continue, Ollama, or editors. URLs rot — verify before touching. |
| `meta/knowledge/nvidia-tflops-derived.md` | Updating GPU specs in `data.gpus.js`. Values are derived — do not overwrite from memory. |
| `meta/knowledge/nvidia-geforce-compare.md` | Read-only verbatim source data. Never modify as a side effect of other work. |
| `meta/benchmarks/` | Empirical data used to calibrate the speed formula (`DECODE_ATTN_EFF`, `gen_eff` ranges). Consult before changing formula constants. |

---

## Change workflow & commit discipline

Work in **small, self-contained increments** — not large multi-feature diffs. Each logically
complete change should be testable, reviewable, and committable on its own. If a diff starts
spanning several unrelated concerns, it has already grown too large — split it. A commit that
touches a dozen files across multiple features is a smell, not a milestone.

The loop for every change:

1. **Make one focused change** — one feature, fix, or refactor at a time, not a session's worth
   of work batched together.
2. **Test it before showing it.** Verify the change actually works — browser-verify any
   HTML/CSS/JS change (see *Browser verification*; screenshots → `meta/cache/screenshots/`), and
   sanity-check data/logic (`node --check`, data-integrity checks, scraper `--verify`). Never
   present an untested change as done, and never claim it works without having run it.
3. **Show the user and wait.** Describe what changed and how you verified it. Do **not** commit yet.
4. **Commit only after the user confirms.** The user tests/approves first; then commit *that*
   increment. Do not commit unprompted, and do not let several confirmed changes pile up into one
   oversized commit — commit each unit as it is confirmed.

Branch before the first commit if on the default branch; keep `main` clean.

---

## Commit conventions

Always credit everyone who contributed to a commit:

```text
Co-Authored-By: Name <email>
Co-Authored-By: Name <email>
```

Include the human who directed the work, any AI agent that produced the code, and any other
contributor involved. No contributor should be invisible in the git history.

---

## Maintenance rules

Every code change must update at least one of:

| File | Update when |
| --- | --- |
| `SPEC.md` | Any described behaviour changes — data structures, formulas, UI layout, constants, file paths, or interaction rules |
| `meta/FEATURES.md` | A feature is implemented (`backlog` → `done`), planned (new entry), or its behaviour changes |
| `meta/BUGS.md` | A bug is discovered (add as `open`) or fixed (`open` → `fixed` with root cause and fix description) |

The rule: if you changed the code, you changed at least one of these files.

---

## Agent memory and private working files

Write any private working state (memory, scratchpads, intermediate results) to `meta/cache/`.
It is gitignored — local only, never committed, but visible on the local filesystem so any agent
can read it. Do not write agent state outside the project folder (e.g. `~/.someagent/`) — that
makes it invisible to other agents and impossible to audit.

---

## Claude Code specifics

*This section applies only if you are running as Claude Code. Other agents: skip this section.*

### `.claude/` directory — not checked in

The repo intentionally ships **no** checked-in `.claude/` config. There is no committed
`.claude/CLAUDE.md` (this `AGENTS.md` is the single agent entry point — read it directly) and no
committed `.claude/settings.json`.

**Do not create or commit `.claude/CLAUDE.md` or `.claude/settings.json`.** Project-level agent
config lives in the tracked docs (`AGENTS.md`, `SPEC.md`, `meta/`), not in `.claude/`.

Machine-specific overrides belong in `.claude/settings.local.json`, which is gitignored — edit it
freely (permissions, local hooks) but never commit it. Common local permissions this project uses:
`WebSearch`, `WebFetch(domain:willitllm.com)`, `WebFetch(domain:ollama.com)` — for the model scraper,
capability verification, and live-site checks.

### Skills

Skills live in `meta/skills/` — portable, readable by any agent.

- `meta/skills/browser-verifier.md` — Playwright + Firefox visual verification
