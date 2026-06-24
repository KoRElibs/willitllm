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
| `meta/knowledge/external-tools.md` | Changing any UI copy or config that references Cline, Continue, Ollama, or editors. URLs rot — verify before touching. |
| `meta/knowledge/nvidia-tflops-derived.md` | Updating GPU specs in `data.gpus.js`. Values are derived — do not overwrite from memory. |
| `meta/knowledge/nvidia-geforce-compare.md` | Read-only verbatim source data. Never modify as a side effect of other work. |
| `meta/benchmarks/` | Empirical data used to calibrate the speed formula (`DECODE_ATTN_EFF`, `gen_eff` ranges). Consult before changing formula constants. |

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

### `.claude/` directory — do not recreate or modify without instruction

`.claude/` at the repo root is a Claude Code project configuration directory.

**Do not recreate it if missing. Do not modify it unless explicitly asked to configure permissions
or hooks. Do not add new entries without user instruction.**

| File | Purpose | Edit? |
| --- | --- | --- |
| `.claude/CLAUDE.md` | Auto-loaded by Claude Code at startup — redirects to this file | No |
| `.claude/settings.json` | Project-level permissions (checked into repo — intentional) | Only if user asks |
| `.claude/settings.local.json` | Machine-specific overrides (gitignored — never commit) | Freely |

Current permissions in `.claude/settings.json`:

- `WebSearch` — for looking up model specs, GPU data, external tool docs
- `WebFetch(domain:willitllm.com)` — for verifying the live site
- `WebFetch(domain:ollama.com)` — for the model scraper and capability verification

### Skills

Skills live in `meta/skills/` — portable, readable by any agent.

- `meta/skills/browser-verifier.md` — Playwright + Firefox visual verification
