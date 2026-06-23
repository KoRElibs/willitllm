# willitllm — AI agent instructions

Read `meta/SPEC.md` before doing anything in this repo.

## Skills

- **Browser verification** — `meta/skills/browser-verifier.md`
  Use after any change to HTML/CSS/JS. Playwright + Firefox are already installed.
  Write script to `/tmp/pw_test.py`, run with `python3 /tmp/pw_test.py`, read `/tmp/shot.png`.
