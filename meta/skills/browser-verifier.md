# Browser verifier for willitllm

Visually verify UI changes by driving Firefox with Playwright.

## Setup (already done)

```bash
pip3 install playwright --break-system-packages
python3 -m playwright install firefox
```

## Pattern for every verification session

```python
import asyncio, subprocess, time
from playwright.async_api import async_playwright

PORT = 7830

async def verify():
    # Start server
    srv = subprocess.Popen(
        ['python3', '-m', 'http.server', str(PORT)],
        cwd='/home/kare/repos/willitllm',
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    time.sleep(0.5)
    try:
        async with async_playwright() as p:
            browser = await p.firefox.launch(headless=True)
            page    = await browser.new_page(viewport={'width': 1280, 'height': 900})
            await page.goto(f'http://localhost:{PORT}')

            # --- YOUR TEST STEPS HERE ---

            await browser.close()
    finally:
        srv.terminate()

asyncio.run(verify())
```

## Common interactions

```python
# Select GPU by visible text
await page.select_option('#vramInput', label='RTX 3090')

# Select target context
await page.select_option('#targetCtx', value='32000')

# Open model combobox and pick a model
await page.click('#modelFace')
await page.fill('#modelSearch', 'llama3.2')
await page.click('.combobox-item:not([hidden])')

# Select variant
await page.select_option('#variantSelect', index=0)

# Screenshot — always save to meta/cache/screenshots/ (gitignored, persists locally)
# Name descriptively: index_baseline.png, coder_after_refactor.png, etc.
# This lets you compare before/after across sessions.
SHOTS = '/home/kare/repos/willitllm/meta/cache/screenshots'
await page.screenshot(path=f'{SHOTS}/index_baseline.png', full_page=False)

# Read element text
text = await page.text_content('#verdict')

# Check element color (e.g. model list item)
color = await page.eval_on_selector('.combobox-item', 'el => el.style.color')

# Wait for something
await page.wait_for_selector('#results:not([hidden])')

# Load a specific URL state
await page.goto(f'http://localhost:{PORT}/#g=RTX+3090&m=llama3.2%3A3b&v=3b&t=32000')
```

## Mobile verification

Any change to layout, controls, or long text must be checked at phone width — the site
has a single `@media (max-width: 600px)` block in `styles.css`, so 600px is the only
breakpoint that exists.

**A ready-made runner already exists: `meta/scripts/mobile_audit.py`.** It walks both
pages across three phone viewports and prints the checks below as JSON. Run that first;
write a one-off script only when you need a state it doesn't cover.

### Device profiles

Use `new_context()` (not `new_page`) so touch and pixel ratio can be emulated. Firefox
supports `viewport`, `device_scale_factor`, and `has_touch`; it does **not** support
`is_mobile` — passing it throws.

```python
DEVICES = {
    'iphone_se':    {'width': 375, 'height': 667, 'dsf': 2},    # smallest common phone
    'iphone_14pro': {'width': 393, 'height': 852, 'dsf': 3},    # modern default
    'pixel_7':      {'width': 412, 'height': 915, 'dsf': 2.6},  # large android
    'fold_closed':  {'width': 344, 'height': 882, 'dsf': 3},    # narrowest realistic
}

ctx = await browser.new_context(
    viewport={'width': d['width'], 'height': d['height']},
    device_scale_factor=d['dsf'],
    has_touch=True,          # makes `(hover: none)` match → touch tooltip path runs
    user_agent='Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) '
               'AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1',
)
page = await ctx.new_page()
```

Tap instead of click when verifying touch behaviour: `await page.tap('#geekToggle')`
(requires `has_touch=True`).

### Screenshots

Take **full-page** shots on mobile — the whole point is what happens below the fold:

```python
await page.screenshot(path=f'{SHOTS}/index_mobile_375_full.png', full_page=True)
```

Name them `<page>_mobile_<width>_<state>.png`. Also grab a viewport-only shot when the
question is "what does the user see first".

### Automated mobile checks

Run these in-page before reading pixels — they catch what a screenshot hides:

```python
# 1. Horizontal overflow (the #1 mobile bug) — plus the elements causing it
overflow = await page.evaluate("""() => {
  const vw = document.documentElement.clientWidth;
  const bad = [];
  document.querySelectorAll('*').forEach(el => {
    const r = el.getBoundingClientRect();
    if (r.width === 0 || getComputedStyle(el).position === 'fixed') return;
    if (r.right > vw + 1 || r.left < -1)
      bad.push({ tag: el.tagName, cls: el.className, right: Math.round(r.right), w: Math.round(r.width) });
  });
  return { scrollWidth: document.documentElement.scrollWidth, clientWidth: vw, offenders: bad.slice(0, 20) };
}""")

# 2. Tap targets below the 44×44 CSS-px accessibility floor
small = await page.evaluate("""() => {
  const out = [];
  document.querySelectorAll('button, a, select, input, [role=button], .cap-pill, .os-tab, .tab-btn').forEach(el => {
    const r = el.getBoundingClientRect();
    if (r.width === 0 || el.hidden) return;
    if (r.height < 44 || r.width < 24)
      out.push({ txt: (el.textContent || '').trim().slice(0, 28), h: Math.round(r.height), w: Math.round(r.width) });
  });
  return out;
}""")

# 3. Text smaller than 12px (unreadable on a phone)
tiny = await page.evaluate("""() => {
  const out = new Map();
  document.querySelectorAll('*').forEach(el => {
    if (!el.offsetParent || !el.textContent.trim() || el.children.length) return;
    const fs = parseFloat(getComputedStyle(el).fontSize);
    if (fs < 12) out.set(el.className || el.tagName, fs);
  });
  return [...out];
}""")

# 4. How far below the fold the primary result sits
fold = await page.evaluate("""() => {
  const el = document.getElementById('verdict');
  return el ? Math.round(el.getBoundingClientRect().top) : null;
}""")
```

### Reference: mobile-specific CSS already in place

`styles.css` `@media (max-width: 600px)` currently stacks `.result-headline`, makes
`.controls` a 2-column grid, hides `.detail-src`, `.coder-score-bar`, and `.coder-ctx`,
and shrinks score/legend type. Check whether a new element needs a rule there — nothing
else in the stylesheet responds to width.

## Reading screenshots

Read the file at `meta/cache/screenshots/<name>.png` after saving.

## Before/after convention

For every change, follow this order exactly — order matters:

1. `rm meta/cache/screenshots/*.png` — wipe first
2. Take **before** shots — current state, before any code change
3. Make the code changes
4. Take **after** shots

```text
meta/cache/screenshots/<page>_before_<feature>.png
meta/cache/screenshots/<page>_after_<feature>.png
```

Example: `index_before_coding-pill.png` / `index_after_coding-pill.png`

The folder then contains exactly two shots per page per change — easy to compare,
easy to show the user before committing. Never wipe after taking before shots.

## Quick one-shot script template

Write a temp script (use the session scratchpad or any `/tmp` path), run it with `python3`, then read the screenshot from `meta/cache/screenshots/`.
