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
