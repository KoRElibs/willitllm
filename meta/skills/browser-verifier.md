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

# Screenshot
await page.screenshot(path='/tmp/shot.png', full_page=False)

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

After `await page.screenshot(path='/tmp/shot.png')`, read `/tmp/shot.png`.

## Quick one-shot script template

Write the script to `/tmp/pw_test.py`, run it with `python3 /tmp/pw_test.py`, then read the screenshot.
