"""Mobile UX audit for willitllm — see meta/skills/browser-verifier.md.

Usage:
    python3 meta/scripts/mobile_audit.py [--tag _before_myfeature]

--tag is appended to every screenshot filename so before/after runs never
overwrite each other. Screenshots land in meta/cache/screenshots/; the
overflow / tap-target / small-text report goes to stdout as JSON.
"""
import asyncio, subprocess, sys, time, json
from playwright.async_api import async_playwright

PORT = 7830
ROOT = '/home/kare/repos/willitllm'
SHOTS = f'{ROOT}/meta/cache/screenshots'
TAG = ''
if '--tag' in sys.argv:
    TAG = sys.argv[sys.argv.index('--tag') + 1]

DEVICES = {
    'iphone_se':    {'width': 375, 'height': 667, 'dsf': 2},
    'iphone_14pro': {'width': 393, 'height': 852, 'dsf': 3},
    'fold_closed':  {'width': 344, 'height': 882, 'dsf': 3},
}

UA = ('Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 '
      '(KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1')

OVERFLOW_JS = """() => {
  const vw = document.documentElement.clientWidth;
  const bad = [];
  document.querySelectorAll('*').forEach(el => {
    const r = el.getBoundingClientRect();
    if (r.width === 0) return;
    if (getComputedStyle(el).position === 'fixed') return;
    if (r.right > vw + 1 || r.left < -1)
      bad.push({tag: el.tagName, cls: String(el.className).slice(0,40), id: el.id,
                right: Math.round(r.right), w: Math.round(r.width), scrollW: el.scrollWidth});
  });
  return {scrollWidth: document.documentElement.scrollWidth, clientWidth: vw,
          bodyScroll: document.body.scrollWidth, offenders: bad.slice(0, 25)};
}"""

TAP_JS = """() => {
  const out = [];
  document.querySelectorAll('button, a, select, input, [role=button]').forEach(el => {
    const r = el.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) return;
    if (r.height < 44 || r.width < 24)
      out.push({txt: (el.textContent||el.id||'').trim().slice(0,30), cls: String(el.className).slice(0,30),
                h: Math.round(r.height), w: Math.round(r.width)});
  });
  return out;
}"""

TINY_JS = """() => {
  const out = {};
  document.querySelectorAll('*').forEach(el => {
    if (!el.offsetParent) return;
    if (!el.textContent.trim() || el.children.length) return;
    const fs = parseFloat(getComputedStyle(el).fontSize);
    if (fs < 12) out[String(el.className||el.tagName).slice(0,40)] = fs;
  });
  return out;
}"""

async def audit_index(ctx, dev_name, dev, report):
    page = await ctx.new_page()
    await page.goto(f'http://localhost:{PORT}')
    await page.wait_for_timeout(400)

    r = report.setdefault(f'index@{dev_name}', {})
    r['hover_none'] = await page.evaluate("() => matchMedia('(hover: none)').matches")
    r['initial_overflow'] = await page.evaluate(OVERFLOW_JS)
    r['initial_tap'] = await page.evaluate(TAP_JS)
    r['initial_tiny'] = await page.evaluate(TINY_JS)
    await page.screenshot(path=f'{SHOTS}/index_mobile_{dev["width"]}_initial{TAG}.png', full_page=True)

    # select GPU + model
    await page.select_option('#vramInput', label='RTX 3090')
    await page.wait_for_timeout(300)
    await page.screenshot(path=f'{SHOTS}/index_mobile_{dev["width"]}_gpu{TAG}.png', full_page=True)

    # open combobox
    await page.click('#modelFace')
    await page.wait_for_timeout(300)
    r['combo_panel'] = await page.evaluate("""() => {
      const p = document.getElementById('modelPanel');
      const r = p.getBoundingClientRect();
      return {top: Math.round(r.top), height: Math.round(r.height), bottom: Math.round(r.bottom),
              vh: window.innerHeight, itemH: Math.round((document.querySelector('.combobox-item')||{getBoundingClientRect:()=>({height:0})}).getBoundingClientRect().height)};
    }""")
    await page.screenshot(path=f'{SHOTS}/index_mobile_{dev["width"]}_combo{TAG}.png', full_page=False)

    await page.fill('#modelSearch', 'devstral:24b')
    await page.wait_for_timeout(300)
    await page.click('.combobox-item:not([hidden])')
    await page.wait_for_timeout(500)

    r['result_overflow'] = await page.evaluate(OVERFLOW_JS)
    r['result_tap'] = await page.evaluate(TAP_JS)
    r['result_tiny'] = await page.evaluate(TINY_JS)
    r['fold'] = await page.evaluate("""() => {
      const g = id => { const e = document.getElementById(id); if (!e) return null;
        const r = e.getBoundingClientRect(); return {top: Math.round(r.top + window.scrollY), h: Math.round(r.height)}; };
      return {vh: window.innerHeight, docH: document.documentElement.scrollHeight,
              verdict: g('verdict'), scorecard: g('scorecard'), cmd: g('ollamaSetup'),
              aside: g('resultAside'), bar: g('barTotal'), geek: g('geekToggle')};
    }""")
    await page.screenshot(path=f'{SHOTS}/index_mobile_{dev["width"]}_result{TAG}.png', full_page=True)
    await page.screenshot(path=f'{SHOTS}/index_mobile_{dev["width"]}_result_fold{TAG}.png', full_page=False)

    # "how to run it" panel (SPEC §7.5b) — the setup block must scroll inside itself,
    # never widen the document, and the OS dropdown must be a real touch target.
    await page.tap('#runToggle')
    await page.wait_for_timeout(400)
    r['run_panel'] = await page.evaluate("""() => {
      const pre = document.getElementById('ollamaSetup');
      const sel = document.getElementById('osSelect');
      const pr = pre.getBoundingClientRect(), sr = sel.getBoundingClientRect();
      return {docW: document.documentElement.scrollWidth, vw: document.documentElement.clientWidth,
              preW: Math.round(pr.width), preScrollW: pre.scrollWidth,
              scrollsInternally: pre.scrollWidth > Math.round(pr.width),
              osValue: sel.value, osH: Math.round(sr.height), osW: Math.round(sr.width),
              toggleH: Math.round(document.getElementById('runToggle').getBoundingClientRect().height)};
    }""")
    await page.screenshot(path=f'{SHOTS}/index_mobile_{dev["width"]}_runopen{TAG}.png', full_page=True)
    await page.tap('#runToggle')
    await page.wait_for_timeout(300)

    # tooltip on touch: score label (span) vs cap pill (button)
    try:
        await page.tap('.score-label')
        await page.wait_for_timeout(250)
        r['tip_span'] = await page.evaluate("""() => {
          const t = document.getElementById('tooltip');
          const r = t.getBoundingClientRect();
          return {hidden: t.hidden, top: Math.round(r.top), left: Math.round(r.left), w: Math.round(r.width),
                  h: Math.round(r.height), right: Math.round(r.right), vw: window.innerWidth, vh: window.innerHeight};
        }""")
        await page.screenshot(path=f'{SHOTS}/index_mobile_{dev["width"]}_tooltip{TAG}.png', full_page=False)
    except Exception as e:
        r['tip_span'] = f'ERR {e}'

    try:
        await page.tap('.cap-pill[data-cap="vision"]')
        await page.wait_for_timeout(250)
        r['tip_pill'] = await page.evaluate("() => document.getElementById('tooltip').hidden")
        await page.tap('.cap-pill[data-cap="vision"]')  # untoggle
        await page.wait_for_timeout(200)
    except Exception as e:
        r['tip_pill'] = f'ERR {e}'

    # geek section
    await page.tap('#geekToggle')
    await page.wait_for_timeout(400)
    r['geek_overflow'] = await page.evaluate(OVERFLOW_JS)
    r['formula_scroll'] = await page.evaluate("""() => {
      const e = document.getElementById('formulaBox');
      if (!e) return null;
      const r = e.getBoundingClientRect();
      return {clientW: Math.round(r.width), scrollW: e.scrollWidth, hidden: e.hidden};
    }""")
    await page.screenshot(path=f'{SHOTS}/index_mobile_{dev["width"]}_geek{TAG}.png', full_page=True)

    # formula tab
    try:
        await page.tap('.tab-btn[data-tab="tabFormula"]')
        await page.wait_for_timeout(300)
        r['formula_tab'] = await page.evaluate("""() => {
          const e = document.getElementById('formulaBox');
          const r = e.getBoundingClientRect();
          const lines = [...e.querySelectorAll('.formula-line, .formula-muted')].map(l => ({w: Math.round(l.getBoundingClientRect().width), sw: l.scrollWidth}));
          return {clientW: Math.round(r.width), scrollW: e.scrollWidth, worst: Math.max(...lines.map(l=>l.sw))};
        }""")
        await page.screenshot(path=f'{SHOTS}/index_mobile_{dev["width"]}_formula{TAG}.png', full_page=False)
    except Exception as e:
        r['formula_tab'] = f'ERR {e}'

    # info sheet
    await page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
    await page.wait_for_timeout(200)
    await page.tap('#infoSheetOpen')
    await page.wait_for_timeout(400)
    r['info_sheet'] = await page.evaluate("""() => {
      const s = document.getElementById('infoSheet');
      const r = s.getBoundingClientRect();
      return {top: Math.round(r.top), left: Math.round(r.left), w: Math.round(r.width), h: Math.round(r.height),
              bottom: Math.round(r.bottom), vw: window.innerWidth, vh: window.innerHeight,
              scrollH: s.scrollHeight, bodyScrollH: document.querySelector('.info-sheet-body').scrollHeight,
              bodyClientH: document.querySelector('.info-sheet-body').clientHeight};
    }""")
    await page.screenshot(path=f'{SHOTS}/index_mobile_{dev["width"]}_infosheet{TAG}.png', full_page=False)
    await page.close()


async def audit_coder(ctx, dev_name, dev, report):
    page = await ctx.new_page()
    await page.goto(f'http://localhost:{PORT}/coder.html')
    await page.wait_for_timeout(400)
    r = report.setdefault(f'coder@{dev_name}', {})
    r['initial_overflow'] = await page.evaluate(OVERFLOW_JS)
    await page.screenshot(path=f'{SHOTS}/coder_mobile_{dev["width"]}_initial{TAG}.png', full_page=True)

    await page.select_option('#vramInput', label='RTX 3090')
    await page.wait_for_timeout(600)
    r['list_overflow'] = await page.evaluate(OVERFLOW_JS)
    r['list_tap'] = await page.evaluate(TAP_JS)
    r['list_tiny'] = await page.evaluate(TINY_JS)
    r['rows'] = await page.evaluate("""() => {
      const rows = [...document.querySelectorAll('.coder-row')];
      return {n: rows.length, docH: document.documentElement.scrollHeight, vh: window.innerHeight,
              first: rows[0] ? {h: Math.round(rows[0].getBoundingClientRect().height)} : null};
    }""")
    await page.screenshot(path=f'{SHOTS}/coder_mobile_{dev["width"]}_list{TAG}.png', full_page=True)
    await page.screenshot(path=f'{SHOTS}/coder_mobile_{dev["width"]}_list_fold{TAG}.png', full_page=False)

    # expand first row
    try:
        await page.tap('.coder-row-header')
        await page.wait_for_timeout(400)
        r['expanded_overflow'] = await page.evaluate(OVERFLOW_JS)
        r['config_scroll'] = await page.evaluate("""() => {
          return [...document.querySelectorAll('.client-config')].map(e => {
            const r = e.getBoundingClientRect();
            return {clientW: Math.round(r.width), scrollW: e.scrollWidth, over: e.scrollWidth - Math.round(r.width)};
          });
        }""")
        await page.screenshot(path=f'{SHOTS}/coder_mobile_{dev["width"]}_expanded{TAG}.png', full_page=True)
        await page.screenshot(path=f'{SHOTS}/coder_mobile_{dev["width"]}_expanded_fold{TAG}.png', full_page=False)
    except Exception as e:
        r['expanded_overflow'] = f'ERR {e}'
    await page.close()


async def main():
    srv = subprocess.Popen(['python3', '-m', 'http.server', str(PORT)], cwd=ROOT,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.0)
    report = {}
    try:
        async with async_playwright() as p:
            browser = await p.firefox.launch(headless=True)
            for name, d in DEVICES.items():
                ctx = await browser.new_context(
                    viewport={'width': d['width'], 'height': d['height']},
                    device_scale_factor=d['dsf'], has_touch=True, user_agent=UA)
                await audit_index(ctx, name, d, report)
                await audit_coder(ctx, name, d, report)
                await ctx.close()
            await browser.close()
    finally:
        srv.terminate()
    print(json.dumps(report, indent=1))

asyncio.run(main())
