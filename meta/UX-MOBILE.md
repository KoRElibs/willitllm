# will-it-llm — mobile UX review

**Tested:** 2026-08-05 · Firefox headless (Playwright), touch emulation on
**Viewports:** 375×667 (iPhone SE), 393×852 (iPhone 14 Pro), 344×882 (Fold closed)
**Reproduce:** `python3 meta/scripts/mobile_audit.py` → screenshots in `meta/cache/screenshots/`

Every measurement below is from that run. Functional defects are also filed in `BUGS.md`
(BUG-21 … BUG-24); this file holds the full picture including design-level findings.

> **Status 2026-08-05:** finding 2 (BUG-22) is **fixed** — the OS tab strip and setup block moved
> out of the result card into the collapsible `#runSection` with a `Show setup for` dropdown
> (SPEC §7.5b), and `.result-main` gained `min-width: 0`. This also resolves finding 8's OS-tab row
> and part of finding 10. Everything else below still stands. `index.html` after-shots:
> `index_mobile_*_result_after_runit.png` / `*_runopen_after_runit.png`.

---

## Blockers

### 1. Coder rows cannot be opened on a touch device — the page's payload is unreachable

`.coder-row-header` is almost entirely made of `[data-tip]` elements. On touch, the
capture-phase handler in `app.shared.js:81` calls `stopPropagation()` and shows a tooltip
instead of letting the tap reach the row's toggle.

Sampled pixel-by-pixel across each header at 393px:

| Row | Tap → tooltip | Tap → expands |
| --- | --- | --- |
| devstral-small-2:24b (★ recommended) | 313px (86%) | 50px |
| devstral:24b | 249px (69%) | 114px |
| granite-code:8b | 241px (66%) | 122px |
| phi4:14b | 241px (66%) | 122px |

The 50px that still work on the top row are not contiguous — they are the 22px of left
padding plus four 8px flex gaps. There is no target a thumb can reliably hit.

Consequence: the ready-to-paste Cline/Continue configs — the reason the page exists — are
effectively inaccessible on a phone. Screenshot `coder_mobile_375_expanded_fold.png` shows
the actual result of tapping the top row: a "Mistral AI · France" tooltip, row still closed.

Filed as [BUG-21](BUGS.md).

### 2. The result card forces the whole page to 534px at every phone width — FIXED

`.result-main` / `.result-cmd` / `.result-aside` are grid items with no `min-width: 0`, so
the track is sized by the min-content of `<pre class="ollama-cmd">` (`white-space: pre`,
497px). Document width is **534px regardless of viewport** — 375, 393 and 344 all produce it.

Verified fix direction: injecting `min-width: 0` on those three under the mobile query takes
the document from 534 → 393px and restores the intended behaviour (command block 325px
visible, 497px scrollable internally).

What the user loses off the right edge (see `index_mobile_393_result_fold.png`):

- the right border of the result card itself
- `it will code!` — the link through to the vibe coder page
- `READING · ~67–~314 WORDS/S` (cut mid-line) and the `ROUGH ESTIMATE` caveat
- the **macOS** and **Windows** OS tabs — 2 of 5 tabs, with no scroll affordance
- the tail of every line in the setup command

Everything else on the page (header, controls, VRAM bar, footer) then sits in a 393px
column inside a 534px canvas that scrolls sideways into dead space.

This also silently defeats the [BUG-07](BUGS.md) fix: `overflow-x: auto` never engages
because the element is never constrained.

Filed as [BUG-22](BUGS.md).

### 3. Coder model names truncate to nothing — including to zero width

`.coder-row-header` is `flex-wrap: nowrap` and `.coder-name` is the only child that shrinks.
Measured at 393px:

| Model | Rendered | Needed |
| --- | --- | --- |
| devstral-small-2:24b (★ recommended) | **0px** | 157px |
| devstral:24b | 49px | 94px |
| granite-code:8b | 57px | 117px |
| granite-code:20b | 57px | 125px |

The badge, the ★ RECOMMENDED tag, the flag, the speed range and the benchmark chip all keep
their full width; the name — the string you have to type after `ollama pull` — is what gets
sacrificed. On the top-ranked row it is gone entirely (`coder_mobile_375_list_fold.png`).

Allowing the header to wrap on mobile also fixes the page's 406px horizontal overflow
(verified: 406 → 393).

Filed as [BUG-23](BUGS.md).

---

## Significant

### 4. The model dropdown opens on a full screen of models that don't fit

List order is parameter-size descending, fixed at build and deliberately never re-sorted
(`index.js:214`). `openCombobox` uses `scrollIntoView({ block: 'nearest' })`, which parks the
auto-selected fitting model at the **bottom** edge of the 300px panel. On a phone the entire
visible list is therefore red ✗ entries (`index_mobile_393_combo_open.png`):

```
✗ llama3.1:405b   ✗ mistral-large:123b   ✗ llama3.2-vision:90b
✗ codellama:70b   ✗ llama3.1:70b         ✗ llama3.3:70b
   command-r:35b  ← auto-selected, at the very bottom edge
```

Six models that cannot run on the selected GPU are the first thing shown. On desktop the
taller panel hides this; on mobile it is the whole first impression. `block: 'center'` (or
`'start'`) would put the usable models on screen without touching the sort order.

### 5. iOS zooms the page on every text field

iOS Safari auto-zooms any focused input with `font-size < 16px` and does not zoom back out.
Both text inputs on the site are 12px:

- `.combobox-search` (`styles.css:206`) — the model search, the primary way to find a model
- `.coder-controls input[type="text"]` (`styles.css:740`) — the Ollama URL field

So the two main entry points each leave the user zoomed in and horizontally panned.
Filed as [BUG-24](BUGS.md).

### 6. The model list fights the keyboard

The panel is `max-height: 300px` anchored under the face at y≈372. With the keyboard up
(~336px on a 393×852 device) the usable area ends at y≈516, but the panel runs to y≈672 —
so the results are behind the keyboard exactly while you are typing to filter them.

Compounding it: 71 items at 27px each (the 44px touch floor is 1.6× that) in a 300px box.

### 7. The Ollama URL field truncates its own default value

180px wide at 375–393px; renders `http://localhost:114` — the port number is cut off. This
is a field whose entire purpose is to be read and corrected.

### 8. Tap targets are below the 44px floor almost everywhere

| Control | Size |
| --- | --- |
| ~~OS tabs (`Generic`/`Linux`/…)~~ | ~~26px tall — and 2 of 5 off-screen~~ — fixed on index (now a 39px dropdown); still present on `coder.html` |
| `▸ details` toggle | 30px tall |
| `it will code!` link | 18px tall |
| footer `about` button | 33 × 14px |
| footer links (`ollama.com`, `vibe coder`) | 14px tall |
| GPU / Context / Variant selects | 36px tall |
| coder row caret `▸` | 6px wide |

Only the capability pills (39px) come close.

### 9. Tooltips are a one-way overlay on touch — and the pills' are unreachable

Tapping a `[data-tip]` opens a 260px panel with **no close control**; it covers the three
scorecard rows below it and only dismisses when you tap somewhere else
(`index_mobile_375_tooltip.png`). Nothing tells you that.

Worse, the handler excludes anything inside a `button` or `a`
(`app.shared.js:84` — `!el.closest('button, a')`), so the capability pills' explanations are
unreachable on a phone entirely. Verified: tapping the `vision` pill leaves the tooltip
hidden. Those three tips (`coding` / `vision` / `thinking`) are exactly the ones a
first-time user needs, and they are desktop-only.

---

## Moderate

### 10. The answer lands ~490px down and nothing scrolls to it

Positions after picking a model, 375×667:

| Element | y (before) | y (after the §7.5b change) |
| --- | --- | --- |
| verdict headline | 489 | 489 |
| scorecard | 536 | 536 |
| speed / context panel | 649 | 649 |
| setup command | 794 | collapsed |
| VRAM bar | 970 | 915 |
| `▸ details` | 1144 | 1089 |

The verdict is the last thing above the fold; everything that explains it is below. Selecting
a model changes nothing in the viewport the user is actually looking at, and there is no
scroll-into-view. On the SE the effect is that the app appears not to have responded.

Collapsing the setup block took ~55px out of the page and shortened the card, but did not move
the verdict itself — that is set by the five control fields above it. Still open.

### 11. Expanded coder config overflows to 726px — nearly 2× the viewport

Once a row is open (forced open, per finding 1), the document goes to 726px at a 393px
viewport. The ① ② ③ prereq strip runs off-screen mid-sentence, and the OS tab strip is cut
after `macOS` (`coder_mobile_393_config_scrolled.png`). The KV toggle buttons
("Quick start · ~50 files" / "More context · ~195 files") wrap to two lines inside their
borders.

### 12. Type below 12px in eight distinct places

`rec-tag` 9px · `aside-label`, `coder-badge`, `coder-bench` 10px · `os-tab`, `geek-toggle`,
`cap-pill`, footer links, `page-nav` 11px · mobile selects 11px (`styles.css:1037`).

At 10px on a phone the ★ RECOMMENDED tag and the benchmark chips are borderline; the
mono face makes them read smaller than the number suggests.

### 13. The Variant field is permanently dimmed on touch

`.field-secondary` sits at `opacity: 0.55` and is restored by `:hover` or `:focus-within`
(`styles.css:130–136`). Touch has no hover, so on a phone it stays dimmed until tapped —
reading as disabled rather than de-emphasised. This is the mobile version of M1 in
`UX-FINDINGS.md`, and it is worse here because the hover escape hatch does not exist.

### 14. Info sheet does not lock the page behind it

No `overscroll-behavior: contain` and no body scroll lock, so scrolling inside the sheet
bleeds through to the page. Most visible on `coder.html`, whose sheet holds the entire
getting-started guide inside 75vh.

---

## Low

### 15. The mobile controls grid is held together by positional selectors

`styles.css:1035` targets `.controls .field:nth-child(3)` and `:nth-child(4)`. There are now
five fields; the layout is correct today (GPU 180px, Context 180px, then three full-width)
only because Variant carries its own `grid-column: 1 / -1`. Reordering or inserting a field
silently breaks it. Not a defect now — a trap.

---

## What holds up well on mobile

Verified, not assumed:

- No horizontal overflow on either page in its **initial** state at any of the three widths.
- The touch-tooltip path itself works — `(hover: none)` matches and tapping a plain
  `[data-tip]` span shows the tip (the problems are placement, dismissal, and the
  button/link exclusion, not the mechanism).
- The controls grid stacks sensibly: GPU and Context paired, capability pills / model /
  variant full-width. Pills are the only controls that already meet the touch floor.
- The `▸ details` panel reflows to one column below 601px and stays inside the viewport at
  all three widths — the widest formula line is 290px at 344px viewport.
- The Estimates tab's formula block does not overflow, even at 344px.
- The info sheet behaves as a proper bottom sheet (75vh cap, internally scrollable body,
  backdrop dismiss).
- The membar and its legend wrap cleanly and stay within the viewport.
- The combobox closes on selection and no JS errors appear on any page or width.

---

## Suggested order of work

1. BUG-22 (`min-width: 0`) — one line, kills the site-wide horizontal scroll.
2. BUG-23 + BUG-21 — both live in `.coder-row-header`; wrapping the header and giving the
   name its own line addresses the truncation, the 406px overflow, and creates the
   contiguous non-`[data-tip]` area the row needs to be tappable.
3. BUG-24 — 12px → 16px on two inputs.
4. Findings 4 and 9 — small, high-leverage (`block: 'center'`; let pills show their tips).
5. The rest as design work.
