# will-it-llm — todo

## UX

- **Tier-based variant selector (mobile)** — replace the variant `<select>` with a tier-first
  control: `Smallest · Fast · Balanced · Quality · Max`. Selecting a tier auto-picks the best
  matching quantization for that model. Power users get a "Custom" fallback to pick the exact
  quant. Removes the need to know quantization names entirely for most users.

- **Searchable model dropdown** — text input that filters the model list in real-time. The
  current `<select>` with 100+ models is painful to navigate. Filter by name, then pick from
  the filtered set. Keyboard-friendly.

- **"What fits?" ranked list mode** — flip the interaction: given the selected GPU, show all
  models ranked by context fit %, with speed and quality scores visible. Lets users discover
  what they can run rather than checking one model at a time. Sortable by fit, speed, quality.
