# will-it-llm — todo

## UX

- **Tier-based variant selector (mobile)** — replace the variant `<select>` with a tier-first
  control: `Smallest · Fast · Balanced · Quality · Max`. Selecting a tier auto-picks the best
  matching quantization for that model. Power users get a "Custom" fallback to pick the exact
  quant. Removes the need to know quantization names entirely for most users.
