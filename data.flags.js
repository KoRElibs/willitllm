// ─────────────────────────────────────────────────────────────────────────────
// FLAGS — origin (country/region) → flag emoji
//
// Single source of truth for the flag shown next to a model's origin. Libraries
// carry only an `origin` string (data.libraries.js); the emoji is looked up here
// so the mapping lives in one place instead of being repeated on every entry.
//
// Shared by both pages. Loaded before app.calc.js.
// ─────────────────────────────────────────────────────────────────────────────
const FLAGS = {
  'USA':         '🇺🇸',
  'France':      '🇫🇷',
  'Canada':      '🇨🇦',
  'UK':          '🇬🇧',
  'India':       '🇮🇳',
  'Singapore':   '🇸🇬',
  'South Korea': '🇰🇷',
  'Spain':       '🇪🇸',
  'UAE':         '🇦🇪',
};

// Returns the flag for an origin: a mapped emoji, 🌍 for an unmapped origin,
// or 👥 (community) when the origin is null/empty.
function flagFor(origin) {
  if (!origin) return '👥';
  return FLAGS[origin] || '🌍';
}
