// ─── APP.FMT — pure formatting helpers
//
// No DOM access, no side effects. Safe to call from any context.
//
// Depends on:  (nothing)
// Provides:    fmtGB, fmtSpeed, fmtSpeedHuman, fmtSpeechPace,
//              fmtCtxWords, fmtCtxPages, fmtTokensHuman, fmtCtx,
//              bar10, colorForScore

function fmtGB(n) { return parseFloat(n.toFixed(1)) + ' GB'; }

function fmtSpeed(lo, hi) {
  const fmt = n => n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(Math.round(n));
  return lo === hi ? `~${fmt(lo)} t/s` : `~${fmt(lo)}–${fmt(hi)} t/s`;
}

// Converts tokens/s → words/s (~0.75 words per token)
function fmtSpeedHuman(lo, hi) {
  const fmt = n => {
    const w = Math.round(n * 0.75);
    return w >= 1000 ? `~${(w / 1000).toFixed(1)}k` : `~${w}`;
  };
  const loS = fmt(lo), hiS = fmt(hi);
  return loS === hiS ? `${loS} words/s` : `${loS}–${hiS} words/s`;
}

// Returns a human-relatable speech pace label, e.g. "3× speech pace"
// Reference: average speech ~2.5 words/s (150 wpm)
function fmtSpeechPace(lo, hi) {
  const SPEECH_WPS = 2.5;
  const avgWps = ((lo + hi) / 2) * 0.75;
  const mult = Math.round(avgWps / SPEECH_WPS);
  if (!isFinite(mult) || mult <= 1) return 'speech pace';
  return `${mult}× speech pace`;
}

// ~0.75 words per token; ~250 words per page (333 tokens/page)
function fmtCtxWords(n) {
  const words = Math.round(n * 0.75 / 500) * 500;
  const w = words >= 1000 ? `${(words / 1000).toFixed(0)}k` : String(words);
  return `≈${w} words`;
}

function fmtCtxPages(n) {
  const pages = Math.round(n / 333 / 5) * 5;
  return `~${pages} pages`;
}

function fmtTokensHuman(n) {
  return `${fmtCtxWords(n)} · ${fmtCtxPages(n)}`;
}

function fmtCtx(n) {
  if (n >= 1000) return Math.round(n / 1000) + 'k';
  return String(n);
}

function bar10(n10) { return '■'.repeat(n10) + '□'.repeat(10 - n10); }

function colorForScore(n5) {
  return n5 >= 4 ? 'var(--green)' : n5 === 3 ? 'var(--amber)' : n5 === 2 ? 'var(--orange)' : 'var(--red)';
}
