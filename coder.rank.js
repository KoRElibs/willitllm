// ─── CODER.RANK — model filtering, ranking, and data aggregation
//
// Depends on:  MODELS, QUANT_INFO (data files),
//              app.calc.js (OVERHEAD_GB, autoKvBpe, calcMaxContext,
//                           calcSpeedEstimates),
//              app.util.js (getLibMeta, getFlashOk, metricLabel)
// Provides:    CODING_ROLES, isCodingModel, fmtCtxCoding,
//              codingRank, releasedRank, whyLine, buildEntries

const CODING_ROLES = ['agent', 'code', 'fim'];

function isCodingModel(lib) {
  return CODING_ROLES.includes(lib.coding_role);
}

// Context in developer units: files when ≥5 files, lines below that.
// ~1000 tokens per average source file; ~3 tokens per line of code.
function fmtCtxCoding(maxCtx) {
  const files = Math.round(maxCtx / 1000);
  if (files >= 5) return `~${Math.round(files / 5) * 5} files`;
  const lines = Math.round(maxCtx / 3 / 100) * 100;
  return `~${lines.toLocaleString()} lines`;
}

// ── Ranking weights ─────────────────────────────────────────────────────────
//
// Capability leads — it's the thing you actually care about (will the model do
// the task), and a benchmark score is an *absolute* measure that's directly
// comparable across model sizes within one metric. Speed is a *saturating*
// usability factor, not a linear good: ≥~30 tok/s reads as "fast enough" for an
// agentic loop, and more barely helps; below that, slowness genuinely hurts (the
// SPEED_TARGET cap encodes this). Context likewise saturates once it comfortably
// holds a working set. We deliberately add NO size/VRAM term — size is already
// reflected (bigger ⇒ slower via speed, doesn't-fit ⇒ filtered out), so a size
// penalty would double-count.
const ROLE_WEIGHTS = {
  agent: { cap: 0.45, speed: 0.30, ctx: 0.25 },  // loops chain tool calls — speed stays strong
  code:  { cap: 0.50, speed: 0.25, ctx: 0.25 },  // chat/generation — capability matters most
  fim:   { cap: 0.00, speed: 0.65, ctx: 0.35 },  // autocomplete — no benchmark fits; latency leads
};
const SPEED_TARGET_TPS  = 30;      // tok/s at which the speed term saturates
const CTX_TARGET_TOKENS = 131072;  // context at which the context term saturates

// Canonical {metric, protocol} per role. A capability score counts in the ranking
// ONLY if the library's capability_metric AND capability_protocol match here — this
// is what guarantees we never compare pass@1 vs pass@5, or SWE-bench vs HumanEval.
// `fim` has no entry: no benchmark cleanly measures autocomplete, so FIM is never
// capability-scored (cap weight is 0). A model whose recorded score isn't the
// canonical pair is treated as "unscored" — honest, rather than falsely comparable.
const ROLE_CANONICAL = {
  agent: { metric: 'swe-bench-verified', protocol: 'pass@1' },
  code:  { metric: 'humaneval',          protocol: 'pass@1' },
};

// The capability % usable for ranking this lib in this role — returned only when the
// recorded {metric, protocol} is the role's canonical pair; null ⇒ unscored.
function usableCapability(lib, role) {
  const c = ROLE_CANONICAL[role];
  if (!c || lib.capability == null) return null;
  return (lib.capability_metric === c.metric && lib.capability_protocol === c.protocol)
    ? lib.capability : null;
}

// Release recency as a sortable integer (YYYY*12 + MM), parsed from the YYMM
// date Mistral/others embed in variant group/tag strings (e.g. "small-2505" →
// 2025-05, "instruct-2512" → 2025-12). Reproducible from already-scraped data;
// no hand-entered dates. Used ONLY as a deterministic tiebreak among models
// that are otherwise equal on the ranked factors ("if comparable, go for date").
function releasedRank(model) {
  let best = 0;
  (model.variants || []).forEach(v => {
    const s = `${v.group || ''} ${v.tag || ''}`;
    const re = /\b(\d{2})(\d{2})\b/g;
    let m;
    while ((m = re.exec(s)) !== null) {
      const yy = +m[1], mm = +m[2];
      if (mm >= 1 && mm <= 12) best = Math.max(best, (2000 + yy) * 12 + mm);
    }
  });
  return best;
}

// Weighted coding score 0–1, capability-led, per role (see ROLE_WEIGHTS).
// Capability is the role's canonical-metric benchmark % (see usableCapability);
// an unscored model falls back to the quantization-quality proxy, scaled down so a
// benchmarked model is preferred over an unscored one. (For `fim`, cap weight is 0,
// so the capability term — fallback included — never affects the score.)
function codingRank(role, lib, quality, genLo, maxCtx) {
  const w   = ROLE_WEIGHTS[role] || ROLE_WEIGHTS.agent;
  const cap = usableCapability(lib, role);
  const capNorm = cap != null
    ? Math.min(1, cap / 100)
    : ((quality || 5) / 10) * 0.5;            // conservative fallback (unscored)
  const speedNorm = Math.min(1, (genLo || 0) / SPEED_TARGET_TPS);
  const ctxNorm   = Math.min(1, maxCtx / CTX_TARGET_TOKENS);
  return capNorm * w.cap + speedNorm * w.speed + ctxNorm * w.ctx;
}

// One plain-language clause explaining a row's standing, relative to the bucket
// leader. Used for the "why" line under the model name.
function whyLine(entry, bucket) {
  const role   = entry.lib.coding_role;
  const top    = bucket[0];
  const label  = metricLabel(entry.lib.capability_metric);
  const cap    = usableCapability(entry.lib, role);
  const topCap = usableCapability(top.lib, role);

  if (entry === top) {
    const second    = bucket[1];
    const secondCap = second ? usableCapability(second.lib, role) : null;
    if (cap != null && (secondCap == null || cap >= secondCap)) {
      return `Best ${label} score that fits your GPU (${cap}%)`;
    }
    return 'Best overall fit for your GPU — capability, speed and context combined';
  }

  // Non-leading rows: state the trade-off versus the recommended model.
  const slower = (entry.speedEsts?.genLo || 0) < (top.speedEsts?.genLo || 0);
  if (cap != null && topCap != null && cap < topCap) {
    return slower
      ? `Lower ${label} (${cap}%) and not faster than the pick`
      : `Faster, but lower ${label} (${cap}%)`;
  }
  if (cap != null && topCap != null && cap > topCap) {
    return `Higher ${label} (${cap}%) but slower / less context here`;
  }
  if (cap != null) return `${label} ${cap}%`;
  return slower
    ? 'Ranked below the pick on speed & context'
    : 'Comparable fit — ranked on speed & context';
}

function buildEntries(vramGB, flashOk) {
  const agent = [];
  const code  = [];
  const fim   = [];

  MODELS.forEach(model => {
    const lib = getLibMeta(model);
    if (!isCodingModel(lib)) return;

    const variant = model.variants?.[0];
    if (!variant) return;

    const weightsGB = variant.weights_gb;
    if (weightsGB >= vramGB - OVERHEAD_GB) return;

    const bpe       = autoKvBpe(model, vramGB, weightsGB, null, flashOk);
    const ctx       = calcMaxContext(model, vramGB, bpe, weightsGB);
    const quantInfo = QUANT_INFO[variant.quantization];
    const speedEsts = calcSpeedEstimates(model, variant, vramGB, quantInfo, ctx.maxCtx, ctx.kvCacheGB, bpe);
    const score     = speedEsts
      ? codingRank(lib.coding_role, lib, quantInfo?.quality, speedEsts.genLo, ctx.maxCtx)
      : 0;
    const ctxF16    = calcMaxContext(model, vramGB, 2, weightsGB);

    const entry  = { model, lib, variant, weightsGB, bpe, ctx, ctxF16, fits: true, speedEsts, quantInfo, score };
    const bucket = lib.coding_role === 'fim'  ? fim
                 : lib.coding_role === 'code' ? code
                 : agent;
    bucket.push(entry);
  });

  // Score desc; release date (newest first) breaks ties among equal scores.
  const sortFn = (a, b) => (b.score - a.score) || (releasedRank(b.model) - releasedRank(a.model));
  agent.sort(sortFn);
  code.sort(sortFn);
  fim.sort(sortFn);
  return { agent, code, fim };
}
