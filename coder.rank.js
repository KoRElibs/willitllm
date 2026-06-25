// ─── CODER.RANK — model filtering, ranking, and data aggregation
//
// Depends on:  MODELS, QUANT_INFO (data files),
//              app.calc.js (OVERHEAD_GB, autoKvBpe, calcMaxContext,
//                           calcSpeedEstimates),
//              app.util.js (getLibMeta, getFlashOk)
// Provides:    CODING_ROLES, isCodingModel, fmtCtxCoding,
//              codingRank, buildEntries

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

// Weighted coding score 0–1: speed 50%, context 30%, quality 20%.
// Speed weighted highest because agentic coding sessions chain 30–100 sequential tool calls.
function codingRank(genLo, maxCtx, quality) {
  const speedNorm = Math.min(1, (genLo || 0) / 30);
  const ctxNorm   = Math.min(1, maxCtx / 65536);
  const qualNorm  = (quality || 5) / 10;
  return speedNorm * 0.5 + ctxNorm * 0.3 + qualNorm * 0.2;
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
    const score     = speedEsts ? codingRank(speedEsts.genLo, ctx.maxCtx, quantInfo?.quality) : 0;
    const ctxF16    = calcMaxContext(model, vramGB, 2, weightsGB);

    const entry  = { model, lib, variant, weightsGB, bpe, ctx, ctxF16, fits: true, speedEsts, quantInfo, score };
    const bucket = lib.coding_role === 'fim'  ? fim
                 : lib.coding_role === 'code' ? code
                 : agent;
    bucket.push(entry);
  });

  const sortFn = (a, b) => b.score - a.score;
  agent.sort(sortFn);
  code.sort(sortFn);
  fim.sort(sortFn);
  return { agent, code, fim };
}
