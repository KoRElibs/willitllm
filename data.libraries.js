// Two consumers:
//   1. Browser — loaded as <script src="data.libraries.js">, exposes LIBRARIES global
//   2. Python scraper (scripts/update_models.py) — strips "const LIBRARIES =" wrapper and parses as JSON
// Keys must therefore be quoted (valid JSON). Do not change to unquoted JS shorthand.
//
// capabilities  — sourced exclusively from ollama.com/library x-test-capability badges.
//                  Values: tools | vision | thinking | embedding | audio
//                  "tools" displays as "AGENT" in the UI (index.html capability pill + coder.html badge).
//                  Omitted when empty. Do NOT set manually — run --capabilities to refresh.
// pulls         — download count string from ollama.com/library x-test-pull-count.
//                  Omitted when not available. Do NOT set manually.
// coding_role   — agent | code | fim. Curated by hand (human judgement); drives the
//                  coder.html model list. Preserved across scraper runs. Omit for non-coding libs.
//                  Note: coding_role:"agent" and capabilities:["tools"] are related but NOT the same.
//                  A model can have tools capability without being a coding agent (e.g. command-r).
//                  Set coding_role:"agent" only for models purpose-trained for coding agent loops.
// capability    — headline benchmark score (0-100) for the model family, used to rank models.
//   capability_metric    RESEARCHED & CITED from an authoritative source (HuggingFace model card,
//   capability_protocol  vendor paper/blog) — NEVER from chat/LLM memory. The ranking only uses a
//   capability_ref       score when its {metric, protocol} EXACTLY matches the role's canonical
//   capability_source    pair (see ROLE_CANONICAL in coder.rank.js); otherwise the model is
//                  "unscored" and ranked on speed + context + recency. Canonical pairs:
//                    agent → swe-bench-verified, pass@1   code → humaneval, pass@1
//                    general (non-coding, index chip only) → mmlu, 5-shot
//                  FIM has NO canonical metric — do not score fim libraries (no benchmark cleanly
//                  measures autocomplete). A non-canonical score (e.g. humaneval-plus, or pass@5)
//                  must be left OUT, not recorded — unscored is honest, a mismatched score is not.
//                  capability_protocol is MANDATORY whenever capability is set. capability_ref
//                  names the exact model measured (scores are family-level — smaller sizes score
//                  lower). capability_source is a mandatory URL. Full procedure:
//                  meta/scripts/update-models.md §"Capability scoring". Preserved across scraper runs.
// (flag emoji lives in data.flags.js, keyed by `origin` — not stored per-library.)
const LIBRARIES = [
  { "library": "starcoder2", "organization": "BigCode", "origin": "France", "source": null, "coding_role": "fim", "pulls": "2.9M" },
  { "library": "command-r", "organization": "Cohere", "origin": "Canada", "source": null, "capabilities": ["tools"], "pulls": "1.4M" },
  { "library": "codegemma", "organization": "Google DeepMind", "origin": "USA", "source": null, "coding_role": "code", "capability": 56.1, "capability_metric": "humaneval", "capability_protocol": "pass@1", "capability_ref": "CodeGemma-7B-it", "capability_source": "https://huggingface.co/google/codegemma-7b-it", "pulls": "3M" },
  { "library": "gemma2", "organization": "Google DeepMind", "origin": "USA", "source": null, "capability": 71.3, "capability_metric": "mmlu", "capability_protocol": "5-shot", "capability_ref": "Gemma 2 9B", "capability_source": "https://arxiv.org/abs/2408.00118", "pulls": "26.1M" },
  { "library": "gemma3", "organization": "Google DeepMind", "origin": "USA", "source": "https://ai.google.dev/gemma/docs/integrations/ollama", "capabilities": ["vision"] },
  { "library": "gemma3n", "organization": "Google DeepMind", "origin": "USA", "source": null, "pulls": "1.8M" },
  { "library": "gemma4", "organization": "Google DeepMind", "origin": "USA", "source": "https://ai.google.dev/gemma/docs/integrations/ollama", "capabilities": ["audio", "thinking", "tools", "vision"] },
  { "library": "translategemma", "organization": "Google DeepMind", "origin": "USA", "source": null, "capabilities": ["vision"] },
  { "library": "smollm2", "organization": "HuggingFace", "origin": "France", "source": null, "capabilities": ["tools"] },
  { "library": "granite-code", "organization": "IBM", "origin": "USA", "source": null, "coding_role": "code", "capability": 57.9, "capability_metric": "humaneval", "capability_protocol": "pass@1", "capability_ref": "granite-8b-code-instruct", "capability_source": "https://arxiv.org/abs/2405.04324" },
  { "library": "granite3-dense", "organization": "IBM", "origin": "USA", "source": null, "capabilities": ["tools"], "pulls": "974.7K" },
  { "library": "granite3.3", "organization": "IBM", "origin": "USA", "source": null, "capabilities": ["tools"], "pulls": "1M" },
  { "library": "moondream", "organization": "M87 Labs", "origin": "USA", "source": "https://github.com/m87-labs/moondream", "capabilities": ["vision"], "pulls": "1.3M" },
  { "library": "codellama", "organization": "Meta", "origin": "USA", "source": null, "coding_role": "code", "capability": 42.7, "capability_metric": "humaneval", "capability_protocol": "pass@1", "capability_ref": "CodeLlama-13B-Instruct", "capability_source": "https://arxiv.org/abs/2308.12950" },
  { "library": "llama3.1", "organization": "Meta", "origin": "USA", "source": null, "capabilities": ["tools"], "capability": 69.4, "capability_metric": "mmlu", "capability_protocol": "5-shot", "capability_ref": "Llama-3.1-8B", "capability_source": "https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md" },
  { "library": "llama3.2", "organization": "Meta", "origin": "USA", "source": null, "capabilities": ["tools"], "capability": 63.4, "capability_metric": "mmlu", "capability_protocol": "5-shot", "capability_ref": "Llama-3.2-3B", "capability_source": "https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md", "pulls": "73.8M" },
  { "library": "llama3.2-vision", "organization": "Meta", "origin": "USA", "source": null, "capabilities": ["vision"], "pulls": "4.7M" },
  { "library": "llama3.3", "organization": "Meta", "origin": "USA", "source": null, "capabilities": ["tools"], "pulls": "4M" },
  { "library": "phi4", "organization": "Microsoft", "origin": "USA", "source": "https://techcommunity.microsoft.com/blog/educatordeveloperblog/welcome-to-the-new-phi-4-models---microsoft-phi-4-mini--phi-4-multimodal/4386037", "coding_role": "code", "capability": 82.6, "capability_metric": "humaneval", "capability_protocol": "pass@1", "capability_ref": "phi-4 (14B)", "capability_source": "https://arxiv.org/abs/2412.08905", "pulls": "7.6M" },
  { "library": "phi4-mini", "organization": "Microsoft", "origin": "USA", "source": null, "capabilities": ["tools"], "pulls": "1.2M" },
  { "library": "falcon3", "organization": "TII", "origin": "UAE", "source": null },
  { "library": "codestral", "organization": "Mistral AI", "origin": "France", "source": null, "coding_role": "fim", "pulls": "1.3M" },
  { "library": "devstral", "organization": "Mistral AI", "origin": "France", "source": null, "capabilities": ["tools"], "coding_role": "agent", "capability": 46.8, "capability_metric": "swe-bench-verified", "capability_protocol": "pass@1", "capability_ref": "Devstral-Small-2505 (24B)", "capability_source": "https://huggingface.co/mistralai/Devstral-Small-2505", "pulls": "965.5K" },
  { "library": "devstral-small-2", "organization": "Mistral AI", "origin": "France", "source": null, "capabilities": ["tools", "vision"], "coding_role": "agent", "capability": 68.0, "capability_metric": "swe-bench-verified", "capability_protocol": "pass@1", "capability_ref": "Devstral-Small-2-24B-Instruct-2512", "capability_source": "https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512" },
  { "library": "magistral", "organization": "Mistral AI", "origin": "France", "source": null, "capabilities": ["thinking", "tools"], "pulls": "1.4M" },
  { "library": "mathstral", "organization": "Mistral AI", "origin": "France", "source": null, "pulls": "525.9K" },
  { "library": "ministral-3", "organization": "Mistral AI", "origin": "France", "source": null, "capabilities": ["tools", "vision"] },
  { "library": "mistral", "organization": "Mistral AI", "origin": "France", "source": null, "capabilities": ["tools"], "pulls": "30.4M" },
  { "library": "mistral-large", "organization": "Mistral AI", "origin": "France", "source": null, "capabilities": ["tools"], "pulls": "1.2M" },
  { "library": "mistral-nemo", "organization": "Mistral AI", "origin": "France", "source": null, "capabilities": ["tools"], "pulls": "5.1M" },
  { "library": "mistral-small", "organization": "Mistral AI", "origin": "France", "source": null, "capabilities": ["tools"], "pulls": "3.1M" },
  { "library": "mistral-small3.1", "organization": "Mistral AI", "origin": "France", "source": null, "capabilities": ["tools", "vision"] },
  { "library": "mistral-small3.2", "organization": "Mistral AI", "origin": "France", "source": null, "capabilities": ["tools", "vision"], "coding_role": "code", "pulls": "2.3M" },
  { "library": "mixtral", "organization": "Mistral AI", "origin": "France", "source": null, "capabilities": ["tools"], "pulls": "2.7M" },
];
