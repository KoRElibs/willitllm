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
// (flag emoji lives in data.flags.js, keyed by `origin` — not stored per-library.)
const LIBRARIES = [
  { "library": "starcoder2", "organization": "BigCode", "origin": "France", "source": null, "coding_role": "fim", "pulls": "2.9M" },
  { "library": "command-r", "organization": "Cohere", "origin": "Canada", "source": null, "capabilities": ["tools"], "pulls": "1.4M" },
  { "library": "codegemma", "organization": "Google DeepMind", "origin": "USA", "source": null, "coding_role": "code", "pulls": "3M" },
  { "library": "gemma2", "organization": "Google DeepMind", "origin": "USA", "source": null, "pulls": "26.1M" },
  { "library": "gemma3", "organization": "Google DeepMind", "origin": "USA", "source": "https://ai.google.dev/gemma/docs/integrations/ollama", "capabilities": ["vision"] },
  { "library": "gemma3n", "organization": "Google DeepMind", "origin": "USA", "source": null, "pulls": "1.8M" },
  { "library": "gemma4", "organization": "Google DeepMind", "origin": "USA", "source": "https://ai.google.dev/gemma/docs/integrations/ollama", "capabilities": ["audio", "thinking", "tools", "vision"] },
  { "library": "translategemma", "organization": "Google DeepMind", "origin": "USA", "source": null, "capabilities": ["vision"] },
  { "library": "smollm2", "organization": "HuggingFace", "origin": "France", "source": null, "capabilities": ["tools"] },
  { "library": "granite-code", "organization": "IBM", "origin": "USA", "source": null, "coding_role": "code" },
  { "library": "granite3-dense", "organization": "IBM", "origin": "USA", "source": null, "capabilities": ["tools"], "pulls": "974.7K" },
  { "library": "granite3.3", "organization": "IBM", "origin": "USA", "source": null, "capabilities": ["tools"], "pulls": "1M" },
  { "library": "moondream", "organization": "M87 Labs", "origin": "USA", "source": "https://github.com/m87-labs/moondream", "capabilities": ["vision"], "pulls": "1.3M" },
  { "library": "codellama", "organization": "Meta", "origin": "USA", "source": null, "coding_role": "code" },
  { "library": "llama3.1", "organization": "Meta", "origin": "USA", "source": null, "capabilities": ["tools"] },
  { "library": "llama3.2", "organization": "Meta", "origin": "USA", "source": null, "capabilities": ["tools"], "pulls": "73.8M" },
  { "library": "llama3.2-vision", "organization": "Meta", "origin": "USA", "source": null, "capabilities": ["vision"], "pulls": "4.7M" },
  { "library": "llama3.3", "organization": "Meta", "origin": "USA", "source": null, "capabilities": ["tools"], "pulls": "4M" },
  { "library": "phi4", "organization": "Microsoft", "origin": "USA", "source": "https://techcommunity.microsoft.com/blog/educatordeveloperblog/welcome-to-the-new-phi-4-models---microsoft-phi-4-mini--phi-4-multimodal/4386037", "coding_role": "code", "pulls": "7.6M" },
  { "library": "phi4-mini", "organization": "Microsoft", "origin": "USA", "source": null, "capabilities": ["tools"], "pulls": "1.2M" },
  { "library": "falcon3", "organization": "TII", "origin": "UAE", "source": null },
  { "library": "codestral", "organization": "Mistral AI", "origin": "France", "source": null, "coding_role": "fim", "pulls": "1.3M" },
  { "library": "devstral", "organization": "Mistral AI", "origin": "France", "source": null, "capabilities": ["tools"], "coding_role": "agent", "pulls": "965.5K" },
  { "library": "devstral-small-2", "organization": "Mistral AI", "origin": "France", "source": null, "capabilities": ["tools", "vision"], "coding_role": "agent" },
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
