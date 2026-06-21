// Two consumers:
//   1. Browser — loaded as <script src="data.libraries.js">, exposes LIBRARIES global
//   2. Python scraper (scripts/update_models.py) — strips "const LIBRARIES =" wrapper and parses as JSON
// Keys must therefore be quoted (valid JSON). Do not change to unquoted JS shorthand.
//
// capabilities  — sourced exclusively from ollama.com/library x-test-capability badges.
//                  Values: tools | vision | thinking | embedding | audio
//                  Omitted when empty. Do NOT set manually — run --capabilities to refresh.
// pulls         — download count string from ollama.com/library x-test-pull-count.
//                  Omitted when not available. Do NOT set manually.
const LIBRARIES = [
  { "library": "notus", "organization": "Argilla", "origin": "Spain", "flag": "\ud83c\uddea\ud83c\uddf8", "source": null, "pulls": "517.3K" },
  { "library": "starcoder2", "organization": "BigCode", "origin": "France", "flag": "\ud83c\uddeb\ud83c\uddf7", "source": null, "pulls": "2.9M" },
  { "library": "dolphin-mixtral", "organization": "Cognitive Computations", "origin": "USA", "flag": "\ud83c\uddfa\ud83c\uddf8", "source": "https://erichartford.com/dolphin-25-mixtral-8x7b", "pulls": "1.8M" },
  { "library": "command-r", "organization": "Cohere", "origin": "Canada", "flag": "\ud83c\udde8\ud83c\udde6", "source": null, "capabilities": ["tools"], "pulls": "1.4M" },
  { "library": "codegemma", "organization": "Google DeepMind", "origin": "USA", "flag": "\ud83c\uddfa\ud83c\uddf8", "source": null, "pulls": "3M" },
  { "library": "gemma2", "organization": "Google DeepMind", "origin": "USA", "flag": "\ud83c\uddfa\ud83c\uddf8", "source": null, "pulls": "25.9M" },
  { "library": "gemma3", "organization": "Google DeepMind", "origin": "USA", "flag": "\ud83c\uddfa\ud83c\uddf8", "source": "https://ai.google.dev/gemma/docs/integrations/ollama", "capabilities": ["vision"] },
  { "library": "gemma4", "organization": "Google DeepMind", "origin": "USA", "flag": "\ud83c\uddfa\ud83c\uddf8", "source": "https://ai.google.dev/gemma/docs/integrations/ollama", "capabilities": ["audio", "thinking", "tools", "vision"] },
  { "library": "translategemma", "organization": "Google DeepMind", "origin": "USA", "flag": "\ud83c\uddfa\ud83c\uddf8", "source": null, "capabilities": ["vision"] },
  { "library": "smollm2", "organization": "HuggingFace", "origin": "France", "flag": "\ud83c\uddeb\ud83c\uddf7", "source": null, "capabilities": ["tools"] },
  { "library": "zephyr", "organization": "HuggingFace", "origin": "France", "flag": "\ud83c\uddeb\ud83c\uddf7", "source": null, "pulls": "1.2M" },
  { "library": "granite3-dense", "organization": "IBM", "origin": "USA", "flag": "\ud83c\uddfa\ud83c\uddf8", "source": null, "capabilities": ["tools"], "pulls": "973.3K" },
  { "library": "neural-chat", "organization": "Intel", "origin": "USA", "flag": "\ud83c\uddfa\ud83c\uddf8", "source": null, "pulls": "1M" },
  { "library": "moondream", "organization": "M87 Labs", "origin": "USA", "flag": "\ud83c\uddfa\ud83c\uddf8", "source": "https://github.com/m87-labs/moondream", "capabilities": ["vision"], "pulls": "1.3M" },
  { "library": "codellama", "organization": "Meta", "origin": "USA", "flag": "\ud83c\uddfa\ud83c\uddf8", "source": null },
  { "library": "llama2", "organization": "Meta", "origin": "USA", "flag": "\ud83c\uddfa\ud83c\uddf8", "source": null, "pulls": "7.1M" },
  { "library": "llama3", "organization": "Meta", "origin": "USA", "flag": "\ud83c\uddfa\ud83c\uddf8", "source": null, "pulls": "24.4M" },
  { "library": "llama3.1", "organization": "Meta", "origin": "USA", "flag": "\ud83c\uddfa\ud83c\uddf8", "source": null, "capabilities": ["tools"] },
  { "library": "llama3.2", "organization": "Meta", "origin": "USA", "flag": "\ud83c\uddfa\ud83c\uddf8", "source": null, "capabilities": ["tools"], "pulls": "73.4M" },
  { "library": "llama3.3", "organization": "Meta", "origin": "USA", "flag": "\ud83c\uddfa\ud83c\uddf8", "source": null, "capabilities": ["tools"], "pulls": "4M" },
  { "library": "phi3", "organization": "Microsoft", "origin": "USA", "flag": "\ud83c\uddfa\ud83c\uddf8", "source": "https://devblogs.microsoft.com/agent-framework/introducing-new-ollama-connector-for-local-models/", "pulls": "17.7M" },
  { "library": "phi4", "organization": "Microsoft", "origin": "USA", "flag": "\ud83c\uddfa\ud83c\uddf8", "source": "https://techcommunity.microsoft.com/blog/educatordeveloperblog/welcome-to-the-new-phi-4-models---microsoft-phi-4-mini--phi-4-multimodal/4386037", "pulls": "7.6M" },
  { "library": "wizardlm2", "organization": "Microsoft", "origin": "USA", "flag": "\ud83c\uddfa\ud83c\uddf8", "source": null, "pulls": "1.1M" },
  { "library": "codestral", "organization": "Mistral AI", "origin": "France", "flag": "\ud83c\uddeb\ud83c\uddf7", "source": null, "coding_role": "fim", "pulls": "1.3M" },
  { "library": "devstral", "organization": "Mistral AI", "origin": "France", "flag": "\ud83c\uddeb\ud83c\uddf7", "source": null, "capabilities": ["tools"], "coding_role": "agent", "pulls": "963.7K" },
  { "library": "devstral-small-2", "organization": "Mistral AI", "origin": "France", "flag": "\ud83c\uddeb\ud83c\uddf7", "source": null, "capabilities": ["tools", "vision"], "coding_role": "agent" },
  { "library": "magistral", "organization": "Mistral AI", "origin": "France", "flag": "\ud83c\uddeb\ud83c\uddf7", "source": null, "capabilities": ["thinking", "tools"], "pulls": "1.4M" },
  { "library": "mathstral", "organization": "Mistral AI", "origin": "France", "flag": "\ud83c\uddeb\ud83c\uddf7", "source": null, "pulls": "525.2K" },
  { "library": "ministral-3", "organization": "Mistral AI", "origin": "France", "flag": "\ud83c\uddeb\ud83c\uddf7", "source": null, "capabilities": ["tools", "vision"] },
  { "library": "mistral", "organization": "Mistral AI", "origin": "France", "flag": "\ud83c\uddeb\ud83c\uddf7", "source": null, "capabilities": ["tools"], "pulls": "30.3M" },
  { "library": "mistral-large", "organization": "Mistral AI", "origin": "France", "flag": "\ud83c\uddeb\ud83c\uddf7", "source": null, "capabilities": ["tools"], "pulls": "1.2M" },
  { "library": "mistral-nemo", "organization": "Mistral AI", "origin": "France", "flag": "\ud83c\uddeb\ud83c\uddf7", "source": null, "capabilities": ["tools"], "pulls": "5M" },
  { "library": "mistral-small", "organization": "Mistral AI", "origin": "France", "flag": "\ud83c\uddeb\ud83c\uddf7", "source": null, "capabilities": ["tools"], "pulls": "3.1M" },
  { "library": "mistral-small3.1", "organization": "Mistral AI", "origin": "France", "flag": "\ud83c\uddeb\ud83c\uddf7", "source": null, "capabilities": ["tools", "vision"] },
  { "library": "mistral-small3.2", "organization": "Mistral AI", "origin": "France", "flag": "\ud83c\uddeb\ud83c\uddf7", "source": null, "capabilities": ["tools", "vision"], "pulls": "2.3M" },
  { "library": "mixtral", "organization": "Mistral AI", "origin": "France", "flag": "\ud83c\uddeb\ud83c\uddf7", "source": null, "capabilities": ["tools"], "pulls": "2.7M" },
  { "library": "nous-hermes2", "organization": "Nous Research", "origin": "USA", "flag": "\ud83c\uddfa\ud83c\uddf8", "source": null, "pulls": "1M" },
  { "library": "openhermes", "organization": "Nous Research", "origin": "USA", "flag": "\ud83c\uddfa\ud83c\uddf8", "source": "https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B", "pulls": "1.1M" },
  { "library": "nuextract", "organization": "NuMind", "origin": "France", "flag": "\ud83c\uddeb\ud83c\uddf7", "source": null, "pulls": "509.7K" },
  { "library": "orca-mini", "organization": "Pankaj Mathur", "origin": "India", "flag": "\ud83c\uddee\ud83c\uddf3", "source": "https://huggingface.co/pankajmathur/orca_mini_3b" },
  { "library": "tinyllama", "organization": "SUTD", "origin": "Singapore", "flag": "\ud83c\uddf8\ud83c\uddec", "source": null, "pulls": "5M" },
  { "library": "stablelm2", "organization": "Stability AI", "origin": "UK", "flag": "\ud83c\uddec\ud83c\udde7", "source": null, "pulls": "976.5K" },
  { "library": "starling-lm", "organization": "UC Berkeley", "origin": "USA", "flag": "\ud83c\uddfa\ud83c\uddf8", "source": "https://starling.cs.berkeley.edu/", "pulls": "926.5K" },
  { "library": "solar", "organization": "Upstage", "origin": "South Korea", "flag": "\ud83c\uddf0\ud83c\uddf7", "source": null, "pulls": "923.5K" },
  { "library": "llava", "organization": "UW-Madison / MSR", "origin": "USA", "flag": "\ud83c\uddfa\ud83c\uddf8", "source": "https://arxiv.org/abs/2304.08485", "capabilities": ["vision"] },
];
