// ─────────────────────────────────────────────────────────────────────────────
// LIBRARY DATABASE
//
// One entry per ollama library slug. Provides display metadata for the UI.
// Maintained by hand — the scraper reads but does not modify this file.
//
// ── CONSUMERS ────────────────────────────────────────────────────────────────
//
//   Browser  — loaded as <script src="data.libraries.js">, exposes LIBRARIES global.
//   Scraper  — scripts/update_models.py strips the "const LIBRARIES =" wrapper
//              and parses the body as JSON. Keys must therefore be quoted strings
//              (valid JSON). Do not change to unquoted JS shorthand.
//
// ── FIELDS ───────────────────────────────────────────────────────────────────
//
// library      Ollama library slug (the part before the colon, e.g. "llama3.2").
//
// organization Display name of the publishing organization.
//
// origin       Country or region of the publishing organization.
//
// flag         Flag emoji for the origin country. null for community projects.
//
// source       URL to the most authoritative source: paper (arxiv), official announcement,
//              or HuggingFace/GitHub page if no paper exists. null only if nothing
//              clearly authoritative can be found.
//
// multimodal   true if the model accepts non-text input (images, audio, etc.).
//              Omit for text-only models.
//
// ─────────────────────────────────────────────────────────────────────────────
const LIBRARIES = [
  { "library": "notus",            "organization": "Argilla",               "origin": "Spain",       "flag": "🇪🇸",  "source": null },
  { "library": "starcoder2",       "organization": "BigCode",               "origin": "France",      "flag": "🇫🇷",  "source": null },
  { "library": "dolphin-mixtral",  "organization": "Cognitive Computations", "origin": "USA",        "flag": "🇺🇸",  "source": "https://erichartford.com/dolphin-25-mixtral-8x7b" },
  { "library": "command-r",        "organization": "Cohere",                "origin": "Canada",      "flag": "🇨🇦",  "source": null },
  { "library": "codegemma",        "organization": "Google DeepMind",       "origin": "USA",         "flag": "🇺🇸",  "source": null },
  { "library": "gemma2",           "organization": "Google DeepMind",       "origin": "USA",         "flag": "🇺🇸",  "source": null },
  { "library": "gemma3",           "organization": "Google DeepMind",       "origin": "USA",         "flag": "🇺🇸",  "source": "https://ai.google.dev/gemma/docs/integrations/ollama", "multimodal": true },
  { "library": "translategemma",   "organization": "Google DeepMind",       "origin": "USA",         "flag": "🇺🇸",  "source": null,        "multimodal": true },
  { "library": "smollm2",          "organization": "HuggingFace",           "origin": "France",      "flag": "🇫🇷",  "source": null },
  { "library": "zephyr",           "organization": "HuggingFace",           "origin": "France",      "flag": "🇫🇷",  "source": null },
  { "library": "granite3-dense",   "organization": "IBM",                   "origin": "USA",         "flag": "🇺🇸",  "source": null },
  { "library": "neural-chat",      "organization": "Intel",                 "origin": "USA",         "flag": "🇺🇸",  "source": null },
  { "library": "moondream",        "organization": "M87 Labs",              "origin": "USA",         "flag": "🇺🇸",  "source": "https://github.com/m87-labs/moondream", "multimodal": true },
  { "library": "codellama",        "organization": "Meta",                  "origin": "USA",         "flag": "🇺🇸",  "source": null },
  { "library": "llama2",           "organization": "Meta",                  "origin": "USA",         "flag": "🇺🇸",  "source": null },
  { "library": "llama3",           "organization": "Meta",                  "origin": "USA",         "flag": "🇺🇸",  "source": null },
  { "library": "llama3.1",         "organization": "Meta",                  "origin": "USA",         "flag": "🇺🇸",  "source": null },
  { "library": "llama3.2",         "organization": "Meta",                  "origin": "USA",         "flag": "🇺🇸",  "source": null,        "multimodal": true },
  { "library": "llama3.3",         "organization": "Meta",                  "origin": "USA",         "flag": "🇺🇸",  "source": null },
  { "library": "phi3",             "organization": "Microsoft",             "origin": "USA",         "flag": "🇺🇸",  "source": "https://devblogs.microsoft.com/agent-framework/introducing-new-ollama-connector-for-local-models/" },
  { "library": "phi4",             "organization": "Microsoft",             "origin": "USA",         "flag": "🇺🇸",  "source": "https://techcommunity.microsoft.com/blog/educatordeveloperblog/welcome-to-the-new-phi-4-models---microsoft-phi-4-mini--phi-4-multimodal/4386037" },
  { "library": "wizardlm2",        "organization": "Microsoft",             "origin": "USA",         "flag": "🇺🇸",  "source": null },
  { "library": "codestral",        "organization": "Mistral AI",            "origin": "France",      "flag": "🇫🇷",  "source": null },
  { "library": "devstral",         "organization": "Mistral AI",            "origin": "France",      "flag": "🇫🇷",  "source": null },
  { "library": "devstral-small-2", "organization": "Mistral AI",            "origin": "France",      "flag": "🇫🇷",  "source": null },
  { "library": "magistral",        "organization": "Mistral AI",            "origin": "France",      "flag": "🇫🇷",  "source": null },
  { "library": "mathstral",        "organization": "Mistral AI",            "origin": "France",      "flag": "🇫🇷",  "source": null },
  { "library": "ministral-3",      "organization": "Mistral AI",            "origin": "France",      "flag": "🇫🇷",  "source": null,        "multimodal": true },
  { "library": "mistral",          "organization": "Mistral AI",            "origin": "France",      "flag": "🇫🇷",  "source": null },
  { "library": "mistral-large",    "organization": "Mistral AI",            "origin": "France",      "flag": "🇫🇷",  "source": null },
  { "library": "mistral-nemo",     "organization": "Mistral AI",            "origin": "France",      "flag": "🇫🇷",  "source": null },
  { "library": "mistral-small",    "organization": "Mistral AI",            "origin": "France",      "flag": "🇫🇷",  "source": null },
  { "library": "mistral-small3.1", "organization": "Mistral AI",            "origin": "France",      "flag": "🇫🇷",  "source": null,        "multimodal": true },
  { "library": "mistral-small3.2", "organization": "Mistral AI",            "origin": "France",      "flag": "🇫🇷",  "source": null,        "multimodal": true },
  { "library": "mixtral",          "organization": "Mistral AI",            "origin": "France",      "flag": "🇫🇷",  "source": null },
  { "library": "nous-hermes2",     "organization": "Nous Research",         "origin": "USA",         "flag": "🇺🇸",  "source": null },
  { "library": "openhermes",       "organization": "Nous Research",         "origin": "USA",         "flag": "🇺🇸",  "source": "https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B" },
  { "library": "nuextract",        "organization": "NuMind",                "origin": "France",      "flag": "🇫🇷",  "source": null },
  { "library": "orca-mini",        "organization": "Pankaj Mathur",         "origin": "India",       "flag": "🇮🇳",  "source": "https://huggingface.co/pankajmathur/orca_mini_3b" },
  { "library": "tinyllama",        "organization": "SUTD",                  "origin": "Singapore",   "flag": "🇸🇬",  "source": null },
  { "library": "stablelm2",        "organization": "Stability AI",          "origin": "UK",          "flag": "🇬🇧",  "source": null },
  { "library": "starling-lm",      "organization": "UC Berkeley",           "origin": "USA",         "flag": "🇺🇸",  "source": "https://starling.cs.berkeley.edu/" },
  { "library": "solar",            "organization": "Upstage",               "origin": "South Korea", "flag": "🇰🇷",  "source": null },
  { "library": "llava",            "organization": "UW-Madison / MSR",     "origin": "USA",         "flag": "🇺🇸",  "source": "https://arxiv.org/abs/2304.08485", "multimodal": true },
];
