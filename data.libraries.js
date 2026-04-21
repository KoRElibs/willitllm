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
// source       URL to the model announcement or documentation page. null if none.
//
// multimodal   true if the model accepts non-text input (images, audio, etc.).
//              Omit for text-only models.
//
// ─────────────────────────────────────────────────────────────────────────────
const LIBRARIES = [
  { "library": "llama3.2",         "organization": "Meta",                  "origin": "USA",         "flag": "🇺🇸",  "source": null,        "multimodal": true },
  { "library": "llama3.1",         "organization": "Meta",                  "origin": "USA",         "flag": "🇺🇸",  "source": null },
  { "library": "mistral",          "organization": "Mistral AI",            "origin": "France",      "flag": "🇫🇷",  "source": null },
  { "library": "gemma3",           "organization": "Google DeepMind",       "origin": "USA",         "flag": "🇺🇸",  "source": "https://ai.google.dev/gemma/docs/integrations/ollama", "multimodal": true },
  { "library": "phi4",             "organization": "Microsoft",             "origin": "USA",         "flag": "🇺🇸",  "source": "https://techcommunity.microsoft.com/blog/educatordeveloperblog/welcome-to-the-new-phi-4-models---microsoft-phi-4-mini--phi-4-multimodal/4386037" },
  { "library": "llama3",           "organization": "Meta",                  "origin": "USA",         "flag": "🇺🇸",  "source": null },
  { "library": "codellama",        "organization": "Meta",                  "origin": "USA",         "flag": "🇺🇸",  "source": null },
  { "library": "mixtral",          "organization": "Mistral AI",            "origin": "France",      "flag": "🇫🇷",  "source": null },
  { "library": "phi3",             "organization": "Microsoft",             "origin": "USA",         "flag": "🇺🇸",  "source": "https://devblogs.microsoft.com/agent-framework/introducing-new-ollama-connector-for-local-models/" },
  { "library": "llava",            "organization": null,                    "origin": null,          "flag": null,   "source": null,        "multimodal": true },
  { "library": "llama2",           "organization": "Meta",                  "origin": "USA",         "flag": "🇺🇸",  "source": null },
  { "library": "neural-chat",      "organization": "Intel",                 "origin": "USA",         "flag": "🇺🇸",  "source": null },
  { "library": "starling-lm",      "organization": null,                    "origin": null,          "flag": null,   "source": null },
  { "library": "orca-mini",        "organization": null,                    "origin": null,          "flag": null,   "source": null },
  { "library": "zephyr",           "organization": "HuggingFace",           "origin": "France",      "flag": "🇫🇷",  "source": null },
  { "library": "dolphin-mixtral",  "organization": null,                    "origin": null,          "flag": null,   "source": null },
  { "library": "nous-hermes2",     "organization": "Nous Research",         "origin": "USA",         "flag": "🇺🇸",  "source": null },
  { "library": "openhermes",       "organization": null,                    "origin": null,          "flag": null,   "source": null },
  { "library": "starcoder2",       "organization": "BigCode / HuggingFace", "origin": "France",      "flag": "🇫🇷",  "source": null },
  { "library": "tinyllama",        "organization": "SUTD",                  "origin": "Singapore",   "flag": "🇸🇬",  "source": null },
  { "library": "command-r",        "organization": "Cohere",                "origin": "Canada",      "flag": "🇨🇦",  "source": null },
  { "library": "granite3-dense",   "organization": "IBM",                   "origin": "USA",         "flag": "🇺🇸",  "source": null },
  { "library": "codegemma",        "organization": "Google DeepMind",       "origin": "USA",         "flag": "🇺🇸",  "source": null },
  { "library": "moondream",        "organization": null,                    "origin": null,          "flag": null,   "source": null,        "multimodal": true },
  { "library": "wizardlm2",        "organization": "Microsoft",             "origin": "USA",         "flag": "🇺🇸",  "source": null },
  { "library": "solar",            "organization": "Upstage",               "origin": "South Korea", "flag": "🇰🇷",  "source": null },
  { "library": "stablelm2",        "organization": "Stability AI",          "origin": "UK",          "flag": "🇬🇧",  "source": null },
  { "library": "smollm2",          "organization": "HuggingFace",           "origin": "France",      "flag": "🇫🇷",  "source": null },
  { "library": "mistral-nemo",     "organization": "Mistral AI",            "origin": "France",      "flag": "🇫🇷",  "source": null },
  { "library": "llama3.3",         "organization": "Meta",                  "origin": "USA",         "flag": "🇺🇸",  "source": null },
  { "library": "gemma2",           "organization": "Google DeepMind",       "origin": "USA",         "flag": "🇺🇸",  "source": null },
  { "library": "translategemma",   "organization": "Google DeepMind",       "origin": "USA",         "flag": "🇺🇸",  "source": null,        "multimodal": true },
  { "library": "codestral",        "organization": "Mistral AI",            "origin": "France",      "flag": "🇫🇷",  "source": null },
  { "library": "mistral-small",    "organization": "Mistral AI",            "origin": "France",      "flag": "🇫🇷",  "source": null },
  { "library": "devstral",         "organization": "Mistral AI",            "origin": "France",      "flag": "🇫🇷",  "source": null },
  { "library": "devstral-small-2", "organization": "Mistral AI",            "origin": "France",      "flag": "🇫🇷",  "source": null },
  { "library": "mistral-small3.1", "organization": "Mistral AI",            "origin": "France",      "flag": "🇫🇷",  "source": null,        "multimodal": true },
  { "library": "mistral-small3.2", "organization": "Mistral AI",            "origin": "France",      "flag": "🇫🇷",  "source": null,        "multimodal": true },
  { "library": "mistral-large",    "organization": "Mistral AI",            "origin": "France",      "flag": "🇫🇷",  "source": null },
  { "library": "mathstral",        "organization": "Mistral AI",            "origin": "France",      "flag": "🇫🇷",  "source": null },
  { "library": "ministral-3",      "organization": "Mistral AI",            "origin": "France",      "flag": "🇫🇷",  "source": null,        "multimodal": true },
  { "library": "magistral",        "organization": "Mistral AI",            "origin": "France",      "flag": "🇫🇷",  "source": null },
  { "library": "notus",            "organization": "Argilla",               "origin": "Spain",       "flag": "🇪🇸",  "source": null },
  { "library": "nuextract",        "organization": "NuMind",                "origin": "France",      "flag": "🇫🇷",  "source": null },
];
