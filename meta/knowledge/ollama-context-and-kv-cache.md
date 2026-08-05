# Ollama — context window size and K/V cache quantization

Authoritative reference for the two Ollama settings this project calculates and emits commands for.
**Read this before changing any setup command in `osKvContent()` / `renderCmd()` (`index.render.js`)
or the assumption note in `coder.rows.js`.**

**Sources:** [docs.ollama.com/faq](https://docs.ollama.com/faq) ·
[docs.ollama.com/context-length](https://docs.ollama.com/context-length)
**Verified:** 2026-08-05. Quoted text is verbatim from those pages; anything marked *(observed)* is
from community reports, not official docs, and should be re-checked before being relied on.

---

## Quick reference

| | Context window | K/V cache quantization |
| --- | --- | --- |
| Variable | `OLLAMA_CONTEXT_LENGTH` | `OLLAMA_KV_CACHE_TYPE` |
| Values | integer token count | `f16` (default) · `q8_0` · `q4_0` |
| Server restart needed | for the env var; not for the other methods | **always** |
| Per-model / per-request override | **yes** — `num_ctx` | **no — global only** |
| Prerequisite | none | Flash Attention |
| Silent-failure risk | low | **high** — see Limitations |

---

## 1. Context window size

Four ways to set it, in ascending precedence:

| # | Method | Scope |
| --- | --- | --- |
| 1 | built-in default | server |
| 2 | `PARAMETER num_ctx` in a Modelfile | that model |
| 3 | `OLLAMA_CONTEXT_LENGTH=8192 ollama serve` | server-wide default |
| 4 | `num_ctx` in the request `options` object, or `/set parameter num_ctx 4096` in the `ollama run` REPL | that request / that REPL session |

> Precedence: API parameter > environment variable > Modelfile PARAMETER > built-in default.

The desktop app also exposes it directly — *"change the slider in the Ollama app under settings to
your desired context length."*

### Defaults are not fixed

The two doc pages disagree, so do not assume a value. The FAQ says *"By default, Ollama uses 4096
tokens."* The context-length page describes a VRAM-scaled default instead:

| VRAM | Default context |
| --- | --- |
| under 24 GB | 4k |
| 24–48 GB | 32k |
| 48 GB and above | 256k |

Cloud models are *"set to their maximum context length by default."* Treat the effective default as
unknown and set it explicitly.

### Guidance from the docs

- *"Setting a larger context length will increase the amount of memory required to run a model."*
- Prefer *"the maximum context length for a model, and avoid offloading the model to CPU."*
- Applications like *"web search, agents, and coding tools"* should get at least 64,000 tokens.

### `OLLAMA_NUM_CTX` does not exist

The name was dropped in Ollama 0.6 and is silently ignored by current releases. This project emitted
it for a long time, so every user who followed those instructions got Ollama's own default rather
than the calculated context — see BUG-26. Never emit it.

---

## 2. K/V cache quantization

> **Purpose:** Quantizes the K/V context cache to reduce memory usage when Flash Attention is enabled.

| Value | Memory vs `f16` |
| --- | --- |
| `f16` | baseline — high precision and memory usage (**default**) |
| `q8_0` | approximately 1/2 |
| `q4_0` | approximately 1/4 |

Set on the server only:

```sh
OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve
```

This is the whole reason the site can promise a longer context on the same card: halving or
quartering the cache is what frees the VRAM.

---

## 3. Flash Attention — the prerequisite

> *"Ollama uses Flash Attention automatically when the selected backend and devices support it."*

Force on with `OLLAMA_FLASH_ATTENTION=1`, off with `=0`. Flash Attention is what makes K/V cache
quantization possible — the two work together.

**Always emit `OLLAMA_FLASH_ATTENTION=1` alongside a non-`f16` cache type.** The docs describe
automatic enablement, but that depends on backend and device support, and *(observed)* setting the
cache type without Flash Attention on results in Ollama silently keeping `f16`. Forcing it costs
nothing on hardware that already had it and closes the failure mode on hardware that did not.

---

## 4. Limitations

These are the things that bite. Most of them fail *silently*.

### K/V cache type is global — this is the big one

> *"Currently this is a global option — meaning all models will run with the specified quantization
> type."*

There is **no per-model and no per-request equivalent** of `num_ctx` for the cache type. Consequences:

- A user running several models through one Ollama server cannot have `q4_0` for a large model and
  `f16` for a small one. Changing it for one changes it for all.
- Any UI that recommends a per-model cache type is implicitly asking the user to reconfigure their
  whole server for that model. Say so rather than implying it is a per-model knob.
- It cannot be carried in an editor config (Continue's `contextLength`, Cline's Context Window) the
  way context length can — those are API options, and no API option exists for cache type.

This asymmetry is why the two settings are handled differently in this codebase: context length can
ride along per request, cache type cannot.

### Server-startup only

Both `OLLAMA_KV_CACHE_TYPE` and `OLLAMA_FLASH_ATTENTION` are read when the server starts. There is
no way to change them in a running server — it must be restarted, which on a normal Linux install
means touching a systemd service, and on macOS/Windows means quitting the desktop app.

### `ollama run` is a client, not the server

Prefixing environment variables to `ollama run` does nothing. It talks to an already-running server,
which read its configuration at startup. Every one of these variables belongs on `ollama serve`, the
systemd drop-in, `launchctl setenv`, or Windows System Properties.

### Port 11434 is held by the service

On a standard Linux install the packaged systemd service owns the port, so a manual `ollama serve`
fails with `address already in use` until the service is stopped.

### Silent fallback to `f16` *(observed)*

Quantized K/V is reported to fall back to `f16` on model architectures that do not support it,
without an error. Setting `q8_0` is a request, not a guarantee — verify actual memory use rather
than assuming.

### Quality impact is not quantifiable in advance

> *"How much the cache quantization impacts the model's response quality will depend on the model
> and the task."*

Community figures — roughly 0.5% perplexity for `q8_0`, 2–5% for `q4_0`, worse at long context — are
*(observed)*, not official. Do not present them as vendor numbers.

### Never set `OLLAMA_KV_CACHE_TYPE=f16`

`f16` is already the default. Emitting it sets the value the server already has, and any ceremony
around doing so — stopping a service, running a foreground server — is unearned. Only a quantized
cache justifies touching the server at all.

---

## 5. Setting server environment variables

**Linux (systemd)** — the documented route is `systemctl edit ollama.service`, adding
`Environment="OLLAMA_HOST=0.0.0.0:11434"` under `[Service]`, then `systemctl daemon-reload` and
`systemctl restart ollama`. That opens an editor, so it cannot be copy-pasted; this project writes
the same drop-in file directly instead (`/etc/systemd/system/ollama.service.d/override.conf` via
`printf | sudo tee`), which is idempotent because `tee` overwrites.

**macOS** — `launchctl setenv OLLAMA_HOST "0.0.0.0:11434"`, then restart the Ollama application.
This project uses an inline variable on a terminal `ollama serve` after quitting the menubar app
instead: it is self-contained, needs no undo, and reopening the app restores the defaults.

**Windows** — quit Ollama, open Settings and search "environment variables", edit the variables for
your account, apply, then start Ollama from the Start menu.

---

## 6. Verifying it actually worked

Because the failure modes are silent, check rather than assume:

- `ollama ps` — the `PROCESSOR` column shows whether the model is on GPU or has spilled to CPU.
  Spilling is the usual sign that the requested context did not fit.
- Compare actual VRAM use against the predicted split. If a `q8_0` recommendation is using roughly
  the `f16` amount of cache, Flash Attention is not on and the setting was ignored.

---

## 7. How this project uses it

See `SPEC.md §7.5b`. The short version:

- **f16 recommended** → nothing to configure. `ollama pull`, `ollama run`, then
  `/set parameter num_ctx <n>` in the REPL. Identical on every platform, so the OS selector hides.
- **q8_0 / q4_0 recommended** → the server must be restarted with
  `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=<type>` and `OLLAMA_CONTEXT_LENGTH=<n>`. The OS
  selector appears because the mechanism differs per platform.
- `coder.html` never emits setup commands — it states the assumption and links to the fit checker,
  so there is one copy to keep correct.
